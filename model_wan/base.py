from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline_long import SelfForcingTrainingPipeline
from utils_long.loss import get_denoising_loss
from utils_long.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from peft import LoraConfig, get_peft_model


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        

        
        # if hasattr(args, "denoising_step_list"):
        #     self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        #     if args.warp_denoising_step:
        #         timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        #         self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        
    def _initialize_models(self, args, device):

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
  
        if getattr(args, "use_lora", False):
            print("Using LoRA for training generator.")
            self.generator.model.requires_grad_(False)

            lora_config = LoraConfig(
                r=getattr(args, "lora_r", 64),
                lora_alpha=getattr(args, "lora_alpha", 64),
                target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
                lora_dropout=getattr(args, "lora_dropout", 0.1),
                bias="none",
            )

            self.generator.model = get_peft_model(self.generator.model, lora_config)
            print("Generator trainable parameters:")
            self.generator.model.print_trainable_parameters()

        else:
            self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        simulated_noisy_input = []
        simulated_noise_input = []
        simulated_timestep_input = []

        for timestep in self.denoising_step_list:
            
            # [B, F, C, H, W]
            noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
            
            # [B, F]
            noisy_timestep = timestep * torch.ones(image_or_video_shape[:2], device=self.device, dtype=torch.long)

            # [B, F, C, H, W]
            if timestep != 0:
                noisy_image = self.scheduler.add_noise(
                    clean_latent.flatten(0, 1),
                    noise.flatten(0, 1),
                    noisy_timestep.flatten(0, 1)
                ).unflatten(0, image_or_video_shape[:2])
            else:
                noisy_image = clean_latent # None

            simulated_noisy_input.append(noisy_image)
            simulated_noise_input.append(noise)
            simulated_timestep_input.append(noisy_timestep)
        
        # [B, T, F, C, H, W]
        
        # 堆成 [B,T,F,…] / [B,T,F]
        simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)       # [B,T,F,C,H,W]
        simulated_noise_input = torch.stack(simulated_noise_input, dim=1)       # [B,T,F,C,H,W]
        simulated_timestep_input = torch.stack(simulated_timestep_input, dim=1) # [B,T,F]

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            image_or_video_shape[0],   # B
            image_or_video_shape[1],   # F
            self.num_frame_per_block,
            uniform_timestep=False
        ) # [B, F]

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        gidx = index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(-1, -1, -1, *image_or_video_shape[2:])
        noisy_input = torch.gather(simulated_noisy_input, dim=1, index=gidx.to(self.device)).squeeze(1)  # [B, F, C, H, W]
        noise_used  = torch.gather(simulated_noise_input, dim=1, index=gidx.to(self.device)).squeeze(1)
        timestep = torch.gather(  
            simulated_timestep_input, dim=1,
            index=index[:, None, :].to(simulated_timestep_input.device)
        ).squeeze(1) # [B,F]

        
        training_target = self.scheduler.training_target(clean_latent, noise_used, timestep)

        #timestep = self.denoising_step_list[index].to(self.device) # [B, F]

        flow_pred, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
        ) # [B, F, C, H, W]


        pred_image_or_video = pred_image_or_video.type_as(noisy_input)

        return flow_pred, training_target, timestep
    
    
    

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )
