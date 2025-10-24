import os
import torch
import torchvision
import json
import decord
from decord import VideoReader, cpu, gpu
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms




class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)
    

class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)
    
    
class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")

class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)
    
    
class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = F.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = F.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image

    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, 
                 frame_processor=lambda x: x, use_gpu=False, gpu_id=0, sampling_strategy="center"):
        """
        Args:
            num_frames: 目标帧数
            time_division_factor: 时间分割因子
            time_division_remainder: 时间分割余数
            frame_processor: 帧处理函数
            use_gpu: 是否使用GPU解码（仅Decord支持）
            gpu_id: GPU设备ID
            sampling_strategy: 采样策略 ("uniform", "random", "sequential")
        """
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.sampling_strategy = sampling_strategy
        
        # 设置 Decord 上下文
        self.ctx = gpu(gpu_id) if self.use_gpu else cpu(0)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 [0,1] 范围的tensor，形状 [C, H, W]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转换到 [-1, 1] 范围
        ])
        
        
    def get_num_frames_decord(self, total_frames):
        """计算实际使用的帧数（Decord版本）"""
        num_frames = min(self.num_frames, total_frames)
        return num_frames
    
    def get_frame_indices(self, total_frames, target_frames):
        """根据采样策略获取帧索引"""
        if self.sampling_strategy == "uniform":
            # 均匀采样
            if total_frames <= target_frames:
                return list(range(total_frames))
            else:
                step = total_frames / target_frames
                return [int(i * step) for i in range(target_frames)]
                
        elif self.sampling_strategy == "random":
            # 随机采样
            indices = np.random.choice(total_frames, target_frames, replace=False)
            return sorted(indices.tolist())
            
        elif self.sampling_strategy == "sequential":
            # 顺序采样（从开头开始）
            return list(range(min(target_frames, total_frames)))
        
        elif self.sampling_strategy == "center":
            # 中心采样
            if total_frames <= target_frames:
                return list(range(total_frames))
            start_index = (total_frames - target_frames) // 2
            return list(range(start_index, start_index + target_frames))
            
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.sampling_strategy}")
        
    def load_video_decord(self, data: str):
        # 创建视频读取器
        vr = VideoReader(data, ctx=self.ctx)
        total_frames = len(vr)
        
        # 计算目标帧数
        target_frames = self.get_num_frames_decord(total_frames)
        
        # 获取帧索引
        frame_indices = self.get_frame_indices(total_frames, target_frames)
        
        # 批量读取帧（Decord的核心优势）
        if len(frame_indices) == 1:
            frames_array = vr[frame_indices[0]].asnumpy()
            frames_array = frames_array[np.newaxis, ...]  # 添加批次维度
        else:
            frames_array = vr.get_batch(frame_indices).asnumpy()
        
        # 转换为PIL图像并处理
        frames_tensor = []
        for frame_array in frames_array:
            frame = Image.fromarray(frame_array)
            frame = self.frame_processor(frame)
            
            frame_tensor = self.transform(frame)
            frames_tensor.append(frame_tensor)
            
            # 释放PIL图像内存
            del frame
        
        # 释放整个frames_array
        del frames_array
            
        video_tensor = torch.stack(frames_tensor, dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor
        
    def __call__(self, data: str):
        return self.load_video_decord(data)
    
    


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        base_path = None, 
        metadata_path = None,
        repeat = 3,
        data_file_keys = ("clip_id",), 
        main_data_operator=lambda x: x,
    ): 
        self.base_path = base_path
        self.load_metadata(metadata_path)
        self.repeat = repeat
        self.data_file_keys = data_file_keys

        self.main_data_operator = main_data_operator
        
    
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.data = metadata
        
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        use_gpu=False, gpu_id=0, sampling_strategy="center",
    ):
        """
        更新的默认视频操作器，使用 Decord 优化
        """
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                    time_division_remainder=time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    use_gpu=use_gpu,
                    gpu_id=gpu_id,
                    sampling_strategy=sampling_strategy,
                )),
            ])),
        ])
    
    def __len__(self):
        return len(self.data) * self.repeat
    
    def __getitem__(self, index):
        data = self.data[index % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                data[key] = self.main_data_operator(data[key])
        return data
        

def cycle(dl):
    while True:
        for data in dl:
            yield data


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from model_wan import SelfForcingWan
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Dataset_fps24")
    parser.add_argument("--dataset_metadata_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.json")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--data_file_keys", type=str, default=("clip_id",))
    parser.add_argument("--max_pixels", type=int, default=2560*1440)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--time_division_factor", type=int, default=4)
    parser.add_argument("--time_division_remainder", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--config_path", type=str, default="/root/ultrawan/configs/self_forcing_dmd.yaml")
    parser.add_argument("--logdir", type=str, default="/root/ultrawan//logs/self_forcing_dmd")
    
    
    args = parser.parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=1)
    
    for data in tqdm(dataloader):
        print(data.keys())
        print(data['detailed_description'])
        print(data['clip_id'].shape)
        break
    # for data in dataset:
    #     print(data['detailed_description'])
    #     print(data['clip_id'])
    #     break
'''
    
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/hpc2hdd/home/htian395/Wenxue/Self-Forcing-Long/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir


    
    # init model
    from model_wan_trainer import WanModel_Trainer
    model_trainer = WanModel_Trainer(config)
    
    
    
    for data in tqdm(dataloader):
        generator_log_dict = model_trainer.train(data)
        print(generator_log_dict)
        #print(data['detailed_description'])
        #print(data['clip_id'])
        #break
        
'''

