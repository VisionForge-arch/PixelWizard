import os
import torch
import torch.utils.data as data
import torchvision
import json
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
import random


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators = [] if operators is None else operators
        
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


class LoadImage(DataProcessingOperator):
    """加载图像的操作符"""
    def __init__(self, image_processor=lambda x: x):
        self.image_processor = image_processor
        
    def __call__(self, data: str):
        image = Image.open(data).convert("RGB")
        image = self.image_processor(image)
        return image


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


class ShiftPadding(DataProcessingOperator):
    """保留原有的ShiftPadding功能"""
    def __init__(self, size=512):
        self.size = size
    
    def __call__(self, img: Image.Image):
        w, h = img.size
        w_new = self.size
        h_new = self.size
        p = Image.new('RGB', (w_new, h_new), (0, 0, 0))
        p.paste(img, (w_new-w, h_new-h, w_new, h_new))
        p.paste(img, (0, h_new-h, w, h_new))
        p.paste(img, (w_new-w, 0, w_new, h))
        p.paste(img, (0, 0, w, h))
        return p


class RandomCrop(DataProcessingOperator):
    """随机裁剪操作符"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, data: Image.Image):
        i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(self.size, self.size))
        return F.crop(data, i, j, h, w)


class DataAugmentation(DataProcessingOperator):
    """数据增强操作符"""
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, data: Image.Image):
        if self.mode == 'train':
            # 随机水平翻转
            if random.randint(0, 1):
                data = F.hflip(data)
            # 随机旋转
            rand_rot = random.randint(0, 3)
            if rand_rot:
                data = F.rotate(data, 90 * rand_rot)
        return data


class ToTensor(DataProcessingOperator):
    """转换为Tensor的操作符"""
    def __init__(self, normalize=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __call__(self, data: Image.Image):
        return self.transform(data)


# 为了兼容旧代码，保留UIEBD_Dataset的包装类
class UIEBD_Dataset_Wrapper(torch.utils.data.Dataset):
    """
    包装UnifiedImageDataset以兼容原有的UIEBD_Dataset接口
    """
    def __init__(self, input_root, gt_root, mode='train', size=224, format='.png'):
        # 创建metadata
        self.input_root = input_root
        self.gt_root = gt_root
        self.mode = mode
        self.size = size
        self.format = format
        
        # 获取所有图像文件
        haze_imgs_dir = os.listdir(input_root)
        self.haze_imgs = [img for img in haze_imgs_dir]
        
        # 创建操作符
        self.input_operator = self._create_operator()
        self.target_operator = self._create_operator()
        
        random.seed(1143)
    
    def _create_operator(self):
        """创建图像处理操作符"""
        # 基本加载
        load_op = LoadImage()
        
        # 根据模式添加处理步骤
        if self.mode == 'train':
            pipeline = load_op >> ShiftPadding(size=self.size) >> DataAugmentation(mode='train')
        else:
            pipeline = load_op
        
        return pipeline
    
    def __len__(self):
        return len(self.haze_imgs)
    
    def __getitem__(self, index):
        img_name = self.haze_imgs[index]
        haze_path = os.path.join(self.input_root, img_name)
        clear_path = os.path.join(self.gt_root, img_name)
        
        # 加载图像
        haze = Image.open(haze_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        
        # 训练模式的处理
        if self.mode == 'train':
            haze = ShiftPadding(self.size)(haze)
            clear = ShiftPadding(self.size)(clear)
            
            # 同步随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = F.crop(haze, i, j, h, w)
            clear = F.crop(clear, i, j, h, w)
            
            # 同步数据增强z
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            if rand_hor:
                haze = F.hflip(haze)
                clear = F.hflip(clear)
            if rand_rot:
                haze = F.rotate(haze, 90 * rand_rot)
                clear = F.rotate(clear, 90 * rand_rot)
        else:
            # 推理模式：也需要resize到指定尺寸
            haze = haze.resize((self.size, self.size), Image.Resampling.BILINEAR)
            clear = clear.resize((self.size, self.size), Image.Resampling.BILINEAR)
        
        # 转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        haze = transform(haze)
        clear = transform(clear)
        
        return haze, clear, img_name


def cycle(dl):
    """无限循环dataloader"""
    while True:
        for data in dl:
            yield data


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, 
                       default='/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/train/raw-890')
    parser.add_argument('--gt_root', type=str, 
                       default='/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/train/reference-890')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--use_wrapper', default=True, type=bool, help='使用包装类（兼容旧接口）')
    parser.add_argument('--metadata_path', type=str, default=None, help='JSON元数据文件路径')
    args = parser.parse_args()
    
        # 使用包装类（兼容旧代码）
    print("使用UIEBD_Dataset_Wrapper（兼容模式）")
    dataset = UIEBD_Dataset_Wrapper(args.input_root, args.gt_root, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for haze, clear, id in dataloader:
        print(f"Haze shape: {haze.shape}, Clear shape: {clear.shape}")
        print(f"IDs: {id}")
        print(haze.min(), haze.max())
        print(clear.min(), clear.max())
        break
