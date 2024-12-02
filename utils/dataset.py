# utils/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=96, upscale_factor=4):
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        
        # LR transform
        self.lr_transform = transforms.Compose([
            transforms.Resize((patch_size//upscale_factor, patch_size//upscale_factor), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        # HR transform
        self.hr_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        # Upscale LR images to match HR size for SRCNN
        self.lr_upscale = transforms.Compose([
            transforms.Resize((patch_size, patch_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, idx):
        # Load images and convert to YCbCr
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert('YCbCr')
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert('YCbCr')
        
        # Extract Y channel
        hr_y, _, _ = hr_img.split()
        lr_y, _, _ = lr_img.split()
        
        # For SRCNN, we need to upscale LR images first
        lr_y_upscaled = self.lr_upscale(lr_y)
        hr_y_tensor = self.hr_transform(hr_y)
        
        return lr_y_upscaled, hr_y_tensor
        
    def __len__(self):
        return len(self.hr_files)