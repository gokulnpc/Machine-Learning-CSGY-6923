# metrics.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.edsr import EDSR
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def process_image(model, lr_img):
    with torch.no_grad():
        # Convert to YCbCr and extract Y channel
        ycbcr = lr_img.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        
        # Transform Y channel
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(y).unsqueeze(0)
        
        # Process through model
        output = model(input_tensor)
        
        # Post-process output
        output = output.squeeze().clamp(0, 1).numpy()
        output_y = Image.fromarray((output * 255).astype(np.uint8))
        
        # Merge channels back
        output_ycbcr = Image.merge('YCbCr', [output_y, cb, cr])
        return output_ycbcr.convert('RGB')

def calculate_ssim(img1, img2):
    # Move channel axis to the end for SSIM calculation
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2, win_size=7)

def evaluate_models(test_image_path):
    # Load models
    models = {
        'SRCNN': SRCNN(),
        'VDSR': VDSR(),
        'EDSR': EDSR()
    }
    
    # Load weights
    for name, model in models.items():
        model.load_state_dict(torch.load(f'checkpoints/{name.lower()}_best.pth', weights_only=True))
        model.eval()
    
    # Load test image
    lr_img = Image.open(test_image_path)
    hr_img = Image.open(test_image_path)  # Using same image as reference
    
    # Results dictionary
    results = {model_name: {} for model_name in models.keys()}
    
    # Process image with each model and calculate metrics
    for name, model in models.items():
        # Generate SR image
        sr_img = process_image(model, lr_img)
        
        # Convert images to tensors for metric calculation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Ensure minimum size for SSIM
            transforms.ToTensor()
        ])
        
        sr_tensor = transform(sr_img)
        hr_tensor = transform(hr_img)
        
        # Calculate metrics
        results[name]['PSNR'] = calculate_psnr(sr_tensor, hr_tensor)
        results[name]['SSIM'] = calculate_ssim(sr_tensor, hr_tensor)
        
        # Save output images
        sr_img.save(f'results/{name.lower()}_output.png')
    
    # Display results
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"{'Model':<10} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<10} {metrics['PSNR']:<15.2f} {metrics['SSIM']:<15.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # PSNR comparison
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), [m['PSNR'] for m in results.values()])
    plt.title('PSNR Comparison')
    plt.ylabel('PSNR (dB)')
    
    # SSIM comparison
    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), [m['SSIM'] for m in results.values()])
    plt.title('SSIM Comparison')
    plt.ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    test_image_path = r"data\DIV2K_train_LR_bicubic_X4\DIV2K_train_LR_bicubic\X4\0001x4.png"  # Replace with your test image path
    evaluate_models(test_image_path)