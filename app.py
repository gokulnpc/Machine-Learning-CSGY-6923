import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import time
import io


# Metrics imports
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Model imports
from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.edsr import EDSR

# Cache for loaded models
model_cache = {}

def load_model(model_name):
    """
    Load super-resolution model with optional scale factor
    
    Args:
        model_name (str): Name of the model (SRCNN, VDSR, EDSR)
        scale_factor (int): Upscaling factor (2, 3, or 4)
    
    Returns:
        torch.nn.Module: Loaded model
    """
    try:
        # Check if model is already in the cache
        if model_name in model_cache:
            return model_cache[model_name]
        
        if model_name == 'SRCNN':
            model = SRCNN()
        elif model_name == 'VDSR':
            model = VDSR()
        else:
            model = EDSR()
            
        # Load pre-trained weights if available
        weight_path = f'checkpoints/{model_name.lower()}_best.pth'
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        else:
            st.warning(f"No pre-trained weights found for the {model_name} model. Using randomly initialized weights.")
        
        model.eval()
        
        # Cache the loaded model
        model_cache[model_name] = model
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        return None

def process_image(image, model):
    # Convert to YCbCr and extract Y channel
    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    
    # Transform Y channel
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    input_tensor = transform(y).unsqueeze(0)
    
    # Process through model
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process output
    output = output.squeeze().clamp(0, 1).numpy()
    output_y = Image.fromarray((output * 255).astype(np.uint8))
    
    # Merge channels back
    output_ycbcr = Image.merge('YCbCr', [output_y, cb, cr])
    output_rgb = output_ycbcr.convert('RGB')
    
    return output_rgb

def calculate_image_metrics(original, enhanced):
    """
    Calculate image quality metrics
    
    Args:
        original (np.ndarray): Original image
        enhanced (np.ndarray): Enhanced image
    
    Returns:
        dict: Quality metrics
    """
    try:
        # Ensure images are the same size
        min_height = min(original.shape[0], enhanced.shape[0])
        min_width = min(original.shape[1], enhanced.shape[1])
        
        # Resize images to the smallest common size
        original = original[:min_height, :min_width]
        enhanced = enhanced[:min_height, :min_width]
        
        # Calculate SSIM with an explicit window size
        win_size = min(7, min(min_height, min_width))
        if win_size % 2 == 0:
            win_size -= 1  # Ensure odd window size
        
        return {
            'PSNR': psnr(original, enhanced),
            'SSIM': ssim(original, enhanced, multichannel=True, win_size=win_size, channel_axis=-1)
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {'PSNR': 0, 'SSIM': 0}

def main():
    st.set_page_config(
        page_title="Super Resolution Comparison",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🚀 Super Resolution Model Comparison")
    st.write("Upload a low-resolution image and compare different super-resolution models.")
    


    
    # File Upload
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a low-resolution image for enhancement"
    )
    
    if uploaded_file is not None:
        # Load input image
        input_image = Image.open(uploaded_file)
        input_array = np.array(input_image)
        
        st.subheader("📸 Original Image")
        st.image(input_image, caption="Low-Resolution Input", use_column_width=True)
        
        # Model Names
        model_names = ['SRCNN', 'VDSR', 'EDSR']
        
        # Performance and Quality Storage
        processing_times = {}
        quality_metrics = {}
        enhanced_images = {}
        
        # Process images
        columns = st.columns(len(model_names))
        for i, model_name in enumerate(model_names):
            with columns[i]:
                st.subheader(f"{model_name} Model")
                
                # Load model
                model = load_model(model_name)
                
                if model:
                    # Time the processing
                    start_time = time.time()
                    enhanced_image = process_image(input_image, model)
                    processing_time = time.time() - start_time
                    
                    if enhanced_image:
                        # Display enhanced image
                        st.image(enhanced_image, caption=f"{model_name} Output", use_column_width=True)
                        
                        # Calculate metrics
                        enhanced_array = np.array(enhanced_image)
                        metrics = calculate_image_metrics(input_array, enhanced_array)
                        
                        # Store results
                        processing_times[model_name] = processing_time
                        quality_metrics[model_name] = metrics
                        enhanced_images[model_name] = enhanced_image
        
        # Performance Metrics Section
        st.subheader("📊 Performance Metrics")
        metric_cols = st.columns(len(model_names))
        
        for i, (model, time_val) in enumerate(processing_times.items()):
            with metric_cols[i]:
                st.metric(f"{model} Processing Time", f"{time_val:.4f} seconds")
        
        # Quality Metrics Section
        st.subheader("🔍 Image Quality Assessment")
        quality_cols = st.columns(len(model_names))
        
        for i, (model, metrics) in enumerate(quality_metrics.items()):
            with quality_cols[i]:
                st.metric(f"{model} PSNR", f"{metrics['PSNR']:.2f} dB")
                st.metric(f"{model} SSIM", f"{metrics['SSIM']:.4f}")
        
        # Download Section
        st.subheader("💾 Download Enhanced Images")
        download_cols = st.columns(len(model_names))
        
        for i, (model, image) in enumerate(enhanced_images.items()):
            with download_cols[i]:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                st.download_button(
                    label=f"Download {model} Image",
                    data=buffered.getvalue(),
                    file_name=f"{model}_enhanced.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()