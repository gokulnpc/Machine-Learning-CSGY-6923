# inference.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.edsr import EDSR

def load_model(model_name):
    if model_name == 'SRCNN':
        model = SRCNN()
    elif model_name == 'VDSR':
        model = VDSR()
    else:
        model = EDSR()
    
    model.load_state_dict(torch.load(f'checkpoints/{model_name.lower()}_best.pth'))
    model.eval()
    return model

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

def main():
    st.title("Super Resolution Model Comparison")
    st.write("Upload a low-resolution image to compare SRCNN, VDSR, and EDSR models")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Load and display input image
        input_image = Image.open(uploaded_file)
        st.subheader("Input Image")
        st.image(input_image, caption="Original Image")
        
        # Process with each model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("SRCNN")
            model = load_model('SRCNN')
            srcnn_output = process_image(input_image, model)
            st.image(srcnn_output, caption="SRCNN Output")
        
        with col2:
            st.subheader("VDSR")
            model = load_model('VDSR')
            vdsr_output = process_image(input_image, model)
            st.image(vdsr_output, caption="VDSR Output")
        
        with col3:
            st.subheader("EDSR")
            model = load_model('EDSR')
            edsr_output = process_image(input_image, model)
            st.image(edsr_output, caption="EDSR Output")

if __name__ == "__main__":
    main()