# train.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DIV2KDataset
from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.edsr import EDSR
import math
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01, min_psnr_improvement=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.min_psnr_improvement = min_psnr_improvement
        self.counter = 0
        self.best_loss = None
        self.best_psnr = None
        self.early_stop = False
        
    def __call__(self, loss, psnr):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_psnr = psnr
        elif (loss > self.best_loss - self.min_delta) and (psnr < self.best_psnr + self.min_psnr_improvement):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(loss, self.best_loss)
            self.best_psnr = max(psnr, self.best_psnr)
            self.counter = 0

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def train_model(model_name, train_loader, val_loader, device, num_epochs=100):
    # Initialize model
    if model_name == 'srcnn':
        model = SRCNN()
    elif model_name == 'vdsr':
        model = VDSR()
    else:
        model = EDSR()
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.00001, min_psnr_improvement=0.1)
    best_psnr = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, (lr_img, hr_img) in enumerate(train_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            output = model(lr_img)
            loss = criterion(output, hr_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_psnr = 0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                output = model(lr_img)
                val_psnr += calculate_psnr(output, hr_img)
        
        val_psnr /= len(val_loader)
        print(f'Epoch: {epoch}, Average Loss: {avg_train_loss:.6f}, Average PSNR: {val_psnr:.2f}dB')
        
        # Early stopping check
        early_stopping(avg_train_loss, val_psnr)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
            print(f'Saved new best model with PSNR: {best_psnr:.2f}dB')

def main():
    # Setup
    device = torch.device('cpu')
    
    # Data paths
    train_hr_dir = 'data/DIV2K_train_HR/DIV2K_train_HR/'
    train_lr_dir = 'data/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4'
    val_hr_dir = 'data/DIV2K_valid_HR/DIV2K_valid_HR'
    val_lr_dir = 'data/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4'
    
    # Create datasets
    train_dataset = DIV2KDataset(train_hr_dir, train_lr_dir, patch_size=48)
    val_dataset = DIV2KDataset(val_hr_dir, val_lr_dir, patch_size=48)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train models
    models = ['edsr']
    for model_name in models:
        print(f'Training {model_name.upper()}...')
        train_model(model_name, train_loader, val_loader, device)

if __name__ == '__main__':
    main()