import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import TransformerAutoencoder, build_index_map, GaitSequenceDataset

# CONFIG
CASIA_B_DIR = r"C:\Users\VARUN\Downloads\Casia_b"
MODEL_PATH = "best_transformer_gait.pth"
SEQ_LEN = 15
IMAGE_SIZE = (64, 64)

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = TransformerAutoencoder().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("Error: Could not load model file.")
        return
    model.eval()
    
    # Get one batch of data
    index = build_index_map(CASIA_B_DIR, SEQ_LEN, step=50, prefix="nm-")
    ds = GaitSequenceDataset(index, SEQ_LEN, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    data = next(iter(loader)).to(device)
    
    # Predict
    with torch.no_grad():
        recon = model(data)
        
    # Plotting
    data = data.cpu().numpy()
    recon = recon.cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    
    for i in range(4):
        # Original (Top Row)
        axes[0, i].imshow(data[i, 7, 0], cmap='gray') # Show middle frame (7th)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Reconstructed (Bottom Row)
        axes[1, i].imshow(recon[i, 7, 0], cmap='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("debug_result.png")
    print("Saved 'debug_result.png'. Open this image and tell me what you see!")
    plt.show()

if __name__ == "__main__":
    visualize()