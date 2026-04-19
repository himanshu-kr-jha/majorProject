import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import your classes from the training script
# (Assuming your training script is named 'train_gait.py')
from train import TransformerAutoencoder, build_index_map, GaitSequenceDataset, ssim_loss_sequence

# CONFIG
CASIA_B_DIR = r"C:\Users\VARUN\Downloads\Casia_b"
MODEL_PATH = "best_transformer_gait.pth"
SEQ_LEN = 15
IMAGE_SIZE = (64, 64)
BATCH = 16

def get_error_distribution(model, loader, device, description):
    errors = []
    mse_fn = nn.MSELoss(reduction='none')
    
    loop = tqdm(loader, desc=description, leave=True)
    
    with torch.no_grad():
        for batch in loop:
            batch = batch.to(device)
            recon = model(batch)
            
            # --- FIX: CLEAN THE BACKGROUND ---
            # Any pixel less than 10% brightness is forced to 0
            recon[recon < 0.1] = 0.0 
            # ---------------------------------

            # Calculate error per sequence
            mse = mse_fn(recon, batch).mean(dim=[1, 2, 3, 4]) 
            
            ssim_scores = []
            for i in range(batch.size(0)):
                s = ssim_loss_sequence(recon[i:i+1], batch[i:i+1])
                ssim_scores.append(s)
            ssim_scores = torch.tensor(ssim_scores).to(device)

            # Combined Anomaly Score
            total_loss = 0.3 * mse + 0.7 * ssim_scores
            errors.extend(total_loss.cpu().numpy())
            
    return np.array(errors)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = TransformerAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 2. Prepare Data
    # Normal Data (Validation set - 'nm')
    print("Loading Normal (NM) data...")
    nm_index = build_index_map(CASIA_B_DIR, SEQ_LEN, step=15, prefix="nm-") # Bigger step for eval speed
    nm_loader = DataLoader(GaitSequenceDataset(nm_index, SEQ_LEN, IMAGE_SIZE), batch_size=BATCH)
    
    # Abnormal Data (Bag 'bg' and Coat 'cl')
    print("Loading Abnormal (BG/CL) data...")
    abnormal_index = build_index_map(CASIA_B_DIR, SEQ_LEN, step=15, prefix="bg-") 
    abnormal_index += build_index_map(CASIA_B_DIR, SEQ_LEN, step=15, prefix="cl-")
    abnormal_loader = DataLoader(GaitSequenceDataset(abnormal_index, SEQ_LEN, IMAGE_SIZE), batch_size=BATCH)

    # 3. Get Errors
    nm_errors = get_error_distribution(model, nm_loader, device)
    ab_errors = get_error_distribution(model, abnormal_loader, device)
    
    # 4. Statistics
    print(f"\n--- RESULTS ---")
    print(f"Normal Error (Mean): {np.mean(nm_errors):.4f} +/- {np.std(nm_errors):.4f}")
    print(f"Abnormal Error (Mean): {np.mean(ab_errors):.4f} +/- {np.std(ab_errors):.4f}")
    
    # 5. Suggest Threshold
    # Rule of thumb: Mean_Normal + 2 * Std_Normal
    suggested_threshold = np.mean(nm_errors) + 2 * np.std(nm_errors)
    print(f"\n>>> SUGGESTED THRESHOLD: {suggested_threshold:.4f}")
    
    # 6. Accuracy at this threshold
    tp = np.sum(ab_errors > suggested_threshold)
    tn = np.sum(nm_errors <= suggested_threshold)
    acc = (tp + tn) / (len(ab_errors) + len(nm_errors))
    print(f"Estimated Accuracy at this threshold: {acc*100:.2f}%")

    # Optional: Plot histogram
    plt.hist(nm_errors, bins=50, alpha=0.5, label='Normal')
    plt.hist(ab_errors, bins=50, alpha=0.5, label='Abnormal')
    plt.axvline(suggested_threshold, color='red', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')
    plt.title('Reconstruction Error Distribution')
    plt.show()

if __name__ == "__main__":
    evaluate()