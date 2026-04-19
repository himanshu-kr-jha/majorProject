import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

##############################################################
# CONFIGURATION
##############################################################

TEST_DIR = r"C:\Users\VARUN\Downloads\ucf\testing"
MODEL_PATH = "best_transformer_gait.pth"
OUTPUT_CSV = "ucf_results.csv"
THRESHOLD = 0.055  # recalibrated: max normal score=0.0513, first clean abnormal=0.060
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################
# MODEL ARCHITECTURE
##############################################################

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.fc = nn.Linear(512*2*2, latent_dim)
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512*2*2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 2, 2)
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    def forward(self, x):
        return x + self.pos[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, seq_len=15):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=latent_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.pos = PositionalEncoding(latent_dim, seq_len)
    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = torch.stack([self.encoder(x[:, t]) for t in range(T)], dim=1)
        feats = self.pos(feats)
        feats = self.transformer(feats)
        outs = [self.decoder(feats[:, t]) for t in range(T)]
        return torch.stack(outs, dim=1)

##############################################################
# PREPROCESSING
##############################################################

def preprocess_ucf_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    frames = []
    
    # SAFETY: Stop scanning if we check 300 frames and find nothing useful
    MAX_SCAN = 300 
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or counter > MAX_SCAN: 
            break
        
        frame = cv2.resize(frame, (320, 240)) 
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        final_frame = cv2.resize(fgmask, (64, 64))
        
        # Only keep frame if it has movement (pixels > 5)
        if np.mean(final_frame) > 5: 
            frames.append(final_frame)
            
        counter += 1
        if len(frames) >= 30: 
            break

    cap.release()
    
    if len(frames) < 15: return None

    mid = len(frames) // 2
    clip = frames[mid-7 : mid+8]
    clip = np.array(clip).astype("float32") / 255.0
    clip = torch.from_numpy(clip).unsqueeze(0).unsqueeze(2) 
    return clip

##############################################################
# MAIN EXECUTION
##############################################################

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = TransformerAutoencoder().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    extensions = ['*.mp4', '*.avi', '*.mpg']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(TEST_DIR, "**", ext), recursive=True))
    
    print(f"Found {len(video_files)} videos.")

    results = []

    # --- PROGRESS BAR IS HERE ---
    pbar = tqdm(video_files)
    
    for video_path in pbar:
        video_name = os.path.basename(video_path)
        
        # This updates the bar text so you know which file is processing
        pbar.set_description(f"Scanning: {video_name[:15]}...") 

        clip = preprocess_ucf_video(video_path)
        
        if clip is None:
            results.append({"Video": video_name, "Score": 0.0, "Prediction": "Error/Too Short"})
            continue

        clip = clip.to(DEVICE)
        
        with torch.no_grad():
            recon = model(clip)
            mse_val = nn.MSELoss()(recon, clip).item()
            
            score = mse_val 
            pred = "ABNORMAL" if score > THRESHOLD else "NORMAL"
            
            results.append({
                "Video": video_name,
                "Score": round(score, 5),
                "Prediction": pred
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*50)
    print(f"DONE! Results saved to {OUTPUT_CSV}")
    print("="*50)
    print(df.head(10)) 

if __name__ == "__main__":
    main()