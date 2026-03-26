"""
CNN-LSTM Attention Gesture Classifier (LOSO Training)
Trains a temporal attention-based neural network on extracted foot/leg CSV features.

USAGE (Default):
  python lstm_attention_loso.py

USAGE (Customized):
  python lstm_attention_loso.py \
      --data_path /path/to/csvs-final \
      --val_prefix javad_ \
      --epochs 100 \
      --batch_size 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
import glob
import argparse
from sklearn.metrics import confusion_matrix


def parse_args():
    """Parse command line arguments and hyperparameters."""
    parser = argparse.ArgumentParser(description="Train the CNN-LSTM Temporal Attention Gesture Model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="/home/parsa/Repos/foot_train/3_person_dataset/3_person_dataset/csvs-final", help="Parent directory containing class folders (1, 2, 3...)")
    parser.add_argument("--model_save_path", type=str, default="./best_gesture_model_LOSO_lstm_8.12.pth", help="Path to save the best trained weights")
    parser.add_argument("--val_prefix", type=str, default="parsa_", help="Filename prefix to hold out for validation (Leave-One-Subject-Out)")
    
    # Model architecture arguments
    parser.add_argument("--input_dim", type=int, default=24, help="Number of feature columns per frame")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of gesture classes")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for LSTM and Attention")
    parser.add_argument("--model_type", type=str, default="cnn_lstm", help="Model type identifier")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=233, help="Fixed sequence length for padding/truncating")
    
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # Multiply by 2 because our LSTM is bidirectional
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_outputs):
        # Calculate an attention score for every single frame
        attn_weights = self.attention(lstm_outputs)
        
        # Convert scores to percentages (softmax) so they sum to 1.0
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Multiply the frames by their weights and sum them up chronologically
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        
        return context_vector, attn_weights


class GestureModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, num_classes=4, model_type='cnn-lstm'):
        super(GestureModel, self).__init__()
        self.model_type = model_type
        
        # Spatial CNN Block (Smooths the X,Y,Z coordinates)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        rnn_input_dim = 64
        
        # Temporal RNN Block (num_layers=1 to prevent memorizing exact bone lengths)
        self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Temporal Attention Brain
        self.attention = TemporalAttention(hidden_dim)
        
        # Classification Head
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # 1. CNN requires channels first: (Batch, 24, Seq_Len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2) 
        
        # 2. LSTM processes the timeline
        out, _ = self.rnn(x) 
            
        # 3. Temporal Attention read of the chronological sequence
        out, attn_weights = self.attention(out) 
        
        # 4. Final Prediction
        out = self.fc_dropout(out)
        return self.fc(out)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET HANDLING & AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    def __init__(self, file_list, class_to_idx, sequence_length=233, augment=False):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.sequence_length = sequence_length
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, cls_name = self.file_list[idx]
        label = self.class_to_idx[cls_name]
        
        try:
            df = pd.read_csv(file_path, header=0)
            data = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            data = np.zeros((self.sequence_length, 24))

        data = data.astype(np.float32)

        # --- ANCHOR POSE NORMALIZATION ---
        # Find resting pose (average of first 5 frames) to smooth out starting jitter.
        # Subtract starting pose so all gestures mathematically start at 0.
        num_anchor_frames = min(5, max(1, len(data)))
        anchor_pose = np.mean(data[:num_anchor_frames], axis=0)
        data = data - anchor_pose

        if self.augment:
            # 1. Random Frame Drop (Simulates slight FPS drops)
            if random.random() < 0.5 and len(data) > 30:
                drop_mask = np.random.rand(len(data)) > 0.1 
                data = data[drop_mask]

            # 2. Additive Noise (Tiny decimal variations)
            if random.random() < 0.5: 
                noise = np.random.normal(0, 0.005, data.shape).astype(np.float32)
                data += noise
            
            # 3. Time Shift
            if random.random() < 0.5:
                shift = np.random.randint(-20, 20) 
                data = np.roll(data, shift, axis=0)
                if shift > 0: data[:shift, :] = 0
                elif shift < 0: data[shift:, :] = 0

            # 4. Amplitude Scaling
            if random.random() < 0.5:
                amp_scale = np.random.uniform(low=0.8, high=1.2)
                data = data * amp_scale

        # --- SEQUENCE PADDING / TRUNCATING ---
        if len(data) < self.sequence_length:
            pad_len = self.sequence_length - len(data)
            # mode='edge' repeats the final posture instead of yo-yoing the gesture.
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='edge')
        else:
            data = data[:self.sequence_length, :]
            
        return torch.tensor(data), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args):
    # Dynamically generate class list based on number of classes (e.g., ['1', '2', ...])
    classes = [str(i) for i in range(1, args.num_classes + 1)]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    all_files = []
    for cls in classes:
        path = os.path.join(args.data_path, cls, "*.csv") 
        files = glob.glob(path)
        for f in files:
            all_files.append((f, cls))
    
    # Dataset Split Logic (LOSO)
    train_files = []
    val_files = []
    
    for f, cls in all_files:
        filename = os.path.basename(f)
        if filename.startswith(args.val_prefix):
            val_files.append((f, cls))
        else:
            train_files.append((f, cls))
            
    print("\n" + "="*65)
    print(" CNN-LSTM Gesture Classifier Pipeline")
    print(f" Dataset Path    : {args.data_path}")
    print(f" Holdout Subject : '{args.val_prefix}' (Validation)")
    print(f" Classes Config  : {args.num_classes}")
    print(f" Training Files  : {len(train_files)}")
    print(f" Validation Files: {len(val_files)}")
    print("="*65 + "\n")

    train_dataset = GestureDataset(train_files, class_to_idx, args.seq_len, augment=True)
    val_dataset = GestureDataset(val_files, class_to_idx, args.seq_len, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureModel(args.input_dim, args.hidden_dim, args.num_classes, args.model_type)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} - Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Acc: {val_acc:.1f}%")

        # Print the confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
             print(f"\nConfusion Matrix ({args.val_prefix} Validation Data):")
             print(confusion_matrix(all_labels, all_preds))
             print("-" * 50 + "\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            
    print(f"\n[OK] Training Complete. Best model saved to: {args.model_save_path}\n")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
