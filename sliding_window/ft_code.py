import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Import directly from your zoo to guarantee weight compatibility
from models_zoo import build_model

# --- 1. Configuration ---
DATA_PATH = '/home/parsa/Repos/foot_train/3_person_dataset/3_person_dataset/angle_test/new_dataset2/final38-1-test/0/all_all'
PRETRAINED_MODEL_PATH = './outputs_sw/best_model_TCN_PHASE1.pth' 
MODEL_TYPE = 'TCN'
WINDOW_SIZE = 60
STEP_SIZE = 5
SUBJECTS = ['A', 'B', 'C', 'D', 'E'] # 'J', 'M', 'P']
CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8']

ARGS = {
    'input_dim': 40,   # Phase 1: 20 XY coords + 20 vels
    'num_classes': 8,
    'hidden_dim': 32,  # Match ablation study
    'lr': 1e-4,  
    'epochs': 15,
    'batch_size': 32,
    'freeze_tcn': False  # False for Full Fine-Tuning, True to freeze temporal convs
}

# --- 2. Dataset & Feature Processing (PHASE 1) ---
class FineTuneDataset(Dataset):
    def __init__(self, file_tuples, window_size=WINDOW_SIZE, step=STEP_SIZE, augment=False):
        self.samples = []
        self.labels = []
        self.video_ids = []
        self.views = []
        self.augment = augment
        
        for vid_id, (file_path, class_id, view, subject) in enumerate(file_tuples):
            label = int(class_id) - 1
            data = self._process_features(file_path)
            
            if data is None:
                continue
                
            if len(data) < window_size:
                pad_len = window_size - len(data)
                data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                
            for start in range(0, len(data) - window_size + 1, step):
                window = data[start:start + window_size]
                self.samples.append(window)
                self.labels.append(label)
                self.video_ids.append(vid_id)
                self.views.append(view)

    def _process_features(self, file_path):
        try:
            df = pd.read_csv(file_path, header=0)
            
            POS_COLS = [
                'L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z',
                'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z',
                'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z',
                'L_Heel_x', 'L_Heel_y', 'L_Heel_z', 'R_Heel_x', 'R_Heel_y', 'R_Heel_z',
                'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z',
                'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z'
            ]
            
            pos_df = df.reindex(columns=POS_COLS, fill_value=0).apply(pd.to_numeric, errors='coerce').fillna(0)

            # PHASE 1: Keep only the 20 X,Y pairs (skip Z)
            xy_cols = []
            for joint in ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Heel', 'R_Heel', 'L_Toe', 'R_Toe']:
                xy_cols += [f'{joint}_x', f'{joint}_y']

            xy_data = pos_df.reindex(columns=xy_cols, fill_value=0).values.astype(np.float32)

            # Add temporal velocities
            if len(xy_data) > 1:
                vel_data = np.diff(xy_data, axis=0)
                vel_data = np.vstack((np.zeros((1, xy_data.shape[1]), dtype=np.float32), vel_data))
            else:
                vel_data = np.zeros_like(xy_data)

            # Total input dim: 20 (pos) + 20 (vel) = 40
            return np.hstack((xy_data, vel_data)).astype(np.float32)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx].copy()
        label = self.labels[idx]
        vid_id = self.video_ids[idx]
        view = self.views[idx]

        if self.augment:
            if np.random.random() < 0.5:  
                # Apply jitter only to the 20 positional coordinates, not the velocities
                jitter = np.random.normal(0, 0.010, (data.shape[0], 20)).astype(np.float32)
                data[:, :20] += jitter 
            if np.random.random() < 0.2:
                for _ in range(np.random.randint(1, 3)):
                    data[np.random.randint(0, len(data) - 1), :20] = 0.0 

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), vid_id, view

# --- 3. Helpers & Evaluation ---
def print_section(title, y_true, y_pred):
    if len(y_true) == 0: return
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc * 100:.2f}% ({sum(np.array(y_true) == np.array(y_pred))} / {len(y_true)} correct)")
    print("Confusion Matrix:")
    print(cm)

def evaluate_video_level(model, loader, device):
    model.eval()
    vid_probs = {}
    vid_truths = {}
    vid_views = {}

    with torch.no_grad():
        for X_batch, y_batch, vid_ids, views in loader:
            X_batch = X_batch.to(device)
            # models_zoo outputs only logits, not a tuple
            logits = model(X_batch, lengths=None)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(vid_ids)):
                v_id = vid_ids[i].item()
                if v_id not in vid_probs:
                    vid_probs[v_id] = []
                    vid_truths[v_id] = y_batch[i].item()
                    vid_views[v_id] = views[i]
                vid_probs[v_id].append(probs[i])

    y_true, y_pred, all_views = [], [], []
    for v_id in vid_probs.keys():
        avg_prob = np.mean(vid_probs[v_id], axis=0)
        pred_class = np.argmax(avg_prob)
        y_pred.append(pred_class)
        y_true.append(vid_truths[v_id])
        all_views.append(vid_views[v_id])

    return np.array(y_true), np.array(y_pred), np.array(all_views)

# --- 4. Main Training Loop ---
def run_loso_finetuning():
    print(f"\n{'='*55}")
    print(f"🚀 INITIALIZING PYTORCH LOSO FINE-TUNING (TCN + PHASE 1)")
    print(f"{'='*55}\n")
    
    all_files = glob.glob(os.path.join(DATA_PATH, "**", "*.csv"), recursive=True)
    parsed_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path).replace('_', '-')
        parts = filename.split('-')
        if len(parts) < 4: continue
        
        subject = parts[0].upper()
        view = parts[1].lower() 
        class_id = parts[2]
        if subject in SUBJECTS:
            parsed_files.append((file_path, class_id, view, subject))

    print(f"Parsed {len(parsed_files)} valid real-world files.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master_y_true, master_y_pred = [], []
    master_y_true_f, master_y_pred_f = [], []
    master_y_true_s, master_y_pred_s = [], []

    for test_subject in SUBJECTS:
        print(f"\n--- FOLD: Testing on Subject {test_subject} ---")
        train_files = [f for f in parsed_files if f[3] != test_subject]
        test_files = [f for f in parsed_files if f[3] == test_subject]
        
        if not train_files or not test_files:
            continue

        train_dataset = FineTuneDataset(train_files, augment=True)
        test_dataset = FineTuneDataset(test_files, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=ARGS['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=ARGS['batch_size'], shuffle=False)

        # Build TCN from models_zoo
        model = build_model(
            arch='TCN', 
            input_dim=ARGS['input_dim'], 
            hidden_dim=ARGS['hidden_dim'], 
            num_classes=ARGS['num_classes']
        ).to(device)
        
        if os.path.exists(PRETRAINED_MODEL_PATH):
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
            print(f"✅ Loaded pretrained weights from {PRETRAINED_MODEL_PATH}")
        else:
            print(f"⚠️ Pretrained weights not found at {PRETRAINED_MODEL_PATH}. Training from scratch!")

        # Freezing logic targeting the TCN blocks
        if ARGS['freeze_tcn']:
            print("❄️ Freezing TCN residual blocks...")
            for param in model.tcn.parameters():
                param.requires_grad = False
        else:
            print("🔥 Full fine-tuning enabled...")

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ARGS['lr'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        print(f"Fine-tuning on {len(train_dataset)} windows for {ARGS['epochs']} epochs...")
        for epoch in range(ARGS['epochs']):
            model.train()
            train_loss = 0
            for X_batch, y_batch, _, _ in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                # models_zoo outputs only logits
                logits = model(X_batch, lengths=None)
                loss = criterion(logits, y_batch)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        torch.save(model.state_dict(), f'fine_tuned_tcn_fold_{test_subject}_NoiseLess.pth')
        
        y_true, y_pred, views = evaluate_video_level(model, test_loader, device)
        fold_acc = accuracy_score(y_true, y_pred) * 100
        print(f"✅ Fold {test_subject} Video-Level Accuracy: {fold_acc:.2f}%")

        master_y_true.extend(y_true)
        master_y_pred.extend(y_pred)
        
        for t, p, v in zip(y_true, y_pred, views):
            if v == 'f':
                master_y_true_f.append(t)
                master_y_pred_f.append(p)
            elif v == 's':
                master_y_true_s.append(t)
                master_y_pred_s.append(p)

    print("\n\n" + "="*50)
    print(" 🏆 FINAL PYTORCH 5-FOLD LOSO REPORT (TCN + PHASE 1)")
    print("="*50)
    
    master_y_true = np.array(master_y_true)
    master_y_pred = np.array(master_y_pred)
    
    print_section("ALL VIEWS (Combined)", master_y_true, master_y_pred)
    print_section("FRONT VIEWS ONLY ('f')", master_y_true_f, master_y_pred_f)
    print_section("SIDE VIEWS ONLY ('s')", master_y_true_s, master_y_pred_s)

    print("\n=== INDIVIDUAL GESTURE ACCURACY ===")
    for c in range(8):
        mask = (master_y_true == c)
        if np.sum(mask) > 0:
            acc = accuracy_score(master_y_true[mask], master_y_pred[mask])
            print(f"Gesture {c+1}: {acc * 100:.2f}% ({sum(master_y_true[mask] == master_y_pred[mask])} / {sum(mask)})")

if __name__ == "__main__":
    run_loso_finetuning()
