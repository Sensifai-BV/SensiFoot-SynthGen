"""
MediaPipe Foot Gesture Feature Extractor
Batch processes a folder of videos to extract normalized leg/foot landmarks.

USAGE:
  python mediapipe_feature_extractor.py \
      --input_dir  /path/to/input_videos \
      --output_dir /path/to/save_location \
      --class_id   2
"""

import cv2
import mediapipe as mp
import csv
import os
import glob
import math
import argparse
import itertools
import multiprocessing
import gc
from concurrent.futures import ProcessPoolExecutor

# --- SAFETY SETTING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract MediaPipe foot/leg features from a folder of videos.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Base directory to save the output CSV folder")
    parser.add_argument("--class_id", required=True, type=int, help="Integer Class ID for the dataset")
    return parser.parse_args()


def worker_process_video(args):
    """Worker function for multiprocessing."""
    video_path, class_id, output_dir = args
    filename = os.path.basename(video_path)
    
    print(f"    [>] Start: {filename}")
    
    try:
        processor = FootGestureProcessor()
        processor.process_video(video_path, class_id, output_dir)
        
        del processor
        gc.collect()
        
        return f"    [OK] Done: {filename}"
    except Exception as e:
        return f"    [X] Error in {filename}: {str(e)}"


class FootGestureProcessor:
    """Processes video frames to extract normalized pose landmarks."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.conf = min_detection_confidence
        self.track_conf = min_tracking_confidence
        
        self.target_indices = {
            25: "L_Knee",  26: "R_Knee",
            27: "L_Ankle", 28: "R_Ankle",
            29: "L_Heel",  30: "R_Heel",
            31: "L_Toe",   32: "R_Toe"
        }

    def process_video(self, input_path, class_id, save_dir):
        cap = cv2.VideoCapture(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{base_name}_features.csv")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.conf,
            min_tracking_confidence=self.track_conf
        ) as pose:

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                header = ["frame", "timestamp", "class_id"]
                for name in self.target_indices.values():
                    header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
                writer.writerow(header)

                frame_idx = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # --- NORMALIZATION ---
                        # 1. Calculate Mid-Hip (Translation Anchor)
                        l_hip, r_hip = landmarks[23], landmarks[24]
                        h_mid_x = (l_hip.x + r_hip.x) / 2.0
                        h_mid_y = (l_hip.y + r_hip.y) / 2.0
                        h_mid_z = (l_hip.z + r_hip.z) / 2.0
                        
                        # 2. Calculate Mid-Shoulder (For Scaling)
                        l_sh, r_sh = landmarks[11], landmarks[12]
                        s_mid_x = (l_sh.x + r_sh.x) / 2.0
                        s_mid_y = (l_sh.y + r_sh.y) / 2.0
                        s_mid_z = (l_sh.z + r_sh.z) / 2.0
                        
                        # 3. Calculate Torso Length (Scale Factor)
                        scale = math.sqrt(
                            (s_mid_x - h_mid_x)**2 + 
                            (s_mid_y - h_mid_y)**2 + 
                            (s_mid_z - h_mid_z)**2
                        ) + 1e-6

                        row = [frame_idx, frame_idx / fps, class_id]

                        for idx in sorted(self.target_indices.keys()):
                            lm = landmarks[idx]
                            
                            # Apply Translation and Scale Normalization
                            norm_x = (lm.x - h_mid_x) / scale
                            norm_y = (lm.y - h_mid_y) / scale
                            norm_z = (lm.z - h_mid_z) / scale
                            
                            row.extend([norm_x, norm_y, norm_z])

                        writer.writerow(row)

                    frame_idx += 1

        cap.release()


def batch_process(input_dir, output_dir, class_id):
    """Handles the folder traversal and multiprocessing pool setup."""
    
    # Generate dynamic output folder name (e.g., class_2_myvideos_csvs)
    # os.path.normpath prevents trailing slashes from returning an empty string
    input_folder_name = os.path.basename(os.path.normpath(input_dir))
    final_output_folder = os.path.join(output_dir, f"class_{class_id}_{input_folder_name}_csvs")
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))

    print("\n" + "="*65)
    print(" MediaPipe Foot Gesture Feature Extractor")
    print(f" Input Dir   : {input_dir}")
    print(f" Output Dir  : {final_output_folder}")
    print(f" Class ID    : {class_id}")
    print(f" Videos Found: {len(video_files)}")
    print("="*65 + "\n")

    if not video_files:
        print("[!] No .mp4 files found in the input directory. Aborting.")
        return

    # Package arguments for the worker pool
    args_list = list(zip(video_files, itertools.repeat(class_id), itertools.repeat(final_output_folder)))

    with ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(worker_process_video, args_list):
            print(result)

    print(f"\n[OK] Batch Processing Complete. Features saved to: {final_output_folder}\n")


def main():
    args = parse_args()
    batch_process(args.input_dir, args.output_dir, args.class_id)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()
