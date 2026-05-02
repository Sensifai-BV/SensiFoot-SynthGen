import cv2
import mediapipe as mp
import csv
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing
import gc
import math
import time 

# --- SAFETY SETTING ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def worker_process_video(args):
    video_path, class_id = args
    filename = os.path.basename(video_path)
    
    try:
        processor = FootGestureProcessor()
        processor.process_video(video_path, class_id)
        
        del processor
        gc.collect()
        
        return f"✅ Done: {filename}"
    except Exception as e:
        return f"❌ Error in {filename}: {str(e)}"

class FootGestureProcessor:
    """
    Multiprocessing-enabled extractor for generating the Feature Dataset.
    Includes Torso Normalization, Kinematic Angles, Foreshortening metrics, 
    and now extracts Shoulders as well!
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.conf = min_detection_confidence
        self.track_conf = min_tracking_confidence
        
        # 1. ADDED SHOULDERS HERE
        self.target_indices = {
            11: "L_Shoulder", 12: "R_Shoulder",
            23: "L_Hip",   24: "R_Hip",
            25: "L_Knee",  26: "R_Knee",
            27: "L_Ankle", 28: "R_Ankle",
            29: "L_Heel",  30: "R_Heel",
            31: "L_Toe",   32: "R_Toe"
        }

    def process_video(self, input_path, class_id=None):
        cap = cv2.VideoCapture(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        if class_id is None:
            folder_name = "inference_ready_raw"
        else:
            folder_name = str(class_id)
            
        save_dir = os.path.join(f"./csv_outputs", folder_name)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{base_name}_features.csv")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.conf,
            min_tracking_confidence=self.track_conf
        ) as pose:

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                header = ["frame", "timestamp", "class_id"]
                
                # 2. FIXED HEADER BUG: Sorting by keys instead of values
                for key in sorted(self.target_indices.keys()): 
                    name = self.target_indices[key]
                    header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
                
                header.extend([
                    "L_Knee_Angle", "R_Knee_Angle", 
                    "L_Hip_Angle", "R_Hip_Angle",
                    "L_Femur_2D_Len", "R_Femur_2D_Len", 
                    "L_Tibia_2D_Len", "R_Tibia_2D_Len"
                ])
                writer.writerow(header)

                frame_idx = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        l_hip, r_hip = landmarks[23], landmarks[24]
                        h_mid_x = (l_hip.x + r_hip.x) / 2.0
                        h_mid_y = (l_hip.y + r_hip.y) / 2.0
                        h_mid_z = (l_hip.z + r_hip.z) / 2.0
                        
                        l_sh, r_sh = landmarks[11], landmarks[12]
                        s_mid_x = (l_sh.x + r_sh.x) / 2.0
                        s_mid_y = (l_sh.y + r_sh.y) / 2.0
                        s_mid_z = (l_sh.z + r_sh.z) / 2.0
                        
                        scale = math.sqrt(
                            (s_mid_x - h_mid_x)**2 + 
                            (s_mid_y - h_mid_y)**2 + 
                            (s_mid_z - h_mid_z)**2
                        ) + 1e-6
                        
                        row = [frame_idx, frame_idx / fps, "" if class_id is None else class_id]
                        norm_coords = {}

                        # Normalizes all 12 target joints (including shoulders now)
                        for idx in sorted(self.target_indices.keys()):
                            lm = landmarks[idx]
                            norm_x = (lm.x - h_mid_x) / scale
                            norm_y = (lm.y - h_mid_y) / scale
                            norm_z = (lm.z - h_mid_z) / scale
                            
                            norm_coords[self.target_indices[idx]] = (norm_x, norm_y, norm_z)
                            row.extend([norm_x, norm_y, norm_z])

                        # 3. Removed manual shoulder extraction from here since the loop above handles it!

                        # Calculate 3D Kinematic Angles
                        def calc_3d_angle(a, b, c):
                            v1 = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
                            v2 = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
                            dot = sum(x*y for x,y in zip(v1, v2))
                            mag1 = math.sqrt(sum(x**2 for x in v1))
                            mag2 = math.sqrt(sum(x**2 for x in v2))
                            if mag1 * mag2 == 0: return 0.0
                            return math.degrees(math.acos(max(min(dot/(mag1*mag2), 1.0), -1.0)))

                        l_knee_angle = calc_3d_angle(norm_coords["L_Hip"], norm_coords["L_Knee"], norm_coords["L_Ankle"])
                        r_knee_angle = calc_3d_angle(norm_coords["R_Hip"], norm_coords["R_Knee"], norm_coords["R_Ankle"])
                        
                        # Hip Angles (Pulls seamlessly from norm_coords now)
                        l_hip_angle = calc_3d_angle(norm_coords["L_Shoulder"], norm_coords["L_Hip"], norm_coords["L_Knee"])
                        r_hip_angle = calc_3d_angle(norm_coords["R_Shoulder"], norm_coords["R_Hip"], norm_coords["R_Knee"])

                        # Calculate 2D Foreshortening
                        def calc_2d_dist(a, b):
                            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

                        l_femur_2d = calc_2d_dist(norm_coords["L_Hip"], norm_coords["L_Knee"])
                        r_femur_2d = calc_2d_dist(norm_coords["R_Hip"], norm_coords["R_Knee"])
                        l_tibia_2d = calc_2d_dist(norm_coords["L_Knee"], norm_coords["L_Ankle"])
                        r_tibia_2d = calc_2d_dist(norm_coords["R_Knee"], norm_coords["R_Ankle"])

                        # Append all 8 engineered features
                        row.extend([
                            l_knee_angle, r_knee_angle, 
                            l_hip_angle, r_hip_angle, 
                            l_femur_2d, r_femur_2d, l_tibia_2d, r_tibia_2d
                        ])
                        writer.writerow(row)

                    frame_idx += 1
        cap.release()

    def batch_process_multiprocess(self, folder_path, class_id=None):
        if not os.path.exists(folder_path):
            print(f"⚠️ Skipping: Folder does not exist -> {folder_path}")
            return

        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        if not video_files:
            print(f"⚠️ Skipping: No MP4 files found in -> {folder_path}")
            return

        print(f"\n🚀 Processing {len(video_files)} videos in {os.path.basename(folder_path)} (Class {class_id})...")

        args_list = list(zip(video_files, itertools.repeat(class_id)))

        with ProcessPoolExecutor(max_workers=3) as executor:
            for result in executor.map(worker_process_video, args_list):
                if "❌" in result:
                    print(result)

        print(f"✅ Finished folder: {folder_path}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    processor = FootGestureProcessor()
    start_time = time.time()
    
    # --- OVERNIGHT BATCH QUEUE ---
    queue = [
        # ("/PATH/TO/SPECIFIC/CLASS/VIDEOS", class_id),
        # ("/PATH/TO/SPECIFIC/CLASS/VIDEOS", class_id),
        # ...
    ]

    print(f"🌙 Starting Overnight Batch Extraction for {len(queue)} folders...")
    
    for folder_path, class_id in queue:
        try:
            processor.batch_process_multiprocess(folder_path, class_id=class_id)
        except Exception as e:
            print(f"❌ CRITICAL ERROR in folder {folder_path}: {e}")
            print("➡️ Moving to the next folder...")
            continue
            
    total_time = (time.time() - start_time) / 60
    print(f"\n🌅 OVERNIGHT EXTRACTION COMPLETE! Total Time: {total_time:.2f} minutes.")