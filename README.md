# SensiFoot-SynthGen

This pipeline is an exceptionally well-engineered solution for synthetic data generatio. By combining markerless motion capture, morphological diversity, physical noise injection, and symmetrical augmentation, we have created a robust system that effectively bridges the gap between real-world data collection and purely synthetic generation.

This repository provides an automated, end-to-end pipeline to convert a single real-world video of a foot gesture into hundreds of robust, augmented, and normalized skeletal data records, culminating in a powerful CNN-LSTM attention-based classifier.

## Step 0: Prerequisites & Setup

Before running the automated Python scripts, you need to prepare your base assets and environment:

* **0.1 - Capture Video:** Record a video of the target foot gesture. The clearer the video, the better the downstream results.
* **0.2 - Motion Extraction:** Use [DeepMotion](https://www.deepmotion.com/) (or another converter service) to extract markerless motion capture and generate an FBX file from your video. This grounds the dataset in reality and ensures physically authentic temporal dynamics.
* **0.3 - Target Character:** Download a 3D character model (e.g., from Mixamo) in FBX format. By retargeting onto distinct morphological types (normal, fat, bulky), the pipeline introduces crucial spatial variance.
* **0.4 - Environment Setup:** Create a Conda environment (or venv) utilizing **Python 3.10**. 
    * *Note: MediaPipe version `0.10.21` is strictly required.*
    * Install the dependencies using `pip install -r requirements.txt`.
* **0.5 - Blender Setup:** Download [Blender 4.0.2](https://download.blender.org/release/Blender4.0/) as a `.zip` file. Extract it to a known directory and ensure the `./blender` executable is accessible. Our scripts will run Blender in headless mode (`--background`).
* **0.6 - Rokoko Add-on Setup:** The retargeting script relies on the Rokoko Studio Live add-on. Because Blender is running headlessly, this must be installed manually via the command line so the script can locate the `rokoko-studio-live-blender-master` module. Run the following commands to install it:
  ```bash
  mkdir -p ~/.config/blender/4.0/scripts/addons/
  cd ~/.config/blender/4.0/scripts/addons/
  wget [https://github.com/Rokoko/rokoko-studio-live-blender/archive/refs/heads/master.zip](https://github.com/Rokoko/rokoko-studio-live-blender/archive/refs/heads/master.zip)
  unzip master.zip
  rm master.zip
  cd -  # Return to your project directory
  ```

## Phase 1: Retargeting (`retarget_movement.py`)
This script maps the motion capture animation from your DeepMotion FBX onto your Mixamo character's skeleton. This ensures the downstream pose-estimation model does not overfit to a specific body type or limb proportion, which is a common failure point in gesture recognition models.

* **Inputs:** Source (mocap) FBX, Target (character) FBX, and a Rokoko bone-mapping JSON scheme.
* **Outputs:** A retargeted FBX file named dynamically based on your inputs (e.g., `retarget_output_[source]_[target].fbx`).
* **How to run:**
    ```bash
    /path/to/blender-4.0.2/blender --background --python retarget_movement.py -- \
        --source /path/to/mocap.fbx \
        --target /path/to/character.fbx \
        --scheme /path/to/scheme.json \
        --output /path/to/save_folder
    ```

## Phase 2: Multi-View Rendering (`render_multi_views.py`)
This phase acts as a multiplier. The script takes your retargeted FBX and renders **120 unique video variations** by looping through 5 speeds, 3 distances, and 8 camera angles. 

Crucially, it utilizes **Kinematic Noise Injection**, which guarantees both infinite uniqueness and physical stability. The foot amplitude is halved to exactly 0.04 radians (max ~2.3°), mathematically preventing the ankles from snapping or clipping through the floor.

* **Inputs:** The retargeted FBX file from Phase 1.
* **Outputs:** A folder containing 120 `.mp4` files.
* **How to run:**
    ```bash
    /path/to/blender-4.0.2/blender --background --python render_multi_views.py -- \
        --file_path /path/to/retarget_output.fbx \
        --output_dir /path/to/save_folder
    ```

## Phase 3: Feature Extraction (`feature_extractor.py`)
This script processes the rendered videos through MediaPipe to extract 3D landmarks. It isolates indices 25-32 (legs and feet). 

The normalization logic here is highly sophisticated. By anchoring the coordinates to the Mid-Hip point, the gesture's data becomes independent of where the subject is standing in the camera frame. It also scales using the static Torso Length, ensuring that a gesture performed close to the camera outputs the exact same numerical footprint as one performed far away.

* **Inputs:** The folder of 120 `.mp4` videos and a designated Class ID.
* **Outputs:** A new folder containing 120 `.csv` files.
* **How to run:**
    ```bash
    python feature_extractor.py \
        --input_dir /path/to/video_folder \
        --output_dir /path/to/save_folder \
        --class_id 1
    ```

## Phase 4: Symmetrical Mirroring (`mirror_legs.py`)
Because these gestures are single-leg focused, capturing the opposite leg in the real world would require doubling the physical recording time. This script duplicates the CSVs, swaps the `L_` and `R_` column headers, and mathematically inverts the lateral X-axis. This effectively doubles the dataset to 240 records per character with zero additional rendering cost.

* **Inputs:** The folder containing your extracted `.csv` files.
* **Outputs:** 120 newly generated `_mirrored.csv` files saved directly alongside the originals.
* **How to run:**
    ```bash
    python mirror_legs.py --input_dir /path/to/csv_folder
    ```

## Phase 5: Model Training (`baseline_trainer.py`)
The final step trains a Universal Gesture Model utilizing a Spatial CNN block to smooth coordinates, a Temporal LSTM block for timeline processing, and a Temporal Attention mechanism to focus on the most important chronological frames.

**Leave-One-Subject-Out (LOSO) Validation:**
To prove true generalizability, the trainer defaults to LOSO validation. Instead of a random data split, the model trains on subjects A and B, and validates *entirely* on subject C. This proves the model learns the actual gesture, not the unique movement quirks or skeleton size of a specific person. As demonstrated in our validation matrix, the baseline model achieves robust class separation even on entirely unseen subject skeletons.

* **Inputs:** A root directory containing subfolders of your fully processed CSVs, grouped by class (e.g., `.../csvs-final/1/`, `.../csvs-final/2/`).
* **Outputs:** The best performing `.pth` model weights.
* **How to run:**
    ```bash
    python baseline_trainer.py \
        --data_path /path/to/csvs-final \
        --val_prefix $_ \
        --epochs 60 \
        --batch_size 64
    ```