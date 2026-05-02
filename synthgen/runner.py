import os
import subprocess
from pathlib import Path

# --- Configuration ---
# Resolve the directory where runner.py is located (the synthgen folder)
CURRENT_DIR = Path(__file__).parent.resolve()

SOURCE_DIR = "/PATH/TO/FBX_FILES"  
NEW_OUTPUT_BASE = "./rendered_outputs_noiseless_csv" 

# Updated to point to the active script in the same directory
RENDER_SCRIPT = CURRENT_DIR / "render_multi_views_noiseless.py"
BLENDER_EXEC = "/PATH/TO/BLENDER_EXECUTABLE/blender" 

# Folders to explicitly skip during the scan
IGNORE_FOLDERS = {
    "series2", "Refinements",
}

def main():
    base_path = Path(SOURCE_DIR)
    output_base_path = Path(NEW_OUTPUT_BASE)
    output_base_name = output_base_path.name
    
    tasks = []
    
    # 1. Scan directory and collect target FBX files
    for item in base_path.iterdir():
        # Ignore files, explicitly ignored folders, and our own output directory
        if item.is_dir() and item.name not in IGNORE_FOLDERS and item.name != output_base_name:
            
            # Find the .fbx file(s) in this directory
            fbx_files = list(item.glob("*.fbx"))
            
            if fbx_files:
                # Grab the first FBX file to ensure exactly one call per folder
                fbx_path = fbx_files[0]
                
                # Replicate the original folder name in the new directory
                target_out_dir = output_base_path / item.name
                target_out_dir.mkdir(parents=True, exist_ok=True)
                
                tasks.append((str(fbx_path), str(target_out_dir)))

    # Sort tasks alphabetically by file path for consistent ordering
    tasks.sort(key=lambda x: x[0])
    
    total_tasks = len(tasks)
    print(f"Found {total_tasks} FBX tasks. Preparing to run in batches of 5...")

    # 2. Chunk the tasks into batches of 5
    batch_size = 5
    batches = [tasks[i:i + batch_size] for i in range(0, total_tasks, batch_size)]
    
    # 3. Execute the batches sequentially, running the internal batch concurrently
    for batch_index, batch in enumerate(batches, start=1):
        print(f"\n--- Starting Batch {batch_index}/{len(batches)} ---")
        
        processes = []
        for fbx_path, out_dir in batch:
            print(f"Launching render for: {fbx_path}")
            
            # Formulate the command
            cmd = [
                BLENDER_EXEC,
                "-b",                 # Run Blender in the background (headless)
                "-P", str(RENDER_SCRIPT),  # Execute the Python script
                "--",                 # Argument separator for the Python script
                "--file_path", fbx_path,
                "--output_dir", out_dir
            ]
            
            # Start the process without blocking the loop
            p = subprocess.Popen(cmd)
            processes.append(p)
            
        # Wait for all processes in the current batch to complete before moving on
        for p in processes:
            p.wait()
            
        print(f"--- Batch {batch_index} Completed ---")

    print("\nAll tasks have been processed successfully!")

if __name__ == "__main__":
    main()