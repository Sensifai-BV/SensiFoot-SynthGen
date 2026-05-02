"""
CSV Leg Mirroring Augmentor
Mirrors the L/R leg landmarks in CSV files and saves them in the same folder.

USAGE:
  python mirror_legs.py --input_dir /path/to/csv_folder
"""

import pandas as pd
import os
import glob
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mirror L/R leg landmarks in CSV files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input CSV files")
    return parser.parse_args()


class MirrorAugmentor:
    """Handles the L/R swapping and mathematical inversion of CSV pose data."""
    
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def process(self, csv_path):
        df = pd.read_csv(csv_path)
        mirrored_df = df.copy()
        
        for col in df.columns:
            if col.startswith('L_'):
                target_col = 'R_' + col[2:]
            elif col.startswith('R_'):
                target_col = 'L_' + col[2:]
            else:
                continue
            
            if target_col not in df.columns:
                print(f"    [!] Warning: Counterpart column {target_col} not found for {col}. Skipping.")
                continue
            
            source_values = df[target_col].values
            
            # Mathematical Inversion: X-coordinates (lateral) must flip signs
            if col.lower().endswith('_x'):
                mirrored_df[col] = -1 * source_values
            else:
                mirrored_df[col] = source_values
                
        filename = os.path.basename(csv_path)
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(self.target_dir, f"{name}_mirrored{ext}")
        
        mirrored_df.to_csv(save_path, index=False)
        return save_path


def main():
    args = parse_args()
    input_dir = args.input_dir

    print("\n" + "="*65)
    print(" CSV Leg Mirroring Augmentor")
    print(f" Target Dir  : {input_dir}")
    print("="*65 + "\n")

    if not os.path.isdir(input_dir):
        print(f"[!] Directory not found: {input_dir}")
        return

    # Grab all CSVs, but filter out files that are already mirrored
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files = [f for f in csv_files if "_mirrored" not in f]

    if not csv_files:
        print(f"[!] No valid, unmirrored CSV files found in {input_dir}.")
        return

    print(f"[*] Found {len(csv_files)} CSV files to process.\n")
    
    augmentor = MirrorAugmentor(input_dir)
    
    for input_file in csv_files:
        try:
            new_file = augmentor.process(input_file)
            print(f"    [OK] Created: {os.path.basename(new_file)}")
        except Exception as e:
            print(f"    [X] Error processing {os.path.basename(input_file)}: {str(e)}")
            
    print(f"\n[OK] Processing complete! Files saved alongside originals in: {input_dir}\n")


if __name__ == "__main__":
    main()
