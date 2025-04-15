import json
import random
import argparse
import os
from tqdm import tqdm

def load_data(file_path, percentage):
    """Loads a specified percentage of data from a line-delimited JSON file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        num_lines_to_keep = int(len(lines) * percentage)
        if num_lines_to_keep > len(lines):
             num_lines_to_keep = len(lines) # Ensure we don't try to keep more lines than exist
        
        # Randomly sample lines if percentage is less than 1.0, otherwise take all
        if percentage < 1.0:
            sampled_lines = random.sample(lines, num_lines_to_keep)
        else:
            sampled_lines = lines[:num_lines_to_keep] # Take the first num_lines_to_keep (which is all lines if percentage=1.0)

        print(f"Loading {num_lines_to_keep} lines ({percentage*100:.1f}%) from {file_path}...")
        for line in tqdm(sampled_lines, desc=f"Parsing {os.path.basename(file_path)}"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
        print(f"Successfully loaded {len(data)} records from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return []
    return data

def main():
    parser = argparse.ArgumentParser(description="Mix and shuffle data from two JSON datasets.")
    parser.add_argument('--input_cv', type=str, default='/home/sarpba/Whisper-Finetune/dataset_CV17/test.json',
                        help='Path to the first input JSON file (CV17).')
    parser.add_argument('--input_custom', type=str, default='/home/sarpba/Whisper-Finetune/dataset_CUSTOM/test.json',
                        help='Path to the second input JSON file (CUSTOM).')
    parser.add_argument('--output_file', type=str, default='/home/sarpba/Whisper-Finetune/dataset/test.json',
                        help='Path to the output mixed JSON file.')
    parser.add_argument('--percent_cv', type=float, default=1.0,
                        help='Percentage of data to use from the first dataset (0.0 to 1.0).')
    parser.add_argument('--percent_custom', type=float, default=1.0,
                        help='Percentage of data to use from the second dataset (0.0 to 1.0).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling.')

    args = parser.parse_args()

    # Validate percentages
    if not (0.0 <= args.percent_cv <= 1.0):
        print("Error: percent_cv must be between 0.0 and 1.0")
        return
    if not (0.0 <= args.percent_custom <= 1.0):
        print("Error: percent_custom must be between 0.0 and 1.0")
        return
        
    # Set random seed
    random.seed(args.seed)

    # Load data
    data_cv = load_data(args.input_cv, args.percent_cv)
    data_custom = load_data(args.input_custom, args.percent_custom)

    # Combine data
    combined_data = data_cv + data_custom
    print(f"\nTotal records loaded: {len(combined_data)}")

    # Shuffle data
    print("Shuffling combined data...")
    random.shuffle(combined_data)
    print("Shuffling complete.")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Save combined and shuffled data
    print(f"Saving mixed data to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(combined_data, desc="Writing output"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(combined_data)} records to {args.output_file}.")
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")

if __name__ == "__main__":
    main()
