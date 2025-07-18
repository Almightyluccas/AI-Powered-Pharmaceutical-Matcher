import pandas as pd
import re
import os


def clean_target_text(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name_lower = name.lower().replace('+', ' ')

    noise_words = [
        # Brands/Labs
        'ems', 'ache', 'prati', 'eurofarma', 'medley', 'neo', 'quimica', 'legrand',
        'cimed', 'teuto', 'germed', 'sandoz', 'gsk', 'pfizer', 'sanofi', 'bayer',
        'ranbaxy', 'med', 'sdt', 'sdz', 'ger', 'am', 'leg', 'bio',
        # Other noise
        'rev', 'generico', 'similar', 'original', 'com', 'c'
    ]

    base_name = name_lower
    noise_pattern = r'\b(' + '|'.join(re.escape(word) for word in noise_words) + r')\b'
    base_name = re.sub(noise_pattern, '', base_name)

    # Standardize quantity notations like '2bl x 15' to '30'
    if bl_match := re.search(r'(\d+)\s*bl\s*x\s*(\d+)', base_name):
        quantity = int(bl_match.group(1)) * int(bl_match.group(2))
        base_name = re.sub(r'\d+\s*bl\s*x\s*\d+', f'{quantity}', base_name)

    # Final cleanup of any remaining special characters and collapse spaces
    base_name = re.sub(r'[^a-z0-9]', ' ', base_name)  # Keep numbers for dosage/qty
    base_name = re.sub(r'\s+', ' ', base_name).strip()

    return base_name


def process_dataset(input_path: str, output_path: str):
    print(f"--- Cleaning Target Text for Dataset: {input_path} ---")

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{input_path}'")
        return

    if 'target_text' not in df.columns:
        print("❌ ERROR: CSV must contain a 'target_text' column.")
        return

    df['target_text'] = df['target_text'].apply(clean_target_text)

    df.drop_duplicates(inplace=True)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ Successfully created a clean dataset with {len(df)} rows.")
    print(f"   -> Saved to: {output_path}")
    print("\n--- Sample of Cleaned Data ---")
    print(df.head().to_string())


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))

    INPUT_DATASET = os.path.join(script_directory, "..", "data", "processed", "initial_training_data.csv")

    OUTPUT_DATASET = os.path.join(script_directory, "..", "data", "processed", "MASTER_TRAINING_DATASET.csv")

    process_dataset(INPUT_DATASET, OUTPUT_DATASET)
