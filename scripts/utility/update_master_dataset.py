import pandas as pd
import os


def update_master_dataset(master_file_path: str, new_data_file_path: str):

    print("--- Starting Master Dataset Update Process ---")

    if os.path.exists(master_file_path):
        print(f"Loading existing master dataset from: {master_file_path}")
        df_master = pd.read_csv(master_file_path)
        initial_count = len(df_master)
    else:
        print(f"Master dataset not found at '{master_file_path}'. A new one will be created.")
        df_master = pd.DataFrame(columns=['input_text', 'target_text'])
        initial_count = 0

    try:
        print(f"Loading new verified data from: {new_data_file_path}")
        df_new = pd.read_csv(new_data_file_path)

        if 'Moleculas_Produto (Y)' in df_new.columns:
            df_new = df_new.rename(columns={
                'Matched_Price_File_Produto (X)': 'input_text',
                'Moleculas_Produto (Y)': 'target_text'
            })

    except FileNotFoundError:
        print(f"❌ ERROR: The new data file was not found at '{new_data_file_path}'.")
        print("   -> Please run the validation script first.")
        return

    if 'input_text' not in df_new.columns or 'target_text' not in df_new.columns:
        print("❌ ERROR: The new data file is missing the required columns.")
        return

    print(f"\nOriginal master dataset size: {initial_count} rows")
    print(f"Found {len(df_new)} new verified rows to add.")

    df_combined = pd.concat([df_master, df_new], ignore_index=True)

    df_combined.drop_duplicates(subset=['input_text'], keep='last', inplace=True)

    final_count = len(df_combined)
    new_rows_added = final_count - initial_count

    df_combined.to_csv(master_file_path, index=False, encoding='utf-8-sig')

    print("\n--- Update Complete ---")
    print(f"✅ Master dataset has been updated.")
    print(f"   -> New rows added: {new_rows_added}")
    print(f"   -> Final dataset size: {final_count} rows")
    print(f"   -> Saved to: {master_file_path}")


# if __name__ == '__main__':
#     # --- Configuration ---
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#
#     # The main "answer key" for your AI
#     MASTER_DATASET_PATH = os.path.join(script_directory, "..", "data", "processed", "MASTER_TRAINING_DATASET.csv")
#
#     # The output from your validation script
#     NEW_VERIFIED_DATA_PATH = os.path.join(script_directory, "..", "data", "processed", "ai", "validated",
#                                           "verified_by_rules.csv")
#
#     update_master_dataset(MASTER_DATASET_PATH, NEW_VERIFIED_DATA_PATH)
