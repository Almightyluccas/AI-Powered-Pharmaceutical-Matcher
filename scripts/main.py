import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_model import train_pharma_translator
    from medicine_matcher_ai import ai_powered_matching
    from validate_ai_matches import validate_ai_matches
    from update_master_dataset import update_master_dataset
except ImportError as e:
    print(f"❌ ERROR: Could not import a required script.")
    print(
        f"   -> Please ensure 'train_model.py', 'medicine_matcher_ai.py', and 'validate_ai_matches.py' exist in the 'scripts' folder.")
    print(f"   -> Details: {e}")
    sys.exit(1)



def get_current_version(version_file: str) -> int:
    try:
        if not os.path.exists(version_file):
            with open(version_file, "w") as f:
                f.write("1")
            print(f"Version file '{version_file}' not found. Initializing at version 1.")
            return 1
        with open(version_file, "r") as f:
            version_str = f.read().strip()
            return int(version_str)
    except (ValueError, FileNotFoundError):
        print(f"Warning: Could not read version from '{version_file}'. Defaulting to 1.")
        with open(version_file, "w") as f:
            f.write("1")
        return 1


def increment_version(version_file: str) -> int:
    current_version = get_current_version(version_file)
    next_version = current_version + 1
    with open(version_file, "w") as f:
        f.write(str(next_version))
    print(f"✅ Version successfully incremented to: {next_version}")
    return next_version


def run_bootstrap_cycle():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    VERSION_FILE = os.path.join(project_root, "version.txt")
    MASTER_DATASET = os.path.join(project_root, "data", "processed", "MASTER_TRAINING_DATASET.csv")
    MOLECULAS_FILE = os.path.join(project_root, "data", "raw", "moleculas.xls")
    PRICE_FILES_DIR = os.path.join(project_root, "data", "raw", "prices")
    NEW_VERIFIED_DATA_PATH = os.path.join(script_directory, "..", "data", "processed", "ai", "validated", "verified_by_rules.csv")
    SIMILARITY_THRESHOLD = 90

    current_version = get_current_version(VERSION_FILE)
    next_version = current_version + 1
    print(f"\n--- Preparing for Bootstrap Cycle ---")
    print(f"Current model version: {current_version}")
    print(f"Next model version to be trained: {next_version}")

    base_model_path = f"models/t5-pharma-translator-v{current_version}"
    new_model_path = f"models/t5-pharma-translator-v{next_version}"

    if current_version == 1 and not os.path.exists(os.path.join(project_root, base_model_path)):
        base_model_path = "t5-base"
    else:
        base_model_path = os.path.join(project_root, base_model_path)

    new_model_path = os.path.join(project_root, new_model_path)

    processed_dir = os.path.join(project_root, "data", "processed")
    standard_log_path = os.path.join(processed_dir, "matching_log_AI.csv")

    print("\n--- Paths Set for This Run ---")
    print(f"  - Base Model for Training: '{base_model_path}'")
    print(f"  - New Model Output Path:   '{new_model_path}'")

    try:
        print("\n--- STAGE 1: Starting Model Training ---")
        train_pharma_translator(MASTER_DATASET, new_model_path, base_model=base_model_path)
        print("--- STAGE 1: Model Training Complete ---")

        print("\n--- STAGE 2: Starting AI-Powered Matching ---")
        ai_powered_matching(new_model_path, MOLECULAS_FILE, PRICE_FILES_DIR, similarity_threshold=SIMILARITY_THRESHOLD)
        print("--- STAGE 2: AI Matching Complete ---")

        print("\n--- STAGE 3: Starting Rule-Based Validation ---")
        validate_ai_matches(standard_log_path)
        print("--- STAGE 3: Validation Complete ---")

        while True:
            choice = input(
                "\nHave you reviewed the 'verified_by_rules.csv' file and want to update the master dataset? (yes/no): ").strip().lower()
            if choice in ['yes', 'y']:
                print("\n--- STAGE 4: Updating Master Dataset ---")
                update_master_dataset(MASTER_DATASET, new_verified_data_path)
                print("--- STAGE 4: Master Dataset Update Complete ---")
                break
            elif choice in ['no', 'n']:
                print("--- STAGE 4: Skipping Master Dataset Update ---")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")



        print("\n--- STAGE 5: Archiving Results ---")
        shutil.move(os.path.join(processed_dir, "moleculas_updated_final_AI.csv"),
                    os.path.join(processed_dir, f"moleculas_updated_final_AI_v{next_version}.csv"))
        shutil.move(standard_log_path, os.path.join(processed_dir, f"matching_log_AI_v{next_version}.csv"))
        print("--- STAGE 5: Archiving Complete ---")

        print("\n--- Workflow Complete ---")
        increment_version(VERSION_FILE)
        print(f"\n🎉 Bootstrap cycle v{current_version} -> v{next_version} completed successfully!")
        print("Your next step is to review the validation files and update your MASTER_TRAINING_DATASET.csv")

    except Exception as e:
        print("\n❌ An error occurred during the bootstrap cycle.")
        print(f"   -> ERROR: {e}")
        print("   -> Version was not incremented. Please check the logs above to debug the issue.")
        print("   -> Once fixed, you can re-run this script to retry the current cycle.")


if __name__ == '__main__':
    run_bootstrap_cycle()
