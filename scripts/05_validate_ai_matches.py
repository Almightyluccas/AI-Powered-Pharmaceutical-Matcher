import pandas as pd
import re
import os

def get_structured_features(name: str):
    """
    Extracts only the critical numerical features (dosage and quantity) from a product name.
    """
    if not isinstance(name, str):
        return {'original': '', 'dosage': 0, 'quantity': 0}

    original_name = name
    name_lower = name.lower()

    # --- Dosage Extraction ---
    dosage_val = 0
    if dosage_match := re.search(r'(\d[\d.,]*)\s*mg\s*/\s*ml', name_lower):
        try: dosage_val = float(dosage_match.group(1).replace(',', '.'))
        except ValueError: dosage_val = 0
    elif dosage_match := re.search(r'(\d[\d.,]*)\s*(mg|g)\b', name_lower):
        try: dosage_val = float(dosage_match.group(1).replace(',', '.'))
        except ValueError: dosage_val = 0

    # --- Quantity Extraction ---
    quantity = 1
    if bl_match := re.search(r'(\d+)\s*bl\s*x\s*(\d+)', name_lower):
        quantity = int(bl_match.group(1)) * int(bl_match.group(2))
    elif c_slash_match := re.search(r'\b(c|com)\s*/\s*(\d+)', name_lower):
        quantity = int(c_slash_match.group(2))
    elif q_match := re.search(r'(\d+)\s*(comp|cap|caps|cps|cpr|cpd|drg|env)\b', name_lower):
        quantity = int(q_match.group(1))
    elif c_match := re.search(r'\bcom\s+(\d+)\b', name_lower):
        quantity = int(c_match.group(2))
    else:
        ml_match = re.search(r'(\d+)\s*ml\b', name_lower)
        if ml_match and quantity == 1:
            quantity = int(ml_match.group(1))
        else:
            g_match = re.search(r'(\d+)\s*g\b', name_lower)
            if g_match and quantity == 1:
                quantity = int(g_match.group(1))

    return {
        'original': original_name,
        'dosage': dosage_val,
        'quantity': quantity
    }

def validate_ai_matches(log_file: str):
    """
    Validates matches from an AI log file based ONLY on dosage and quantity,
    and produces three distinct output files for training, review, and reporting.
    """
    print("--- Starting Rule-Based Validation of AI Matches (Dosage & Quantity Only) ---")

    # --- 1. Load the AI's Match Log ---
    try:
        print(f"Loading AI match log: {log_file}")
        df_log = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"❌ ERROR: The log file was not found at '{log_file}'.")
        return

    if 'Moleculas_Produto (Y)' not in df_log.columns or 'Matched_Price_File_Produto (X)' not in df_log.columns:
        print("❌ ERROR: Log file is missing required columns.")
        return

    # Create lists to hold the different outputs
    training_data_rows = []
    verified_with_price_rows = []
    rejected_rows = []

    print(f"Validating {len(df_log)} matches...")
    for index, row in df_log.iterrows():
        molecula_features = get_structured_features(row['Moleculas_Produto (Y)'])
        matched_features = get_structured_features(row['Matched_Price_File_Produto (X)'])

        is_verified = True
        rejection_reason = []

        if molecula_features['dosage'] != matched_features['dosage']:
            is_verified = False
            rejection_reason.append(f"Dosage Mismatch ({molecula_features['dosage']} vs {matched_features['dosage']})")

        if molecula_features['quantity'] != matched_features['quantity']:
            is_verified = False
            rejection_reason.append(f"Quantity Mismatch ({molecula_features['quantity']} vs {matched_features['quantity']})")

        # --- Decision and Logging ---
        if is_verified:
            # Add to the list for the clean training data file
            training_data_rows.append({
                'input_text': row['Matched_Price_File_Produto (X)'],
                'target_text': row['Moleculas_Produto (Y)']
            })
            # Add the full row to the detailed verified log
            verified_with_price_rows.append(row.to_dict())
        else:
            # Add the names and the reason for rejection to the rejected log
            rejected_rows.append({
                'Moleculas_Produto (Y)': row['Moleculas_Produto (Y)'],
                'Matched_Price_File_Produto (X)': row['Matched_Price_File_Produto (X)'],
                'Rejection_Reason': " & ".join(rejection_reason)
            })

    # --- 4. Save Output Files ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "processed", "ai", "validated")
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Validation Complete ---")

    # Save the clean training data file
    if training_data_rows:
        df_training = pd.DataFrame(training_data_rows).drop_duplicates()
        output_training = os.path.join(output_dir, "verified_by_rules.csv")
        df_training.to_csv(output_training, index=False, encoding='utf-8-sig')
        print(f"\n✅ Created clean training data file with {len(df_training)} pairs.")
        print(f"   -> Saved to: {output_training}")
    else:
        print("\nℹ️ No matches passed verification. No training data file was created.")

    # Save the detailed verified log with prices
    if verified_with_price_rows:
        df_verified_detailed = pd.DataFrame(verified_with_price_rows)
        output_verified_detailed = os.path.join(output_dir, "verified_matches_with_prices.csv")
        df_verified_detailed.to_csv(output_verified_detailed, index=False, encoding='utf-8-sig')
        print(f"✅ Created detailed log with {len(df_verified_detailed)} verified matches (including prices).")
        print(f"   -> Saved to: {output_verified_detailed}")
    else:
        print("ℹ️ No detailed verified log to save.")

    # Save the rejected matches
    if rejected_rows:
        df_rejected = pd.DataFrame(rejected_rows)
        output_rejected = os.path.join(output_dir, "rejected_by_rules.csv")
        df_rejected.to_csv(output_rejected, index=False, encoding='utf-8-sig')
        print(f"❌ Found {len(df_rejected)} rejected matches for review.")
        print(f"   -> Saved to: {output_rejected}")
    else:
        print("ℹ️ No matches were rejected by the rules.")


if __name__ == '__main__':
    # --- Configuration ---
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # This should be the output from your AI matching script
    AI_LOG_FILE = os.path.join(script_directory, "..", "data", "processed", "ai", "matching_log_AI.csv")

    validate_ai_matches(AI_LOG_FILE)
