import pandas as pd
import re
import os

def parse_currency_to_float(value: str | float) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), 2) if value > 0 else None

    if not isinstance(value, str):
        return None

    try:
        cleaned_value = value.replace('R$', '').replace('.', '').replace(',', '.').strip()
        return round(float(cleaned_value), 2)
    except (ValueError, AttributeError):
        return None

def create_verified_dataset(moleculas_updated_file: str, log_file: str):

    print("--- Starting Price Verification for Dataset Creation ---")

    try:
        print(f"Loading main data file: {moleculas_updated_file}")
        df_updated = pd.read_csv(moleculas_updated_file)

        print(f"Loading log file: {log_file}")
        df_log = pd.read_csv(log_file)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required file. {e}")
        return

    product_col_updated = df_updated.columns[0]

    if len(df_updated.columns) <= 6:
        print(f"❌ ERROR: The '{moleculas_updated_file}' file has 6 or fewer columns, so no price columns can be identified.")
        return

    original_price_cols = df_updated.columns[3:-3]

    print(f"Using Product Column from '{moleculas_updated_file}': '{product_col_updated}'")
    print(f"Using these columns for original price verification: {list(original_price_cols)}")

    price_lookup = {}
    print("\nBuilding a lookup map of original prices...")
    for index, row in df_updated.iterrows():
        product_name = row[product_col_updated]
        prices = {parse_currency_to_float(row[col]) for col in original_price_cols}
        prices.discard(None)
        price_lookup[product_name] = prices

    training_data = []
    detailed_verified_log = []
    print("Verifying matches by comparing assigned price to original prices...")

    for index, log_row in df_log.iterrows():
        molecula_name = log_row['Moleculas_Produto (Y)']
        assigned_price = parse_currency_to_float(log_row['Assigned_Price (Z)'])

        if assigned_price is None:
            continue

        original_prices_set = price_lookup.get(molecula_name)

        if original_prices_set and assigned_price in original_prices_set:

            training_data.append({
                'input_text': log_row['Matched_Price_File_Produto (X)'],
                'target_text': log_row['Moleculas_Produto (Y)']
            })

            detailed_verified_log.append({
                'Moleculas_Produto': log_row['Moleculas_Produto (Y)'],
                'Matched_Produto': log_row['Matched_Price_File_Produto (X)'],
                'Verified_Price': log_row['Assigned_Price (Z)'],
                'Price_Source_File': log_row['Price_Source_File']
            })

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    if not training_data:
        print("\nℹ️ No matches could be verified by price. No training data file was created.")
    else:
        df_training = pd.DataFrame(training_data).drop_duplicates()
        output_file_train = os.path.join(output_dir, "verified_training_data.csv")
        df_training.to_csv(output_file_train, index=False, encoding='utf-8-sig')
        print(f"\n✅ Successfully created training dataset with {len(df_training)} pairs.")
        print(f"File saved as: {output_file_train}")
        print("\n--- Sample of Training Data ---")
        print(df_training.head().to_string())

    if not detailed_verified_log:
        print("\nℹ️ No detailed verified log to save.")
    else:
        df_detailed = pd.DataFrame(detailed_verified_log).drop_duplicates()
        output_file_detailed = os.path.join(output_dir, "price_verified_log.csv")
        df_detailed.to_csv(output_file_detailed, index=False, encoding='utf-8-sig')
        print(f"\n✅ Successfully created a detailed log with {len(df_detailed)} price-verified matches.")
        print(f"File saved as: {output_file_detailed}")
        print("\n--- Sample of Detailed Verified Log ---")
        print(df_detailed.head().to_string())


if __name__ == '__main__':
    script_directory = os.path.dirname(__file__)

    MOLECULAS_UPDATED_FILE = os.path.join(script_directory, "..", "data", "processed", "ai", "moleculas_updated_final_AI.csv")
    MATCHING_LOG_FILE = os.path.join(script_directory, "..", "data", "processed", "ai", "matching_log_AI.csv")

    # NONE AI VERSION
    # MOLECULAS_UPDATED_FILE = os.path.join(script_directory, "..", "data", "processed", "moleculas_updated_final.csv")
    # MATCHING_LOG_FILE = os.path.join(script_directory, "..", "data", "processed", "matching_log.csv")

    create_verified_dataset(MOLECULAS_UPDATED_FILE, MATCHING_LOG_FILE)
