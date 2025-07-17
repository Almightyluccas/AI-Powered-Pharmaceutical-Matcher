import pandas as pd
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import re
from rapidfuzz import process, fuzz

def load_model_and_tokenizer(model_path: str):
    """Loads a fine-tuned T5 model and its tokenizer."""
    print(f"Loading custom model from: {model_path}")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded successfully onto device: {device}")
        return model, tokenizer, device
    except OSError:
        print(f"❌ ERROR: Could not find a trained model at '{model_path}'.")
        return None, None, None

def standardize_names_with_ai(texts: list, model, tokenizer, device, batch_size: int = 16) -> list:
    """Uses the trained AI model to translate a list of product names with robust error handling."""
    if not texts:
        return []

    # Ensure all inputs are valid, non-empty strings
    original_texts = [str(text) for text in texts if text and isinstance(text, str) and str(text).strip()]
    if not original_texts:
        return []

    prefixed_texts = ["translate: " + text for text in original_texts]
    predictions = []

    print(f"Standardizing {len(original_texts)} names with the AI model...")
    for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Processing Batches"):
        batch = prefixed_texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

            # FIX: Switched to a simpler, more robust generation method (greedy search)
            # This avoids the hangs caused by beam search on certain hardware/data.
            outputs = model.generate(**inputs, max_length=128)

            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in outputs]
            predictions.extend(decoded_preds)
        except Exception as e:
            print(f"⚠️ Warning: An error occurred during a batch generation: {e}. Skipping batch.")
            # Add empty strings for the failed batch to maintain list length
            predictions.extend([''] * len(batch))

    # Safety Net: Fallback for any failed predictions
    final_predictions = []
    for original, pred in zip(original_texts, predictions):
        if pred and pred.strip():
            final_predictions.append(pred)
        else:
            print(f"⚠️ Warning: AI model failed on input: '{original}'. Using original text as fallback.")
            final_predictions.append(original.lower())

    return final_predictions

def find_column_by_substring(df_columns: list, substrings: list) -> str | None:
    """Finds the first column in a list that contains any of the given substrings."""
    for sub in substrings:
        for col in df_columns:
            if sub.lower() in str(col).lower():
                return col
    return None

def parse_currency_to_float(value: str | float) -> float | None:
    """Converts a currency string or a number to a float."""
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    try:
        cleaned_value = value.replace('R$', '').replace('.', '').replace(',', '.').strip()
        return float(cleaned_value)
    except (ValueError, AttributeError):
        return None

def find_excel_header_row(file_path, num_rows_to_check=20):
    """Finds the header row in an Excel file."""
    try:
        df_temp = pd.read_excel(file_path, header=None, nrows=num_rows_to_check)
        header_keywords = ['produto', 'descrição', 'ean', 'código', 'codprod', 'descricao']
        for i, row in df_temp.iterrows():
            non_empty_cells = row.dropna()
            if len(non_empty_cells) < 3: continue
            string_cells = [cell for cell in non_empty_cells if isinstance(cell, str)]
            if (len(string_cells) / len(non_empty_cells)) > 0.6 and any(kw in str(cell).lower() for cell in string_cells for kw in header_keywords):
                return i
        return 0
    except Exception:
        return 0

def is_price_column(column_series, column_name):
    """Robustly determines if a column is a price column."""
    column_name_lower = str(column_name).lower()
    if any(kw in column_name_lower for kw in ['%', 'desconto', 'desc']):
        return False
    sample = column_series.dropna().head(20)
    if sample.empty:
        return False
    price_like_count = sum(1 for value in sample if parse_currency_to_float(value) is not None and parse_currency_to_float(value) >= 1)
    return (price_like_count / len(sample)) >= 0.5

def run_ai_powered_matching(model_path: str, moleculas_file: str, price_files_dir: str, similarity_threshold: int = 95):
    """The main function to perform the entire matching process using the trained AI model."""
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if not model:
        return

    # --- 1. Load and Standardize Moleculas File ---
    print("\n--- Processing Main Moleculas File ---")
    try:
        df_moleculas = pd.read_excel(moleculas_file, sheet_name="GENERICO")
        product_col_moleculas = find_column_by_substring(df_moleculas.columns, ['Descrição', 'Produto'])
        if not product_col_moleculas:
            print("❌ ERROR: Could not find product column in moleculas file.")
            return

        df_moleculas.dropna(subset=[product_col_moleculas], inplace=True)
        df_moleculas_std = pd.DataFrame({'original_name': df_moleculas[product_col_moleculas].unique()})
        df_moleculas_std['standardized_name'] = standardize_names_with_ai(
            df_moleculas_std['original_name'].tolist(), model, tokenizer, device
        )

        df_final_output = df_moleculas.copy()

    except Exception as e:
        print(f"❌ ERROR: Failed to process {moleculas_file}. {e}")
        return

    # --- 2. Process Each Price File ---
    print("\n--- Processing Price Files ---")
    verified_log = []

    for filename in os.listdir(price_files_dir):
        if not filename.endswith(('.xlsx', '.xls')):
            continue

        file_path = os.path.join(price_files_dir, filename)
        print(f"\nProcessing file: {filename}")
        try:
            header_row = find_excel_header_row(file_path)
            df_price = pd.read_excel(file_path, header=header_row)

            product_col_price = find_column_by_substring(df_price.columns, ['DESCRIÇÃO', 'PRODUTO', 'Prod_Nome'])
            if not product_col_price:
                print(f"  Warning: Could not find a product column in {filename}. Skipping.")
                continue

            df_price.dropna(subset=[product_col_price], inplace=True)
            if df_price.empty:
                print(f"  Warning: No valid product data in {filename} after cleaning. Skipping.")
                continue

            price_cols = [col for col in df_price.columns if col != product_col_price and is_price_column(df_price[col], col)]
            if not price_cols:
                print(f"  Warning: No valid price columns found in {filename}. Skipping.")
                continue

            df_price_std = pd.DataFrame({'original_name': df_price[product_col_price].unique()})
            df_price_std['standardized_name'] = standardize_names_with_ai(
                df_price_std['original_name'].tolist(), model, tokenizer, device
            )
            price_choices = df_price_std.dropna(subset=['standardized_name'])

            # --- 3. Perform Matching ---
            price_col_name = f'Price_{os.path.splitext(filename)[0]}'
            df_final_output[price_col_name] = None

            for index, molecula_row in df_moleculas_std.iterrows():
                std_name_query = molecula_row['standardized_name']
                original_molecula_name = molecula_row['original_name']

                result = process.extractOne(
                    std_name_query, price_choices['standardized_name'],
                    scorer=fuzz.ratio, score_cutoff=similarity_threshold
                )

                if result:
                    matched_std_name, score, choice_index = result
                    original_price_name = price_choices.iloc[choice_index]['original_name']

                    matched_price_rows = df_price[df_price[product_col_price] == original_price_name]
                    if not matched_price_rows.empty:
                        matched_price_row = matched_price_rows.iloc[0]
                        best_price = min((p for col in price_cols if (p := parse_currency_to_float(matched_price_row.get(col))) is not None), default=float('inf'))

                        if best_price != float('inf'):
                            molecula_idx_list = df_final_output.index[df_final_output[product_col_moleculas] == original_molecula_name].tolist()
                            if molecula_idx_list:
                                df_final_output.loc[molecula_idx_list[0], price_col_name] = f"R$ {best_price:.2f}".replace('.', ',')

                            verified_log.append({
                                'Moleculas_Produto (Y)': original_molecula_name,
                                'Matched_Price_File_Produto (X)': original_price_name,
                                'Assigned_Price (Z)': f"R$ {best_price:.2f}".replace('.', ','),
                                'Similarity_Score': f"{score:.2f}% (AI + Fuzzy)",
                                'Price_Source_File': filename,
                            })

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")

    # --- 4. Save Final Files ---
    print("\n--- AI Matching Complete ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "processed", "ai")
    os.makedirs(output_dir, exist_ok=True)

    output_file_main = os.path.join(output_dir, "moleculas_updated_final_AI.csv")
    df_final_output.to_csv(output_file_main, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully created the main updated file: {output_file_main}")

    if verified_log:
        df_log = pd.DataFrame(verified_log)
        output_file_log = os.path.join(output_dir, "matching_log_AI.csv")
        df_log.to_csv(output_file_log, index=False, encoding='utf-8-sig')
        print(f"✅ Successfully created the detailed AI-powered matching log: {output_file_log}")
        print("\n--- Sample of AI Verified Matches ---")
        print(df_log.head().to_string())
    else:
        print("ℹ️ No matches were found by the AI model.")


if __name__ == '__main__':
    # --- Configuration ---
    script_directory = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_directory, "..", "models", "t5-pharma-translator-v8")
    MOLECULAS_FILE = os.path.join(script_directory, "..", "data", "raw", "moleculas.xls")
    PRICE_FILES_DIR = os.path.join(script_directory, "..", "data", "raw", "prices")

    SIMILARITY_THRESHOLD = 90

    run_ai_powered_matching(MODEL_PATH, MOLECULAS_FILE, PRICE_FILES_DIR, similarity_threshold=SIMILARITY_THRESHOLD)
