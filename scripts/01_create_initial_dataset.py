import pandas as pd
import re
import os
from rapidfuzz import fuzz

def get_standardized_features(name):
    """
    Extracts structured features and creates a highly standardized base name
    for the most accurate comparison.
    """
    if not isinstance(name, str):
        return {'original': '', 'base_name': '', 'dosage': 0, 'quantity': 0}

    original_name = name
    name = name.lower().replace('+', ' ')

    # Pre-standardize specific known abbreviations
    name = re.sub(r'val\.', 'valerato ', name)

    # --- Feature Extraction (before extensive manipulation) ---
    dosage_val = 0
    if dosage_match := re.search(r'(\d[\d.,]*)\s*(mg|ml|g)\b', name):
        try:
            dosage_val = float(dosage_match.group(1).replace(',', '.'))
        except ValueError:
            dosage_val = 0

    quantity = 1
    if bl_match := re.search(r'(\d+)\s*bl\s*x\s*(\d+)', name):
        quantity = int(bl_match.group(1)) * int(bl_match.group(2))
    elif c_match := re.search(r'\b(c|com)\s+(\d+)\b', name):
        quantity = int(c_match.group(2))
    elif q_match := re.search(r'(\d+)\s*(comp|cap|cps|cpr|drg|comprimido|capsula)', name):
        quantity = int(q_match.group(1))
    elif g_match := re.search(r'(\d+)\s*g\b', name) and dosage_val == 0:
        # Handle cases like "POM 20G" as quantity if no mg/ml dosage is found
        quantity = int(g_match.group(1))

    # --- Text Standardization for Fuzzy Matching ---
    replacements = {
        r'\bc\b': 'com', r'\bcomp\b': 'comprimido', r'\bcompr\b': 'comprimido', r'\bcpr\b': 'comprimido',
        r'\bcps\b': 'capsula', r'\bcap\b': 'capsula', r'\bdrg\b': 'dragea', r'\bsusp\b': 'suspensao',
        r'\bgts\b': 'gotas', r'\bsol\b': 'solucao', r'\bsl\b': 'solucao', r'\bcr\b': 'creme',
        r'\bpom\b': 'pomada', r'\boft\b': 'oftalmica', r'\bof\b': 'oftalmica',
    }
    for old, new in replacements.items(): name = re.sub(old, new, name)

    form_groups = {
        'unidade': ['comprimido', 'capsula', 'dragea'],
        'liquido': ['suspensao', 'gotas', 'solucao'],
        'topico': ['creme', 'pomada', 'gel']
    }
    for group_name, forms in form_groups.items():
        for form in forms: name = name.replace(form, group_name)

    name = re.sub(r'[^a-z\s]', '', name) # Remove all non-alpha characters and numbers

    noise_words = [
        'mg', 'ml', 'g', 'nova', 'bio', 'rev', 'ems', 'neo', 'quimica', 'eurofarma', 'medley',
        'ache', 'generico', 'similar', 'original', 'plus', 'dc', 'vit', 'vitamina', 'teu',
        'geo', 'oral', 'po', 'bl', 'x', 'com', 'frasco', 'caixa', 'unidade', 'liquido', 'topico',
        'oftalmica', 'am', 'ger'
    ]
    words = [word for word in name.split() if word not in noise_words]
    base_name = ' '.join(words)

    return {
        'original': original_name, 'base_name': base_name.strip(),
        'dosage': dosage_val, 'quantity': quantity
    }

def find_excel_header_row(file_path, num_rows_to_check=20):
    try:
        df_temp = pd.read_excel(file_path, header=None, nrows=num_rows_to_check)
        header_keywords = ['produto', 'descrição', 'ean', 'código', 'codprod', 'descricao']
        for i, row in df_temp.iterrows():
            non_empty_cells = row.dropna()
            if len(non_empty_cells) < 3: continue
            string_cells = [cell for cell in non_empty_cells if isinstance(cell, str)]
            is_text_heavy = (len(string_cells) / len(non_empty_cells)) > 0.6
            has_keyword = any(kw in str(cell).lower() for cell in string_cells for kw in header_keywords)
            if is_text_heavy and has_keyword: return i
        return 0
    except Exception as e:
        print(f"Error trying to find header for {file_path}: {e}")
        return 0

def is_price_column(column_series, column_name):
    column_name_lower = str(column_name).lower()
    non_price_keywords = ['%', 'desconto', 'desc', 'percentual']
    if any(keyword in column_name_lower for keyword in non_price_keywords): return False
    known_price_keywords = ['pmc', 'pf', 'liquido', 'preco', 'valor', 'vlr']
    if any(keyword in column_name_lower for keyword in known_price_keywords): return True
    sample = column_series.dropna().head(20)
    if sample.empty: return False
    price_like_count = 0
    currency_pattern = re.compile(r'r\$\s*[\d.,]+', re.IGNORECASE)
    for value in sample:
        if isinstance(value, str) and currency_pattern.search(value.strip()): price_like_count += 1
        elif isinstance(value, (int, float)) and value >= 1: price_like_count += 1
    return (price_like_count / len(sample)) >= 0.6

def parse_currency_to_float(value):
    if isinstance(value, str):
        cleaned_value = value.replace('R$', '').replace('.', '').replace(',', '.').strip()
        try:
            parsed_float = float(cleaned_value)
            return parsed_float if parsed_float > 0 else None
        except ValueError: return None
    elif isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None

def update_medicine_prices_dynamic(moleculas_file_path, moleculas_sheet_name, price_files_directory):
    match_log = []
    print("--- Starting Price Update Process ---")
    try:
        df_moleculas = pd.read_excel(moleculas_file_path, sheet_name=moleculas_sheet_name)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading moleculas file: {e}")
        return None, None
    product_col_moleculas = 'Descrição' if 'Descrição' in df_moleculas.columns else 'Produto'
    df_moleculas['Structured_Features'] = df_moleculas[product_col_moleculas].apply(get_standardized_features)

    price_sources = {}
    known_product_column_keywords = ['produto', 'descrição', 'prod_nome', 'item', 'nome']
    print(f"\n--- Loading All Price Files from '{price_files_directory}' ---")
    for filename in os.listdir(price_files_directory):
        if not filename.endswith(('.xlsx', '.xls')): continue
        file_path = os.path.join(price_files_directory, filename)
        try:
            sheet_name = os.path.splitext(filename)[0]
            header_row = find_excel_header_row(file_path)
            df_price = pd.read_excel(file_path, header=header_row)
            product_col = next((c for c in df_price.columns for kw in known_product_column_keywords if str(c).lower() in kw), df_price.columns[0])
            price_cols = [c for c in df_price.columns if str(c).lower() != product_col.lower() and is_price_column(df_price[c], c)]
            if price_cols:
                print(f"  Loaded '{filename}'. Product: '{product_col}', Found Price Columns: {price_cols}")
                df_price['Structured_Features'] = df_price[product_col].apply(get_standardized_features)
                price_sources[sheet_name] = {'df': df_price, 'product_col': product_col, 'price_cols': price_cols, 'filename': filename}
                df_moleculas[f'Price_{sheet_name}'] = None
            else:
                print(f"  Warning: Skipped '{filename}'. No reliable price columns were found.")
        except Exception as e:
            print(f"  An error occurred while loading {filename}: {e}")

    print("\n--- Starting Price Matching (Expert System V3) ---")
    fuzzy_threshold = 95
    for index, row in df_moleculas.iterrows():
        molecula_features = row['Structured_Features']
        if not molecula_features or not molecula_features['base_name']: continue

        for source_name, source_data in price_sources.items():
            best_match_row, highest_similarity = None, 0
            for price_idx, price_row in source_data['df'].iterrows():
                price_features = price_row['Structured_Features']
                if not price_features or not price_features['base_name']: continue
                if molecula_features['dosage'] != price_features['dosage'] or \
                        molecula_features['quantity'] != price_features['quantity']:
                    continue

                name_similarity = fuzz.token_set_ratio(molecula_features['base_name'], price_features['base_name'])
                if name_similarity > highest_similarity:
                    highest_similarity = name_similarity
                    best_match_row = price_row

            if best_match_row is not None and highest_similarity >= fuzzy_threshold:
                best_price = float('inf')
                for col_name in source_data['price_cols']:
                    parsed_price = parse_currency_to_float(best_match_row.get(col_name))
                    if parsed_price is not None: best_price = min(best_price, parsed_price)

                if best_price != float('inf'):
                    formatted_price = f"R$ {best_price:.2f}".replace('.', ',')
                    df_moleculas.at[index, f'Price_{source_name}'] = formatted_price
                    match_log.append({
                        'Moleculas_Produto (Y)': molecula_features['original'],
                        'Matched_Price_File_Produto (X)': best_match_row['Structured_Features']['original'],
                        'Assigned_Price (Z)': formatted_price,
                        'Similarity_Score': f"{highest_similarity:.2f}% (Verified)",
                        'Price_Source_File': source_data['filename'],
                    })

    print("\n--- Price Matching Complete ---")
    df_moleculas.drop(columns=['Structured_Features'], inplace=True, errors='ignore')
    log_df = pd.DataFrame(match_log)
    return df_moleculas, log_df

# --- Configuration and Execution ---
moleculas_file_path = r"../data/raw/moleculas.xls"
moleculas_main_sheet_name = "GENERICO"
price_files_directory = r"../data/raw/prices"

updated_moleculas_df, log_df = update_medicine_prices_dynamic(
    moleculas_file_path, moleculas_main_sheet_name, price_files_directory
)

if updated_moleculas_df is not None:
    output_file_path = "../data/processed/moleculas_updated_final.csv"
    updated_moleculas_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Successfully created the main updated file: {output_file_path}")
if log_df is not None and not log_df.empty:
    log_file_path = "../data/processed/matching_log.csv"
    log_df.to_csv(log_file_path, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully created the detailed matching log: {log_file_path}")
    print("\n--- Sample of Matches Logged ---")
    print(log_df.head().to_string())
else:
    print("\nℹ️ No matches were found, so no matching log was created.")