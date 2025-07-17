# AI-Powered Pharmaceutical Matcher

## 1. Project Overview

This project is an end-to-end data pipeline designed to solve a complex, real-world business problem: the automated reconciliation of pharmaceutical product data from multiple, inconsistent supplier price lists. The system intelligently matches product names despite significant variations in descriptions, abbreviations, and formatting, a task that was previously done manually, leading to hours of work and potential for costly errors.

The solution evolved from a complex rule-based "expert system" into a sophisticated AI-powered "translator" by fine-tuning a T5 transformer model. This AI core can understand the semantic context of product names, allowing it to standardize messy, unstructured text into a clean, consistent format for high-accuracy matching.

---

## 2. The Problem

Pharmaceutical suppliers provide price lists with no standardized format. The same product can be described in many different ways, for example:

* `VALSARTANA 80 MG 30 COMP GERM`
* `VALSARTANA 80 MG COMP REV 2BLX15`
* `ACETILCISTEINA INF 120ML XRP EURO`
* `ACETILCISTEINA 20MG/ML XP IN 120ML G GER`

Manually matching thousands of these entries is incredibly inefficient and prone to human error, where a mistake in dosage or quantity could have significant financial consequences.

---

## 3. The Solution: An Iterative AI Approach

This project demonstrates a complete development cycle, moving from a traditional rule-based system to a more powerful and scalable machine learning solution.

### Stage 1: Rule-Based "Expert System"

The initial approach involved building a complex Python script with a large set of hand-crafted rules (using RegEx) to extract features like dosage, quantity, brand, and form. While effective for known patterns, this system was brittle and failed when new, unseen abbreviations or formats were introduced.

### Stage 2: AI-Powered "Translator"

To overcome the limitations of a rule-based system, the project pivoted to a machine learning approach. The core of the new system is a **fine-tuned T5 Transformer model** that acts as an intelligent "translator."

* **The Goal:** The model was trained to translate messy, inconsistent product names (`input_text`) into a perfectly clean, standardized format (`target_text`).

* **The Training:** A high-quality dataset was created by identifying "gold standard" matches from the initial data. This dataset was then used to fine-tune the `t5-base` model, teaching it the specific patterns and vocabulary of pharmaceutical product descriptions.

* **The Result:** The final script uses this custom AI model to standardize all product names first, then performs a simple and fast exact match on the clean text, resulting in much higher accuracy and robustness.

### Stage 3: The Bootstrapping Loop for Continuous Improvement

The project implements a "bootstrapping" workflow to continuously improve the AI model.

1. The current best model is used to process all the data.

2. The results are reviewed by a human expert to "grade" the AI's work.

3. New, correct matches discovered by the AI are added to the master training dataset.

4. The model is re-trained with this larger, more diverse dataset, creating an even smarter version.

This iterative cycle has proven highly effective, increasing the number of verified matches from an initial **48** to **over 230**—a nearly **500% improvement** in performance.

---

## 4. Tech Stack

* **Language:** Python

* **Data Manipulation:** Pandas

* **Machine Learning:** PyTorch

* **AI Models & Pipelines:** Hugging Face Transformers, Datasets, Accelerate

* **Core Model:** T5 (Text-to-Text Transfer Transformer)

* **File Handling & Automation:** OS, RegEx

---

## 5. Project Structure

The project is organized into a clean, professional structure to separate data, models, and code.

```
/AI-Powered-Pharmaceutical-Matcher/
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── moleculas.xls
│   │   └── 📂 prices/
│   │       └── ... (original price files)
│   └── 📂 processed/
│       ├── MASTER_TRAINING_DATASET.csv
│       ├── model_predictions_vX.csv
│       └── 📂 validation_reports/
│           └── ... (verified & rejected logs)
├── 📂 models/
│   ├── t5-pharma-translator-v1/
│   └── t5-pharma-translator-v2/
├── 📂 scripts/
│   ├── 01_create_initial_dataset.py
│   ├── 02_train_model.py
│   ├── 03_bootstrap_new_data.py
│   ├── 04_run_final_matching.py
│   └── 05_validate_ai_matches.py
├── 📄 .gitignore
└── 📄 README.md
```

---

## 6. Workflow & How to Use

The project follows a clear, step-by-step workflow.

1. **Data Preparation (Optional):** Run `01_create_initial_dataset.py` to generate the first batch of training data using rule-based verification.

2. **Model Training:** Run `02_train_model.py`. This script takes the master training dataset and fine-tunes the T5 model, saving the new, smarter version to the `models/` directory.

3. **Bootstrapping (The Improvement Loop):**

    * Run `03_bootstrap_new_data.py` to use the latest model to process all raw data and generate a `model_predictions.csv` file.

    * Manually review this predictions file, correct any errors, and add the new, verified pairs to your `MASTER_TRAINING_DATASET.csv`.

    * Repeat Step 2 to train an even better model.

4. **Final Matching:** Run `04_run_final_matching.py` to use your best-trained model to perform the final, end-to-end matching task and generate the `moleculas_updated_final_AI.csv` report.

5. **Validation (Optional):** Run `05_validate_ai_matches.py` to use the old rule-based system as a final check on the AI's output, separating the results into verified and rejected files for review.
