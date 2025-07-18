import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import os

print(torch.cuda.is_available())

def train_pharma_translator(csv_path: str, model_output_dir: str, base_model: str = "t5-base"):
    print(f"Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{csv_path}' was not found.")
        print("Please make sure your training data is in the correct directory.")
        return

    if 'input_text' not in df.columns or 'target_text' not in df.columns:
        print("❌ ERROR: Your CSV must contain 'input_text' and 'target_text' columns.")
        return

    df['input_text'] = "translate: " + df['input_text'].astype(str)
    df['target_text'] = df['target_text'].astype(str)

    dataset = Dataset.from_pandas(df)

    print(f"Initializing tokenizer and model from '{base_model}'...")

    try:
        tokenizer = T5Tokenizer.from_pretrained(base_model)
        model = T5ForConditionalGeneration.from_pretrained(base_model)
    except OSError:
        print(f"❌ ERROR: Could not find the base model at '{base_model}'.")
        print("If this is your first time, make sure the base_model is 't5-base'.")
        print("If fine-tuning, ensure the path to your previous model is correct.")
        return

    def tokenize_function(examples):
        inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=128)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], padding="max_length", truncation=True, max_length=128)

        inputs['labels'] = labels['input_ids']
        return inputs

    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(model_output_dir, "training_checkpoints"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir='./logs',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n--- Starting Model Training ---")
    print(f"Training with {len(dataset)} examples for {training_args.num_train_epochs} epochs.")
    print("This may take a significant amount of time. You can monitor progress below.")

    trainer.train()

    print("\n--- Training Complete! ---")
    print(f"Saving your custom model to: {model_output_dir}")

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"\n✅ Your model is trained and ready! Find it in the '{model_output_dir}' folder.")


# if __name__ == '__main__':
#     # --- Configuration ---
#     # This script will look for your dataset here:
#     DATASET_PATH = "../data/exceptions_caught.csv"
#
#     # This is where your new, custom model will be saved:
#     MODEL_OUTPUT_PATH = "../models/t5-pharma-translator-v12"
#
#     # For the very first training, we use the base Google model.
#     # For re-training/fine-tuning, you would change this to MODEL_OUTPUT_PATH
#     # to start from your previously trained model.
#     BASE_MODEL = "../models/t5-pharma-translator-v11"
#
#     train_pharma_translator(DATASET_PATH, MODEL_OUTPUT_PATH, base_model=BASE_MODEL)
