import os
import torch
import pandas as pd
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from argparse import ArgumentParser

# disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Define the data collator
def data_collator(data):
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
        "attention_mask": torch.stack(
            [torch.tensor(f["attention_mask"]) for f in data]
        ),
        "labels": torch.tensor([f["label"] for f in data]),
    }

# Define the compute_metrics function
def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions.argmax(-1)
    acc = acc_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels)
    recall = recall_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Configuration
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_path', default='./agency_dataset/', required=False) 
    parser.add_argument('-m', '--model_type', default="bert-base-uncased", required=False) 
    parser.add_argument('-cp', '--checkpoint_path', default='./checkpoints', required=False) 
    parser.add_argument('-lr', '--learning_rate', default=2e-5, required=False)
    parser.add_argument('-e', '--num_epochs', default=10, required=False)
    parser.add_argument('-tb', '--train_bsz', default=8, required=False)
    parser.add_argument('-eb', '--eval_bsz', default=16, required=False)
    parser.add_argument('-wd', '--weight_decay', default=0.01, required=False)
    args = parser.parse_args()

    train_df = pd.read_csv(args.dataset_path + "train.csv")
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('\n Length:', len(train_df))
    train_dataset = datasets.Dataset.from_pandas(train_df)

    val_df = pd.read_csv(args.dataset_path + "val.csv")
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    val_dataset = datasets.Dataset.from_pandas(val_df)

    test_df = pd.read_csv(args.dataset_path + "test.csv")
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    test_dataset = datasets.Dataset.from_pandas(test_df)

    acc_metric = datasets.load_metric("accuracy")
    precision_metric = datasets.load_metric("precision")
    recall_metric = datasets.load_metric("recall")
    f1_metric = datasets.load_metric("f1")

    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = train_dataset.map(
        tokenize_function, batched=True, batch_size=len(train_dataset)
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_function, batched=True, batch_size=len(val_dataset)
    )

    tokenized_test_dataset = test_dataset.map(
        tokenize_function, batched=True, batch_size=len(test_dataset)
    )

    # Load the BERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_type, num_labels=2
    )

    # Define the training arguments
    training_args = TrainingArguments(
        # To turn off wandb
        report_to=None,
        output_dir=args.checkpoint_path,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,  # enable mixed precision training
        gradient_accumulation_steps=2,  # accumulate gradients for every 2 batches
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()
    print('\n\n\n Testing -------------------------------- \n')
    trainer.evaluate(eval_dataset=tokenized_test_dataset)