import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from model import MambaTokenizerFast


def reg_compute_metrics(eval_preds):
    # Computes evaluation metrics for regression
    predictions, labels = eval_preds
    predictions = predictions.squeeze()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    rmse = np.sqrt(mse)

    return {"mse": mse, "mae": mae, "r2": r2, "rmse": rmse}


def cls_compute_metrics(eval_preds):
    # Computes evaluation metrics for binary classification
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)

    probs = logits[:, 1] if logits.shape[1] > 1 else logits.flatten()
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def show_results(results):
    # Displays the results of the evaluation
    print("\nFinal evaluation results:")
    print("-" * 40)
    for metric, value in sorted(results.items()):
        if metric.startswith("eval_"):
            metric = metric[5:]  # Remove 'eval_' prefix
        print(f"  {metric.upper():10s}: {value:.4f}")
    print("-" * 40)


def main():
    # load pretrained weights from huggingface hub
    model_checkpoint = "suzuki-2001/plant-genomic-jamba"
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        trust_remote_code=True,
    )

    # load pre-trained genomic-jamba model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True,
        num_labels=1,
        problem_type="regression",
    )

    # check model parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("Model parameters (M):", round(num_params, 2))

    # plant genomic benchmark
    task_name = "promoter_strength.leaf"
    dataset = load_dataset(
        "InstaDeepAI/plant-genomic-benchmark",
        task_name=task_name,
        trust_remote_code=True,
    )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["sequence"],
            padding="max_length",
            max_length=170,
            truncation=True,
        )
        tokenized["label"] = examples["label"]
        return tokenized

    # tokenize single-nucleotide resolution
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
    )

    # training config
    run_name = f"mambaAttn_pretrained_human-ref_50M_{task_name}"
    training_args = TrainingArguments(
        output_dir=f"./runs/{run_name}",
        max_steps=1000,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        weight_decay=0.1,
        learning_rate=6e-4,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=250,
        save_steps=250,
        fp16=False,
        bf16=True,
        optim="schedule_free_radam",  # no learning rate scheduling
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        run_name=run_name,
        remove_unused_columns=False,
        save_total_limit=2,
        seed=42,
        adam_beta1=0.9,
        adam_beta2=0.95,
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        # max_grad_norm=1.0,
        # report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=reg_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    eval_results = trainer.evaluate(tokenized_dataset["test"])

    show_results(eval_results)


if __name__ == "__main__":
    main()
