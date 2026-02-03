import yaml
import logging
from datasets import load_dataset, DatasetDict
import os
from models.model import load_model
from data.preprocess import tokenize_train, tokenize_eval
from trainers.causal_lm_trainer import CausalLMTrainer
from metrics.compute_metrics import compute_metrics
from transformers import TrainingArguments, EarlyStoppingCallback
from utils.io import prepare_output_dirs
from utils.logging import init_logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Load configuration
config_path = "configs/deepseek/deepseek_finetune.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set up output directories and logging
paths = prepare_output_dirs(config, use_timestamp=True)
init_logging(os.path.join(paths["logs"], "train.log"))
logger = logging.getLogger(__name__)
logger.info(f"Using config: {config_path}")

# Load model and tokenizer
model, tokenizer = load_model(config)
logger.info(f"Model loaded from: {config['model_path']}")

# Load and split dataset
raw_dataset = load_dataset("json", data_files=config["data_file"])["train"]
dataset = DatasetDict({
    "train": raw_dataset.filter(lambda x: x["split"] == "train"),
    "val": raw_dataset.filter(lambda x: x["split"] == "validation"),
    "test": raw_dataset.filter(lambda x: x["split"] == "test"),
})
logger.info("Dataset loaded: train=%d, val=%d, test=%d",
            len(dataset["train"]), len(dataset["val"]), len(dataset["test"]))

# Tokenize
use_vuln_info = config.get("use_vulnerability_info", False)
tokenized_dataset = DatasetDict({
    "train": dataset["train"].map(lambda x: tokenize_train(x, tokenizer, use_vuln_info), batched=True, remove_columns=dataset["train"].column_names),
    "val": dataset["val"].map(lambda x: tokenize_eval(x, tokenizer, use_vuln_info), batched=True, remove_columns=dataset["val"].column_names),
    "test": dataset["test"].map(lambda x: tokenize_eval(x, tokenizer, use_vuln_info), batched=True, remove_columns=dataset["test"].column_names),
})
logger.info("Tokenization complete.")

# Training setup
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=config["eval_steps"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    gradient_accumulation_steps=config["grad_accum_steps"],
    num_train_epochs=config["num_epochs"],
    learning_rate=float(config["learning_rate"]),
    fp16=config.get("fp16", True),
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=2,
    dataloader_num_workers=4,
    report_to="tensorboard",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=config.get("early_stop_patience", 2),
)

trainer = CausalLMTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metrics=("bleu", "rouge"), output_dir=paths["predictions"], dataset_split="eval"),
    callbacks=[early_stopping],
)

# Train
logger.info("Starting training.")
trainer.train()
logger.info("Training done.")

# Save model
model.save_pretrained(paths["checkpoints"])
tokenizer.save_pretrained(paths["tokenizer"])
logger.info("Model saved.")

# Evaluate
logger.info("Evaluating on test set...")
prediction_output = trainer.predict(tokenized_dataset["test"])
test_metrics = compute_metrics(
    eval_preds=(prediction_output.predictions, prediction_output.label_ids),
    tokenizer=tokenizer,
    metrics=("bleu", "rouge"),
    output_dir=paths["predictions"],
    dataset_split="test"
)

for key, value in test_metrics.items():
    logger.info(f"{key}: {value:.4f}")