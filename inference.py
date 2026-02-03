import yaml
import logging
import os
from datasets import load_dataset, DatasetDict
from models.model import load_model
from data.preprocess import tokenize_eval,tokenize_train
from transformers import TrainingArguments
from trainers.causal_lm_trainer import CausalLMTrainer
from metrics.compute_metrics import compute_metrics
from utils.io import prepare_output_dirs
from utils.logging import init_logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load configuration
config_path = "configs/deepseek/deepseek_baseline.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set up output directories and logging
paths = prepare_output_dirs(config, use_timestamp=True)
init_logging(os.path.join(paths["logs"], "eval.log"))
logger = logging.getLogger(__name__)
logger.info(f"Loaded config file: {config_path}")

# Load model and tokenizer
model, tokenizer = load_model(config)
logger.info(f"Successfully loaded model from: {config['model_path']}")

# Load and split dataset
raw_dataset = load_dataset("json", data_files=config["data_file"])["train"]
dataset = DatasetDict({
    "test": raw_dataset.filter(lambda x: x["split"] == "test"),
})
logger.info(f"Test data loaded: {len(dataset['test'])}")

# Tokenize
use_vuln_info = config.get("use_vulnerability_info", False)
tokenized_dataset = DatasetDict({
    "test": dataset["test"].map(lambda x: tokenize_eval(x, tokenizer, use_vuln_info), batched=True, remove_columns=dataset["test"].column_names),
})
logger.info("Tokenization complete.")

# inference setup
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_eval_batch_size=config["eval_batch_size"],
    dataloader_num_workers=4,
    report_to="none",
)

trainer = CausalLMTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metrics=("bleu", "rouge"), output_dir=paths["predictions"], dataset_split="test"),
)

# inference
logger.info("Starting inference...")
predictions = trainer.predict(tokenized_dataset["test"])

# Evaluate
test_metrics = compute_metrics(
    eval_preds=(predictions.predictions, predictions.label_ids),
    tokenizer=tokenizer,
    metrics=("bleu", "rouge"),
    output_dir=paths["predictions"],
    dataset_split="test"
)

logger.info("Test set metrics:")
for key, value in test_metrics.items():
    logger.info(f"{key}: {value:.4f}")
