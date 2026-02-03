import os
import datetime
import json

def prepare_output_dirs(config, use_timestamp=True):
    """
    Create output directories for logs, checkpoints, tokenizer, and predictions.
    """
    base_output = config["output_dir"]

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = os.path.join(base_output, timestamp)

    paths = {
        "base": base_output,
        "logs": os.path.join(base_output, "logs"),
        "checkpoints": os.path.join(base_output, "checkpoints"),
        "tokenizer": os.path.join(base_output, "tokenizer"),
        "predictions": os.path.join(base_output, "predictions"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Update the config
    config["output_dir"] = base_output

    return paths

def save_predictions(predictions, labels, output_dir, file_prefix="predictions"):
    """
    Save predictions and labels as a JSONL file.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{file_prefix}.jsonl")

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for pred, label in zip(predictions, labels):
            line = {"pred": pred, "label": label}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    return jsonl_path