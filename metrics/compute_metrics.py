import evaluate
from metrics.bleu.smooth_bleu import bleu_fromstr
from data.preprocess import decode_and_clean
import os
# Load the official HuggingFace rouge metric once
rouge = evaluate.load("metrics/rouge")

def compute_metrics(eval_preds, tokenizer, metrics=("bleu", "rouge"), output_dir=None, dataset_split="eval"):
    """
    Compute selected metrics from raw predictions and labels.
    """
    preds, labels = eval_preds

    prefix = os.path.join(output_dir, f"{dataset_split}_") if output_dir else f"{dataset_split}_"
    # Decode token IDs and clean up formatting
    clean_preds, clean_labels = decode_and_clean(preds, labels, tokenizer, prefix)

    return compute(clean_preds, clean_labels, metrics)


def compute(preds, labels, metrics):
    results = {}

    if "bleu" in metrics:
        results["bleu"] = bleu_fromstr(preds, labels,rmstop=True)

    if "rouge" in metrics:
        rouge_result = rouge.compute(
            predictions=preds, 
            references=labels,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        for key in ["rouge1", "rouge2", "rougeL"]:
            results[key] = round(rouge_result[key], 4)

    return results