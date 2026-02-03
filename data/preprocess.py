import numpy as np
import json
from torch.utils.data import Dataset

def tokenize_train(examples, tokenizer, use_vulnerability_info=False):
    """
    Tokenize training data for causal language modeling.
    """
    input_ids_list, attention_mask_list, label_ids_list = [], [], []

    for diff, vul, review in zip(
        examples["diff_hunk"], examples["vulnerable_line"], examples["review_message"]
    ):
        input_text = get_instruction() + "\n"
        input_text += f"Code Change:\n{diff}\n"
        if use_vulnerability_info:
            input_text += f"Vulnerable Line:\n{format_vulnerable_lines(vul)}\n"
        input_text += "Review Comment:\n"

        full_text = input_text + review + tokenizer.eos_token

        tokenized_full = tokenizer(full_text, padding="max_length", truncation=True, max_length=512)
        tokenized_input = tokenizer(input_text, truncation=True, max_length=512)

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        #input_len = len(tokenized_input["input_ids"])
        #pad_len = 512 - len(tokenizer(full_text, truncation=True, max_length=512)["input_ids"])
        #target_start = pad_len + input_len
    
        input_len = sum(tokenized_input["attention_mask"])
        full_len = sum(attention_mask)
        target_start = 512 - full_len + input_len
        labels = [-100] * target_start + input_ids[target_start:]
        labels = labels[:512]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        label_ids_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": label_ids_list,
    }


def tokenize_eval(examples, tokenizer, use_vulnerability_info=False):
    """
    Tokenize validation/test data.
    """
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for diff, vul, review in zip(
        examples["diff_hunk"], examples["vulnerable_line"], examples["review_message"]
    ):
        input_text = get_instruction() + "\n"
        input_text += f"Code Change:\n{diff}\n"
        if use_vulnerability_info:
            input_text += f"Vulnerable Line:\n{format_vulnerable_lines(vul)}\n"
        input_text += "Review Comment:\n"

        tokenized_input = tokenizer(
            input_text, padding="max_length", truncation=True, max_length=512
        )

        review_text = review + tokenizer.eos_token

        with tokenizer.as_target_tokenizer():
            label_enc = tokenizer(
                review_text, padding="max_length", truncation=True, max_length=512
            )

        labels = [
            (lid if mask == 1 else -100)
            for lid, mask in zip(label_enc["input_ids"], label_enc["attention_mask"])
        ]

        input_ids_list.append(tokenized_input["input_ids"])
        attention_mask_list.append(tokenized_input["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }





def format_vulnerable_lines(vul_lines):
    """
    Convert a list of vulnerable lines into a string.
    """
    return "\n".join([f"{v['change_type'].upper()}: {v['code_line']}" for v in vul_lines])

def get_instruction():
    return "Given the code change and the vulnerable line below, write a concise review comment."

def decode_and_clean(preds, labels, tokenizer, prefix=""):
    """
    Decode model predictions and labels, clean them by removing instruction prefixes,
    and save both raw and cleaned outputs in JSONL format.

    Args:
        preds (np.ndarray): Model prediction token IDs.
        labels (np.ndarray): Ground-truth label token IDs.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        prefix (str): Optional prefix for saved file names.

    Returns:
        Tuple[List[str], List[str]]: Cleaned predictions and labels as strings.
    """
    # Replace -100 with pad_token_id for decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode token IDs to strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Normalize whitespace
    decoded_preds = [' '.join(p.strip().split()) for p in decoded_preds]
    decoded_labels = [' '.join(l.strip().split()) for l in decoded_labels]

    # Save original decoded output to JSONL
    with open(f"{prefix}decoded_preds.jsonl", "w", encoding="utf-8") as f:
        for pred, label in zip(decoded_preds, decoded_labels):
            json_line = {"pred": pred, "label": label}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    # Strip prefix like "Review Comment:" if present
    clean_preds = [p.split("Review Comment:")[-1].strip() if "Review Comment:" in p else p for p in decoded_preds]
    clean_labels = [l.split("Review Comment:")[-1].strip() if "Review Comment:" in l else l for l in decoded_labels]

    # Save cleaned output to JSONL
    with open(f"{prefix}clean_preds.jsonl", "w", encoding="utf-8") as f:
        for pred, label in zip(clean_preds, clean_labels):
            json_line = {"pred": pred, "label": label}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    return clean_preds, clean_labels