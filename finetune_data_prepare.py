import glob
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, AutoTokenizer
import argparse
from omegaconf import OmegaConf


def tokenize_inputs(tokenizer, examples, input_column, output_column, max_length=256, dataset_eos_token = "</s>"):
    # hacky backward compatible
    different_eos = tokenizer.eos_token != dataset_eos_token
    out = {"labels": [], "input_ids": []}
    for prompt, response in zip(examples[input_column], examples[output_column]):
        if different_eos:
            if response.count(f"{dataset_eos_token} \n") > 0 or response.count(f"{dataset_eos_token}\n") > 0:
                response = response.replace(f"{dataset_eos_token} \n", f"{tokenizer.eos_token} \n") 

        prompt_len = len(tokenizer(prompt + "\n", return_tensors="pt")["input_ids"][0])

        # hack if our prompt is super long
        # we need to include some labels so we arbitrarily trunacate at max_length // 2
        # if the length is too long
        if prompt_len >= max_length // 2:
            # if prompt is too long, truncate
            # but make sure to truncate to at max 1024 tokens
            new_len = min(max_length // 2, len(prompt) // 2)
            prompt = prompt[:new_len]
            # get new prompt length
            prompt_len = tokenizer(prompt + "\n", return_tensors="pt", max_length=max_length // 2, truncation=True).input_ids.ne(tokenizer.pad_token_id).sum().item()

        assert prompt_len <= max_length // 2, f"prompt length {prompt_len} exceeds max length {max_length}"

        input_tokens = tokenizer(prompt + "\n" + response + tokenizer.eos_token,
                                 truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].squeeze()

        labels = input_tokens.clone()
        labels[:prompt_len] = -100
        if len(labels) < max_length:
            # pad to max_length with -100
            labels = torch.cat([labels, torch.full((max_length - len(labels),), -100)])

        assert (labels == -100).sum() < len(labels), f"Labels are all -100, something wrong. prompt length {prompt_len} exceeds max length {max_length}" 
        
        if (labels == -100).sum() == len(labels) - 1:
            print(prompt)
            print(response)
            raise

        input_tokens = tokenizer.pad({"input_ids": input_tokens}, padding="max_length", max_length=max_length)["input_ids"]
        out["labels"].append(labels)
        out["input_ids"].append(input_tokens)

    out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}

    return out


def load_data(config, tokenizer):
    dataset_path = config["dataset_path"]

    if os.path.exists(dataset_path):
        if os.path.isdir(dataset_path):
            files = glob.glob(os.path.join(dataset_path, "*_clean.jsonl"))
        else:
            files = [dataset_path]

        print(f"Reading files {files}")

        dataset = load_dataset("json", data_files=files, split="train")

    else:
        dataset = load_dataset(dataset_path)
        #dataset = load_dataset(dataset_path, revision= config["data_version"], split=config["n_samples"]) #"train")

    if dataset.get("test"):
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif dataset.get("validation"):
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        dataset = dataset.train_test_split(test_size=.05, seed=config["seed"])
        train_dataset, val_dataset = dataset["train"], dataset["test"]

    if config["streaming"] is False:
        kwargs = {"num_proc": config["num_proc"]}
    else:
        kwargs = {}

    # tokenize inputs and return labels and attention mask
    train_dataset = train_dataset.map(
        lambda ele: tokenize_inputs(tokenizer, ele, config["input_column"], config["output_column"],
                                    max_length=config["max_length"], dataset_eos_token = config["eos_token"]),
        batched=True,
        remove_columns=["source", "prompt", "id"],
        **kwargs
    )
    val_dataset = val_dataset.map(
        lambda ele: tokenize_inputs(tokenizer, ele, config["input_column"], config["output_column"],
                                    max_length=config["max_length"], dataset_eos_token = config["eos_token"]),
        batched=True,
        remove_columns=["source", "prompt","id"],
        **kwargs
    )

    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")

    # create dataloader with default data collator since we already have labels

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config["train_batch_size"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config["train_batch_size"],
    )

    return train_dataloader, val_dataloader

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation for finetuning.')
    parser.add_argument('--data_path', type=str, default=None, help='path to store the processed data downloaded from huggingface')
    parser.add_argument('--num_proc', type=int, default=8, help='number of processes to use to prepare dataset')
    parser.add_argument('--dataset', type=str, help='specify the dataset you want to use from huggingface datasets')
    parser.add_argument('--repo_name', type=str, default="Gptfinetune0/gpt-j", help='specify the hugging face repository from which to get tokenizer.') 
    parser.add_argument('--input_colum', type=str, help='specify the column in the hugging face dataset to use as input to the model.')
    parser.add_argument('--target_column', type=str, help='specify the column in hugging face repository to use as target label or model response.')
  
    args = parser.parse_args()
    
    
    if not args.dataset :
        raise ValueError("Provide the huggingface dataset to process.")
    elif not args.input_column:
        raise ValueError("Input column in dataset to be used by model has not been provided.")
    elif not args.target_column:
        raise ValueError("Output column in dataset to be used as model's output/response has not been provided")
    else:
        config = config = OmegaConf.load("../config.yaml")
        tokenizer = AutoTokenizer.from_pretrained("Gptfinetune0/gpt-j")
        train_dataloader, test_dataloader  =  load_dataset(config, tokenizer)
        print(len(train_dataloader), len(test_dataloader))

