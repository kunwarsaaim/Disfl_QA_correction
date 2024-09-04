import itertools
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel

from dataset import load_test_dataset, load_val_dataset
from metrics import evaluate_metrics


def inference(model, tokenizer, test_dataset, batch_size=64):
    model.eval()
    test_data_loader = DataLoader(
        test_dataset["text"], batch_size=batch_size, shuffle=False
    )

    inference_results = []

    for batch in tqdm(test_data_loader, desc="Processing Batches"):

        test_tokens = tokenizer(
            batch, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        input_lengths = test_tokens["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **test_tokens,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1,
            )

            new_tokens = outputs[:, input_lengths:]
            new_tokens = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        inference_results.append(new_tokens)
    inference_results = list(itertools.chain(*inference_results))
    return inference_results


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="lora_model-Phi-3.5-mini-instruct-disfl_qa"
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load model in 4-bit precision",
    )
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer.padding_side = "left"

    FastLanguageModel.for_inference(model)

    val_dataset = load_val_dataset(tokenizer)

    test_dataset = load_test_dataset(tokenizer)

    inference_val = inference(model, tokenizer, val_dataset, batch_size=args.batch_size)

    val_metrics = evaluate_metrics(inference_val, val_dataset["original question"])

    print("Validation metrics: ", val_metrics)

    with open(f"val_metrics-{args.model_path.split('/')[-1]}.txt", "w") as f:
        f.write(str(val_metrics))

    inference_test = inference(
        model, tokenizer, test_dataset, batch_size=args.batch_size
    )

    test_metrics = evaluate_metrics(inference_test, test_dataset["original question"])

    print("Test metrics: ", test_metrics)

    with open(f"test_metrics-{args.model_path.split('/')[-1]}.txt", "w") as f:
        f.write(str(test_metrics))
