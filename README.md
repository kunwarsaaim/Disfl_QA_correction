# Question Rewrite Model for Disfluency Correction

This repository contains the implementation of a question rewrite model developed for the Disfl QA benchmark dataset. The primary objective of this project is to build a model that rewrites noisy, disfluent questions into clean, fluent ones, thereby improving the accuracy of question-answering systems in conversational workflows.

## Installation

Install the required packages using the following command:

```bash
pip install torch
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes triton evaluate
pip install rouge_score
```

## Usage

To train the model, run the following command:

```bash
python train.py --model_name <huggingface_model_name> --max_seq_length 256 --load_in_4bit --rank 8
```

To evaluate the model, run the following command:

```bash
python evaluate.py --model_path <peft_model_path> --max_seq_length 256 --load_in_4bit --rank 8
```

The following metrics will be calculated:

- rouge-1
- rouge-l
- bleu
- exact match

## Test Dataset Metrics

| Model-LORA            | Model                                       | Quantization | Rank | ROUGE-1 | ROUGE-L | BLEU   | Exact Match |
| --------------------- | ------------------------------------------- | ------------ | ---- | ------- | ------- | ------ | ----------- |
| Llama-3.1-8B-Instruct | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | 4bit         | 8    | 0.9487  | 0.9353  | 0.8816 | 0.6780      |
| Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct       | fp16         | 8    | 0.9458  | 0.9336  | 0.8734 | 0.6580      |
| Phi-3.5-mini-instruct | unsloth/Phi-3.5-mini-instruct               | 4bit         | 8    | 0.9589  | 0.9490  | 0.9028 | 0.7340      |
| Phi-3.5-mini-instruct | unsloth/Phi-3.5-mini-instruct               | fp16         | 8    | 0.9603  | 0.9502  | 0.9037 | 0.7392      |

## Validation Dataset Metrics

| Model-LORA            | Model                                       | Quantization | Rank | ROUGE-1 | ROUGE-L | BLEU   | Exact Match |
| --------------------- | ------------------------------------------- | ------------ | ---- | ------- | ------- | ------ | ----------- |
| Llama-3.1-8B-Instruct | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | 4bit         | 8    | 0.9582  | 0.9460  | 0.8979 | 0.7290      |
| Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct       | fp16         | 8    | 0.9533  | 0.9396  | 0.8931 | 0.7170      |
| Phi-3.5-mini-instruct | unsloth/Phi-3.5-mini-instruct               | 4bit         | 8    | 0.9710  | 0.9628  | 0.9253 | 0.8010      |
| Phi-3.5-mini-instruct | unsloth/Phi-3.5-mini-instruct               | fp16         | 8    | 0.9721  | 0.9646  | 0.9274 | 0.7970      |
