from datasets import load_dataset


def formatting_prompts_func(example, tokenizer):
    instructions = "Remove disfluency from the following question:"
    inputs = example["disfluent question"]
    outputs = example["original question"]

    message = [
        {"role": "user", "content": instructions + "\n\n" + inputs},
        {"role": "assistant", "content": outputs},
    ]
    chat_inputs = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )
    return {"text": chat_inputs}


def formatting_prompt_test_data(example, tokenizer):
    instructions = "Remove disfluency from the following question:"
    inputs = example["disfluent question"]

    message = [
        {"role": "user", "content": instructions + "\n\n" + inputs},
    ]
    chat_inputs = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    return {"text": chat_inputs}


def load_train_and_val_datasets(tokenizer):
    train_dataset = load_dataset("google-research-datasets/disfl_qa", "train")
    val_dataset = load_dataset("google-research-datasets/disfl_qa", "validation")

    train_dataset = train_dataset.map(
        formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}
    )
    val_dataset = val_dataset.map(
        formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}
    )
    return train_dataset, val_dataset


def load_test_dataset(tokenizer):
    test_dataset = load_dataset("google-research-datasets/disfl_qa", "test")
    test_dataset = test_dataset.map(
        formatting_prompt_test_data, fn_kwargs={"tokenizer": tokenizer}
    )
    return test_dataset
