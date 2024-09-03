from argparse import ArgumentParser

from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from dataset import load_train_and_val_datasets
from model import get_lora_adapter, load_model_and_tokenizer

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="unsloth/Phi-3.5-mini-instruct"
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = get_lora_adapter(model, rank=args.rank)

    train_dataset, val_dataset = load_train_and_val_datasets(tokenizer)

    if args.load_in_4bit:
        optim = "adamw_4bit"
    else:
        optim = "adamw"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=optim,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            seed=3407,
            output_dir=f"outputs-{args.model_name}",
        ),
    )

    # if tokenizer.name_or_path.split("/")[-1] == "Phi-3.5-mini-instruct":
    #     instruction_part = "<|user|>\n"
    #     response_part = "<|assistant|>\n"
    # if (
    #     tokenizer.name_or_path.split("/")[-1] == "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    #     or tokenizer.name_or_path.split("/")[-1] == "Meta-Llama-3.1-8B-Instruct"
    # ):
    #     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
    #     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part=instruction_part,
    #     response_part=response_part,
    # )

    trainer_stats = trainer.train()

    model.save_pretrained(f"lora_model-{args.model_name.split('/')[-1]}-disfl_qa")
    tokenizer.save_pretrained(f"lora_model-{args.model_name.split('/')[-1]}-disfl_qa")
