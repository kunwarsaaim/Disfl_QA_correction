from argparse import ArgumentParser

from transformers import TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
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
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether to load model in 4-bit precision",
    )
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    if args.load_in_4bit:
        quant = "4bit"
    else:
        quant = "fp16"
    model = get_lora_adapter(model, rank=args.rank)

    train_dataset, val_dataset = load_train_and_val_datasets(tokenizer)

    if args.load_in_4bit:
        optim = "adamw_8bit"
    else:
        optim = "adamw_torch"

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # Number of evaluations with no improvement
        early_stopping_threshold=0.05,  # Minimum change to qualify as an improvement
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=3,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=optim,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=112,
            save_strategy="steps",
            save_steps=112,
            save_total_limit=3,
            load_best_model_at_end=True,
            seed=3407,
            output_dir=f"outputs-{args.model_name}-{quant}-disfl_qa",
        ),
        callbacks=[early_stopping_callback],
    )

    trainer_stats = trainer.train()

    model.save_pretrained(
        f"lora_model-{args.model_name.split('/')[-1]}-{quant}-disfl_qa"
    )
    tokenizer.save_pretrained(
        f"lora_model-{args.model_name.split('/')[-1]}-{quant}-disfl_qa"
    )
