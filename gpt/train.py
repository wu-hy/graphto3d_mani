# main.py

import os
import logging
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments
)
import wandb

# Prevent parallelism issues with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import custom modules
from training_config import config_params
from datacollator import MyDataCollator
from data_processing import load_and_prepare_dataset, load_and_prepare_dataset_from_json
from trainer import CustomTrainer  # Assuming this file exists in your project

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize wandb if desired
    wandb.init(project="gpt2-training", config=config_params)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config_params["tokenizer_file"])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    tokenized_datasets = load_and_prepare_dataset_from_json(
        # config_params["dataset_path"], 
        "/cluster/project/cvg/students/shangwu/graphto3d/GT/3DSSG_processed_files/relationships_train_clean.json",
        tokenizer, 
        config_params["context_length"],
        config_params["seed"]
    )
    # tokenized_datasets = load_and_prepare_dataset(
    #     config_params["dataset_path"], 
    #     tokenizer, 
    #     config_params["context_length"],
    #     config_params["seed"]
    # )

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=config_params["context_length"],
        n_layer=config_params["n_layer"],
        n_head=config_params["n_head"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_embd=config_params["n_emb"]
    )

    model = GPT2LMHeadModel(config)
    
    data_collator = MyDataCollator(tokenizer, mlm=False)
    
    train_args = TrainingArguments(
        output_dir=config_params["output_dir"],
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=config_params["steps"],
        logging_steps=config_params["steps"],
        logging_first_step=True,
        save_total_limit=5,
        save_steps=config_params["steps"],
        learning_rate=config_params["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        weight_decay=0.01,
        seed=config_params["seed"],
        load_best_model_at_end=True,
        no_cuda=False,
    )
    
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
