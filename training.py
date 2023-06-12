import os
import argparse

import wandb
import torch

from accelerate import Accelerator

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict

wandb.login()

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )   

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")

    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_output_length", type=int, default=2048)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000)

    return parser.parse_args()

if __name__ == "__main__" :
    args = get_args()
    set_seed(args.seed)

    tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    datasets=load_dataset(
        "neulab/conala", 
        use_auth_token=True,
        num_proc=args.num_workers 
    )
    
    datasets=datasets.filter(lambda x : x["intent"] is not None)
    datasets=datasets.filter(lambda x : x["rewritten_intent"] is not None)

    def f(x):
        x["intent"] = x["intent"]+"\n"+x["snippet"]
        return x
    
    datasets = datasets.map(lambda x : f(x))
    
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["intent"],
            max_length=args.max_input_length,
            truncation=True
        )
        labels = tokenizer(
            examples["rewritten_intent"],
            max_length=args.max_output_length,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets=datasets.map(preprocess, batched=True)
    tokenized_datasets=tokenized_datasets.remove_columns(datasets["train"].column_names)

    model=AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index}
    )
    
    model_name = args.model_name_or_path.split("/")[-1]
    
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-finetuned-"+args.dataset_name.split("/")[-1],
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.num_warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.log_freq,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        dataloader_drop_last=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="wandb",
        run_name="run"+model_name+"-"+args.dataset_name.split("/")[-1],
        save_total_limit=2,
        ddp_find_unused_parameters=False,
    )


    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback]
    )

    print("Training...")
    trainer.train()