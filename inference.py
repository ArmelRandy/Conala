import time
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset, where the dataset is a list of instructions (str)
    """
    def __init__(self, tokenizer, sentences):
        self.dataset = tokenizer([sentence for sentence in sentences], padding=True, truncation=True, max_length=2048, return_tensors="pt")
        self.length = len(sentences)
    def __iter__(self):
        for i in range(self.length):
            yield {
                "input_ids" : self.dataset.input_ids[i],
                "attention_mask" : self.dataset.attention_mask[i]
            }

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=8)

    return parser.parse_args()

if __name__ == "__main__" :
    args = get_args()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model=model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model = model.to(accelerator.device)
    dataset = load_dataset("neulab/conala", "mined")
    total = len(dataset["train"])
    
    for start in tqdm(range(0, total, args.batch_size)):
        end = start+args.batch_size
        sentences = [intent+"\n"+snippet for (intent, snippet) in zip( dataset["train"]["intent"][start:end], dataset["train"]["snippet"][start:end] )]
        train_data = TokenizedDataset(tokenizer, sentences)
        dataloader = DataLoader(train_data, batch_size=accelerator.num_processes)
        with open("predicted_conala_mined.jsonl", "a") as fout:
            print("Generation ...")
            start_time = time.time()
            state = accelerator.state
            for step, batch_ in tqdm(enumerate(dataloader)) :
                for key in batch_ :
                    batch_[key] = batch_[key].to(state.device)
                with state.split_between_processes(batch_, apply_padding=True) as batch :
                    with torch.no_grad():
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        outputs = accelerator.unwrap_model(model).generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            temperature=0.2,
                            top_p=0.9,
                            repetition_penalty=1.2,
                            max_new_tokens=100
                        )
                        outputs = accelerator.pad_across_processes(
                            outputs, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        outputs = accelerator.gather(outputs)
                        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        if accelerator.is_main_process :
                            for prompt in decoded_outputs :
                                fout.write(
                                    json.dumps(
                                        {"rewritten_intent" : prompt}
                                    )+"\n"
                            )
        duration = time.time() - start_time
        print("duration = "+str(duration))


