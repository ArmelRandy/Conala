# Conala Translation

A repository on how to perform a completion of the conala-corpus.

# What is this about?
This repository is dedicated to the effort of creating an instruction dataset based on the [conala-corpus](https://conala-corpus.github.io). As a matter of fact, conala is a dataset 
obtained by crawling Stack Overflow. It is available on Hugging Face Hub under the name [neulab/conala](https://huggingface.co/datasets/neulab/conala). The dataset is divided into 2 parts :
- **curated** : It contains high quality data and is divided into a training set and a test set. 
- **mined** : It contains the data that we not processed and so they are likely to be noisier. This represents the biggest part of the whole corpus.

# Table
1. [Overview of the method](#overview)
  - [The problem](#the-problem)
  - [Our approach](#our-approach)
2. [Quickstart](#quickstart)
  - [Step by step installation with conda](#step-by-step-installation-with-conda)
  - [Training](#training)
  - [Inference](#inference)
3. [Acknowledgements](#acknowledgments)

# Overview
Conala is a dataset whose each example correspond to a specific code-related problem or use-case. The dataset provides multiple information but we are particularly interested in 3 columns: 
- *intent* : Natural Language intent (i.e., the title of a Stack Overflow question)
- *snippet* : A code snippet that implements the intent.
- *rewritten_intent* : Crowdsourced revised intents that try to better reflect the full meaning of the code, typically done by incorporating variable names and function arguments that appeared in the code into the intent.

From the description above, we notice that the *rewritten_intent* is the column that is close enough to be considered an instruction in an instruction fine-tuning setting. However we can not make
use of conala-curated as a fine-tuning dataset because the dataset is pretty small (2,379 training and 500 test examples) and it is a benchmark on which we can evaluate LLM. The idea that we have
was to use conala-mined. 

## The problem
conala-mined consists of almost 600K examples. For this part of the dataset, there is no *rewritten_intent*, what can we do about it?
- Stick with the *intent* and frame it as the instruction : this method is not ideal. I rapid scan of the conala-curated and conala-mined makes it clear that the intent is not informative enough to describe
the code snippet. 
- Try to reconstruct *rewritten_intent* : this method is more viable and it is what we decided to go with. But how can we reconstruct the column *rewritten_intent*?

## Our approach
In order to reconstruct the column *rewritten_intent* we had to understand what it is about. As its description suggest, it is a revised version of the intent. This implies that we may have to make use of the
provided column *intent*. Another important thing that is mentionned in its description is the fact that this more accurate description may include variable names and function arguments that appear in the code snippet.
Which means that we can not expect to reconstruct the *rewrittent_intent* by solely relying on the *intent*. We then have the idea to create a mapping [intent + snippet] -> rewritten_intent.

We did that mapping by fine-tuning a LLM, namely [google-ul2](https://huggingface.co/google/ul2) on the sequence to sequence task that we created
```
### Input
<intent>+"\n"+<snippet>
### Output
<rewritten_intent>
```
We used conala-curated training and test set as our task's training and validation set respectively.


# Quickstart
In this section, we describe how to train a translator/transcriber and how to make the inference as fast as possible with the help of 🤗's [accelerate](https://github.com/huggingface/accelerate)

## Step by step installation with conda

## Training
Our training procedure makes use of the 🤗[PEFT](https://github.com/huggingface/peft) library to perform parameter-efficient fine-tuning. This is particularly important as UL2 is a 20B parameter model.
The training command is as follows
```
accelerate launch training.py \
    --model_name_or_path="google/ul2" \
    --max_input_length 2048 \
    --max_output_length 2048 \
    --max_steps 10000 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type="linear" \
    --num_warmup_steps 500 \
    --weight_decay 0.01 \
    --log_freq 100 \
    --eval_freq 100 \
    --save_freq 1000 \
```
The code was specifically written for the purpose of conala but it is robust to a change in the model used. Feel free to try using any seq2seqLM e.g. T5, BART etc.

## Inference
The burden of the translation of conala is concentrated on the training procedure. Nevertheless, the inference is to take care of. It has to be done properly in order to alleviate the time it
could require because we have 600K rewritten intents to recover. The command to run the inference on the whole conala-mined is

```
accelerate launch inference.py
```
The predictions are going to be saved in a json file.

# Acknowledgments
- [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github.io)
