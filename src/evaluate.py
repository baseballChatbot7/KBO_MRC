#%%
# !pip install torch==1.7.1
# !pip install datasets==1.4.1
# !pip install transformers==4.3.3
# !pip install sentencepiece==0.1.95
# !pip install nltk
# !pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
# !pip install nltk
# !pip install sentencepiece # tokenizer불러올때 필요
import nltk 
import torch
nltk.download('punkt')
from models.bart_model import KoBARTConditionalGeneration

#%%
from datasets import load_dataset, load_metric, load_from_disk
# datasets = load_dataset('squad_kor_v1')
metric = load_metric('squad')
from pytorch_lightning.utilities.seed import seed_everything
seed_everything(42)
# %%
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
from transformers import BartModel, BartForConditionalGeneration

tokenizer = get_kobart_tokenizer()
added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:", "질문:"]})

CKPT_PATH = '/opt/ml/code/pytorch_lightning_examples/2_QA/logs/runs/2021-05-14/05-29-48/checkpoints/last.ckpt'
model = KoBARTConditionalGeneration.load_from_checkpoint(checkpoint_path=CKPT_PATH)
# %%
max_source_length = 1024
max_target_length = 128
padding = False
preprocessing_num_workers=12
num_beams = 2
max_train_samples = 16
max_val_samples = 200
batch_size = 128
num_train_epochs = 4
# %%
def preprocess_function(examples):
    inputs = [
        f"제목: {t} 질문: {q}  글: {c} </s>"
        for t, q, c in zip(examples['title'], examples["question"], examples["text"])
    ]
    targets = [f'{a} </s>' for a in examples["answers"]]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )  # , return_tensors='pt')

        # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True
        )  # , return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# %%
dataset = load_from_disk('/opt/ml/input/data/data/retriever_dataset_split')
dataset = dataset.sort('document_id')
dataset = dataset.train_test_split(test_size=0.2)
column_names = dataset["train"].column_names
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_dataset = train_dataset.select(range(max_train_samples))
train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
        )
# %%
# eval_dataset = datasets["validation"]
eval_dataset = eval_dataset.select(range(max_val_samples))
eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

# %%
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
# %%
label_pad_token_id = tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            # model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )
# %%
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decoded_labels is for rouge metric, not used for f1/em metric
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    for f, r in zip(decoded_labels, decoded_preds):
        print(f, '/', r)
    formatted_predictions = [{"id": ex['document_id'], "prediction_text": decoded_preds[i]} for i, ex in enumerate(eval_dataset)]
    references = [{"id": ex["document_id"], "answers": ex["answers"]} for ex in eval_dataset]
    # for f, r in zip(formatted_predictions, references):
        # print(r['answers'], '/', f['prediction_text'])

    result = metric.compute(predictions=formatted_predictions, references=references)
    return result
# %%
args = Seq2SeqTrainingArguments(
    output_dir='kobart/outputs', 
    do_train=True, 
    do_eval=True, 
    predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    save_total_limit=1
)
# %%
trainer = Seq2SeqTrainer(
    model=model.model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# %%
# train_result = trainer.train(resume_from_checkpoint=None)
# %%
# print(train_result)
metrics = trainer.evaluate(
    max_length=max_target_length, num_beams=num_beams, metric_key_prefix='eval'
)
# %%
print(metrics)