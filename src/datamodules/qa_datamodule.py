#%%
import os
from typing import Optional, Dict, Any
from collections import OrderedDict
from datasets import load_dataset, load_from_disk

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch
from functools import partial
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from transformers import DataCollatorWithPadding, default_data_collator
from src.datamodules.processing import post_processing_function, prepare_train_features, prepare_validation_features
#%%

class QADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = '/opt/ml/input/data/data/train_dataset_split',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
        max_seq_length: int = 384,
        max_answer_length: int = 30,
        doc_stride: int = 128,
        **kwargs,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.doc_stride = doc_stride
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.padding = "max_length"
        self.example_id_strings = OrderedDict()

    def train_dataloader(self):
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    

    
    @property
    def collate_fn(self):
        return default_data_collator if self.pad_to_max_length else DataCollatorWithPadding(self.tokenizer)     

    @property
    def pad_to_max_length(self):
        return self.padding == "max_length"
    
    @staticmethod
    def convert_to_train_features(*args, **kwargs):
        return prepare_train_features(*args, **kwargs)

    @staticmethod
    def convert_to_validation_features(*args, example_id_strings, **kwargs):
        return prepare_validation_features(*args, example_id_strings=example_id_strings, **kwargs)

    def setup(self, stage):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        train = stage == "fit"
        column_names = dataset["train" if train else "validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        self.answer_column_name = answer_column_name

        kwargs = {
            "tokenizer": self.tokenizer,
            "pad_on_right": self.tokenizer.padding_side == "right",
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "answer_column_name": answer_column_name,
            "max_length": self.max_seq_length,
            "doc_stride": self.doc_stride,
            "padding": self.padding,
        }

        prepare_train_features = partial(self.convert_to_train_features, **kwargs)

        if train:
            # dataset["tain"] = dataset["train"].select(range(64))
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=12,
                remove_columns=column_names,
                load_from_cache_file=True,
            )
            # dataset['train'].save_to_disk('/content/drive/MyDrive/data/mrc/processed_train')
            # dataset['train'] = load_from_disk('/content/drive/MyDrive/data/mrc/processed_train')

        if "test" not in dataset:
            # dataset["validation"] = dataset["validation"].select(range(64))
            kwargs.pop("answer_column_name")
            prepare_validation_features = partial(
                self.convert_to_validation_features, example_id_strings=self.example_id_strings, **kwargs
            )
            dataset["validation_original"] = dataset["validation"]  # keep an original copy for computing metrics
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=1,
                remove_columns=column_names,
                load_from_cache_file=True,
            )
        return dataset
    
    def load_dataset(self):
        # dataset = load_dataset("squad_kor_v1")
        dataset = load_from_disk('/content/drive/MyDrive/data/mrc/train_dataset')
        return dataset
    
    def split_dataset(self, dataset):
        return dataset

    def postprocess_func(
        self,
        dataset: Dataset,
        validation_dataset: Dataset,
        original_validation_dataset: Dataset,
        predictions: Dict[int, torch.Tensor],
    ) -> Any:
        return post_processing_function(
            datasets=dataset,
            predictions=predictions,
            answer_column_name=self.answer_column_name,
            features=validation_dataset,
            examples=original_validation_dataset,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=self.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir='/content/drive/MyDrive/mrc/1_generator_qa/outputs',
        )