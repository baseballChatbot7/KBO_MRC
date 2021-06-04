from typing import Any, List, Dict
from functools import partial

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from datasets import load_metric
from transformers import AdamW, get_cosine_schedule_with_warmup, AutoModelForQuestionAnswering, AutoTokenizer
from src.utils.metric import SquadMetric
from src.models.modules.qa_model import CustomQAModel
from src.utils import template_utils

log = template_utils.get_logger(__name__)

class Reader(LightningModule):
    def __init__(
        self, 
        lr: float = 0.001, 
        weight_decay: float = 0.0005, 
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = CustomQAModel.from_pretrained("/content/drive/MyDrive/mrc/1_generator_qa/kbo_custom_model")
        self.tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/mrc/1_generator_qa/kbo_custom_model")

        self.metric_hist = {
            "train/loss": [],
            "val/loss": [],
        }

    def setup(self, stage):
        self.configure_metrics(stage)
        self.configure_optimizers()

    def forward(self, inputs):
        return self.model(
            **inputs
        )

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch.pop("offset_mapping")
        example_ids = batch.pop("example_id")
        outputs = self.model(**batch)
        self.metric.update(example_ids, outputs.start_logits, outputs.end_logits)

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.metric.compute()
        # self.log_dict(metric_dict, prog_bar=True)
        self.log("val/f1", metric_dict["f1"], prog_bar=True)
        self.log("val/em", metric_dict["exact_match"], prog_bar=True)

    def configure_metrics(self, stage: str):
        dataset = self.trainer.datamodule
        validation_dataset = dataset.ds["validation"]
        original_validation_dataset = dataset.ds["validation_original"]
        postprocess_func = partial(
            dataset.postprocess_func,
            dataset=dataset.ds,
            validation_dataset=validation_dataset,
            original_validation_dataset=original_validation_dataset,
        )
        example_id_strings = dataset.example_id_strings
        self.metric = SquadMetric(postprocess_func=postprocess_func, example_id_strings=example_id_strings)

    def configure_optimizers(self):
        param_optimzier = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimzier if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimzier if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )

        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        num_training_steps = int(
            data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs
        )
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        log.info(' ### SAVE TRASFORMERS ### ')
        save_path = '/content/drive/MyDrive/mrc/1_generator_qa/kbo_custom_model_fine_tuned'
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)