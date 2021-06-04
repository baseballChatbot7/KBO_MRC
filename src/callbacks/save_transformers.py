from typing import Dict, Any
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from random import randint
from src.utils import template_utils

log = template_utils.get_logger(__name__)

class SaveLoadTransformers(ModelCheckpoint):
    def __init__(self, **kwargs):
        super(SaveLoadTransformers, self).__init__(**kwargs)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]):
        log.info('### SAVE TRANSFORMERS ###')
        transformers_model = pl_module.model
        tokenizer = pl_module.tokenizer
        transformers_model.save_pretrained('/content/drive/MyDrive/mrc/1_generator_qa/kbo_custom_model')
        tokenizer.save_pretrained('/content/drive/MyDrive/mrc/1_generator_qa/kbo_custom_model')
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath
        }
    
    # def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]):        
    #     log.info('### LOAD TRANSFORMERS ###')
    #     pl_module.model = AutoModelForQuestionAnswering.from_pretrained("./kbo_qa_model/")
    #     pl_module.tokenizer = AutoTokenizer.from_pretrained("./kbo_qa_model/")