from __future__ import annotations
import os
from abc import ABC
from typing import List
import json
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd
from dataclasses import dataclass
from cgi import test
import copy
import logging
import random
import h5py
import numpy as np

from datasets import load_metric, Metric
from datetime import datetime
import json
from isort import file
from loguru import logger
from sympy import true
from tap import Tap
from torch.optim import AdamW
import numpy as np
from torch import nn
from torch.nn import Module
import torchaudio
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import Optional, Tuple  # 将wav2vec processor 和 model 合并
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Dict, List
import os
from torch.utils.tensorboard import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "3"




@dataclass
class AudioInputExample: 
    """
    Input Example for a single example
    """
    id: str = ""
    file: str = ""
    text: str = ""


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""

    def get_train_dataset(self) -> Dataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()




class AudioDataProcessor(DataProcessor):
    """AudioDataProcessor
    """

    def __init__(self, data_dir, is_debug: bool) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.is_debug = is_debug

    def _read(self, file: str) -> List[AudioInputExample]:
        examples = []
        examples = []
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            examples = [AudioInputExample(**self.string2dict(item)) for item in data]
        # index = [i for i in range(len(examples)) if examples[i].id=='T0055G7072S0215']
        # examples = [examples[index[0]]]
        return examples

    def _load_dataset(self, mode: str = 'train.csv') -> Dataset:
        file = os.path.join(self.data_dir, mode)
        examples = self._read(file)
        indices = [i for i in range(len(examples))] 
        return Subset(examples, indices) 

    def get_train_dataset(self) -> Dataset:
        return self._load_dataset('train.list')
    
    def string2dict(self, item: str) -> List:
        return {"id":item.split('\"')[3], "file":item.split('\"')[7], "text": item.split('\"')[11]}

class Config(Tap):
    audio_data_dir: str = '/home/users/jiangjin/jiangjin_bupt/ASR_CORRECTION/Cross_modal/TAP/data/zh/AISHELL-1/audio-feature/'
    audio_pretrained_model: str = 'ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt'
    is_debug = False
    batch_size = 80
    device: str = 'cuda'
    max_seq_length: int = 64
    def get_device(self):
        """return the device"""
        return torch.device(self.device)
  

class Extractor:
    def __init__(
        self, config: Config,
        audio_processor: DataProcessor,
        # text_tokenizer: PreTrainedTokenizer,
        audio_tokenizer: Tuple[Wav2Vec2Processor, Wav2Vec2Model],
        f_wav2vec_path,
        
    ) -> None: 
        self.config = config

        self.audio_tokenizer = audio_tokenizer
        # self.text_tokenizer = text_tokenizer
        self.resampler = torchaudio.transforms.Resample()

        self.audio_train_dataloader = self.create_dataloader(
            dataset=audio_processor.get_train_dataset(),
            shuffle=False,
            collate_fn=self.convert_audio_examples_to_features,
        )
        self.f_wav2vec_path = f_wav2vec_path

    def create_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def convert_audio_examples_to_features(self, audio_examples: List[AudioInputExample]):
        "load audio from disk"
        speechs = []
        for example in audio_examples:
            speech, _ = torchaudio.load(example.file)
            speech = self.resampler(speech).squeeze().numpy()
            speechs.append(speech)

        # audio_tokenizer 为 wav2vecprocessor 可以将语音转化为向量
        speechs = self.audio_tokenizer[0](
            raw_speech=speechs,
            sampling_rate=16_000,
            return_tensors="pt",
            padding='max_length',
            max_length=50000,
            truncation=True,
        )
        speechs = speechs.to(self.config.get_device())
        with torch.no_grad():
            speechs = self.audio_tokenizer[1](**speechs)
        # 这里感觉可以 输出 list(last_hidden_states.shape)的info
        speechs = speechs.last_hidden_state
        # texts = [example.text for example in audio_examples]
        encoded_features = []
        # encoded_features = self.text_tokenizer.batch_encode_plus(
        #     texts,
        #     max_length=self.config.max_seq_length,
        #     padding='max_length',
        #     return_tensors='pt',
        #     return_attention_mask=True
        # )
        
        if not os.path.exists(self.f_wav2vec_path):
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'w')
        else:
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'a')
        for i in range(len(speechs)):
            # print(i)
            f_wav2vec.create_dataset(audio_examples[i].id, data=speechs[i].detach().cpu().numpy())

        f_wav2vec.close()
        return  speechs, encoded_features

    def result_save(self,):
        print(f'Extractor begin...')
        # self.audio_tokenizer[0].to(self.config.get_device())
        # self.audio_tokenizer[1].to(self.config.get_device())
        self.audio_tokenizer[1].cuda()
        self.train_bar = tqdm(total=len(self.audio_train_dataloader))      
        
        # f_wav2vec=h5py.File("wav2vec_feature.hdf5","w")
        # group_speech=self.f_wav2vec.create_group("speechs")
        # group_label=self.f_wav2vec.create_group("labels")
        for audio_batch in self.audio_train_dataloader:
            self.train_bar.update()



if __name__ == "__main__":
    config: Config = Config().parse_args(known_only=True)
    

    wav2vec_pretrained_model = '/home/users/jiangjin/jiangjin_bupt/peking/huawei_asr/Model_data-all/MODEL/Wav2vec_path/'
    extractor = Extractor(
        config,
        audio_processor=AudioDataProcessor(
            config.audio_data_dir, config.is_debug),
        # text_tokenizer=AutoTokenizer.from_pretrained(config.pretrained_model),
        audio_tokenizer=(Wav2Vec2Processor.from_pretrained(
            wav2vec_pretrained_model), Wav2Vec2Model.from_pretrained(wav2vec_pretrained_model)),
        f_wav2vec_path="./aishell_wav2vec_feature.h5",
    )
    extractor.result_save()