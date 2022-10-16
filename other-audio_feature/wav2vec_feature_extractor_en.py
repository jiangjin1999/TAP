from __future__ import annotations
import os
from abc import ABC
from typing import List
import json
from sqlalchemy import false
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


os.environ["CUDA_VISIBLE_DEVICES"] = "2"




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

    def _load_dataset(self,) -> Dataset:
        file = os.path.join(self.data_dir)
        examples = self._read(file)
        indices = [i for i in range(len(examples))] 
        return Subset(examples, indices) 

    def get_train_dataset(self) -> Dataset:
        return self._load_dataset()
    
    def string2dict(self, item: str) -> List:
        return {"id":item.split('\"')[3], "file":self.rewrite_audio_path(item.split('\"')[7]), "text": item.split('\"')[11]}
        
    def rewrite_audio_path(sefl, file_path: str):
        file_path = file_path.split('LibriSpeech')
        file_path = os.path.join(config.audio_data_current_path+file_path[1])
        return file_path

class Config(Tap):
    # dataset_path: str = '/home/users/jiangjin/jiangjin_bupt/peking/huawei_asr/Model_data-all/DATA/audio_data/librispeech/'
    audio_data_dir: str =  '/home/users/jiangjin/jiangjin_bupt/peking/huawei_asr/Model_data-all/DATA/audio_data/librispeech/train_other_500/data.list'#'../librispeech/train_clean/data.list'
    audio_pretrained_model: str = "facebook/wav2vec2-base-960h"
    audio_data_current_path: str = '/home/users/jiangjin/jiangjin_bupt/Python/ASR/wenet/examples/librispeech/s0/export/LibriSpeech/'
    dataset_name: str = 'librispeech_train_other_500' # 例子
    is_debug = False
    batch_size = 128
    device: str = 'cuda'
    max_seq_length: int = 64
    audio_max_length: int = 65000
    wav2vec_pretrained_model = '/home/users/jiangjin/jiangjin_bupt/peking/huawei_asr/Model_data-all/MODEL/Wav2vec_en_path/' #'../../../MODEL/Wav2vec_en_path/'
    dataset_path = '/home/data/jiangjin/wav2vec_h5_file/'
    f_wav2vec_path= dataset_path + dataset_name + "_wav2vec_feature.h5"
    
    is_use_DDP = False
    # arg for ddp
    local_rank = '0'
    def get_device(self):
        """return the device"""
        if config.is_use_DDP is True:
            return torch.device(self.device, int(local_rank))
        else:
            return torch.device(self.device)
  

class Extractor:
    def __init__(
        self, config: Config,
        audio_processor: DataProcessor,
        # text_tokenizer: PreTrainedTokenizer,
        audio_tokenizer: Wav2Vec2Processor,
        wav2vec_model: Wav2Vec2Model,
        f_wav2vec_path,
        
    ) -> None: 
        self.config = config
        self.audio_tokenizer = audio_tokenizer
        # self.wav2vec_model = wav2vec_model
        # self.text_tokenizer = text_tokenizer
        self.resampler = torchaudio.transforms.Resample()
        if self.config.is_use_DDP is True:
            wav2vec_model = wav2vec_model.to(self.config.get_device())
            self.wav2vec_model = DDP(wav2vec_model, device_ids=[int(self.config.local_rank)], output_device=[int(self.config.local_rank)], find_unused_parameters=True)
        else:
            self.wav2vec_model = wav2vec_model.to(self.config.get_device())
        if self.config.is_use_DDP is True:
            self.audio_train_dataloader = self.create_DDP_dataloader(
                dataset=audio_processor.get_train_dataset(),
                shuffle=False,
                collate_fn=self.convert_audio_examples_to_features,
            )
        else:   
            self.audio_train_dataloader = self.create_dataloader(
                dataset=audio_processor.get_train_dataset(),
                shuffle=False,
                collate_fn=self.convert_audio_examples_to_features,
            )
        self.f_wav2vec_path = f_wav2vec_path

    def create_DDP_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        if self.config.is_use_DDP is True:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset)
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle, # self.config.shuffle,
                collate_fn=collate_fn,
                # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
            )

    def create_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # self.config.shuffle,
            collate_fn=collate_fn,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        )

    def convert_audio_examples_to_features(self, audio_examples: List[AudioInputExample]):
        "load audio from disk"
        speechs = []
        for example in audio_examples:
            speech, _ = torchaudio.load(example.file)
            speech = self.resampler(speech).squeeze().numpy()
            speechs.append(speech)

        # audio_tokenizer 为 wav2vecprocessor 可以将语音转化为向量
        speechs = self.audio_tokenizer(
            raw_speech=speechs,
            sampling_rate=16_000,
            return_tensors="pt",
            padding='max_length',
            max_length=65000,
            truncation=True,
        )
        speechs = speechs.to(self.config.get_device())

        with torch.no_grad():
            speechs = self.wav2vec_model(**speechs)
        # 这里感觉可以 输出 list(last_hidden_states.shape)的info
        speechs = speechs.last_hidden_state
        # texts = [example.text for example in audio_examples]
        # encoded_features = []
        # encoded_features = self.text_tokenizer.batch_encode_plus(
        #     texts,
        #     max_length=self.config.max_seq_length,
        #     padding='max_length',
        #     return_tensors='pt',
        #     return_attention_mask=True
        # )


        # if not os.path.exists(self.f_wav2vec_path):
        #     f_wav2vec = h5py.File(self.f_wav2vec_path, 'w')
        # else:
        #     print('__________start open H5file-_____________')
        #     f_wav2vec = h5py.File(self.f_wav2vec_path, 'a')
        #     print('__________end open H5file-_____________')
        # print('__________start save-_____________')
        # for i in range(len(speechs)):
        #     # print(i)
        #     print('__________start save loop-_____________')
        #     f_wav2vec.create_dataset(audio_examples[i].id, data=speechs[i].detach().cpu().numpy())
        #     print('__________end save loop-_____________')

        # print('__________end save-_____________')
        # print('__________start close H5file-_____________')
        # f_wav2vec.close()
        # print('__________start close H5file-_____________')        


        if not os.path.exists(self.f_wav2vec_path):
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'w')
        else:
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'a')
        for i in range(len(speechs)):
            # print(i)
            if self.config.local_rank=='0':
                f_wav2vec.create_dataset(audio_examples[i].id, data=speechs[i].detach().cpu().numpy())
        f_wav2vec.close()

        
        return  speechs, audio_examples

    def result_save(self,):
        print(f'Extractor begin...')
        # self.audio_tokenizer[0].to(self.config.get_device())
        # self.audio_tokenizer[1].to(self.config.get_device())
        # self.audio_tokenizer[1].cuda()

        self.train_bar = tqdm(total=len(self.audio_train_dataloader))      
        
        # f_wav2vec=h5py.File("wav2vec_feature.hdf5","w")
        # group_speech=self.f_wav2vec.create_group("speechs")
        # group_label=self.f_wav2vec.create_group("labels")
        speechs_list = []
        for audio_batch in self.audio_train_dataloader:
            speech_tmp_list = []
            speechs, audio_examples = audio_batch
            speech_tmp_list.append(speechs)
            speech_tmp_list.append(audio_examples)
            speechs_list.append(speechs_list)
            

            self.train_bar.update()



if __name__ == "__main__":
    config: Config = Config().parse_args(known_only=True)

    if config.is_use_DDP is True:
        # 新增1:依赖
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        # # 新增：从外面得到local_rank参数
        # import argparse
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--local_rank", default=-1)
        # FLAGS = parser.parse_args()
        # local_rank = FLAGS.local_rank
        local_rank = config.local_rank

        # print(local_rank)

        # 新增：DDP backend初始化
        torch.cuda.set_device('cuda:'+str(local_rank))
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    
    extractor = Extractor(
        config,
        audio_processor=AudioDataProcessor(
            config.audio_data_dir, config.is_debug),
        audio_tokenizer=Wav2Vec2Processor.from_pretrained(
            config.wav2vec_pretrained_model),
        wav2vec_model=Wav2Vec2Model.from_pretrained(config.wav2vec_pretrained_model),   
        f_wav2vec_path=config.f_wav2vec_path,
    )
    extractor.result_save()