import random
import torch
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank

from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.replay_memory = deque(maxlen=args.capacity)
        self.bs = args.batch_size
        if args.model_type in ["gpt2", "llama"]:
            self.data = namedtuple("Generation", \
               field_names=["input_ids", "attention_mask", "position_ids", "label", "loss_mask"])
        else:
            self.data = namedtuple("Generation", \
               field_names=["input_ids", "attention_mask", "label", "loss_mask"])
            
    def __len__(self):
        return len(self.replay_memory)
    
    def sample(self):
        data = random.sample(self.replay_memory, k=self.bs)
        input_ids = torch.stack([d.input_ids for d in data], dim=0)
        attention_mask = torch.stack([d.attention_mask for d in data], dim=0)
        label = torch.stack([d.label for d in data], dim=0)
        loss_mask = torch.stack([d.loss_mask for d in data], dim=0)
        
        if self.args.model_type in ["gpt2", "llama"]:
            position_ids = torch.stack([d.position_ids for d in data], dim=0)
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids
            }
        else:
            model_data = {
                "input_ids": input_ids, "attention_mask": attention_mask
            }
            
        no_model_data = {
            "label": label, "loss_mask": loss_mask
        }
        return model_data, no_model_data
        
    
    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        return model_data, no_model_data
    
    def move_to_memory(self, model_data, no_model_data):
        device = torch.device("cpu")
        model_data_cpu, no_model_data_cpu = {}, {}
        for k in model_data:
            model_data_cpu[k] = model_data[k].to(device)
        
        for k in no_model_data:
            no_model_data_cpu[k] = no_model_data[k].to(device)
        
        for idx in range(model_data_cpu["input_ids"].size(0)):
            if self.args.model_type in ["gpt2", "llama"]:
                e = self.data(model_data_cpu["input_ids"][idx], model_data_cpu["attention_mask"][idx], model_data_cpu["position_ids"][idx],
                              no_model_data_cpu["label"][idx], no_model_data_cpu["loss_mask"][idx])
            else:
                e = self.data(model_data_cpu["input_ids"][idx], model_data_cpu["attention_mask"][idx],
                              no_model_data_cpu["label"][idx], no_model_data_cpu["loss_mask"][idx])
            self.replay_memory.append(e)