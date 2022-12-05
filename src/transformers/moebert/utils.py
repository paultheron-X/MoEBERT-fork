import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple


def use_experts(
    layer_idx,
    perm = False
):  # This is the function that we want to choose if we don't want to use experts for a layer and instead have a normal bert layer.
    return not perm


def process_ffn(model):
    if "perm" in model.config.moebert_route_method:
        if model.config.model_type == "bert":
            inner_model = model.bert
        else:
            raise ValueError("Model type not recognized.")

        for i in range(model.config.num_hidden_layers):
            model_layer = inner_model.encoder.layer[i]
            if model_layer.perm:
                model_layer._init_experts(model_layer)

    else:
        if model.config.model_type == "bert":
            inner_model = model.bert
        else:
            raise ValueError("Model type not recognized.")

        for i in range(model.config.num_hidden_layers):
            model_layer = inner_model.encoder.layer[i]
            if model_layer.use_experts:
                model_layer.importance_processor.load_experts(model_layer)


class ImportanceProcessor:
    def __init__(self, config, layer_idx, num_local_experts, local_group_rank):
        self.num_experts = config.moebert_expert_num  # total number of experts
        self.num_local_experts = num_local_experts  # number of experts on this device
        self.local_group_rank = local_group_rank  # rank in the current process group
        self.intermediate_size = config.moebert_expert_dim  # FFN hidden dimension
        self.share_importance = config.moebert_share_importance  # number of shared FFN dimension

        importance = ImportanceProcessor.load_importance_single(config.moebert_load_importance)[layer_idx, :]
        self.importance = self._split_importance(importance)

        self.is_moe = False  # safety check

    @staticmethod
    def load_importance_single(importance_files):
        with open(importance_files, "rb") as file:
            data = pickle.load(file)
            data = data["idx"]
        return np.array(data)

    def _split_importance(self, arr):
        result = []
        top_importance = arr[:self.share_importance]
        remain = arr[self.share_importance:]
        all_experts_remain = []
        for i in range(self.num_experts):
            all_experts_remain.append(remain[i::self.num_experts])
        all_experts_remain = np.array(all_experts_remain)

        for i in range(self.num_local_experts):
            temp = all_experts_remain[self.num_local_experts * self.local_group_rank + i]
            temp = np.concatenate((top_importance, temp))
            temp = temp[:self.intermediate_size]
            result.append(temp.copy())
        result = np.array(result)
        return result

    def load_experts(self, model_layer):
        expert_list = model_layer.experts.experts
        fc1_weight_data = model_layer.intermediate.dense.weight.data
        fc1_bias_data = model_layer.intermediate.dense.bias.data
        fc2_weight_data = model_layer.output.dense.weight.data
        fc2_bias_data = model_layer.output.dense.bias.data
        layernorm_weight_data = model_layer.output.LayerNorm.weight.data
        layernorm_bias_data = model_layer.output.LayerNorm.bias.data
        for i in range(self.num_local_experts):
            idx = self.importance[i]
            expert_list[i].fc1.weight.data = fc1_weight_data[idx, :].clone()
            expert_list[i].fc1.bias.data = fc1_bias_data[idx].clone()
            expert_list[i].fc2.weight.data = fc2_weight_data[:, idx].clone()
            expert_list[i].fc2.bias.data = fc2_bias_data.clone()
            expert_list[i].LayerNorm.weight.data = layernorm_weight_data.clone()
            expert_list[i].LayerNorm.bias.data = layernorm_bias_data.clone()
        del model_layer.intermediate
        del model_layer.output
        self.is_moe = True


class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size, dropout):
        nn.Module.__init__(self)

        # first layer
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # second layer
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor):
        input_tensor = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class FeedForwardPermutation(nn.Module):
    def __init__(self, config, intermediate_size, dropout):
        nn.Module.__init__(self)

        # first layer
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # second layer
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, permutations):
        """In this case, FFN (the expert) is a 2-layered feedforward network of the form B*ReLU(A*h), where A \in R^{3072,768},  B \in R^{768, 3072}, h \in R^768.

        When you convert to MoE, this splits into 4 pieces:
        B1*ReLU(A1*h),  B2*ReLU(A2*h)​, B3*ReLU(A3*h)​, B4*ReLU(A4*h)​

        where A1, A2, A3, A4 \in R^{768,768​}, and B1, B2, B3, B4 \in R^{768,768​},

        What we need to do it apply this operation s= P*A*h, where P\in R^{3072,3072​} is the output of the permutation block, 

        split this s into 4 equal parts [s1,s2,s3,s4] and apply:

        B1*ReLU(s1),  B2*ReLU(s2)​, B3*ReLU(s3​), B4*ReLU(s4)​
        """
        input_tensor = hidden_states
        print("hidden_states", hidden_states.shape)
        Ah = self.fc1(hidden_states)
        # multiply by permutation matrix to get s (beware of the batch dimension)
        s = torch.matmul(permutations, Ah.transpose(0,1)) # (3072, 3072) * (3072, batch_size) = (1, 3072, batch_size)
        s = s.squeeze(0) # (3072, batch_size)
        s = s.transpose(0,1) # (3072, batch_size) 
        # split s into 4 equal parts
        s1, s2, s3, s4 = torch.split(s, 768, dim=1)
        print("s1", s1.shape) # (batch_size, 768)
        
        # apply the 4 experts to the 4 parts of s and split fc2 into 4 equal parts
        B1 = self.fc2[:768, :] # (768, 768)
        B2 = self.fc2[768:2*768, :] # (768, 768)
        B3 = self.fc2[2*768:3*768, :] # (768, 768)
        B4 = self.fc2[3*768:, :] # (768, 768)
        print("B1", B1.shape) # (768, 768)

        # apply the 4 experts to the 4 parts of s
        hidden_states1 = self.intermediate_act_fn(torch.matmul(s1, B1.transpose(0,1))) # (batch_size, 768) * (768, 768) = (batch_size, 768)
        hidden_states2 = self.intermediate_act_fn(torch.matmul(s2, B2.transpose(0,1))) # (batch_size, 768) * (768, 768) = (batch_size, 768)
        hidden_states3 = self.intermediate_act_fn(torch.matmul(s3, B3.transpose(0,1))) # (batch_size, 768) * (768, 768) = (batch_size, 768)
        hidden_states4 = self.intermediate_act_fn(torch.matmul(s4, B4.transpose(0,1))) # (batch_size, 768) * (768, 768) = (batch_size, 768)
        
        # concatenate the 4 outputs
        hidden_states = torch.cat((hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1) # (batch_size, 3072)
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


@dataclass
class MoEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None


@dataclass
class MoEModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None
