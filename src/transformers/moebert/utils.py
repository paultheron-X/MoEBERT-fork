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

#torch.autograd.set_detect_anomaly(True)

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
        
class ImportanceProcessorPermute(ImportanceProcessor):
    def __init__(self, config, layer_idx, num_local_experts, local_group_rank):
        super().__init__(config, layer_idx, num_local_experts, local_group_rank)
        self.is_perm = False

    @staticmethod
    def load_importance_single(importance_files):
        super().load_importance_single(importance_files)
    
    def _split_importance(self, arr):
        super()._split_importance(arr)
        
    def load_experts(self, model_layer):
        expert_list = model_layer.experts.experts
        fc1_weight_data = model_layer.intermediate.dense.weight.data
        fc1_bias_data = model_layer.intermediate.dense.bias.data
        fc2_weight_data = model_layer.output.dense.weight.data
        fc2_bias_data = model_layer.output.dense.bias.data
        layernorm_weight_data = model_layer.output.LayerNorm.weight.data
        layernorm_bias_data = model_layer.output.LayerNorm.bias.data
        i = 0 # Single ffn
        idx = self.importance[i]
        first_512 = idx[:512]
        # keep the remaining indices, such that we have in total 3072 indices
        remaining = idx[512:3072 - 4 * 512]
        # Duplicate the top 512 importance in 4 for all experts
        fc1_weight_data_ = fc1_weight_data[first_512, :].clone()
        fc1_bias_data_ = fc1_bias_data[first_512].clone()
        fc2_weight_data_ = fc2_weight_data[:, first_512].clone()
        fc2_bias_data_ = fc2_bias_data.clone()
        #layernorm_weight_data_ = layernorm_weight_data.clone()
        #layernorm_bias_data_ = layernorm_bias_data.clone()
        
        fc1_weight_data_begin = fc1_weight_data_.repeat(4, 1)
        fc1_bias_data_begin = fc1_bias_data_.repeat(4)
        fc2_weight_data_begin = fc2_weight_data_.repeat(1, 4)
        fc2_bias_data_begin = fc2_bias_data_.repeat(4)
        #layernorm_weight_data = layernorm_weight_data_.repeat(4)
        #layernorm_bias_data = layernorm_bias_data_.repeat(4)
        
        # fill the rest of the importance with the remaining indices such that the total size is the same
        fc1_weight_data_end = fc1_weight_data[remaining, :].clone()
        fc1_bias_data_end = fc1_bias_data[remaining].clone()
        fc2_weight_data_end = fc2_weight_data[:, remaining].clone()
        fc2_bias_data_end = fc2_bias_data.clone()
        
        fc1_weight_data_fin = torch.cat((fc1_weight_data_begin, fc1_weight_data_end), 0)
        fc1_bias_data_fin = torch.cat((fc1_bias_data_begin, fc1_bias_data_end), 0)
        fc2_weight_data_fin = torch.cat((fc2_weight_data_begin, fc2_weight_data_end), 1)
        fc2_bias_data_fin = torch.cat((fc2_bias_data_begin, fc2_bias_data_end), 0)
        #layernorm_weight_data_fin = layernorm_weight_data_.repeat(8)
        
        expert_list[i].fc1.weight.data = fc1_weight_data_fin.clone()
        expert_list[i].fc1.bias.data = fc1_bias_data_fin.clone()
        expert_list[i].fc2.weight.data = fc2_weight_data_fin.clone()
        expert_list[i].fc2.bias.data = fc2_bias_data_fin.clone()
        expert_list[i].LayerNorm.weight.data = layernorm_weight_data.clone()
        expert_list[i].LayerNorm.bias.data = layernorm_bias_data.clone()
        del model_layer.intermediate
        del model_layer.output
        self.is_moe = True
        self.is_perm = True
        

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

class FeedForwardPermutationBis(nn.Module):
    def __init__(self, config, intermediate_size, dropout):
        nn.Module.__init__(self)

        # first layer
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # second layer
        self.fc2_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_4 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, permutations):
       
        input_tensor = hidden_states
        #print("hidden_states", hidden_states.shape)
        Ah = self.fc1(hidden_states)
        Ah = self.intermediate_act_fn(Ah)
        
        # split permutations(3072, 3072) into 4 equal parts (768, 3072)
        #print("permutations", permutations.shape)
        permutations = permutations.squeeze(0)
        P1, P2, P3, P4 = torch.split(permutations, 768, dim=0)
        #print("P1", P1.shape) # (768, 3072)
        
        # p_i * A
        s_1 = torch.matmul(P1, Ah.transpose(0,1)) # 
        s_1 = s_1.squeeze(0) #
        s_1 = s_1.transpose(0,1) # 
        s_2 = torch.matmul(P2, Ah.transpose(0,1)) #
        s_2 = s_2.squeeze(0) #
        s_2 = s_2.transpose(0,1) #
        s_3 = torch.matmul(P3, Ah.transpose(0,1)) #
        s_3 = s_3.squeeze(0) #
        s_3 = s_3.transpose(0,1) #
        s_4 = torch.matmul(P4, Ah.transpose(0,1)) #
        s_4 = s_4.squeeze(0) #
        s_4 = s_4.transpose(0,1) #
        
        # apply the 4 experts to the 4 parts of s and split fc2 into 4 equal parts
        B1 = self.fc2_1 # (768, 768)
        B2 = self.fc2_2 # (768, 768)
        B3 = self.fc2_3 # (768, 768)
        B4 = self.fc2_4 # (768, 768)
        
        # apply the 4 experts to the 4 parts of s
        s1 = B1(s_1)
        s2 = B2(s_2)
        s3 = B3(s_3)
        s4 = B4(s_4)
        
        # apply the final layer
        hs_1 = self.dropout(s1)
        hs_2 = self.dropout(s2)
        hs_3 = self.dropout(s3)
        hs_4 = self.dropout(s4)
        hs_1 = self.LayerNorm(hs_1 + input_tensor)
        hs_2 = self.LayerNorm(hs_2 + input_tensor)
        hs_3 = self.LayerNorm(hs_3 + input_tensor)
        hs_4 = self.LayerNorm(hs_4 + input_tensor)
        return  (hs_1, hs_2, hs_3, hs_4)


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
        self.fc2_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2_4 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, permutations):
       
        input_tensor = hidden_states
        #print("hidden_states", hidden_states.shape)
        Ah = self.fc1(hidden_states)
        Ah = self.intermediate_act_fn(Ah)
        
        # split permutations(3072, 3072) into 4 equal parts (768, 3072)
        #print("permutations", permutations.shape)
        permutations = permutations.squeeze(0)
        P1, P2, P3, P4 = torch.split(permutations, 768, dim=0)
        #print("P1", P1.shape) # (768, 3072)
        
        # p_i * A
        s_1 = torch.matmul(P1, Ah.transpose(0,1)) # 
        s_1 = s_1.squeeze(0) #
        s_1 = s_1.transpose(0,1) # 
        s_2 = torch.matmul(P2, Ah.transpose(0,1)) #
        s_2 = s_2.squeeze(0) #
        s_2 = s_2.transpose(0,1) #
        s_3 = torch.matmul(P3, Ah.transpose(0,1)) #
        s_3 = s_3.squeeze(0) #
        s_3 = s_3.transpose(0,1) #
        s_4 = torch.matmul(P4, Ah.transpose(0,1)) #
        s_4 = s_4.squeeze(0) #
        s_4 = s_4.transpose(0,1) #
        
        # apply the 4 experts to the 4 parts of s and split fc2 into 4 equal parts
        B1 = self.fc2_1 # (768, 768)
        B2 = self.fc2_2 # (768, 768)
        B3 = self.fc2_3 # (768, 768)
        B4 = self.fc2_4 # (768, 768)
        
        # apply the 4 experts to the 4 parts of s
        s1 = B1(s_1)
        s2 = B2(s_2)
        s3 = B3(s_3)
        s4 = B4(s_4)
        
        # apply the final layer
        hs_1 = self.dropout(s1)
        hs_2 = self.dropout(s2)
        hs_3 = self.dropout(s3)
        hs_4 = self.dropout(s4)
        hs_1 = self.LayerNorm(hs_1 + input_tensor)
        hs_2 = self.LayerNorm(hs_2 + input_tensor)
        hs_3 = self.LayerNorm(hs_3 + input_tensor)
        hs_4 = self.LayerNorm(hs_4 + input_tensor)
        return  (hs_1, hs_2, hs_3, hs_4)


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
