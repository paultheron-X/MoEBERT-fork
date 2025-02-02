import copy
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import SoftTreeGate, SoftTreePermutedGate, TopKGate, SampleKSoftmaxUnbiasedWithTrimmedLassoGate
from .permutations import NoPermutations, LearnPermutations, LearnPermutationsBase


class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert, route_method, vocab_size, hash_list, gamma, entropy_reg, perm_epoch, trimmed_lasso_reg, k=1):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        elif route_method == "hash-random":
            self.hash_list = self._random_hash_list(vocab_size)
        elif route_method == "hash-balance":
            self.hash_list = self._balance_hash_list(hash_list)
        elif route_method == "topk":
            config = {
                "k": k,
                "jitter": False,
                "nb_experts": num_experts,
                "task" : 0,
                "input_dim": hidden_size,
            }
            self.gate = TopKGate(config)
        elif route_method == "hash-p":
            self.hash_list = self._random_hash_list(vocab_size)
            config_perm = {
                "nb_experts": 4,
                "k": 1,
                "steps_per_epoch" : 1000,
                "epochs_for_learning_permutation": perm_epoch,
                "learn_k_permutations" : 1,
                "noise_factor" : 0.0,
                "perm_entropy_reg" : 1e-4

            }
            self.gate_perm = LearnPermutationsBase(config_perm)
        elif route_method == "trimmed_lasso":
            config = {
                "task": 0,
                "nb_experts": num_experts,
                "k": k,
                "jitter": False,
                "input_dim": hidden_size,
                "use_routing_input": False,
                "tau": 1.0,
                "trimmed_lasso_reg": trimmed_lasso_reg,
                'num_training_samples': 100,  # don't care
                'biasness': 'small-random',
                'replace': False,
            }
            self.gate = SampleKSoftmaxUnbiasedWithTrimmedLassoGate(config)
        elif route_method in ["soft-tree"]:
            config = {
                "k": k,
                "nb_experts": 4,
                "gamma": gamma,  # gamma = [0.01, 0.1, 1]
                "input_dim": hidden_size,
                "temperature": 1.0,
                "entropy_reg": entropy_reg,  # [5e-2, 1e-1, 5e-1, 1, 5, 10] 1.0
                # "temperature": 1.0,
            }
            self.gate = SoftTreeGate(config)
            pass
        elif route_method in ["soft-tree-oldpgate"]: 
            # This is the mode were we learn the permutation accross the 4 experts, with the weights initialized with importance processor 
            # We call the Learn Permutations gate, and we use the soft tree gate the same as in the Conditional Computing Repo. 
            # BEWARE: Do not put 'perm' in the name of the route_method, as it will be confused with the permutation gate. 
            #           This will break everything while initializing the MoEBERT layers
            config_perm = {
                "nb_experts": 4,
                "k": 1,
                "steps_per_epoch" : 1000,
                "epochs_for_learning_permutation": perm_epoch,
                "learn_k_permutations" : 1,
                "noise_factor" : 0.0,
                "perm_entropy_reg" : 1e-4

            }
            config = {
                "k": 1,
                "nb_experts": 4,
                "gamma": gamma,  # gamma = [0.01, 0.1, 1]
                "input_dim": hidden_size,
                "temperature": 1.0,
                "entropy_reg": entropy_reg,  # [5e-2, 1e-1, 5e-1, 1, 5, 10] 1.0
                # "temperature": 1.0,
            }
            
            self.gate = SoftTreePermutedGate(config)
            self.gate_perm = LearnPermutationsBase(config_perm)
            pass
        elif route_method in ["soft-tree-perm"]:
            config_perm = {
                "nb_experts": 3072,
                "k": 1,
                "steps_per_epoch" : 1000,
                "epochs_for_learning_permutation": 1.0,
                "learn_k_permutations" : 1,
                "noise_factor" : 0.05,
                "perm_entropy_reg" : 1e-1
            }
            
            #self.gate_perm = NoPermutations(config_perm)
            self.gate_perm = LearnPermutations(config_perm)
            config = {
                "k": 1,
                "nb_experts": 4,
                "gamma": gamma,  # gamma = [0.01, 0.1, 1]
                "input_dim": hidden_size,
                "temperature": 1.0,
                "entropy_reg": entropy_reg,  # [5e-2, 1e-1, 5e-1, 1, 5, 10] 1.0
                # "temperature": 1.0,
            }
            #self.gate = SoftTreePermutedGate(config)
            self.gate = SoftTreeGate(config)

        else:
            raise KeyError("Routing method not supported.")

    def _random_hash_list(self, vocab_size):
        hash_list = torch.randint(low=0, high=self.num_experts, size=(vocab_size,))
        return hash_list

    def _balance_hash_list(self, hash_list):
        with open(hash_list, "rb") as file:
            result = pickle.load(file)
        result = torch.tensor(result, dtype=torch.int64)
        return result

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        y_agg = self.gate.forward(x)
        logits_gate = y_agg.view(bsz, seq_len, -1)  # This returns the logits for each expert
        prob_gate = F.softmax(logits_gate, dim=-1)  # This returns the probabilities for each expert
        gate = torch.argmax(prob_gate, dim=-1)  # This returns the expert that has the highest probability

        order = gate.argsort(0)  # This returns the order of the experts in the batch
        num_tokens = (
            F.one_hot(gate, self.num_experts).gt(0).sum(0)
        )  # This returns the number of tokens for each expert
        gate_load = num_tokens.clone()  # This returns the number of tokens for each expert
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x
            return input_x

        x = [forward_expert(x[i], prob_gate[i], i) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, balance_loss, gate_load

    def _forward_soft_tree_gate(self, x):
        bsz, seq_len, dim = x.size()  # bsz = 1, seq_len = 512, dim = 768

        x = x.view(-1, dim)

        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x

        h = [forward_expert(x, i) for i in range(self.num_experts)]

        # pass the hidden states to the gate
        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x))
        # print("y_agg", y_agg.shape)
        # print("soft_averages", soft_averages.shape)
        # print("hard_averages", hard_averages.shape)

        # print(y_agg)
        # print("soft_averages", soft_averages)
        # print("hard_averages", hard_averages)

        x = y_agg.view(bsz, seq_len, dim)

        return x, regularization_loss, s_concat

    def _forward_soft_tree_gate_perm(self, x):
        bsz, seq_len, dim = x.size()  # bsz = 1, seq_len = 512, dim = 768

        perm, reg = self.gate_perm.forward(x)

        x = x.view(-1, dim)

        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x

        h = [forward_expert(x, i) for i in range(self.num_experts)]

        # pass the hidden states to the gate
        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x, perm))
        # print("y_agg", y_agg.shape)
        # print("soft_averages", soft_averages.shape)
        # print("hard_averages", hard_averages.shape)

        # print(y_agg)
        # print("soft_averages", soft_averages)
        # print("hard_averages", hard_averages)

        x = y_agg.view(bsz, seq_len, dim)

        return x, regularization_loss + reg.to(regularization_loss.device), s_concat

    def _forward_hash_random_perm(self, x, input_ids):
        bsz, seq_len, dim = x.size()
        
        # Create a routing method that integrates the hash function and the permutation
        perm, reg = self.gate_perm.forward(x) # perm is a 1D tensor of size (num_experts, num_experts)
        perm = perm.to(x.device) # move perm to the same device as x
        reg = reg.to(x.device) # move reg to the same device as x
        
        x = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        self.hash_list = self.hash_list.to(x.device)  # move hash_list to the same device as x
        gate = self.hash_list[input_ids.view(-1)]  # gate is a 1D tensor of size (bsz * seq_len)
        
        # 1. forward the input through the experts
        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x
        
        h = [forward_expert(x, i) for i in range(self.num_experts)]  # h is a list of length num_experts, each element is a 2D tensor of size (bsz * seq_len, dim)
        
        # 2. Mix the perm weights and the hash function
        h = [torch.unsqueeze(t, -1) for t in h]  # h is now a list of length num_experts, each element is a 3D tensor of size (bsz * seq_len, 1, dim)
       
        h = torch.cat(h, dim=2)  # h is now a 3D tensor of size (bsz * seq_len, num_experts, dim)
        
        hash_one_hot = F.one_hot(gate, num_classes=self.num_experts)  # gate_one_hot is a 2D tensor of size (bsz * seq_len, num_experts)
        hash_perm = torch.matmul(hash_one_hot.float(), perm)  # gate_perm is a 2D tensor of size (bsz * seq_len, num_experts)
        
        hash_perm = torch.unsqueeze(hash_perm, -1)  # gate_perm is now a 3D tensor of size (bsz * seq_len, num_experts, 1)
        
        # reshape hash_perm to ((bsz * seq_len, 1, 1, num_experts)
        hash_perm = hash_perm.view(bsz * seq_len, 1, 1, self.num_experts)
        
        # reshape h to (bsz * seq_len, dim, 1, nb_experts)
        h = h.view(bsz * seq_len, dim, 1, self.num_experts)
        
        gate_load = torch.sum(hash_perm, dim=0)  # gate_load is a 2D tensor of size (num_experts, 1)
        
        h = torch.sum(h * hash_perm, dim=[2, 3])  # (bsz * seq_len, dim, 1, nb_experts) -> (bsz * seq_len, dim)
        
        # 3. reform the tensor
        x = h.view(bsz, seq_len, dim)
        
        return x, reg, gate_load  # return x, balance_loss, gate_load
        
    def _forward_soft_tree_gate_perm_bis(self, x):
        
        # Do the process described above
        
        bsz, seq_len, dim = x.size()  # bsz = 1, seq_len = 512, dim = 768
        
        perm, reg = self.gate_perm.forward(x)
        
        x = x.view(-1, dim)
        
        def forward_expert_perm(input_x, expert_idx, perm):
            input_x = self.experts[expert_idx].forward(input_x, perm)
            return input_x

        h = list(forward_expert_perm(x, 0, perm))

        # pass the hidden states to the gate
        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x))
        # print("y_agg", y_agg.shape)
        # print("soft_averages", soft_averages.shape)
        # print("hard_averages", hard_averages.shape)

        # print(y_agg)
        # print("soft_averages", soft_averages)
        # print("hard_averages", hard_averages)

        x = y_agg.view(bsz, seq_len, dim)

        return x, regularization_loss + reg, s_concat
        

    def _forward_soft_tree_gate_sentence(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)

        bsz, seq_len, dim = x.size()  # bsz = 1, seq_len = 512, dim = 768

        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x

        h = [forward_expert(x_average, i) for i in range(self.num_experts)]

        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x_average))

        x = y_agg.view(bsz, seq_len, dim)

        return x, regularization_loss, s_concat

    def _forward_gate_sentence(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_sentences = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sentences.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_sentences.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_sentences.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sentences.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x.unsqueeze(-1)
            return input_x

        result = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                result.append(forward_expert(x[i], prob_gate[i], i))
        result = torch.vstack(result)
        result = result[order.argsort(0)]  # restore original order

        return result, balance_loss, gate_load

    def _forward_sentence_single_expert(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        gate_load = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        x = self.experts[gate.cpu().item()].forward(x)
        return x, 0.0, gate_load

    def _forward_hash(self, x, input_ids):
        bsz, seq_len, dim = x.size()  # bsz is batch size, seq_len is sequence length, dim is hidden size

        x = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        self.hash_list = self.hash_list.to(x.device)  # move hash_list to the same device as x
        gate = self.hash_list[input_ids.view(-1)]  # gate is a 1D tensor of size (bsz * seq_len)

        order = gate.argsort(0)  # order is a 1D tensor of size (bsz * seq_len)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)  # num_tokens is a 1D tensor of size (num_experts)
        gate_load = num_tokens.clone()  # gate_load is a 1D tensor of size (num_experts)
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        x = [
            self.experts[i].forward(x[i]) for i in range(self.num_experts)
        ]  # x is a list of tensors of size (num_tokens[i], dim)
        x = torch.vstack(x)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)  # x is now a 3D tensor of size (bsz, seq_len, dim)

        return x, 0.0, gate_load  # return x, balance_loss, gate_load
    
    def _forward_topk(self, x):
        # TOFILL 
        bsz, seq_len, dim = x.size()  # bsz is batch size, seq_len is sequence length, dim is hidden size
        
        x = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        
        def forward_expert(input_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            return input_x

        h = [forward_expert(x, i) for i in range(self.num_experts)]

        # pass the hidden states to the gate
        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x))
        # print("y_agg", y_agg.shape)
        # print("soft_averages", soft_averages.shape)
        # print("hard_averages", hard_averages.shape)

        # print(y_agg)
        # print("soft_averages", soft_averages)
        # print("hard_averages", hard_averages)

        x = y_agg.view(bsz, seq_len, dim)

        return x, regularization_loss, s_concat
        

    def forward(self, x, input_ids, attention_mask):
        if self.route_method == "gate-token":
            x, balance_loss, gate_load = self._forward_gate_token(x)
            # find the meaning of these functions, when is it used? and meaning of variables
        elif self.route_method == "soft-tree":
            x, balance_loss, gate_load = self._forward_soft_tree_gate(x)
        elif self.route_method == "soft-tree-perm":
            x, balance_loss, gate_load = self._forward_soft_tree_gate_perm_bis(x)
        elif self.route_method == "soft-tree-oldpgate":
            x, balance_loss, gate_load = self._forward_soft_tree_gate_perm(x)
        elif self.route_method == "topk":
            x, balance_loss, gate_load = self._forward_topk(x)
        elif self.route_method == "trimmed_lasso":
            x, balance_loss, gate_load = self._forward_topk(x)
        elif self.route_method == "hash-p":
            x, balance_loss, gate_load = self._forward_hash_random_perm(x, input_ids)
        elif self.route_method == "soft-tree-sentence":
            x, balance_loss, gate_load = self._forward_soft_tree_gate_sentence(x, attention_mask)
        elif self.route_method == "gate-sentence":
            if x.size(0) == 1:
                x, balance_loss, gate_load = self._forward_sentence_single_expert(x, attention_mask)
            else:
                x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        elif self.route_method in ["hash-random", "hash-balance"]:
            x, balance_loss, gate_load = self._forward_hash(x, input_ids)
        else:
            raise KeyError("Routing method not supported.")

        return x, balance_loss, gate_load
