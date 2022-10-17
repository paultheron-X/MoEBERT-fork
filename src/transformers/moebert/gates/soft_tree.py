import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

EPSILON = 1e-6


class SmoothStep(nn.Module):
    """A smooth-step function.
    For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.
    See https://arxiv.org/abs/2002.07772 for more details on this function.
    """

    def __init__(self, gamma=1.0):
        """Initializes the layer.
        Args:
          gamma: Scaling parameter controlling the width of the polynomial region.
        """
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, x):
        return torch.where(
            x <= self._lower_bound,
            torch.zeros_like(x),
            torch.where(
                x >= self._upper_bound,
                torch.ones_like(x),
                self._a3 * (x**3) + self._a1 * x + self._a0,
            ),
        )


class SoftTreeGate(nn.Module):
    """An ensemble of soft decision trees.

    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.

    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with
        num_trees units, each corresponding to the hyperplane of one tree.

    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(
        self,
        config,
        node_index=0,
        depth_index=0,
        name="Node-Root",
    ):
        """Initializes the layer.
        Args:
          input_dim: The dimension of the input tensor.
          num_trees: The number of trees in the ensemble.
          leaf_dims: The dimension of the output vector.
          gamma: The scaling parameter for the smooth-step function.
        """
        super(SoftTreeGate, self).__init__()

        self.nb_experts = config["nb_experts"]
        self.max_depth = (int)(np.ceil(np.log2(self.nb_experts)))
        self.k = config["k"]

        #         #print("=========self.nb_experts:", self.nb_experts)
        #         #print("=========self.max_depth:", self.max_depth)
        self.node_index = node_index
        #         #print("=========self.node_index:", self.node_index)
        self.depth_index = depth_index
        #         self.max_split_nodes = 2**self.max_depth - 1
        self.max_split_nodes = self.nb_experts - 1
        #         self.leaf = node_index >= self.max_split_nodes
        self.leaf = node_index >= self.nb_experts - 1
        #         assert self.nb_experts == 2**self.max_depth # to check number of experts is a power of 2

        self.gamma = config["gamma"]
        
        
        self.activation = SmoothStep(self.gamma)
        
        self.input_dim = config["input_dim"]

        self.entropy_reg = config["entropy_reg"]
        if "temperature" in config.keys():
            self.iterations = 0
            self.temperature = torch.tensor(config["temperature"], dtype=torch.float32)
        else:
            self.temperature = None

        self.balanced_splitting = config.get("balanced_splitting", False)
        self.balanced_splitting_penalty = config.get("balanced_splitting_penalty", 0.0)
        self.exp_decay_mov_ave = config.get("exp_decay_mov_ave", 0.0)

        if not self.leaf:
            self.selector_layer = nn.Linear(self.input_dim, self.k, bias=False)
            self.selector_layer.weight = self._z_initializer(self.selector_layer.weight)
            
            self.left_child = SoftTreeGate(
                config,
                node_index=2 * node_index + 1,
                depth_index=depth_index + 1,
                name="Node-Left",
            )
            self.right_child = SoftTreeGate(
                config,
                node_index=2 * node_index + 2,
                depth_index=depth_index + 1,
                name="Node-Right",
            )
            if self.balanced_splitting:
                self.alpha_ave_past_tensor = torch.tensor(0.0, dtype=torch.float32)
                self.alpha_ave_past = nn.Parameter(self.alpha_ave_past_tensor, requires_grad=False)

        else:
            self.output_layer = nn.Linear(self.input_dim, self.k)
            self.output_layer.weight = self._w_initializer(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.0)
        
        if self.node_index==0:
            self.permutation_mask = torch.tensor(np.array([np.identity(self.nb_experts)[np.random.permutation(np.arange(self.nb_experts)),:] for _ in range(self.k)]), dtype=torch.float32)
    
    def _z_initializer(self, x):
        return nn.init.uniform_(x, -self.gamma / 100, self.gamma / 100)      
    
    def _w_initializer(self, x):
       return nn.init.uniform_(x, a = -0.05, b = 0.05)  

    
    def _compute_balanced_split_loss(self, prob, current_prob):
        exp_decay_mov_ave = self.exp_decay_mov_ave * (1.0**self.depth_index)

        penalty = self.balanced_splitting_penalty / (1.0**self.depth_index)

        alpha_ave_current = torch.sum(prob * current_prob, dim=0) / (
            torch.sum(prob * torch.ones_like(current_prob), dim=0) + EPSILON
        )
        alpha_ave = (
            1 - exp_decay_mov_ave
        ) * alpha_ave_current.float() + exp_decay_mov_ave * self.alpha_ave_past.float()


        self.alpha_ave_past = nn.Parameter(alpha_ave, requires_grad=False)

        loss = -penalty * (0.5 * torch.log(alpha_ave + EPSILON) + 0.5 * torch.nn.math.log(1 - alpha_ave + EPSILON))
        loss = torch.mean(loss, dim=-1)
        loss = torch.sum(loss)
        #         #print("===========bal-split loss:", loss)
        return loss

    def _compute_entropy_regularization_per_expert(
        self,
        prob,
        entropy_reg,
    ):
        # Entropy regularization is defined as: sum_{b \in batch_size} sum_{i \in [k]} -sum_{i=1}^n p_{bi}*log(p_{bi})
        regularization = entropy_reg * torch.mean(torch.sum(-prob * torch.log(prob + EPSILON), dim=1))
        return regularization

    def forward(self, inputs, training=True, prob=1.0):
        ##print(inputs)

        #h, x, permutation_weights = inputs 
        h, x = inputs

        # #print("\ninput of softmax gate: ",len(h), h[0].shape, x.shape)
        assert all([h[i].shape[1] == h[i + 1].shape[1] for i in range(len(h) - 1)])

        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        h = [torch.unsqueeze(t, -1) for t in h]

        # h: (bs, dim_exp_i, nb_experts)
        h = torch.concat(h, dim=2)
        
        #print("h concat shape: ", h.shape)

        if not self.leaf:
            #print('hi')
            # shape = (batch_size, k)
            current_prob = self.selector_layer(x)  # (batch_size, k)
            current_prob = self.activation.forward(current_prob)  # (batch_size, k)
            
            s_left_child = self.left_child.forward(inputs, training=training, prob=current_prob * prob)
            s_right_child = self.right_child.forward(inputs, training=training, prob=(1 - current_prob) * prob)
            
            #print("s_left_child: ", s_left_child.shape)
            #print("s_right_child: ", s_right_child.shape)
            
            s_bj = torch.cat([s_left_child, s_right_child], dim=2)
            
            if self.node_index == 0: # root node
                #print('Using root node routing')

                #print('test s bj', s_bj.shape)

                #print('h.shape@', h.shape)
                h = torch.unsqueeze(h, dim=2) 
                #print('h.shape@', h.shape)


                s_bj = torch.reshape(s_bj, shape=[s_bj.shape[0], -1])  # (b, k*nb_experts)
                s_bj = torch.softmax(s_bj, dim=-1)  # (b, k*nb_experts)
                
                #print("s_bj shape: ", s_bj.shape)
                #print(s_bj)
                
                w_concat = torch.reshape(
                    s_bj, shape=[s_bj.shape[0], self.k, self.nb_experts]
                )  # (b, k, nb_experts)
                w_concat = torch.unsqueeze(w_concat, dim=1)  # (b, 1, k, nb_experts)

                # w_concat: (b, 1, k, nb_experts), perm_mask: [k, nb_experts, nb_experts]
                
                w_permuted = torch.einsum("bijk,jkl->bijl", w_concat, self.permutation_mask.to(w_concat.device))
                w_permuted = torch.sum(w_permuted, dim=2, keepdim=True)  # (b, 1, 1, nb_experts)
                w_permuted = w_permuted / torch.sum(w_permuted, dim=-1, keepdim=True)  # (b, 1, 1, nb_experts)

                # h:(b, dim_exp_i, 1, nb_experts) * w_permuted: (b, 1, 1, nb_experts)
                y_agg = torch.sum(h * w_permuted, dim=[2, 3])  # (b, dim_exp_i, 1, nb_experts) -> (b, dim_exp_i)

                # Compute s_bj
                s_concat = torch.where(
                    torch.less(w_permuted, 1e-5),
                    torch.ones_like(w_permuted),
                    torch.zeros_like(w_permuted),
                )  # (b, 1, 1, nb_experts)
                s_avg = torch.mean(s_concat, dim=-1)  # (b, 1, 1)
                
                #print(s_concat.shape)

                avg_sparsity = torch.mean(s_avg)  # average over batch
                

                soft_averages = torch.mean(w_permuted, dim=[0, 1, 2])  # (nb_experts,)
                hard_averages = torch.mean(torch.ones_like(s_concat) - s_concat, dim=[0, 1, 2])  # (nb_experts,)
                
            
                return y_agg, soft_averages, hard_averages, s_concat # For root node, return the aggregated output and the sparsity of the weights, and the sparsity of the weights for each expert
            else:
                #print('Using internal node routing')
                return s_bj  # , s_bj_sp
        else:
            #print('Using leaf node routing')
            # prob's shape = (b, k)
            # Computing a_bij,    a_bij shape = (b, k)
            a_bij = self.output_layer(x)  # (b, k) # Note we do not have access to j as j represents leaves

            prob = torch.unsqueeze(prob, dim=-1)  # (b, k, 1)
            a_bij = torch.unsqueeze(a_bij, dim=-1)  # (b, k, 1)
            
            log_prob = torch.where(
                prob < 1e-5, -torch.ones_like(prob) * torch.inf, torch.log(prob + + 1e-6 )
            )
            #                s_bj = torch.reduce_logsumexp(a_bij+log_prob, dim=-1, keepdim=True) # (b, 1)
            #                s_bj_sp = torch.reduce_logsumexp(a_bij+torch.math.log(prob),dim=-1,keepdim=True)
            s_bj = a_bij + log_prob  # (b, k, 1)

            return s_bj  # ,s_bj_sp


if __name__ == "__main__":
    config = {
        "use_routing_input": False,
        "nb_experts": 4,
        "gamma": 1.0,
        "k": 2,
        "task" : "classification",
        "z_initializer": None,
        "w_initializer": None,
        "entropy_reg": 1,
    }
    s = SoftTreeGate(config)
    
    h = [
        np.random.random((8, 10)) for _ in range(config["nb_experts"])
    ]
    x = np.random.random((8, 5))
    y = s([h, x])
    #print(y)
    #print(y.shape)        