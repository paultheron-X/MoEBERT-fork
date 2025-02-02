"""Our implementation of Top-k routing

Ref: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In ICLR 2017.

The implementation is based on the description in Section 2.1.
"""

"""A PyTorch layer for Top-k routing gate."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGate(nn.Module):
    """A custom layer for selecting a sparse mixture of experts.
    Let f_1, f_2, ..., f_n be the experts. The layer returns:
              g_1 * f_1 + g_2 * f_2 + ... + g_n * f_n,
    where the mixture weights satisfy:
        (1) cardinality constraint: ||g||_0 <= k
        (2) simplex constraint: g_1, ..., g_n >= 0 and g_1 + ... + g_n = 1.
    The number of non-zeros in the mixture weights can be directly controlled.
    The layer is trained using first-order methods like SGD.
    
    Input:
        The inputs should be as follows:
            inputs: Tuple of the form: (f, routing_inputs, permutation_weights)
                f: list of experts f_i, each with same shape.
                routing_inputs: 2D tensor of input examples
                permutation_weights: identity or permutation from Permutation-based Local Search.
            training: 
                Ignored
            indices:
                Ignored
        
    Output:
        Tensor, with the same shape as the expert tensors.
    """
    def __init__(self, config, task=0):
        super(TopKGate, self).__init__()
        self.task = config["task"]
        self.nb_experts = config["nb_experts"]
        self.k = int(config["k"])
        self.jitter = config["jitter"]
        
        self.layer = nn.Linear(config["input_dim"], self.nb_experts, bias=True)

        if self.jitter:
            self.jitter_weights = nn.Parameter(torch.Tensor(self.nb_experts, config["input_dim"]))

    def forward(self, inputs, training=True):
        f, x = inputs
        assert all([f[i].shape[1] == f[i + 1].shape[1] for i in range(len(f) - 1)])

        f = [t.unsqueeze(-1) for t in f]
        f = torch.cat(f, dim=2)

        gate_logits = self.layer(x)
        

        if self.jitter and training:
            gate_logits += torch.randn_like(gate_logits) * F.softplus(torch.matmul(x, self.jitter_weights.t()))

        topk = torch.topk(gate_logits, self.k)

        row_range = torch.arange(gate_logits.shape[0], device=gate_logits.device)
        row_tensor = row_range.unsqueeze(-1).expand(-1, self.k)
        topk_row_col_indices = torch.stack([row_tensor, topk.indices], dim=2)

        topk_scattered = (-torch.inf*torch.ones_like(gate_logits)).scatter_(1, topk.indices, topk.values).unsqueeze(-1)
        
        #print('topk_scattered', topk_scattered)
        #print('topk_scattered shape', topk_scattered.shape)
        
        g = F.softmax(topk_scattered, dim=1)
        #print('g shape', g.shape)
        #print('g', g)
        
        #permutation_weights = permutation_weights.mean(dim=0)
        #g_permuted = torch.einsum("bik,ij->bjk", g, permutation_weights)
        #g_permuted = g_permuted / g_permuted.sum(dim=1, keepdim=True)

        g_permuted = g  # no permutation
        #print(torch.matmul(f, g_permuted).shape)
        y = torch.matmul(f, g_permuted).view(-1, f.shape[1])
        
        # Calculate metrics
        metrics = {}
        s_concat = torch.where(g_permuted < 1e-5, torch.ones_like(g_permuted), torch.zeros_like(g_permuted))
        metrics['avg_sparsity'] = torch.mean(s_concat)

        soft_averages = torch.mean(g_permuted, dim=[0]) # (nb_experts,)
        hard_averages = torch.mean(torch.ones_like(s_concat) - s_concat, dim=[0]) # (nb_experts,)

        soft_averages_for_all_experts_list = soft_averages.view(-1).split(self.nb_experts)
        for j, le in enumerate(soft_averages_for_all_experts_list):
            metrics[f'soft_averages_for_task_{self.task+1}_for_expert_{j}'] = le.mean()

        hard_averages_for_all_experts_list = hard_averages.view(-1).split(self.nb_experts)
        for j, le in enumerate(hard_averages_for_all_experts_list):
            metrics[f'hard_averages_for_task_{self.task+1}_for_expert_{j}'] = le.mean()

        simplex_constraint = torch.mean(torch.sum(g_permuted, dim=1))
        metrics[f'simplex_sum_for_task_{self.task+1}'] = simplex_constraint

        simplex_constraint_fails = torch.sum(torch.sum(g_permuted, dim=1), dim=[1]) # (b, )
        simplex_constraint_fails = torch.where(simplex_constraint_fails < 1.0-1e-5, torch.ones_like(simplex_constraint_fails), torch.zeros_like(simplex_constraint_fails)) # (b, nb_gates)
        simplex_constraint_fails = torch.mean(simplex_constraint_fails, dim=0)
        metrics[f'simplex_constraint_fails_for_task_{self.task+1}'] = simplex_constraint_fails

        return y, soft_averages, hard_averages, s_concat, 0 # reg loss

def test_forward():
    # Set up config and instantiate the TopKGate
    config = {
        "task": 0,
        "nb_experts": 5,
        "k": 2,
        "jitter": False,
        "input_dim": 10
    }
    gate = TopKGate(config)

    # Test data
    batch_size = 8
    num_experts = config["nb_experts"]
    input_dim = config["input_dim"]
    expert_dim = 20

    # Generate some experts
    experts = [Linear(input_dim, expert_dim) for _ in range(num_experts)]

    # Generate some inputs
    x = torch.randn(batch_size, input_dim)
    f = [expert(x) for expert in experts]

    # Pass inputs through the gate
    y, soft_averages, hard_averages, s_concat, reg_loss = gate((f, x))

    print('input size', x.size())
    print('input', x)
    
    # Check output size
    print("Checking output size...")
    print('y', y.size())
    print('y', y)
    
    # Check metrics
    print('soft_averages', soft_averages.size())
    
    print('hard_averages', hard_averages.size())
    
    print('s_concat', s_concat.size())

    #print("All tests passed!")

if __name__ == "__main__":
    import torch
    from torch.nn import Linear
    
    with torch.no_grad():
        test_forward()

