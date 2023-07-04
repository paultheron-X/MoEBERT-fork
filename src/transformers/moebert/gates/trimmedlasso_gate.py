import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SampleKSoftmaxUnbiasedWithTrimmedLassoGate(nn.Module):
    """
    This gate is based on sampling:
    The high-level idea is as follows:
    During training:
        - Compute logits:
            - O = Ax+b/tau  --- this means o_i = (a_i^T dot x + b_i)/tau
        - Compute g = softmax(O)
        - Sample (without replacement) k times a set of indices s(x) from g for each x.
            - Note cardinality of s(x) is k.
        - On the selected indices in s(x), compute adjusted logits such that
            - One of the logits j in s(x) is randomly chosen to not be adjusted.
                - o_{j} = o_{j}
            - The remaining logits i in s(x), i not equal to j are adjusted
                - o_{j} = o_{j} - log(k*g_{j})
         - Remaining logits (outside s(x)) are set to -inf.
         - Compute g' via Softmax on adjusted logits.
         The idea of adjustment is to ensure the new g' (sparse) has a small bias in comparison to g (dense).

    We also impose trimmed lasso reg on g: \sum_{i>k} g_i for each sample. This encourage g to accumulate mass in Topk components during the course of training for each sample.

    During inference:
        - Compute logits:
            - O = Ax+b/tau  --- this means o_i = (a_i^T dot x + b_i)/tau
        - Compute g = softmax(O)
        - Select Topk indices from g. (inference is deterministic)
        - On the selected indices in s(x), compute adjusted logits for all j in s such that
                - o_{j} = o_{j} - log(k*g_{j})
         - Remaining logits (outside s(x)) are set to -inf.
         - Compute g' via Softmax on adjusted logits.

    """

    def __init__(self, config, task=0):
        super(SampleKSoftmaxUnbiasedWithTrimmedLassoGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        self.jitter = config["jitter"] if config["jitter"] is not None else False
        self.epsilon = 1e-6
        self.tau = float(config["tau"])
        self.trimmed_lasso_reg = config["trimmed_lasso_reg"]
        self.num_training_samples = config["num_training_samples"]
        self.biasness = config["biasness"] if "biasness" in config else "zero"
        self.replace = config["replace"] if "replace" in config else False
        print("===================replace=========================", self.replace)

        if "temperature" in config.keys():
            self.iterations = 0
            self.temperature = config["temperature"]
        else:
            self.temperature = None

        self.use_bias = config["use_bias"] if self.use_routing_input else False

        self.gate_weights = nn.Linear(config["input_dim"], self.nb_experts, bias=self.use_bias)

        self.g_sparse = nn.Linear(self.num_training_samples, self.nb_experts, bias=False)
        
        self.g_sparse.weight.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if self.use_bias:
            nn.init.zeros_(self.gate_weights.bias)
        nn.init.zeros_(self.g_sparse.weight)

    
    def _get_mask_torch(self, w, k, replace):
        p = w.clone()
        p = p.float()
        choices = []
        batch_size, num_experts = p.shape
        indices = torch.arange(batch_size).to(p.device)
        complete_mask = torch.zeros_like(p)
        for i in range(k):
            zero_indices = (p.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            c = p.cumsum(dim=1)
            u = torch.rand(len(c), 1).to(p.device)
            choice = (u < c).int().argmax(dim=1)

            mask = torch.zeros_like(p)
            mask[indices, choice] = 1.0
            mask[zero_indices, :] = 0.0

            complete_mask += mask

            if not replace:
                p[indices, choice] = 0
                p = p / (p.sum(dim=1, keepdim=True) + 1e-10)  # added a small constant to prevent division by zero
                p[torch.isnan(p)] = 0

        return complete_mask


    def _sampled_softmax(self, gate_logits, g, k=2, biasness="zero", replace=True, deterministic=False):
        if not deterministic:
            indices_mask = self._get_mask_torch(g, k, replace)
        else:
            topk = torch.topk(gate_logits, k)
            indices_mask = torch.zeros_like(gate_logits).scatter(1, topk.indices, 1.0)

        gate_logits_with_neg_infs = torch.where(
            indices_mask == 0.0, -np.inf * torch.ones_like(gate_logits), gate_logits
        )
        adjusted_gate_logits = gate_logits_with_neg_infs + torch.log(indices_mask)

        if not deterministic:
            if biasness == "zero":
                adjusted_gate_logits -= torch.log(
                    torch.clamp(torch.tensor(k, dtype=torch.float32) * (g + 1e-10), min=1e-10)
                )
            elif biasness == "small-random":
                indices_mask_uniform = torch.where(
                    indices_mask > 0.0, torch.ones_like(indices_mask), torch.zeros_like(indices_mask)
                )
                indices_mask_uniform = indices_mask_uniform / indices_mask_uniform.sum(dim=1, keepdims=True)
                random_indices_mask = self._get_mask_torch(indices_mask_uniform, 1, False)
                adjusted_gate_logits -= torch.where(
                    random_indices_mask > 0.0,
                    torch.zeros_like(g),
                    torch.log(torch.clamp(torch.tensor(k - 1, dtype=torch.float32) * (g + 1e-10), min=1e-10)),
                )
            elif biasness == "large":
                pass
            else:
                raise ValueError("biasness {} is not supported".format(biasness))
        else:
            adjusted_gate_logits -= torch.log(
                torch.clamp(torch.tensor(k, dtype=torch.float32) * (g + 1e-10), min=1e-10)
            )

        g_on_sampled_mask = F.softmax(
            torch.where(indices_mask == 0.0, -np.inf * torch.ones_like(adjusted_gate_logits), adjusted_gate_logits),
            dim=-1,
        )
        return g_on_sampled_mask

    def _topk(self, g, k):
        num_rows = g.size(0)
        topk_values, topk_indices = torch.topk(g, k, dim=1)
        topk_tensor = torch.zeros_like(g)
        row_indices = torch.arange(num_rows).unsqueeze(-1).expand_as(topk_indices)
        topk_tensor.scatter_(1, topk_indices, topk_values)
        return topk_tensor


    def forward(self, inputs, training, indices=None, sample_indices=None):
        h, x = inputs
        assert all([h[i].size(1) == h[i + 1].size(1) for i in range(len(h) - 1)])

        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        h = [t.unsqueeze(-1) for t in h]
        h = torch.cat(h, dim=2)

        k = self.k

        gate_logits = self.gate_weights(x)

        if self.jitter and training:
            gate_logits += torch.randn(gate_logits.shape, device=gate_logits.device) * self.epsilon

        gate_logits /= self.tau

        g = F.softmax(gate_logits, dim=1)
        prob_mass_sorted = torch.mean(g.sort(dim=1, descending=True).values, dim=0)

        if training:
            g_topk = self._topk(g, k)
            trimmed_lasso_loss = torch.mean(torch.sum(torch.abs(g - g_topk), dim=1), dim=0)
            self.loss = self.trimmed_lasso_reg * trimmed_lasso_loss

        if training:
            g_on_sampled_mask = self._sampled_softmax(
                gate_logits, g, k=k, biasness=self.biasness, replace=self.replace, deterministic=False
            )
        else:
            g_on_sampled_mask = self._sampled_softmax(
                gate_logits, g, k=k, biasness=self.biasness, replace=self.replace, deterministic=True
            )
        g_on_sampled_mask = g_on_sampled_mask.unsqueeze(-1)

        g_permuted = g_on_sampled_mask

        if training:
            g_sparse_previous = self.g_sparse.weight[sample_indices, :]
            mask_previous = torch.where(
                g_sparse_previous == 0.0, torch.zeros_like(g_sparse_previous), torch.ones_like(g_sparse_previous)
            )

            g_sparse_current = g_permuted.squeeze(dim=2)
            mask_current = torch.where(
                g_sparse_current == 0.0, torch.zeros_like(g_sparse_current), torch.ones_like(g_sparse_current)
            )
            mask_diff = torch.abs(mask_current - mask_previous)
            routing_consistency = torch.mean(mask_diff)
            self.routing_consistency_metric = routing_consistency

            self.g_sparse.weight[sample_indices, :] = g_sparse_current

        y = torch.matmul(h, g_permuted).view(-1, h.size(1))

        s_concat = torch.where(g_permuted < 1e-5, torch.ones_like(g_permuted), torch.zeros_like(g_permuted))

        self.avg_sparsity_metric = torch.mean(s_concat)

        soft_averages = torch.mean(g_permuted, dim=[0])
        hard_averages = torch.mean(torch.ones_like(s_concat) - s_concat, dim=[0])
        soft_averages_for_all_experts_list = torch.split(soft_averages.view(-1), self.nb_experts)

        prob_mass_sorted_list = torch.split(prob_mass_sorted.view(-1), self.nb_experts)

        simplex_constraint = torch.mean(torch.sum(g_permuted, dim=1))

        simplex_constraint_fails = torch.sum(torch.sum(g_permuted, dim=1), dim=1)

        simplex_constraint_fails = torch.where(
            simplex_constraint_fails < 1.0 - 1e-5,
            torch.ones_like(simplex_constraint_fails),
            torch.zeros_like(simplex_constraint_fails),
        )
        simplex_constraint_fails = torch.mean(simplex_constraint_fails, dim=0)

        self.simplex_constraint_fails_metric = simplex_constraint_fails

        return y, soft_averages, hard_averages


def test_forward():
    # Set up config and instantiate the TopKGate
    config = {
        "task": 0,
        "nb_experts": 5,
        "k": 2,
        "jitter": False,
        "input_dim": 10,
        "use_routing_input": False,
        "tau": 1.0,
        "trimmed_lasso_reg": 0.0,
        'num_training_samples': 100,
        
    }
    gate = SampleKSoftmaxUnbiasedWithTrimmedLassoGate(config)

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
    y, soft_averages, hard_averages, s_concat, reg_loss = gate((f, x), training=True)

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