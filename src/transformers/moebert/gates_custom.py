import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf


class TopKSoftmaxGate_tensorflow(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
        task=0,
    ):
        super(TopKSoftmaxGate_tensorflow, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        # self.regularization_coef = config[“regularization_coef”]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        
        if "temperature" in config.keys():
            self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
            self.temperature = (
                tf.constant(config["temperature"]) if config["temperature"] is not None else config["temperature"]
            )
        else:
            self.temperature = None
        if self.use_routing_input:
            self.use_bias = config["use_bias"]

        
    def build(
        self,
        input_shape
    ):
        if not self.use_routing_input:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts,)   
            )
        else:
            self.expert_weights = self.add_weight(
                name="expert_weights",
                shape=(self.nb_experts, input_shape[1][1])
            )   
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.nb_experts,),
                    initializer=tf.keras.initializers.Zeros()
                )

            
    def call(
        self,
        inputs,           # inputs = (h,x), h being a list of tensors, all of the same size
        training
    ):
        h, x, permutation_weights = inputs
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))
               
        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        # h = [tf.reshape(t, [-1, t.shape[1], 1]) for t in h]
        h = [tf.expand_dims(t, -1) for t in h]
        # h: (bs, dim_exp_i, nb_experts)
        h = tf.concat(h, axis=2)
        
        if self.temperature is not None:
            p = tf.shape(self.expert_weights)[0]
            self.iterations.assign_add(1)
            scheduler = (
                1.0-tf.math.exp(
                    -tf.cast(
                        self.temperature, self.expert_weights.dtype
                    ) * tf.cast(
                        self.iterations, dtype=self.expert_weights.dtype
                    )
                )
            )
            k = p - tf.cast(
                tf.math.round( 
                    tf.cast(
                        p - tf.constant(self.k, dtype=p.dtype), 
                        dtype=self.expert_weights.dtype
                    ) * scheduler
                ),
                dtype=p.dtype
            )
            self.add_metric(
                tf.cast(k, dtype=tf.float32),
                name='k'
            )
            # tf.print("k: ", k)
        else:
            k = self.k
        
        if not self.use_routing_input:
            topk = tf.math.top_k(
                tf.reshape(tf.expand_dims(self.expert_weights, 1), [-1]),
                # k=self.k,
                k=k,
            )
            topk_scattered = tf.scatter_nd(
                tf.reshape(topk.indices, [-1, 1]),
                topk.values,
                [self.nb_experts]
            )
            topk_prep = tf.where(
                tf.math.equal(topk_scattered, tf.constant(0.0)),
                -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
                topk_scattered
            )
            # softmaxes: (nb_experts, 1)
            softmaxes = tf.nn.softmax(
                tf.expand_dims(topk_prep, 1),  # else, we get an error in the softmax activation
                axis=0
            )
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes # tf.reshape(topk_scattered, [-1, 1])
                ),
                [-1, h.shape[1]]
            )
            
            self.add_metric(
                tf.reduce_mean(
                    tf.where(
                        tf.math.less(softmaxes, 1e-5),
                        tf.ones_like(softmaxes),
                        tf.zeros_like(softmaxes)
                    )
                ),
                name='avg_sparsity'
            )

            return y
        
        else:
            # expert_weights: (bs, nb_experts)
            expert_weights = tf.matmul(
                x,
                tf.transpose(self.expert_weights)
            )

            if self.use_bias:
#                 tf.print("========Bias added", summarize=-1, output_stream=sys.stdout)
                expert_weights += tf.expand_dims(self.bias, axis=0)
            
            # print("expert_weights: ", expert_weights)
            topk = tf.math.top_k(
                expert_weights,
                # k=self.k,
                k=k
            )
            
            num_rows = tf.shape(expert_weights)[0]
            row_range = tf.range(num_rows)
            # row_tensor = tf.tile(row_range[:,None], (1, self.k))
            row_tensor = tf.tile(row_range[:,None], (1, k))
            topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)
            
            topk_scattered = tf.expand_dims(
                tf.scatter_nd(
                    topk_row_col_indices,
                    topk.values,
                    expert_weights.shape
                ),
                -1
            )
            # tf.print("topk scattered: ",topk_scattered, summarize=-1)
            # print("topk: ",topk_scattered)
            # softmaxes = tf.keras.activations.softmax(
            softmaxes = tf.nn.softmax(
                tf.where(
                    tf.math.equal(topk_scattered, tf.constant(0.0)),
                    -np.inf * tf.ones_like(topk_scattered),  # we add the mask here
                    topk_scattered
                ),
                axis=1
            ) # (bs, nb_experts, 1)

#             tf.print("gate softmaxes shape: ",tf.shape(softmaxes), summarize=-1)
#             tf.print("gate softmaxes: ",tf.reduce_sum(tf.reduce_sum(softmaxes, axis=1)), summarize=-1)
            # print("softmaxes: ", softmaxes)
    
            # softmaxes: (bs, nb_experts, 1), perm_mask: [k, nb_experts, nb_experts]
            permutation_weights = tf.reduce_mean(permutation_weights, axis=0) # [nb_experts, nb_experts]
            softmaxes_permuted = tf.einsum('bik,ij->bjk', softmaxes, permutation_weights)
            
            # h: (bs, dim_exp, nb_experts), softmaxes: (bs, nb_experts)    
            # y: (bs, dim_exp)
            y = tf.reshape(
                tf.matmul(
                    h,
                    softmaxes_permuted 
                ),
                [-1, h.shape[1]]
            )
            
            s_concat = tf.where(
                tf.math.less(softmaxes_permuted, 1e-5),
                tf.ones_like(softmaxes_permuted),
                tf.zeros_like(softmaxes_permuted)
            )

            self.add_metric(
                tf.reduce_mean(s_concat),
                name='avg_sparsity'
            )
            soft_averages = tf.reduce_mean(softmaxes_permuted, axis=[0]) # (nb_experts,)
            hard_averages = tf.reduce_mean(tf.ones_like(s_concat)-s_concat, axis=[0]) # (nb_experts,)
            soft_averages_for_all_experts_list = tf.split(
                tf.reshape(soft_averages, [-1]),
                self.nb_experts
            )
            [self.add_metric(le, name='soft_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(soft_averages_for_all_experts_list)]
            hard_averages_for_all_experts_list = tf.split(
                tf.reshape(hard_averages, [-1]),
                self.nb_experts
            )
            [self.add_metric(le, name='hard_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(hard_averages_for_all_experts_list)] 
            
            simplex_constraint = tf.reduce_mean(
                tf.reduce_sum(softmaxes_permuted, axis=1),
            )
#             tf.print("========simplex_constraint:", simplex_constraint)
            self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
            simplex_constraint_fails = tf.reduce_sum(
                tf.reduce_sum(softmaxes_permuted, axis=1),
                axis=[1]
            ) # (b, )

            simplex_constraint_fails = tf.where(
                tf.math.less(simplex_constraint_fails, 1.0-1e-5),
                tf.ones_like(simplex_constraint_fails),
                tf.zeros_like(simplex_constraint_fails)
            ) # (b, nb_gates)
            simplex_constraint_fails = tf.reduce_mean(simplex_constraint_fails, axis=0)
            self.add_metric(simplex_constraint_fails, name='simplex_constraint_fails_for_task_{}'.format(self.task+1))
            
            return y, soft_averages, hard_averages
        
        
    def get_config(self):
        config = super(TopKSoftmaxGate_tensorflow, self).get_config()
        config.update({
            "k": self.k
        })
        return config

class TopKSoftmaxGate_pytorch(nn.Module):
    def __init__(self, nb_experts, k = 2, use_bias=True, temperature=1.0):
        super(TopKSoftmaxGate_pytorch, self).__init__()
        self.nb_experts = nb_experts
        self.k = k
        if temperature != 0.0:
            self.iterations =  nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.temperature = (temperature)
        
        self.expert_weights = nn.Parameter(torch.Tensor(self.nb_experts, 1))
        
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.nb_experts))
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expert_weights)
        if self.use_bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, h, x, permutation_weights):  
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))

        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        # h = [tf.reshape(t, [-1, t.shape[1], 1]) for t in h]
        h = torch.stack(h, dim=2) # (bs, dim_exp, nb_experts)

        # h: (bs, dim_exp_i, nb_experts)
        h = torch.concat(h, axis=2)
        
        if self.temperature != 0.0:
            p = self.expert_weights.shape[0]
            self.iterations += 1
            scheduler = 1.0 - torch.exp(-self.iterations / self.temperature)
            k = p - torch.floor(scheduler * p)
            
        else:
            k = self.k
            
        


class DSelectkGate(nn.Layer):
    def __init__(self,
                 num_nonzeros,
                 expert_num,
                 input_shape=None,
                 gamma=1.0,
                 entropy_reg=None,
                 z_initializer=None,
                 w_initializer=None):
        super(DSelectkGate, self).__init__()
        raise NotImplementedError("DSelectkGate is not implemented yet")
        self._num_nonzeros = num_nonzeros
        self._smooth_step = SmoothStep(gamma)
        self._entropy_reg = entropy_reg
        self._z_initializer = z_initializer or torch.nn.initializer.Uniform(
            -gamma / 100.0, gamma / 100.0)
        self._w_initializer = w_initializer or torch.nn.initializer.Uniform()
        self._expert_num = expert_num
        self._num_binary = math.ceil(math.log2(expert_num))
        self._power_of_2 = (expert_num == 2**self._num_binary)
        if input_shape is None:
            z_logits = self.create_parameter(
                shape=[self._num_nonzeros, 1, self._num_binary],
                attr=torch.ParamAttr(initializer=self._z_initializer))
            self._z_logits = z_logits
            self.add_parameter("z_logits", z_logits)

            w_logits = self.create_parameter(
                shape=[self._num_nonzeros, 1],
                attr=torch.ParamAttr(initializer=self._w_initializer))
            self._w_logits = w_logits
            self.add_parameter("w_logits", w_logits)
        else:
            self._z_logits = torch.nn.Linear(
                in_features=input_shape,
                out_features=self._num_nonzeros * self._num_binary,
                weight_attr=torch.ParamAttr(initializer=self._z_initializer),
                bias_attr=torch.ParamAttr(initializer=self._z_initializer))
            self._w_logits = torch.nn.Linear(
                in_features=input_shape,
                out_features=self._num_nonzeros,
                weight_attr=torch.ParamAttr(initializer=self._w_initializer),
                bias_attr=torch.ParamAttr(initializer=self._w_initializer))

        self._binary_codes = torch.unsqueeze(
            self.dec2bin(
                torch.arange(
                    start=0, end=expert_num, dtype=torch.int64),
                self._num_binary),
            axis=0)

    def dec2bin(self, x, bits):
        mask = torch.arange(bits - 1, -1, -1, dtype=torch.float32)
        mask = torch.cast(2**mask, dtype=torch.int64)
        return torch.not_equal(
            x.unsqueeze(-1).bitwise_and(mask),
            torch.full(
                shape=[1], fill_value=0, dtype=torch.int64))

    def forward(self, inputs, training=False):
        if isinstance(inputs, tuple):
            experts, routing_inputs = inputs
        else:
            experts, routing_inputs = inputs, None

        if routing_inputs is None:
            # static gating
            expert_weights, selector_outputs = self._compute_expert_weights()
            output = torch.add_n(inputs=[
                expert_weights[i] * experts[i] for i in range(len(experts))
            ])
        else:
            # per-example gating
            expert_weights, selector_outputs = self._compute_example_conditioned_expert_weights(
                routing_inputs)
            output = torch.add_n(inputs=[
                torch.reshape(expert_weights[:, i], [-1, 1]) * experts[i]
                for i in range(len(experts))
            ])

        return output

    def _compute_expert_weights(self):
        """Computes the weight vector for the experts.
        Args: None.
        Returns:
          A tuple: (expert_weights, selector_outputs).
            expert_weights is the final weight vector of the experts.
            selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
            represents the outputs of the i-th single-expert selector.
        """
        # Shape = (num_nonzero, 1, num_binary)
        smooth_step_activations = self._smooth_step(self._z_logits)

        # Shape = (num_nonzero, num_experts)
        selector_outputs = torch.prod(
            torch.where(self._binary_codes, smooth_step_activations,
                         1 - smooth_step_activations),
            axis=2)

        # Weights for the single-expert selectors: shape = (num_nonzero, 1)
        selector_weights = F.softmax(self._w_logits, axis=0)
        expert_weights = torch.sum(selector_weights * selector_outputs,
                                    axis=0)

        return expert_weights, selector_outputs

    def _compute_example_conditioned_expert_weights(self, routing_inputs):
        """Computes the example-conditioned weights for the experts.
        Args:
            routing_inputs: a tensor of shape=(batch_size, num_features) containing
            the input examples.
        Returns:
            A tuple: (expert_weights, selector_outputs).
            expert_weights is a tensor with shape=(batch_size, num_experts),
            containing the expert weights for each example in routing_inputs.
            selector_outputs is a tensor with
            shape=(batch_size, num_nonzero, num_experts), which contains the outputs
            of the single-expert selectors for all the examples in routing_inputs.
        """
        sample_logits = torch.reshape(
            self._z_logits(routing_inputs),
            [-1, self._num_nonzeros, 1, self._num_binary])
        smooth_step_activations = self._smooth_step(sample_logits)

        # Shape = (batch_size, num_nonzeros, num_experts).
        selector_outputs = torch.prod(
            torch.where(
                torch.unsqueeze(self._binary_codes, 0),
                smooth_step_activations, 1 - smooth_step_activations),
            axis=3)

        # Weights for the single-expert selectors
        # Shape = (batch_size, num_nonzeros, 1)
        selector_weights = torch.unsqueeze(self._w_logits(routing_inputs), 2)
        selector_weights = F.softmax(selector_weights, axis=1)

        # Sum over the signle-expert selectors. Shape = (batch_size, num_experts).
        expert_weights = torch.sum(selector_weights * selector_outputs,
                                    axis=1)

        return expert_weights, selector_outputs
