import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils import control_flow_util 



class SampleKSoftmaxUnbiasedWithTrimmedLassoGate(tf.keras.layers.Layer):
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
    def __init__(
        self,
        config,
        task=0,
    ):
        super(SampleKSoftmaxUnbiasedWithTrimmedLassoGate, self).__init__()
        self.task = config["task"]
        self.use_routing_input = config["use_routing_input"]
        # self.regularization_coef = config[“regularization_coef”]
        self.nb_experts = config["nb_experts"]
        self.k = config["k"]
        self.jitter = (config["jitter"] if config["jitter"] is not None else False)
        self.epsilon = tf.cast(1e-6, dtype=tf.float32)
        self.tau = (float)(config["tau"])
        self.trimmed_lasso_reg = config["trimmed_lasso_reg"]
        self.num_training_samples = config["num_training_samples"]
        self.biasness = (config["biasness"] if "biasness" in config else "zero")
        self.replace = (config["replace"] if "replace" in config else False)
        print("===================replace=========================", self.replace)
        
        
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
        self.gate_weights = self.add_weight(
            name="gate_weights",
            shape=(self.nb_experts, input_shape[1][1]),
#                 initializer=tf.keras.initializers.Zeros()
        )   
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.nb_experts,),
                initializer=tf.keras.initializers.Zeros()
            )

        self.g_sparse = self.add_weight(
            name="g_sparse",
            shape=(self.num_training_samples, self.nb_experts),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )                     
  

    def _get_mask_numpy(self, w, k, replace):
        p = np.copy(w)
        p = p.astype(np.float32)
        choices = []
        batch_size, np_experts = p.shape
        indices = np.arange(batch_size)
        complete_mask = np.zeros_like(p)
        for i in np.arange(k):
            zero_indices = np.where(p.sum(axis=1)==0)[0]
            c = p.cumsum(axis=1)
            u = np.random.rand(len(c), 1)
            choice = (u < c).argmax(axis=1)


            mask = np.zeros_like(p)
            mask[indices, choice] = 1.0
            mask[zero_indices, :] = 0.0

            complete_mask += mask

            if not replace:
                p[indices, choice] = 0
                p = p/p.sum(axis=1, keepdims=True)
                p = np.nan_to_num(p)
                
        return complete_mask

    @tf.function
    def _get_mask_tf(self, weights, k, replace):
        mask_batch = tf.numpy_function(
            func=self._get_mask_numpy,
            inp=[weights, k, replace],
            Tout=tf.float32
        )
        return mask_batch 

    def _sampled_softmax(self, gate_logits, g, k=2, biasness="zero", replace=True, deterministic=False):
        if not deterministic:
            indices_mask = self._get_mask_tf(g, k, replace)
            indices_mask = tf.cast(
                indices_mask,
                dtype=gate_logits.dtype
            )
        else:
            topk = tf.math.top_k(
                gate_logits,         
                k=k
            )

            indices_mask = tf.reduce_sum(
                tf.one_hot(
                   topk.indices,
                   depth=self.nb_experts
                ), 
                axis=1
            ) 
            
        gate_logits_with_neg_infs = tf.where(
            tf.math.equal(indices_mask, tf.constant(0.0)),
            -np.inf * tf.ones_like(gate_logits),  # we add the mask here
            gate_logits
        )
        adjusted_gate_logits = gate_logits_with_neg_infs + tf.math.log(indices_mask) # important for sampling with replacement
        
        if not deterministic:
            if biasness=="zero":      
                adjusted_gate_logits -= tf.math.log(
                    tf.cast(k, dtype=tf.float32)*(g + 1e-10)
                )
            elif biasness=="small-random":
                indices_mask_uniform = tf.where(
                    tf.math.greater(indices_mask, 0.0),
                    tf.ones_like(indices_mask),
                    tf.zeros_like(indices_mask)
                )
                indices_mask_uniform = indices_mask_uniform / tf.reduce_sum(indices_mask_uniform, axis=1, keepdims=True)

                random_indices_mask = self._get_mask_tf(indices_mask_uniform, 1, False) 
                random_indices_mask = tf.cast(
                    random_indices_mask,
                    dtype=gate_logits.dtype
                )
            #     print("random_indices_mask:", random_indices_mask)

                adjusted_gate_logits -= tf.where(
                    tf.math.greater(random_indices_mask, 0.0),
                    tf.zeros_like(g),
                    tf.math.log(tf.cast(k-1, dtype=tf.float32)*(g+1e-10))
                )
            #         print("adjusted_gate_logits:", adjusted_gate_logits)
            elif biasness=="large":
                pass
            else:
                raise ValueError("biasness {} is not supported".format(biasness))
        else:
            adjusted_gate_logits -= tf.math.log(
                tf.cast(k, dtype=tf.float32)*(g + 1e-10)
            )
            
        g_on_sampled_mask = tf.nn.softmax(
            tf.where(
                tf.math.equal(indices_mask, tf.constant(0.0)),
                -np.inf * tf.ones_like(adjusted_gate_logits),  # we add the mask here
                adjusted_gate_logits
            ),
            axis=-1
        )
        return g_on_sampled_mask


    def _topk(self, g, k):
        topk = tf.math.top_k(
            g,
            k=k
        )

        num_rows = tf.shape(g)[0]
        row_range = tf.range(num_rows)
        # row_tensor = tf.tile(row_range[:,None], (1, self.k))
        row_tensor = tf.tile(row_range[:,None], (1, k))
        topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)

        topk_scattered = tf.scatter_nd(
            topk_row_col_indices,
            topk.values,
            g.shape
        )
        return topk_scattered

    def call(
        self,
        inputs,           # inputs = (h,x), h being a list of tensors, all of the same size
        training,
        indices=None,
        sample_indices=None
    ):
        h, x, permutation_weights = inputs
        assert(all([h[i].shape[1] == h[i+1].shape[1] for i in range(len(h)-1)]))
               
        # h: [(bs, dim_exp_i, 1) for i in range(nb_experts)] with dim_exp_i = dim_exp = constant
        # h = [tf.reshape(t, [-1, t.shape[1], 1]) for t in h]
        h = [tf.expand_dims(t, -1) for t in h]
        # h: (bs, dim_exp_i, nb_experts)
        h = tf.concat(h, axis=2)
# -

        k = self.k


        # expert_weights: (bs, nb_experts)
        gate_logits = tf.matmul(
            x,
            tf.transpose(self.gate_weights)
        )
#             tf.print("expert_weights:", expert_weights[0,:], summarize=-1)

        if self.use_bias:
#                 tf.print("========Bias added", summarize=-1, output_stream=sys.stdout)
            gate_logits += tf.expand_dims(self.bias, axis=0)
#                 tf.print("expert_weights+bias:", expert_weights[0,:], summarize=-1)

        if self.jitter and training:
            gate_logits += tf.random.normal(gate_logits.shape)*self.epsilon


        gate_logits /= self.tau

        g = tf.nn.softmax(gate_logits, axis=1) # (bs, nb_experts)
        prob_mass_sorted = tf.reduce_mean(tf.sort(g, axis=1, direction='DESCENDING'), axis=0)

        if training:
            g_topk = self._topk(g, k)
            trimmed_lasso_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(g-g_topk), axis=1), axis=0)
            self.add_loss(self.trimmed_lasso_reg*trimmed_lasso_loss)

        
        if training:
            g_on_sampled_mask = self._sampled_softmax(gate_logits, g, k=k, biasness=self.biasness, replace=self.replace, deterministic=False)
        else:
            g_on_sampled_mask = self._sampled_softmax(gate_logits, g, k=k, biasness=self.biasness, replace=self.replace, deterministic=True)
        g_on_sampled_mask = tf.expand_dims(g_on_sampled_mask, axis=-1)

        g_permuted = g_on_sampled_mask

        if training:
            g_sparse_previous = tf.gather(self.g_sparse, sample_indices, axis=0) # (b, nb_experts)
            mask_previous = tf.where(
                tf.math.equal(g_sparse_previous, 0.0),
                tf.zeros_like(g_sparse_previous),
                tf.ones_like(g_sparse_previous)
            )
            
            g_sparse_current = tf.squeeze(g_permuted, axis=2)
            mask_current = tf.where(
                tf.math.equal(g_sparse_current, 0.0),
                tf.zeros_like(g_sparse_current),
                tf.ones_like(g_sparse_current)
            )
            mask_diff = tf.abs(mask_current - mask_previous)
            routing_consistency = tf.reduce_mean(mask_diff)
            self.add_metric(routing_consistency, "routing_consistency_for_task{}".format(self.task+1))
#             tf.print("======routing_consistency:", routing_consistency, summarize=-1)
            self.g_sparse.assign(tf.tensor_scatter_nd_update(self.g_sparse, sample_indices[:,None], g_sparse_current))
#             self.embedding.assign(tf.tensor_scatter_nd_update(self.embedding, sample_indices[:,None], x))

        y = tf.reshape(
            tf.matmul(
                h,
                g_permuted 
            ),
            [-1, h.shape[1]]
        )

        s_concat = tf.where(
            tf.math.less(g_permuted, 1e-5),
            tf.ones_like(g_permuted),
            tf.zeros_like(g_permuted)
        )

        self.add_metric(
            tf.reduce_mean(s_concat),
            name='avg_sparsity'
        )
        soft_averages = tf.reduce_mean(g_permuted, axis=[0]) # (nb_experts,)
        hard_averages = tf.reduce_mean(tf.ones_like(s_concat)-s_concat, axis=[0]) # (nb_experts,)
        soft_averages_for_all_experts_list = tf.split(
            tf.reshape(soft_averages, [-1]),
            self.nb_experts
        )
        [self.add_metric(le, name='soft_averages_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(soft_averages_for_all_experts_list)]

        prob_mass_sorted_list = tf.split(tf.reshape(prob_mass_sorted, [-1]), self.nb_experts) 
        [self.add_metric(le, name='prob_mass_sorted_for_task_{}_for_expert_{}'.format(self.task+1, j)) for j, le in enumerate(prob_mass_sorted_list)]

        simplex_constraint = tf.reduce_mean(
            tf.reduce_sum(g_permuted, axis=1),
        )
#             tf.print("========simplex_constraint:", simplex_constraint)
        self.add_metric(simplex_constraint, name='simplex_sum_for_task_{}'.format(self.task+1))
        simplex_constraint_fails = tf.reduce_sum(
            tf.reduce_sum(g_permuted, axis=1),
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
        config = super(SampleKSoftmaxUnbiasedWithTrimmedLassoGate, self).get_config()
        config.update({
            "k": self.k
        })
        return config
