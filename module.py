import tensorflow as tf
import numpy as np

def safe_sqrt(x):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, 1e-6, 1e8))

class Net(object):
    def __init__(self, dims, FLAGS):

        self.variables = {}
        self.wd_loss = 0

        ''' Initialize input placeholders '''
        self.x  = tf.compat.v1.placeholder("float", shape=[None, dims[1]], name='x') # Features
        self.t  = tf.compat.v1.placeholder("float", shape=[None, 1], name='t')   # Treatent
        self.y_ = tf.compat.v1.placeholder("float", shape=[None, 1], name='y_')  # Outcome
        self.do_in = tf.compat.v1.placeholder("float", name='dropout_in')
        self.do_out = tf.compat.v1.placeholder("float", name='dropout_out')
        self.p_t = tf.compat.v1.placeholder("float", name='p_treated')
        self.y_0_median = tf.compat.v1.placeholder("float", name='y_0_median')
        self.y_1_median = tf.compat.v1.placeholder("float", name='y_1_median')

        self.i0 = tf.to_int32(tf.where(self.t < 0.50)[:, 0])
        self.i1 = tf.to_int32(tf.where(self.t > 0.50)[:, 0])

        self.num = tf.shape(self.x)[0]

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        elif FLAGS.nonlin.lower() == 'tanh':
            self.nonlin = tf.nn.tanh
        else:
            self.nonlin = tf.nn.relu
        
        self._build_graph(dims, FLAGS)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) 
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        return var

    def _build_graph(self, dims, FLAGS):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        n, dim_input, dim_in, dim_out = dims

        r_coef_y = FLAGS.p_coef_y
        r_coef_mu = FLAGS.p_coef_mu
        r_coef_mi = FLAGS.p_coef_mi  

        weights_in = []
        biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            self.bn_biases = []
            self.bn_scales = []

        with tf.name_scope("representation"):
            h_rep_I, h_rep_norm_I, weights_in_I, biases_in_I = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            h_rep_C, h_rep_norm_C, weights_in_C, biases_in_C = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            h_rep_A, h_rep_norm_A, weights_in_A, biases_in_A = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            self.h_rep_I = h_rep_I
            self.h_rep_C = h_rep_C
            self.h_rep_A = h_rep_A
            self.h_rep_norm_I = h_rep_norm_I
            self.h_rep_norm_C = h_rep_norm_C
            self.h_rep_norm_A = h_rep_norm_A
            self.h_rep = tf.concat((h_rep_I, h_rep_C, h_rep_A), axis=1)
            self.h_rep_norm = tf.concat((h_rep_norm_I, h_rep_norm_C, h_rep_norm_A), axis=1)

        weights_in = weights_in_I + weights_in_C + weights_in_A
        biases_in = biases_in_I + biases_in_C + biases_in_A
        self.weights_in_I = weights_in_I
        self.weights_in_C = weights_in_C
        self.weights_in_A = weights_in_A

        with tf.name_scope("weight"):
            sample_weight = 1.0
        self.sample_weight = sample_weight

        with tf.name_scope("output"):
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output_graph(
                tf.concat([h_rep_norm_C, h_rep_norm_A], 1), self.t, 2 * dim_in, dim_out, self.do_out, FLAGS)
        self.weights_out = weights_out
        self.weights_pred = weights_pred

        with tf.compat.v1.variable_scope('mi_it'):
            self.lld_it, self.bound_it, self.mu_it, self.logvar_it, self.ws_it = self._mi_net(
                inp=h_rep_norm_I,
                outp=self.t,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='max',
                name='it')

        with tf.compat.v1.variable_scope('mi_iy'):
            self.lld_iy, self.bound_iy, self.mu_iy, self.logvar_iy, self.ws_iy = self._mi_net(
                inp=h_rep_norm_I,
                outp=self.y_,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='min',
                name='iy')

        with tf.compat.v1.variable_scope('mi_ct'):
            self.lld_ct, self.bound_ct, self.mu_ct, self.logvar_ct, self.ws_ct = self._mi_net(
                inp=h_rep_norm_C,
                outp=self.t,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='max',
                name='ct')

        with tf.compat.v1.variable_scope('mi_cy'):
            self.lld_cy, self.bound_cy, self.mu_cy, self.logvar_cy, self.ws_cy = self._mi_net(
                inp=h_rep_norm_C,
                outp=self.y_,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='max',
                name='cy')

        with tf.compat.v1.variable_scope('mi_at'):
            self.lld_at, self.bound_at, self.mu_at, self.logvar_at, self.ws_at = self._mi_net(
                inp=h_rep_norm_A,
                outp=self.t,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='min',
                name='at')

        with tf.compat.v1.variable_scope('mi_ay'):
            self.lld_ay, self.bound_ay, self.mu_ay, self.logvar_ay, self.ws_ay = self._mi_net(
                inp=h_rep_norm_A,
                outp=self.y_,
                dim_in=dim_in,
                dim_out=1,
                mi_min_max='max',
                name='ay')

        L_MI_C_lld = self.lld_ct + self.lld_cy
        L_MI_C_bound = self.bound_ct + self.bound_cy
        L_MI_C = L_MI_C_lld + L_MI_C_bound

        L_MI_I_lld = self.lld_it + self.lld_iy
        L_MI_I_bound = self.bound_it + self.bound_iy
        L_MI_I = L_MI_I_lld + L_MI_I_bound

        L_MI_A_lld = self.lld_at + self.lld_ay
        L_MI_A_bound = self.bound_at + self.bound_ay
        L_MI_A = L_MI_A_lld + L_MI_A_bound

        if r_coef_mu > 0:
            h_i_center = h_rep_norm_I - tf.reduce_mean(h_rep_norm_I, axis=0, keepdims=True)
            h_c_center = h_rep_norm_C - tf.reduce_mean(h_rep_norm_C, axis=0, keepdims=True)
            h_a_center = h_rep_norm_A - tf.reduce_mean(h_rep_norm_A, axis=0, keepdims=True)

            batch_size = tf.cast(tf.shape(h_i_center)[0], tf.float32)
            batch_size = tf.maximum(batch_size, 1.0)

            cov_ic = tf.matmul(h_i_center, h_c_center, transpose_a=True) / batch_size
            cov_ia = tf.matmul(h_i_center, h_a_center, transpose_a=True) / batch_size
            cov_ca = tf.matmul(h_c_center, h_a_center, transpose_a=True) / batch_size

            ortho_ic = tf.reduce_sum(tf.square(cov_ic))
            ortho_ia = tf.reduce_sum(tf.square(cov_ia))
            ortho_ca = tf.reduce_sum(tf.square(cov_ca))
            L_O = ortho_ic + ortho_ia + ortho_ca
        else:
            zero_scalar = tf.constant(0.0, dtype=tf.float32)
            ortho_ic = zero_scalar
            ortho_ia = zero_scalar
            ortho_ca = zero_scalar
            L_O = zero_scalar
        self.loss_ortho = L_O

        self.loss_mi_total = L_MI_C + L_MI_I + L_MI_A
        self.imb_dist_C = L_MI_C
        self.imb_dist_I = L_MI_I
        self.imb_dist_A = L_MI_A

        if FLAGS.loss == 'l1':
            L_R = tf.reduce_mean(sample_weight*tf.abs(self.y_-y))
            pred_error = -tf.reduce_mean(tf.abs(self.y_-y))
        elif FLAGS.loss == 'log':
            y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
            labels = tf.concat((1.0 - self.y_, self.y_), axis=1)
            logits = tf.concat((-y, y), axis=1)
            loss_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_per_sample = tf.reduce_mean(loss_per_sample, axis=1, keepdims=True)

            L_R = tf.reduce_mean(sample_weight * loss_per_sample)
            pred_error = tf.reduce_mean(loss_per_sample)
        else:
            L_R = tf.reduce_mean(sample_weight * tf.square(self.y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - y)))
        if r_coef_y > 0:
            L_R = r_coef_y * L_R


        # Representation weight decay is disabled (p_coef_lambda not used)
        R_W = 0.0

        tot_loss = L_R + (r_coef_mi * self.loss_mi_total) + (r_coef_mu * L_O) + R_W
        val_error = pred_error

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        if FLAGS.loss == 'log':
            y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
            self.output = y_prob

            label = y_prob
            one = tf.ones_like(label)
            zero = tf.zeros_like(label)
            self.output_discrete = tf.where(label < 0.5, x=zero, y=one)
        else:
            self.output = y
            self.output_discrete = y
        self.stage2_loss = tot_loss
        self.tot_loss = tot_loss
        self.val_loss = val_error
        self.imb_loss = r_coef_mi * self.loss_mi_total
        self.imb_dist = L_MI_C + L_MI_I + L_MI_A
        self.pred_loss = pred_error

        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.biases_in = biases_in
        self.biases_out = biases_out
        self.bias_pred = bias_pred

    def _build_representation_graph(self, dim_input, dim_in, dim_out, FLAGS):
        weights_in = [];
        biases_in = []

        h_in = [self.x]
        for i in range(0, FLAGS.n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0 / dim_input * tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.multiply(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        self.bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        self.bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, self.bn_biases[-1], self.bn_scales[-1],1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], self.do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep

        return h_rep, h_rep_norm, weights_in, biases_in

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * FLAGS.n_out)

        weights_out = [];
        biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=FLAGS.weight_init / np.sqrt(dims[i])),
                'y_w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1],
                                                              stddev=FLAGS.weight_init / np.sqrt(dim_out)), 'y_w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'y_b_pred')

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:
            rep0 = tf.gather(rep, self.i0)
            rep1 = tf.gather(rep, self.i1)

            y0, weights_out0, weights_pred0, biases_out0, bias_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1, biases_out1, bias_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([self.i0, self.i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
            biases_out = biases_out0 + biases_out1
            bias_pred = bias_pred0 + bias_pred1
        else:
            h_input = tf.concat([rep, t], 1)
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output(h_input, dim_in + 1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _fc_net(self, inp, dim_out, act_fun):
        init = tf.contrib.layers.xavier_initializer()
        return tf.contrib.layers.fully_connected(
            inputs=inp,
            num_outputs=dim_out,
            activation_fn=act_fun,
            weights_initializer=init
        )

    def _mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name=None):
        """Mutual information network adapted from AutoIV."""
        h_mu = self._fc_net(inp, dim_in // 2, tf.nn.elu)
        mu = self._fc_net(h_mu, dim_out, None)

        h_var = self._fc_net(inp, dim_in // 2, tf.nn.elu)
        logvar = self._fc_net(h_var, dim_out, tf.nn.tanh)

        new_order = tf.random_shuffle(tf.range(self.num))
        outp_rand = tf.gather(outp, new_order)

        loglikeli = -tf.reduce_mean(
            tf.reduce_sum(
                -(outp - mu) ** 2 / tf.exp(logvar) - logvar,
                axis=-1
            )
        )

        pos = -(mu - outp) ** 2 / tf.exp(logvar)
        neg = -(mu - outp_rand) ** 2 / tf.exp(logvar)

        if name == 'iy':
            t_rand = tf.gather(self.t, new_order)
            sigma = 1.0
            w = tf.exp(-tf.square(self.t - t_rand) / (2 * sigma ** 2))
            w_soft = tf.nn.softmax(w, axis=0)
        else:
            w_soft = 1.0 / tf.cast(self.num, tf.float32)

        if mi_min_max == 'min':
            pn = 1.0
        elif mi_min_max == 'max':
            pn = -1.0
        else:
            raise ValueError("mi_min_max must be 'min' or 'max'")

        bound = pn * tf.reduce_mean(w_soft * (pos - neg))

        return loglikeli, bound, mu, logvar, w_soft
