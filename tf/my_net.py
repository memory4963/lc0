import tensorflow.keras as keras
import tensorflow as tf
import attention_policy_map as apm


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.DEFAULT_ACTIVATION = 'relu'
        self.l2reg = keras.regularizers.l2(l=0.5 * (0.0001))
        self.virtual_batch_size = None
        self.RESIDUAL_FILTERS = 64
        self.SE_ratio = 2
        self.encoder_layers = 0
        self.RESIDUAL_BLOCKS = 6
        self.pol_embedding_size = 64
        self.pol_encoder_dff = 128
        self.pol_encoder_heads = 4
        self.pol_encoder_d_model = 64
        self.policy_d_model = 64
        self.model_dtype = tf.float32

        self.input_conv = self.create_conv_block(filter_size=3, output_channels=self.RESIDUAL_FILTERS, name='input', bn_scale=True)
        self.res_blocks = []
        for i in range(self.RESIDUAL_BLOCKS):
            res_block = self.create_residual_block(self.RESIDUAL_FILTERS, 'residual_{}'.format(i + 1))
            se = self.create_squeeze_excitation(self.RESIDUAL_FILTERS, 'residual_{}'.format(i + 1) + '/se')
            ase = ApplySqueezeExcitation()
            activate = keras.Sequential((
                keras.layers.Activation(self.DEFAULT_ACTIVATION)
            ))
            self.res_blocks.append((res_block, se, ase, activate))

        self.policy_dense = keras.layers.Dense(self.pol_embedding_size,
                                           kernel_initializer='glorot_normal',
                                           kernel_regularizer=self.l2reg,
                                           activation='selu',
                                           name='policy/embedding')

        self.weight_q = keras.layers.Dense(self.pol_encoder_d_model,
                                  name='policy/enc_layer/mha/wq',
                                  kernel_initializer='glorot_normal')
        self.weight_k = keras.layers.Dense(self.pol_encoder_d_model,
                                  name='policy/enc_layer/mha/wk',
                                  kernel_initializer='glorot_normal')
        self.weight_v = keras.layers.Dense(self.pol_encoder_d_model,
                                  name='policy/enc_layer/mha/wv',
                                  kernel_initializer='glorot_normal')
        self.mha_out_layer = keras.layers.Dense(
            self.pol_embedding_size, name="policy/enc_layer/mha/dense",
            kernel_initializer='glorot_normal')
        self.ln_before_ffn = keras.layers.LayerNormalization(epsilon=1e-6, name="policy/enc_layer/ln1")
        self.ffn = self.create_ffn(self.pol_embedding_size, self.pol_encoder_dff, 'glorot_normal', 'policy/enc_layer/ffn')
        self.ln_after_ffn = keras.layers.LayerNormalization(epsilon=1e-6, name="policy/enc_layer/ln2")

        self.promotion_weight_q = keras.layers.Dense(self.policy_d_model,
                                            kernel_initializer='glorot_normal',
                                            name='policy/attention/wq')
        self.promotion_weight_k = keras.layers.Dense(self.policy_d_model,
                                            kernel_initializer='glorot_normal',
                                            name='policy/attention/wk')
        self.promotion_offset_layer = keras.layers.Dense(4, kernel_initializer='glorot_normal',
            name='policy/attention/ppo', use_bias=False)
        self.aapm = ApplyAttentionPolicyMap()

        self.value_net = keras.Sequential((
            self.create_conv_block(1, 32, 'value'),
            tf.keras.layers.Flatten(),
            keras.layers.Dense(128, kernel_initializer='glorot_normal',
                            kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                            name='value/dense1')
        ))
        self.value_dense2 = keras.layers.Dense(3, kernel_initializer='glorot_normal',
                            kernel_regularizer=self.l2reg,
                            bias_regularizer=self.l2reg,
                            name='value/dense2')

        self.moves_left_net = keras.Sequential((
            self.create_conv_block(1, 8, 'moves_left'),
            tf.keras.layers.Flatten(),
            keras.layers.Dense(128, kernel_initializer='glorot_normal',
                            kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                            name='moves_left/dense1'),
            keras.layers.Dense(1, kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='relu',
                                          name='moves_left/dense2')
        ))


    def call(self, inputs):
        # input shape: None, 112, 8, 8
        flow = self.input_conv(inputs)
        for i in range(self.RESIDUAL_BLOCKS):
            res_block = self.res_blocks[i]
            res_out = res_block[0](flow)
            excited = res_block[1](res_out)
            res_out = res_block[2]([res_out, excited])
            flow = keras.layers.add([flow, res_out])
            flow = res_block[3](flow)

        # policy
        attn_wts = []
        tokens = tf.transpose(flow, perm=[0, 2, 3, 1])
        tokens = tf.reshape(tokens, [-1, 64, self.RESIDUAL_FILTERS])
        tokens = self.policy_dense(tokens)

        attn_output, attn_wts_l = self.mha(tokens, self.pol_embedding_size, self.pol_encoder_d_model, self.pol_encoder_heads)
        attn_output = self.ln_before_ffn(tokens + attn_output)
        ffn_output = self.ffn(attn_output)
        tokens = self.ln_after_ffn(attn_output + ffn_output)
        attn_wts.append(attn_wts_l)

        pro_q = self.promotion_weight_q(tokens)
        pro_k = self.promotion_weight_k(tokens)
        h_fc1 = self.apply_promotion_logits(pro_q, pro_k, attn_wts)

        # value
        h_fc2 = self.value_net(flow)
        h_fc3 = self.value_dense2(h_fc2)

        # moves_left
        h_fc5 = self.moves_left_net(flow)

        return h_fc1, h_fc3, h_fc5, attn_wts

    def create_conv_block(self, filter_size, output_channels, name, bn_scale=False):
        return keras.Sequential((
            keras.layers.Conv2D(output_channels, filter_size,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_first',
                name=name+'/conv2d'),
            self.create_bn(name=name+'/bn', scale=bn_scale),
            keras.layers.Activation(self.DEFAULT_ACTIVATION)
            ))

    def create_bn(self, name, scale=False):
        return tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            axis=1,
            center=True,
            scale=scale,
            virtual_batch_size=self.virtual_batch_size,
            name=name)

    def create_residual_block(self, channels, name):
        return keras.Sequential((
            keras.layers.Conv2D(channels, 3,
                    use_bias=False,
                    padding='same',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=self.l2reg,
                    data_format='channels_first',
                    name=name + '/1/conv2d'),
            self.create_bn(name+'/1/bn', scale=False),
            keras.layers.Activation(self.DEFAULT_ACTIVATION),
            keras.layers.Conv2D(channels, 3,
                    use_bias=False,
                    padding='same',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=self.l2reg,
                    data_format='channels_first',
                    name=name + '/2/conv2d'),
            self.create_bn(name+'/2/bn', scale=True),
            ))
        # sequeeze excitation

    # init
    def create_squeeze_excitation(self, channels, name):
        assert channels % self.SE_ratio == 0

        return keras.Sequential((
            keras.layers.GlobalAveragePooling2D(
                data_format='channels_first'),
            keras.layers.Dense(channels // self.SE_ratio,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                name=name + '/se/dense1'),
            keras.layers.Activation(self.DEFAULT_ACTIVATION),
            keras.layers.Dense(2 * channels,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                name=name + '/se/dense2')
        ))
        # ApplySqueezeExcitation

    # call
    def encoder_layer(self, inputs, emb_size: int, d_model: int,
                      num_heads: int, dff: int, name: str):
        initializer = None
        if self.encoder_layers > 0:
            # DeepNorm
            alpha = tf.cast(tf.math.pow(2. * self.encoder_layers, 0.25),
                            self.model_dtype)
            beta = tf.cast(tf.math.pow(8. * self.encoder_layers, -0.25),
                           self.model_dtype)
            xavier_norm = tf.keras.initializers.VarianceScaling(
                scale=beta, mode='fan_avg', distribution='truncated_normal')
            initializer = xavier_norm
        else:
            alpha = 1
            initializer = "glorot_normal"
        # multihead attention
        attn_output, attn_wts = self.mha(inputs,
                                         emb_size,
                                         d_model,
                                         num_heads,
                                         initializer,
                                         name=name + "/mha")
        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(self.dropout_rate,
                                              name=name +
                                              "/dropout1")(attn_output)
        # skip connection + layernorm
        out1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln1")(inputs * alpha + attn_output)
        # feed-forward network
        ffn_output = self.ffn(out1,
                              emb_size,
                              dff,
                              initializer,
                              name=name + "/ffn")
        ffn_output = tf.keras.layers.Dropout(self.dropout_rate,
                                             name=name +
                                             "/dropout2")(ffn_output)
        out2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=name + "/ln2")(out1 * alpha + ffn_output)
        return out2, attn_wts

    # init
    def create_ffn(self, emb_size: int, dff: int, initializer, name: str):
        activation = "selu"
        return keras.Sequential((
                    keras.layers.Dense(dff,
                                       name=name + "/dense1",
                                       kernel_initializer=initializer,
                                       activation=activation),
                    keras.layers.Dense(emb_size,
                                    name=name + "/dense2",
                                    kernel_initializer=initializer)
                ))

    # call
    def apply_promotion_logits(self, queries, keys, attn_wts):
        # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
        dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1],
                                  self.model_dtype))  # constant for scaling
        promotion_keys = keys[:, -8:, :]
        # queen, rook, bishop, knight order
        promotion_offsets = self.promotion_offset_layer(promotion_keys)
        promotion_offsets = tf.transpose(promotion_offsets,
                                         perm=[0, 2, 1]) * dk  # Bx4x8
        # knight offset is added to the other three
        promotion_offsets = promotion_offsets[:, :
                                              3, :] + promotion_offsets[:,
                                                                        3:4, :]

        # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
        matmul_qk = tf.matmul(
            queries, keys,
            transpose_b=True)  # Bx64x64 (from 64 queries, 64 keys)

        # q, r, and b promotions are offset from the default promotion logit (knight)
        n_promo_logits = matmul_qk[:, -16:-8,
                                   -8:]  # default traversals from penultimate rank to promotion rank
        q_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 0:1, :],
                                        axis=3)  # Bx8x8x1
        r_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 1:2, :],
                                        axis=3)
        b_promo_logits = tf.expand_dims(n_promo_logits +
                                        promotion_offsets[:, 2:3, :],
                                        axis=3)
        promotion_logits = tf.concat(
            [q_promo_logits, r_promo_logits, b_promo_logits],
            axis=3)  # Bx8x8x3
        promotion_logits = tf.reshape(
            promotion_logits,
            [-1, 8, 24])  # logits now alternate a7a8q,a7a8r,a7a8b,...,

        # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
        promotion_logits = promotion_logits / dk  # Bx8x24 (8 from-squares, 3x8 promotions)
        policy_attn_logits = matmul_qk / dk  # Bx64x64 (64 from-squares, 64 to-squares)

        attn_wts.append(promotion_logits)
        attn_wts.append(policy_attn_logits)

        # APPLY POLICY MAP: output becomes Bx1856
        h_fc1 = self.aapm(policy_attn_logits, promotion_logits)
        return h_fc1

    # call
    def mha(self, inputs, emb_size: int, d_model: int, num_heads: int):
        assert d_model % num_heads == 0

        depth = d_model // num_heads
        q = self.weight_q(inputs)
        k = self.weight_k(inputs)
        v = self.weight_v(inputs)
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size, num_heads, depth)
        k = self.split_heads(k, batch_size, num_heads, depth)
        v = self.split_heads(v, batch_size, num_heads, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        if num_heads > 1:
            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])
            scaled_attention = tf.reshape(
                scaled_attention,
                (batch_size, -1, d_model))  # concatenate heads

        # final dense layer
        output = self.mha_out_layer(scaled_attention)
        return output, attention_weights

    # call
    def split_heads(self, inputs, batch_size: int, num_heads: int, depth: int):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        # (batch_size, num_heads, 64, depth)
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    # call
    def scaled_dot_product_attention(self, q, k, v):

        # 0 h 64 d, 0 h d 64
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        heads = scaled_attention_logits.shape[1]

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, scaled_attention_logits


class ApplySqueezeExcitation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyAttentionPolicyMap(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ApplyAttentionPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(apm.make_map())

    def call(self, logits, pp_logits):
        logits = tf.concat([
            tf.reshape(logits, [-1, 64 * 64]),
            tf.reshape(pp_logits, [-1, 8 * 24])
        ],
                           axis=1)
        return tf.matmul(logits, tf.cast(self.fc1, logits.dtype))
