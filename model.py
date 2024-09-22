from utils import *


class OrthoRegularization(keras.layers.Layer):
    """
    Custom Regularization layer applied on the activations of the latent space, representing the extracted abundance maps,
    to induce linear independence between them with a relaxed orthogonality constraint
    """
    def __init__(self, params):
        super(OrthoRegularization, self).__init__()
        self.batch_size = params['batch_size']
        self.orth_factor = params['orth_factor']
        # initializing None variables outside of call function
        self.abd_maps = None
        self.AAt = None
        self.diag = None
        self.norms = None

    def call(self, x, **kwargs):
        """
        The input of this layer correspond to the abundance maps of a given patch
        The shape is [n1, n2, k] where n1, n2 is the patch dimensions and k the number of elements ;
        except we are using batches, so the input shape is [BatchSize, n1, n2, k]
        We want to reshape the inputs into 2d matrices where each row describes an element : [k, n1 x n2]
        for batches : [BS, k, n1 x n2]
        :param x:
        """
        # turning [BatchSize, n1, n2, k] into [BatchSize, k, n1, n2]
        self.abd_maps = tf.transpose(x, perm=[0, 3, 1, 2])

        # turning [BatchSize, k, n1, n2] into [BatchSize, k, n1xn2]
        self.abd_maps = tf.reshape(self.abd_maps, [
            self.batch_size,  # BS
            x.get_shape().as_list()[-1],  # k
            x.get_shape().as_list()[1] * x.get_shape().as_list()[2]])  # n1 x n2

        # A * transpose(A)
        self.AAt = tf.linalg.matmul(self.abd_maps, self.abd_maps, transpose_a=False, transpose_b=True)

        # extracting AAt's diagonal (to subtract it later)
        self.diag = tf.linalg.set_diag(tf.zeros_like(self.AAt), tf.linalg.diag_part(self.AAt))
        # we have a norm for each batch
        # tf.norm is L2 norm by default, equivalent to tf.sqrt(tf.reduce_sum(tf.square(
        self.norms = tf.norm(tf.subtract(self.AAt, self.diag), axis=[-1, -2])

        # mean of the norms over batches, multiplied by the orth_factor
        self.add_loss(self.orth_factor * tf.math.reduce_mean(self.norms))
        return


class SumToOne(keras.layers.Layer):
    """
    Layer imposing the Abundances Sum-to-one Constraint (ASC) with the softmax function
    """
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.params = params

    def call(self, x, **kwargs):
        x = tf.nn.softmax(self.params['smax_temp'] * x)
        return x


class LKA(keras.layers.Layer):
    """
    Large Kernel Attention block, composed of three successive convolution layers  : \n
    Depth-Wise \n
    Dilated Depth-Wise \n
    Point Wise \n
    Returns the product between the input and the obtained feature maps
    """
    def __init__(self, dim):
        super(LKA, self).__init__()
        self.dw_conv = keras.layers.Conv2D(filters=dim, kernel_size=5, strides=1, padding='same', groups=dim)
        self.dw_d_conv = keras.layers.Conv2D(filters=dim, kernel_size=1, strides=1, padding='same', dilation_rate=3, groups=dim)
        self.pw_conv = keras.layers.Conv2D(filters=dim, kernel_size=1, strides=1, padding='same')

    def call(self, x, *args, **kwargs):
        u = tf.identity(x)
        attn = self.dw_conv(x)
        attn = self.dw_d_conv(attn)
        attn = self.pw_conv(attn)
        return u * attn


class AttBlockLKA(keras.layers.Layer):
    """
    Attention block, composed of successive layers: \n
    Batch Normalization \n
    1x1 Conv \n
    LKA \n
    1x1 Conv \n
    Returns the sum between the input and the obtained feature maps
    """
    def __init__(self, dim):
        super(AttBlockLKA, self).__init__()
        self.bn = keras.layers.BatchNormalization()
        self.conv1_1x1 = keras.layers.Conv2D(filters=dim, kernel_size=1)
        self.lka = LKA(dim)
        self.conv2_1x1 = keras.layers.Conv2D(filters=dim, kernel_size=1)

    def call(self, x, *args, **kwargs):
        skip = tf.identity(x)
        x = self.bn(x)
        x = self.conv1_1x1(x)
        x = tf.nn.gelu(x)
        x = self.lka(x)
        x = self.conv2_1x1(x)
        return x + skip, x


class MyEncoder(keras.Model):
    def __init__(self, params):
        super(MyEncoder, self).__init__()
        self.params = params
        self.code = None  # initializing code outside of call function
        self.hidden_layer_one = keras.layers.Conv2D(filters=self.params['e_filters'],
                                                    kernel_size=self.params['e_size'],
                                                    activation=self.params['activation'], strides=1, padding='same',
                                                    kernel_initializer=keras.initializers.orthogonal(),
                                                    use_bias=False)

        self.hidden_layer_two = keras.layers.Conv2D(filters=self.params['num_endmembers'], kernel_size=1,
                                                    activation=self.params['activation'], strides=1, padding='same',
                                                    kernel_initializer=keras.initializers.orthogonal(),
                                                    use_bias=False)

        self.asc_layer = SumToOne(params=self.params, name='ASC')

        self.ortho_regul_layer = OrthoRegularization(params=self.params)

        self.LKA_block_1 = AttBlockLKA(dim=self.params['e_filters'])
        self.LKA_block_2 = AttBlockLKA(dim=self.params['num_endmembers'])

    def call(self, input_patch, **kwargs):
        # input patch has shape [BatchSize, PatchDim1, PatchDim2, NumSpectra]

        # conv1 output has shape [BatchSize, PatchDim1, PatchDim2, NumFilters]
        self.code = self.hidden_layer_one(input_patch)
        self.code, blc = self.LKA_block_1(self.code)

        # conv2 output has shape [BatchSize, PatchDim1, PatchDim2, NumEndmembers]
        self.code = self.hidden_layer_two(self.code)
        F = self.code
        self.code, Ff = self.LKA_block_2(self.code)

        self.code = self.asc_layer(self.code)

        if self.params['orth_factor'] != 0:
            self.ortho_regul_layer(self.code)

        return self.code


class Decoder(keras.Model):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.output_layer = keras.layers.Dense(units=params['n_bands'],
                                               activation='linear',
                                               kernel_constraint=keras.constraints.non_neg(),
                                               kernel_initializer=keras.initializers.orthogonal(),
                                               )

    def call(self, code, **kwargs):
        return self.output_layer(code)

    def getEndmembers(self):
        return self.output_layer.get_weights()


class Autoencoder(keras.Model):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.params = params
        self.encoder = MyEncoder(self.params)
        self.decoder = Decoder(self.params)

    def call(self, patch, **kwargs):
        return self.decoder(self.encoder(patch))

    def getAbundances(self, msi):
        """
        Returns the latent space, i.e. the output of the encoder, i.e. the abundance maps, extracted from an MSI
        :param msi:
        :return abundance maps:
        """
        return np.squeeze(self.encoder.predict(np.expand_dims(msi, 0), verbose=1))

    def getOutput(self, abundances):
        """
        Returns the reconstructed output, i.e. the output of the decoder, from abundance maps
        :param abundances:
        :return reconstructed output:
        """
        return np.squeeze(self.decoder.predict(np.expand_dims(abundances, 0), verbose=0))

    def train(self, patches):
        """
        regular Tensorflow .fit function, just made easier to write and read
        :param patches:
        """
        return self.fit(patches, patches, epochs=self.params['nb_epochs'], batch_size=self.params['batch_size'], verbose=1)
