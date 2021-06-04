# beta-VAE galaxy model by @egorssed
import tensorflow as tf
from keras.layers import (Input, Dense, BatchNormalization, Flatten, Reshape,
                          Lambda, Conv2D, Conv2DTranspose,LeakyReLU)
from keras.models import Model
from keras.optimizers import Adam
from keras import initializers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops,smart_cond
import keras.backend as K


image_size = 64
npix = image_size * image_size
batch_size = 32
latent_dim = 64
start_lr = 1e-6

def encoder_function(input_img):
    #He initialization for Relu activated layers
    Heinitializer = initializers.HeNormal()

    x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
               use_bias=False, kernel_initializer=Heinitializer)(input_img)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same',
               use_bias=False, kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=256, kernel_size=4, strides=2, padding='same',
               use_bias=False, kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
               use_bias=False, kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=4096, kernel_size=4, strides=1, padding='valid',
               use_bias=False, kernel_initializer=Heinitializer)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)


    #Predict mean of standard distribution and logarithm of variance
    #Initialize initial logvar to be very small,
    #So latent variables do not become completely random and lose the meaning
    Logvar_initializer=initializers.Constant(value=-10)
    Xavierinitializer=initializers.GlorotNormal()

    z_mean = Dense(latent_dim, kernel_initializer=Xavierinitializer,
                   bias_initializer=Xavierinitializer)(x)
    z_log_var = Dense(latent_dim, kernel_initializer=Xavierinitializer,
                      bias_initializer=Logvar_initializer)(x)

    return z_mean, z_log_var

def get_decoder(activation):
    def decoder_function(z):
        #He initialization for Relu activated layers
        Heinitializer = initializers.HeNormal()

        x = Reshape(target_shape=(1, 1, 64))(z)

        x = Conv2DTranspose(filters=512, kernel_size=4, strides=1,
                            padding='valid', use_bias=False,
                            kernel_initializer=Heinitializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2DTranspose(filters=256, kernel_size=4, strides=2,
                            padding='same', use_bias=False,
                            kernel_initializer=Heinitializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2DTranspose(filters=128, kernel_size=4, strides=2,
                            padding='same', use_bias=False,
                            kernel_initializer=Heinitializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2DTranspose(filters=64, kernel_size=4, strides=2,
                            padding='same', use_bias=False,
                            kernel_initializer=Heinitializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        #Xavier intialization for differentiable functions activated layers
        Xavierinitializer=initializers.GlorotNormal()
        if activation=='':
            decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,
                                      padding='same', use_bias=False,
                                      kernel_initializer=Xavierinitializer)(x)
        else:
            decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,
                                      padding='same', use_bias=False,
                                      activation=activation,
                                      kernel_initializer=Xavierinitializer)(x)

        return decoded
    return decoder_function

def get_reconstruction_loss(loss_type='chi_sq'):

    #Chose original loss function
    if (loss_type == 'chi_sq') or (loss_type == 'mse'):
        loss_function = tf.math.squared_difference
    if loss_type == 'binary_crossentropy':
        loss_function = K.binary_crossentropy

    def reconstruction_loss_function(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        reconstruction_loss = loss_function(y_true, y_pred)

        #cast mse to chi_square by adding weights
        if (loss_type == 'chi_sq'):
            #Poisson weights (sigma=sqrt(image))
            weights = tf.math.pow(tf.sqrt(tf.abs(y_true) + 1e-5), -1)
            weights = math_ops.cast(weights, y_pred.dtype)
            reconstruction_loss = reconstruction_loss * weights

        return K.mean(reconstruction_loss, axis=-1)

    return reconstruction_loss_function

def get_model_loss(beta_vae=1e-3, loss_type='chi_sq'):
    reconstruction_loss_function = get_reconstruction_loss(loss_type)

    def model_loss_function(x, decoded):
        #reconstruction quality
        flattened_x = K.reshape(x, shape=(len(x), npix))
        flattened_decoded = K.reshape(decoded, shape=(len(decoded), npix))
        reconstruction_loss = (npix * reconstruction_loss_function(flattened_x,
                               flattened_decoded))

        #KL divergence regularization quality
        mean = models['z_meaner'](x)
        logvar = models['z_log_varer'](x)
        KL_loss = 0.5 * K.sum(1 + logvar - K.square(mean) - K.exp(logvar), axis=-1)

        Beta_VAE_Loss = (reconstruction_loss - beta_vae * KL_loss) / npix
        return Beta_VAE_Loss

    return model_loss_function, reconstruction_loss_function

def create_vae(beta_vae=1e-3, activation='softplus', loss_type='chi_sq'):

    #Reparametrization trick
    def reparameterize(args):
        mean, logvar = args
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar / 2) + mean

    decoder_function = get_decoder(activation)
    model_loss_function, reconstruction_loss_function = get_model_loss(beta_vae, loss_type)


    #VAE
    models = {}

    #Encoder
    input_img = Input(batch_shape=(batch_size, image_size, image_size, 1))

    z_mean, z_log_var = encoder_function(input_img)

    l = Lambda(reparameterize, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"]  = Model(input_img, l, name='Encoder')
    models["z_meaner"] = Model(input_img, z_mean, name='Enc_z_mean')
    models["z_log_varer"] = Model(input_img, z_log_var, name='Enc_z_log_var')

    #Decoder
    z = Input(shape=(latent_dim, ))
    decoded = decoder_function(z)

    models["decoder"] = Model(z, decoded, name='Decoder')
    models["vae"]     = Model(input_img, models["decoder"](models["encoder"](input_img)), name="VAE")

    return models,model_loss_function, reconstruction_loss_function

def get_Neural_Network(beta_vae=1e-3, activation='softplus', loss_type='chi_sq'):
    with tf.device('/device:GPU:0'):
        models, model_loss_function, reconstruction_loss_function = create_vae(beta_vae, activation, loss_type)
        models["vae"].compile(optimizer=Adam(learning_rate=start_lr, beta_1=0.5,
                              beta_2=0.999, clipvalue=0.1),
                              loss=model_loss_function)
        return models, model_loss_function, reconstruction_loss_function
