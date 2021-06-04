import os
import numpy as np

# Basic JAX
import jax
from jax import random, tree_map, numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

# JAX neural networks
import flax.linen as nn
from flax import optim
from flax.core import freeze, unfreeze
from jax.nn.initializers import he_normal

from .keras import get_Neural_Network


class Encoder(nn.Module):
    zdim: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Common arguments
        kwargs = {'kernel_size': (4, 4),
                  'strides': (2, 2),
                  'padding': 'SAME',
                  'use_bias': False,
                  'kernel_init': he_normal()}

        # x = np.reshape(x, (64, 64, 1))
        x = x[..., None]

        # Layer 1
        x = nn.Conv(features=64, **kwargs)(x)
        x = nn.leaky_relu(x, 0.2)

        # Layer 2
        x = nn.Conv(features=128, **kwargs)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x, 0.2)

        # Layer 3
        x = nn.Conv(features=256, **kwargs)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x, 0.2)

        # Layer 4
        x = nn.Conv(features=512, **kwargs)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x, 0.2)

        # Layer 5
        x = nn.Conv(features=4096, kernel_size=(4, 4), strides=(1, 1),
                    padding='VALID', use_bias=False, kernel_init=he_normal())(x)
        x = nn.leaky_relu(x, 0.2)

        # Flatten
        x = x.flatten()

        # Predict latent variables
        z_mean = nn.Dense(features=self.zdim)(x)
        z_logvar = nn.Dense(features=self.zdim)(x)

        return z_mean, z_logvar

class Decoder(nn.Module):
    zdim: int

    @nn.compact
    def __call__(self, z, train: bool = True):
        # Common arguments
        conv_kwargs = {'kernel_size': (4, 4),
                       'strides': (2, 2),
                       'padding': 'SAME',
                       'use_bias': False,
                       'kernel_init': he_normal()}
        norm_kwargs = {'use_running_average': not train,
                       'momentum': 0.99,
                       'epsilon': 0.001,
                       'use_scale': True,
                       'use_bias': True}

        z = np.reshape(z, (1, 1, self.zdim))

        # Layer 1
        z = nn.ConvTranspose(features=512, kernel_size=(4, 4), strides=(1, 1),
                             padding='VALID', use_bias=False,
                             kernel_init=he_normal())(z)
        z = nn.BatchNorm(**norm_kwargs)(z)
        z = nn.leaky_relu(z, 0.2)

        # Layer 2
        z = nn.ConvTranspose(features=256, **conv_kwargs)(z)
        z = nn.BatchNorm(**norm_kwargs)(z)
        z = nn.leaky_relu(z, 0.2)

        # Layer 3
        z = nn.ConvTranspose(features=128, **conv_kwargs)(z)
        z = nn.BatchNorm(**norm_kwargs)(z)
        z = nn.leaky_relu(z, 0.2)

        # Layer 4
        z = nn.ConvTranspose(features=64, **conv_kwargs)(z)
        z = nn.BatchNorm(**norm_kwargs)(z)
        z = nn.leaky_relu(z, 0.2)

        # Layer 5
        z = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2),
                             padding='SAME', use_bias=False,
                             kernel_init=nn.initializers.xavier_normal())(z)
        # x = nn.sigmoid(z)
        x = nn.softplus(z)

        return jnp.rot90(np.squeeze(x), k=2)  # Rotate to match TF output

    def load_keras_model(self, checkpoint, prng_key=None):
        # Create the Keras beta-VAE
        keras_nn = get_Neural_Network(1e-3, 'softplus', 'chi_sq')
        models, model_loss_function, reconstruction_loss_function = keras_nn
        # Load weights into keras model from a given checkpoint
        # base_dir = os.path.dirname(os.getcwd())
        # checkpoint = os.path.join(base_dir, 'data', f'epoch_{epoch}', 'Model')
        # assert os.path.exists(checkpoint), "Path does not exist : " + checkpoint
        models['vae'].load_weights(checkpoint)
        decoder_weights = models['decoder'].get_weights()
        decoder_weights = [jnp.array(w) for w in decoder_weights]

        # Initialise
        if prng_key is None:
            prng_key = random.PRNGKey(42)
        init_data = jnp.ones((1, 1, self.zdim))
        params = self.init(prng_key, init_data)

        # Replace weights by the ones from keras
        unfrozen_params = unfreeze(params)
        unfrozen_params['params']['ConvTranspose_0']['kernel'] = np.swapaxes(decoder_weights[0], 2, 3)
        unfrozen_params['params']['BatchNorm_0']['scale'] = decoder_weights[1]
        unfrozen_params['params']['BatchNorm_0']['bias'] = decoder_weights[2]
        unfrozen_params['batch_stats']['BatchNorm_0']['mean'] = decoder_weights[3]
        unfrozen_params['batch_stats']['BatchNorm_0']['var'] = decoder_weights[4]
        unfrozen_params['params']['ConvTranspose_1']['kernel'] = np.swapaxes(decoder_weights[5], 2, 3)
        unfrozen_params['params']['BatchNorm_1']['scale'] = decoder_weights[6]
        unfrozen_params['params']['BatchNorm_1']['bias'] = decoder_weights[7]
        unfrozen_params['batch_stats']['BatchNorm_1']['mean'] = decoder_weights[8]
        unfrozen_params['batch_stats']['BatchNorm_1']['var'] = decoder_weights[9]
        unfrozen_params['params']['ConvTranspose_2']['kernel'] = np.swapaxes(decoder_weights[10], 2, 3)
        unfrozen_params['params']['BatchNorm_2']['scale'] = decoder_weights[11]
        unfrozen_params['params']['BatchNorm_2']['bias'] = decoder_weights[12]
        unfrozen_params['batch_stats']['BatchNorm_2']['mean'] = decoder_weights[13]
        unfrozen_params['batch_stats']['BatchNorm_2']['var'] = decoder_weights[14]
        unfrozen_params['params']['ConvTranspose_3']['kernel'] = np.swapaxes(decoder_weights[15], 2, 3)
        unfrozen_params['params']['BatchNorm_3']['scale'] = decoder_weights[16]
        unfrozen_params['params']['BatchNorm_3']['bias'] = decoder_weights[17]
        unfrozen_params['batch_stats']['BatchNorm_3']['mean'] = decoder_weights[18]
        unfrozen_params['batch_stats']['BatchNorm_3']['var'] = decoder_weights[19]
        unfrozen_params['params']['ConvTranspose_4']['kernel'] = np.swapaxes(decoder_weights[20], 2, 3)
        self.params = freeze(unfrozen_params)


class VAE(nn.Module):
    xdim: int
    zdim: int

    def setup(self):
        self.encoder = Encoder(self.zdim)
        self.decoder = Decoder(self.zdim)

    def __call__(self, x, z_rng, train: bool = True):
        z_mean, z_logvar = self.encoder(x, train)
        z = self.reparameterize(z_rng, z_mean, z_logvar)
        x_rec = self.decoder(z, train)
        return x_rec, z_mean, z_logvar

    @staticmethod
    def reparameterize(rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    def load_keras_model(self, checkpoint, prng_key=None):
        # Create the Keras beta-VAE
        keras_nn = get_Neural_Network(1e-3, 'softplus', 'chi_sq')
        models, model_loss_function, reconstruction_loss_function = keras_nn
        # Load weights into keras model from the given checkpoint
        models['vae'].load_weights(checkpoint).expect_partial()
        encoder_weights = models['encoder'].get_weights()
        decoder_weights = models['decoder'].get_weights()
        # Recast as JAX device arrays to enable autodiff through the model
        encoder_weights = [jnp.array(w) for w in encoder_weights]
        decoder_weights = [jnp.array(w) for w in decoder_weights]

        # Initialise
        if prng_key is None:
            prng_key = random.PRNGKey(42)
        init_data = jnp.ones((self.xdim, self.zdim))
        key, subkey1, subkey2 = random.split(prng_key, 3)
        params = self.init(subkey1, init_data, z_rng=subkey2)

        # Replace encoder weights
        unfrozen_params = unfreeze(params)
        unfrozen_params['params']['encoder']['Conv_0']['kernel'] = encoder_weights[0]
        unfrozen_params['params']['encoder']['Conv_1']['kernel'] = encoder_weights[1]
        unfrozen_params['params']['encoder']['BatchNorm_0']['scale'] = encoder_weights[2]
        unfrozen_params['params']['encoder']['BatchNorm_0']['bias'] = encoder_weights[3]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_0']['mean'] = encoder_weights[4]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_0']['var'] = encoder_weights[5]
        unfrozen_params['params']['encoder']['Conv_2']['kernel'] = encoder_weights[6]
        unfrozen_params['params']['encoder']['BatchNorm_1']['scale'] = encoder_weights[7]
        unfrozen_params['params']['encoder']['BatchNorm_1']['bias'] = encoder_weights[8]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_1']['mean'] = encoder_weights[9]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_1']['var'] = encoder_weights[10]
        unfrozen_params['params']['encoder']['Conv_3']['kernel'] = encoder_weights[11]
        unfrozen_params['params']['encoder']['BatchNorm_2']['scale'] = encoder_weights[12]
        unfrozen_params['params']['encoder']['BatchNorm_2']['bias'] = encoder_weights[13]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_2']['mean'] = encoder_weights[14]
        unfrozen_params['batch_stats']['encoder']['BatchNorm_2']['var'] = encoder_weights[15]
        unfrozen_params['params']['encoder']['Conv_4']['kernel'] = encoder_weights[16]
        unfrozen_params['params']['encoder']['Dense_0']['kernel'] = encoder_weights[17]
        unfrozen_params['params']['encoder']['Dense_0']['bias'] = encoder_weights[18]
        unfrozen_params['params']['encoder']['Dense_1']['kernel'] = encoder_weights[19]
        unfrozen_params['params']['encoder']['Dense_1']['bias'] = encoder_weights[20]

        # Replace decoder weights
        unfrozen_params['params']['decoder']['ConvTranspose_0']['kernel'] = np.swapaxes(decoder_weights[0], 2, 3)
        unfrozen_params['params']['decoder']['BatchNorm_0']['scale'] = decoder_weights[1]
        unfrozen_params['params']['decoder']['BatchNorm_0']['bias'] = decoder_weights[2]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_0']['mean'] = decoder_weights[3]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_0']['var'] = decoder_weights[4]
        unfrozen_params['params']['decoder']['ConvTranspose_1']['kernel'] = np.swapaxes(decoder_weights[5], 2, 3)
        unfrozen_params['params']['decoder']['BatchNorm_1']['scale'] = decoder_weights[6]
        unfrozen_params['params']['decoder']['BatchNorm_1']['bias'] = decoder_weights[7]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_1']['mean'] = decoder_weights[8]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_1']['var'] = decoder_weights[9]
        unfrozen_params['params']['decoder']['ConvTranspose_2']['kernel'] = np.swapaxes(decoder_weights[10], 2, 3)
        unfrozen_params['params']['decoder']['BatchNorm_2']['scale'] = decoder_weights[11]
        unfrozen_params['params']['decoder']['BatchNorm_2']['bias'] = decoder_weights[12]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_2']['mean'] = decoder_weights[13]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_2']['var'] = decoder_weights[14]
        unfrozen_params['params']['decoder']['ConvTranspose_3']['kernel'] = np.swapaxes(decoder_weights[15], 2, 3)
        unfrozen_params['params']['decoder']['BatchNorm_3']['scale'] = decoder_weights[16]
        unfrozen_params['params']['decoder']['BatchNorm_3']['bias'] = decoder_weights[17]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_3']['mean'] = decoder_weights[18]
        unfrozen_params['batch_stats']['decoder']['BatchNorm_3']['var'] = decoder_weights[19]
        unfrozen_params['params']['decoder']['ConvTranspose_4']['kernel'] = np.swapaxes(decoder_weights[20], 2, 3)
        self.params = freeze(unfrozen_params)

    def encode(self, x):
        if not hasattr(self, 'params'):
            raise AttributeError("No params available. First initialise the" +
                                 " model else or load a Keras checkpoint.")
        return self.apply(self.params, x, method=self._encode)

    def _encode(self, x):
        return self.encoder(x, train=False)

    def decode(self, z):
        if not hasattr(self, 'params'):
            raise AttributeError("No params available. First initialise the" +
                                 " model else or load a Keras checkpoint.")
        return self.apply(self.params, z, method=self._decode)

    def _decode(self, z):
        return self.decoder(z, train=False)
