from distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

class LWTA_Conv2D_Activation(tf.keras.layers.Layer):
    
    def call(self,inputs,training=None):
        
        ksize = int(inputs.shape[-1]//2)
        lam = inputs

        if training:
            layer_loss = 0.
            # reshape weight to calculate probabilities
            lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], ksize, 2])

            prbs = tf.nn.softmax(lam_re) + 1e-5
            prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)

            # draw relaxed sample and apply activation
            xi = concrete_sample(prbs, 0.5)

            #apply activation
            out = lam_re * xi
            out = tf.reshape(out, tf.shape(input=lam))

            # add the relative kl terms
            kl_xi = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=concrete_kl(tf.ones_like(lam_re)/2, prbs, xi), axis=[1]))

            layer_loss = layer_loss + tf.math.reduce_mean(kl_xi)/100000

        else:

            layer_loss = 0.

            # calculate probabilities of activation
            lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], ksize, 2])
            prbs = tf.nn.softmax(lam_re) + 1e-5
            prbs /= tf.reduce_sum(input_tensor=prbs,axis=-1, keepdims=True)

            # draw sample for activated units
            out = lam_re * concrete_sample(prbs, 0.01)
            out = tf.reshape(out, tf.shape(input=lam))

        self.add_loss(layer_loss)

        return out, prbs
