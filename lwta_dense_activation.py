from distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

class LWTA_Dense_Activation(tf.keras.layers.Layer):
    
    def call(self,inputs,training=None):
        
        lam = inputs
        K = int(inputs.shape[-1]//2)
        U = 2
         
        if training:
            layer_loss = 0.
                    
            # reshape weight for LWTA
            lam_re = tf.reshape(lam, [-1,K,U])
        
            # calculate probability of activation and some stability operations
            prbs = tf.nn.softmax(lam_re) + 1e-4
            prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)
        
            # relaxed categorical sample
            xi = concrete_sample(prbs, 0.67)
        
            #apply activation
            out  = lam_re * xi
            out = tf.reshape(out, tf.shape(input=lam))
        
            # kl for the relaxed categorical variables
            kl_xi = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=concrete_kl(tf.ones([1,K,U])/U, prbs, xi), axis=[1]))
            # print(kl_xi) #negative #something very small
            tf.compat.v1.add_to_collection('kl_loss', kl_xi)
            # self.add_loss(tf.math.reduce_mean(kl_xi)/60000)
            layer_loss = layer_loss + tf.math.reduce_mean(kl_xi)/100000
            tf.compat.v2.summary.scalar(name='kl_xi', data=kl_xi)

        else:

            layer_loss = 0.

            lam_re = tf.reshape(lam, [-1, K, U])
            prbs = tf.nn.softmax(lam_re) + 1e-4
            prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)

           # apply activation
            out = lam_re*concrete_sample(prbs, 0.01)
            out = tf.reshape(out, tf.shape(input=lam))
        
        self.add_loss(layer_loss)

        return out, prbs