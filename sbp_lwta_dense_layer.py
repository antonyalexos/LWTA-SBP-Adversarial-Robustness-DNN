#here we are going to build our own custom layer
from distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

import keras.backend as K

class SB_Layer(tf.keras.layers.Layer):
  
  def __init__(self,K=5,U=2,bias=True,sbp=True,temp_bern=0.67,temp_cat=0.67, activation='lwta', **kwargs):
    super(SB_Layer, self).__init__(**kwargs)
    self.tau = 1e-2
    self.K = K
    self.U = U
    self.bias = bias
    self.sbp = sbp
    self.temp_bern = temp_bern
    self.temp_cat = temp_cat
    self.activation = activation

  def build(self, input_shape):
    self.mW = self.add_weight(shape=(input_shape[-1],self.K*self.U),
                              initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                              trainable=True,
                              dtype=tf.float32,
                              name='mW')

    self.sW = self.add_weight(shape=(input_shape[-1],self.K*self.U),
                              initializer = tf.compat.v1.initializers.random_normal(-5.,1e-2),
                              constraint = lambda x: tf.clip_by_value(x, -7.,x),
                              trainable=True,
                              dtype=tf.float32,
                              name='sW')

    # variables and construction for the stick breaking process (if active)
    if self.sbp==True:    
      # posterior concentration variables for the IBP
      self.conc1 = self.add_weight(shape=([self.K]),
                                   initializer = tf.compat.v1.constant_initializer(self.K),
                                   constraint=lambda x: tf.clip_by_value(x, -6., x),
                                   trainable=True,
                                   dtype = tf.float32,
                                   name = 'sb_t_u_1')
      
      self.conc0 = self.add_weight(shape=([self.K]),
                                   initializer = tf.compat.v1.constant_initializer(2.),
                                   constraint=lambda x: tf.clip_by_value(x, -6., x),
                                   trainable=True,
                                   dtype = tf.float32,
                                   name = 'sb_t_u_2')
      
      # posterior probabilities z
      self.t_pi = self.add_weight(shape=[input_shape[-1],self.K], 
                                  initializer =  tf.compat.v1.initializers.random_uniform(-.1, .1),
                                  constraint = lambda x: tf.clip_by_value(x, -5.,600.),
                                  dtype = tf.float32,
                                  trainable=True,
                                  name = 'sb_t_pi')
      
    self.biases = 0
    if self.bias:
      self.biases = self.add_weight(shape=(self.K*self.U,),
                                    initializer=tf.compat.v1.constant_initializer(0.1),
                                    trainable=True,
                                    name='bias')
      
    # super(SB_Layer, self).build(input_shape)
    self.built = True 

  def call(self,inputs,training=None):
    
    sW_softplus = tf.nn.softplus(self.sW)

    if training:

      # reparametrizable normal sample
      eps = tf.stop_gradient(tf.random.normal([inputs.get_shape()[1], self.K*self.U]))
      # W = self.mW + eps * self.sW
      W = self.mW + eps * sW_softplus

      z = 1.
      layer_loss = 0.
      
      #sbp
      if self.sbp==True:

        # posterior concentration variables for the IBP
        conc1_softplus = tf.nn.softplus(self.conc1)
        conc0_softplus = tf.nn.softplus(self.conc0)

        # stick breaking construction
        q_u = kumaraswamy_sample(conc1_softplus, conc0_softplus, sample_shape = [inputs.get_shape()[1],self.K])
        pi = tf.math.cumprod(q_u)

        # posterior probabilities z
        t_pi_sigmoid = tf.nn.sigmoid(self.t_pi)

        # sample relaxed bernoulli
        z_sample = bin_concrete_sample(t_pi_sigmoid,self.temp_bern)
        z = tf.tile(z_sample, [1,self.U])
        re = z*W
        
        # kl terms for the stick breaking construction
        kl_sticks = tf.reduce_sum(input_tensor=kumaraswamy_kl(tf.ones_like(conc1_softplus), tf.ones_like(conc0_softplus),
                                              conc1_softplus, conc0_softplus, q_u))
        kl_z = tf.reduce_sum(input_tensor=bin_concrete_kl(pi, t_pi_sigmoid, self.temp_bern, z_sample))
    
        tf.compat.v1.add_to_collection('kl_loss', kl_sticks) #positive something very big
        tf.compat.v1.add_to_collection('kl_loss', kl_z) #negative something very big
        # self.add_loss(tf.math.reduce_mean(kl_sticks)/60000)
        layer_loss = layer_loss + tf.math.reduce_mean(kl_sticks)/60000
        layer_loss = layer_loss + tf.math.reduce_mean(kl_z)/60000
        # self.add_loss(tf.math.reduce_mean(kl_z)/60000)

        tf.compat.v2.summary.scalar(name='kl_sticks', data=kl_sticks)
        tf.compat.v2.summary.scalar(name='kl_z', data=kl_z)
        
        # cut connections if probability of activation less than tau
        tf.compat.v2.summary.scalar(name='sparsity', data=tf.reduce_sum(input_tensor=tf.cast(tf.greater(t_pi_sigmoid/(1.+t_pi_sigmoid), self.tau), tf.float32))*self.U)
        # sparsity = tf.reduce_sum(input_tensor=tf.cast(tf.greater(t_pi_sigmoid/(1.+t_pi_sigmoid), self.tau), tf.float32))*self.U
        
      else:  
        re = W

      # add the kl for the weights to the collection
      # kl_weights = tf.reduce_sum(input_tensor=normal_kl(tf.zeros_like(self.mW), tf.ones_like(sW_softplus),self.mW, sW_softplus))
      kl_weights = - 0.5 * tf.reduce_mean(2*sW_softplus - tf.square(self.mW) - sW_softplus**2 + 1, name = 'kl_weights')

      
      tf.compat.v1.add_to_collection('kl_loss', kl_weights) #something very big
      # self.add_loss(tf.math.reduce_mean(kl_weights)/60000)
      layer_loss = layer_loss + tf.math.reduce_mean(kl_weights)/60000
      tf.compat.v2.summary.scalar(name='kl_weights', data=kl_weights)

      # dense calculation
      lam = tf.matmul(inputs, re) + self.biases


      if self.activation=='lwta':
        assert self.U>1, 'The number of competing units should be larger than 1'
        
        # reshape weight for LWTA
        lam_re = tf.reshape(lam, [-1,self.K,self.U])
        
        # calculate probability of activation and some stability operations
        prbs = tf.nn.softmax(lam_re) + 1e-4
        prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)
        
        # relaxed categorical sample
        xi = concrete_sample(prbs, self.temp_cat)
        
        #apply activation
        out  = lam_re * xi
        out = tf.reshape(out, tf.shape(input=lam))
        
        # kl for the relaxed categorical variables
        kl_xi = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=concrete_kl(tf.ones([1,self.K,self.U])/self.U, prbs, xi), axis=[1]))
        # print(kl_xi) #negative #something very small
        tf.compat.v1.add_to_collection('kl_loss', kl_xi)
        # self.add_loss(tf.math.reduce_mean(kl_xi)/60000)
        layer_loss = layer_loss + tf.math.reduce_mean(kl_xi)/60000
        tf.compat.v2.summary.scalar(name='kl_xi', data=kl_xi)

      elif self.activation == 'relu':
        out = tf.nn.relu(lam)
      elif self.activation=='maxout':
        lam_re = tf.reshape(lam, [-1,self.K,self.U])
        out = tf.reduce_max(input_tensor=lam_re, axis=-1)
      else:
        out = lam

    #test branch in the layer. It is activated automatically in the model. TF does the work ;)
    else:
        #this is very different from the original
      # we use re for accuracy and z for compression (if sbp is active)
      re = 1.
      z = 1.
      layer_loss = 0.
      #sbp
      if self.sbp==True:

        # posterior probabilities z
        t_pi_sigmoid = tf.nn.sigmoid(self.t_pi)

        mask = tf.cast(tf.greater(t_pi_sigmoid, self.tau), tf.float32)
        z = tfd.Bernoulli(probs = mask*t_pi_sigmoid, name="q_z_test", dtype=tf.float32).sample()
        z = tf.tile(z, [1,self.U])
         
        re = tf.tile(mask * t_pi_sigmoid, [1, self.U])
        
      lam = tf.matmul(inputs, re*self.mW) + self.biases

      if self.activation == 'lwta':

        # reshape and calulcate winners
        lam_re = tf.reshape(lam, [-1, self.K, self.U])
        prbs = tf.nn.softmax(lam_re) + 1e-4
        prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)

        # apply activation
        out = lam_re*concrete_sample(prbs, 0.01)
        out = tf.reshape(out, tf.shape(input=lam))

      elif self.activation == 'relu':
                
          out = tf.nn.relu(lam)
          
      elif self.activation=='maxout':
          
          lam_re =  tf.reshape(lam, [-1, self.K, self.U])
          out = tf.reduce_max(input_tensor=lam_re, axis=-1)
          
      else:
          out = lam
    self.add_loss(layer_loss)
    # return out, self.mW, z*self.mW, z*self.sW**2, z
    return out