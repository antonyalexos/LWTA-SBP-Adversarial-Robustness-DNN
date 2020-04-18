from distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.python.util import tf_inspect

class SB_Conv2d(tf.keras.layers.Layer):

  def __init__(self, ksize, padding='SAME', strides=[1,1,1,1], bias = True, sbp=False, temp_bern=0.67, temp_cat=0.67, activation='lwta',dynamic=True,**kwargs):
    super(SB_Conv2d, self).__init__(**kwargs)
    self.tau = 1e-2
    self.ksize = ksize
    self.padding = padding
    self.strides = strides
    self.bias = bias
    self.sbp = sbp
    self.temp_bern = temp_bern
    self.temp_cat = temp_cat
    self.activation = activation
 
    # K  = ksize[-2]
    # U  = ksize[-1]

  def build(self, input_shape):
      
    self.mW = self.add_weight(shape=(self.ksize[0], self.ksize[1], input_shape[3], self.ksize[-2]*self.ksize[-1]),
                              initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                              trainable=True,
                              dtype= tf.float32,
                              name='mW1')
    

    
    self.sW = self.add_weight(shape=(self.ksize[0], self.ksize[1],  input_shape[3], self.ksize[-2]*self.ksize[-1]),
                              trainable=True,
                              initializer=tf.constant_initializer(-5.),
                              constraint = lambda x: tf.clip_by_value(x, -7., x ),
                              dtype= tf.float32,
                              name='sW1')
    
    # variables and construction for the stick breaking process
    if self.sbp:
      
      # posterior concentrations for the Kumaraswamy distribution
      self.conc1 = self.add_weight(shape = ([self.ksize[-2]]),
                                   initializer = tf.constant_initializer(3.),
                                   constraint=lambda x: tf.clip_by_value(x, -6., x),
                                   dtype = tf.float32,
                                   trainable=True,
                                   name = 'sb_t_u_1')
            
      self.conc0 = self.add_weight(shape = ([self.ksize[-2]]),
                                  initializer = tf.constant_initializer(1.),
                                  constraint=lambda x: tf.clip_by_value(x, -6., x),
                                  dtype = tf.float32,
                                   trainable=True,
                                  name = 'sb_t_u_2')

      # posterior bernooulli (relaxed) probabilities
      self.t_pi = self.add_weight(shape = ([self.ksize[-2]]),
                                  initializer =  tf.compat.v1.initializers.random_uniform(-5., 1.),
                                  constraint = lambda x: tf.clip_by_value(x, -7., 600.),\
                                  dtype = tf.float32,
                                  trainable=True,
                                  name = 'sb_t_pi')
    
    self.biases=0.
    if self.bias:
        self.biases = self.add_weight(shape=(self.ksize[-2]*self.ksize[-1],),
                               initializer=tf.constant_initializer(0.0),
                               trainable=True,
                               name='bias')
            
    self.built = True 
    
  def call(self,inputs,training=None):
      
    sW_softplus = tf.nn.softplus(self.sW)
    
    if training:

      layer_loss = 0.
      z = 1.
      # reparametrizable normal sample
      eps = tf.stop_gradient(tf.random.normal(self.mW.get_shape()))
      W = self.mW + eps*sW_softplus
      
      re = tf.ones_like(W)
      
      # stick breaking construction
      if self.sbp==True:
          
        conc1_softplus = tf.nn.softplus(self.conc1)
        conc0_softplus= tf.nn.softplus(self.conc0)
        
        
        # stick breaking construction
        q_u = kumaraswamy_sample(conc1_softplus, conc0_softplus, sample_shape = [inputs.get_shape()[1],self.ksize[-2]])
        pi = tf.math.cumprod(q_u)

        # posterior bernooulli (relaxed) probabilities
        t_pi_sigmoid = tf.nn.sigmoid(self.t_pi)
        
        z_sample = bin_concrete_sample(t_pi_sigmoid, self.temp_bern)
        z = tf.tile(z_sample,[self.ksize[-1]])
        re = z*W
                      
        kl_sticks = tf.reduce_sum(kumaraswamy_kl(tf.ones_like(conc1_softplus), tf.ones_like(conc0_softplus),
                                                conc1_softplus, conc0_softplus, q_u))
        kl_z = tf.reduce_sum(bin_concrete_kl(pi, t_pi_sigmoid, self.temp_bern, z_sample))
      
        tf.compat.v1.add_to_collection('kl_loss', kl_sticks)
        tf.compat.v1.add_to_collection('kl_loss', kl_z)
          
        layer_loss = layer_loss + tf.math.reduce_mean(kl_sticks)/60000
        layer_loss = layer_loss + tf.math.reduce_mean(kl_z)/60000
          
        tf.compat.v2.summary.scalar('kl_sticks', kl_sticks)
        tf.compat.v2.summary.scalar('kl_z', kl_z)
          
        # if probability of activation is smaller than tau, it's inactive
        tf.compat.v2.summary.scalar('sparsity', tf.reduce_sum(tf.cast(tf.greater(t_pi_sigmoid/(1.+t_pi_sigmoid), self.tau), tf.float32))*self.ksize[-1])
        # spasrity = tf.reduce_sum(tf.cast(tf.greater(t_pi_sigmoid/(1.+t_pi_sigmoid), self.tau), tf.float32))*self.ksize[-1]
        
          
      # add the kl terms to the collection
      # kl_weights = tf.reduce_sum(normal_kl(tf.zeros_like(self.mW), tf.ones_like(sW_softplus), \
                                            # self.mW, sW_softplus, W))
                                            
      kl_weights = - 0.5 * tf.reduce_mean(2*sW_softplus - tf.square(self.mW) - sW_softplus**2 + 1, name = 'kl_weights')
      

      tf.compat.v1.add_to_collection('losses',  kl_weights)
      tf.compat.v2.summary.scalar('kl_weights', kl_weights)
                   
      layer_loss = layer_loss + tf.math.reduce_mean(kl_weights)/60000
           
      # convolution operation
      lam = tf.nn.conv2d(inputs, re, strides=(self.strides[0],self.strides[1]), padding = self.padding) + self.biases

      if self.activation=='lwta':
        assert self.ksize[-1]>1, 'The number of competing units should be larger than 1'

      # reshape weight to calculate probabilities
        lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], self.ksize[-2], self.ksize[-1]])
      
        prbs = tf.nn.softmax(lam_re) + 1e-5
        prbs /= tf.reduce_sum(input_tensor=prbs, axis=-1, keepdims=True)
                  
        # draw relaxed sample and apply activation
        xi = concrete_sample(prbs, self.temp_cat)
        
        #apply activation
        out = lam_re * xi
        out = tf.reshape(out, tf.shape(input=lam))
                  
        # add the relative kl terms
        kl_xi = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=concrete_kl(tf.ones_like(lam_re)/self.ksize[-1], prbs, xi), axis=[1]))
              
        tf.compat.v1.add_to_collection('kl_loss', kl_xi)
        tf.compat.v2.summary.scalar('kl_xi', kl_xi)
        
        layer_loss = layer_loss + tf.math.reduce_mean(kl_xi)/60000
        
      elif self.activation == 'relu':
        out = tf.nn.relu(lam)
      elif self.activation=='maxout':
        lam_re =  tf.reshape(lam, [-1,lam.get_shape()[1], lam.get_shape()[2],self.ksize[-2],self.ksize[-1]])
        out = tf.reduce_max(lam_re, -1, keepdims=False) 
      elif self.activation=='none':
        out = lam
      else:
        print('Activation:', self.activation, 'not implemented.')
        out = lam

    else:
        
        re = tf.ones_like(self.mW)
        z = 1.
        layer_loss = 0.
        
        # if sbp is active calculate mask and draw samples
        if self.sbp:
            
            # posterior probabilities z
            t_pi_sigmoid = tf.nn.sigmoid(self.t_pi)
            
            mask = tf.cast(tf.greater(t_pi_sigmoid, self.tau), tf.float32)
            z = tfd.Bernoulli(probs = mask*t_pi_sigmoid, name="q_z_test", dtype=tf.float32).sample()
            z = tf.tile(z, [self.ksize[-1]])
            re = tf.tile(mask*t_pi_sigmoid,[self.ksize[-1]])
        
        # convolution operation
        lam = tf.nn.conv2d(inputs, re * self.mW, strides=(self.strides[0],self.strides[1]) , padding = self.padding) + self.biases
        
        if self.activation == 'lwta':
            # calculate probabilities of activation
            lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], self.ksize[-2],self.ksize[-1]])
            prbs = tf.nn.softmax(lam_re) + 1e-5
            prbs /= tf.reduce_sum(input_tensor=prbs,axis=-1, keepdims=True)
            
            # draw sample for activated units
            out = lam_re * concrete_sample(prbs, 0.01)
            out = tf.reshape(out, tf.shape(input=lam))
            
        elif self.activation == 'relu':
            # apply relu
            out = tf.nn.relu(lam)
        
        elif self.activation=='maxout':
            # apply maxout operation
            lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], self.ksize[-2], self.ksize[-1]])
            out = tf.reduce_max(input_tensor=lam_re, axis=-1)
        elif self.activation=='none':
            out = lam
        else:
            print('Activation:', activation,' not implemented.')
            out = lam
        
    self.add_loss(layer_loss)
      # return self.out, self.mW, self.z*self.mW, self.z*self.sW**2, self.z
    return out

  def get_config(self):
    """Returns the config of the layer.
    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).
    Returns:
        Python dictionary.
    """
    all_args = tf_inspect.getfullargspec(self.__init__).args
    config = {'name': self.name, 'trainable': self.trainable}
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    if hasattr(self, 'dtype'):
      config['dtype'] = self.dtype
    if hasattr(self, 'dynamic'):
      # Only include `dynamic` in the `config` if it is `True`
      if self.dynamic:
        config['dynamic'] = self.dynamic
      elif 'dynamic' in all_args:
        all_args.remove('dynamic')
    expected_args = config.keys()
    # Finds all arguments in the `__init__` that are not in the config:
    extra_args = [arg for arg in all_args if arg not in expected_args]
    # Check that either the only argument in the `__init__` is  `self`,
    # or that `get_config` has been overridden:
    if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
      raise NotImplementedError('Layers with arguments in `__init__` must '
                                'override `get_config`.')
    # TODO(reedwm): Handle serializing self._dtype_policy.
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.
    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).
    Arguments:
        config: A Python dictionary, typically the
            output of get_config.
    Returns:
        A layer instance.
    """
    return cls(**config)
  
  
  
  
  


