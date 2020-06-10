import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
Normal = tfd.Normal
Beta = tfd.Beta


# =============================================================================
# Some helper functions
# =============================================================================
    
def bin_concrete_sample(a, temp, eps=1e-8):
    """" 
    Sample from the binary concrete distribution
    """
    U = tf.random.uniform(tf.shape(a), minval = 0., maxval=1.)
    L = tf.math.log(U+eps) - tf.math.log(1.-U+eps) 
    X = tf.nn.sigmoid((L + tf.math.log(a))/temp)
    
    return tf.clip_by_value(X, 1e-4, 1.-1e-4)

def concrete_sample(a, temp, eps = 1e-8):
    """
    Sample from the Concrete distribution
    """
    U = tf.random.uniform(tf.shape(a), minval = 0., maxval=1.)
    G = - tf.math.log(-tf.math.log(U+eps)+eps)
    t = (tf.math.log(a) + G)/temp 
    out = tf.nn.softmax(t,-1)
    out += eps
    out /= tf.reduce_sum(out, -1, keepdims=True)
    return out*tf.stop_gradient(tf.cast(a>0., tf.float32))
    
def bin_concrete_kl(pr_a, post_a, post_temp, post_sample):
    """
    Calculate the binary concrete kl using the sample
    """
    p_log_prob = bin_concrete_log_mass(pr_a, post_temp, post_sample)
    q_log_prob = bin_concrete_log_mass(post_a,post_temp, post_sample)
   
    return -(p_log_prob - q_log_prob)
   

def concrete_kl(pr_a, post_a, post_sample):
    """
    Calculate the KL between two relaxed discrete distributions, using MC samples.
    This approach follows " The concrete distribution: A continuous relaxation of 
    discrete random variables" [Maddison et al.] and the rationale for this approximation
    can be found in eqs (20)-(22)
    
    Parameters:
        pr: tensorflow distribution
            The prior discrete distribution.
        post: tensorflow distribution
            The posterior discrete distribution
            
    Returns:
        kl: float
            The KL divergence between the prior and the posterior discrete relaxations
    """

    p_log_prob = tf.math.log(pr_a)
    q_log_prob = tf.math.log(post_a+1e-4)
   
    return -(p_log_prob - q_log_prob)


def kumaraswamy_sample(conc1, conc0, sample_shape):
    x = tf.random.uniform(sample_shape, minval=0.01, maxval=0.99)
        
    q_u = (1-(1-x)**(1./conc0))**(1./conc1)
    
    return q_u

def kumaraswamy_log_pdf(a, b, x):
    return tf.math.log(a) +tf.math.log(b) + (a-1.)*tf.math.log(x)+ (b-1.)*tf.math.log(1.-x**a)

def kumaraswamy_kl(prior_alpha, prior_beta,a,b, x):
    """
    Implementation of the KL distribution between a Beta and a Kumaraswamy distribution.
    Code refactored from the paper "Stick breaking DGMs". Therein they used 10 terms to 
    approximate the infinite taylor series.
    
    Parameters:
        prior_alpha: float/1d, 2d
            The parameter \alpha  of a prior distribution Beta(\alpha,\beta).
        prior_beta: float/1d, 2d
            The parameter \beta of a prior distribution Beta(\alpha, \beta).
        a: float/1d,2d
            The parameter a of a posterior distribution Kumaraswamy(a,b).
        b: float/1d, 2d
            The parameter b of a posterior distribution Kumaraswamy(a,b).
            
    Returns:
        kl: float
            The KL divergence between Beta and Kumaraswamy with given parameters.
    
    """
    
    q_log_prob = kumaraswamy_log_pdf(a, b, x)
    p_log_prob  = Beta(prior_alpha, prior_beta).log_prob(x)

    return -(p_log_prob-q_log_prob)

def normal_kl(m1,s1,m2,s2, sample):
    p_log_prob = Normal(m1, s1).log_prob(sample)
    q_log_prob = Normal(m2, s2).log_prob(sample)
   
    return  -(p_log_prob - q_log_prob)

#####################
## EXTRA
#########

def sample_lognormal(mu, sigma):
    U = tf.random_normal(tf.shape(mu))
    normal_sample = mu + U*sigma
    log_normal_sample = tf.exp(normal_sample)
    
    return tf.clip_by_value(log_normal_sample, 1e-3, log_normal_sample)

def lognormal_kl(mu, sigma):
    return 0.5*(mu**2 + sigma**2 -1. ) - 2*tf.math.log(sigma)

def exponential_sample(rate, eps = 1e-8):
    U = tf.random_uniform(tf.shape(rate), minval = np.finfo(np.float32).tiny, maxval=1.)
    
    return -tf.math.log(U+eps)/(rate + eps)

def exponential_kl(rate0, rate):
    return tf.math.log(rate) - tf.math.log(rate0) + rate0/rate - 1.


def sas_kl(alpha, gamma, mu, sigma ):
    # maybe it's not alpha and it's alpha/alpha+1 and the same for gamma
    safe_one_minus_alpha = tf.clip_by_value(1.-alpha, 1e-3, 1.-1e-3)
    safe_alpha = tf.clip_by_value(alpha, 1e-2, 1.-1e-3)
    
    return 0.5*gamma*(-1.- 2*tf.math.log(sigma) + tf.square(mu) + tf.square(sigma))\
            + (1.-gamma)* (tf.math.log(1.-gamma) - tf.math.log(safe_one_minus_alpha))\
              + gamma*(tf.math.log(gamma)-tf.math.log(safe_alpha))
            
def sas_kl_2(mu, sigma, post_sample):
    kl_w = 0.5*post_sample *( -1.- 2*tf.math.log(sigma) + tf.square(mu) + tf.square(sigma))
    #kl_z = bin_concrete_kl(alpha, 0.5, gamma, 0.67, post_sample )
    
    return kl_w 


def concrete_mass(a, temp, x):
    # it's the log prob of the exp relaxed, so we exp it to take the log prob
    # of the relaxed
    n= tf.cast(tf.shape(a)[-1], tf.float32)
    log_norm = (tf.lgamma(n)
                      + (n - 1.)
                      * tf.math.log(temp))
    
    log_un = tf.nn.log_softmax(tf.math.log(a+1e-4) -x*temp)
    log_un = tf.reduce_sum(log_un,-1, keep_dims=True)
    
    pr = tf.clip_by_value(log_norm + log_un, -10., -1e-2)
         
    return tf.exp(pr)


def bin_concrete_log_mass(a, temp, x):
    log_pr = tf.math.log(temp) + tf.math.log(a + 1e-4 ) + (-temp-1) * tf.math.log(x) + (-temp-1)*tf.math.log(1-x)
    log_pr -= 2 * (tf.math.log(a + 1e-4) - temp* tf.math.log(x) - temp*tf.math.log(1-x))
    
    return log_pr

def beta_function(a,b):
    """
    Calculation of the Beta function using the lgamma (log gamma) implementation of tf.
    
    Parameters:
        a: 1d or 2d tensor
            The first parameter of the beta function
        b: 1d or 2d tensor
            The second parameter of the beta function
            
    Returns:
        out: same as input size
            The calculated beta function for given a and b
    """
    
    return tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a+b))

def _log_prob(loc, scale, x):
    return _log_unnormalized_prob(loc, scale, x) - _log_normalization(scale)

def _log_unnormalized_prob(loc,scale, x):
    return -0.5 * tf.square(_z(loc, scale, x))

#missing the log(2pi) term since we dont really care
def _log_normalization(scale):
    return  tf.math.log(scale)

def _z(loc, scale, x):
    """Standardize input `x` to a unit normal."""
    return (x - loc) / scale

