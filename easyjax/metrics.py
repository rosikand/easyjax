"""
File: metrics.py 
-------------- 
Jax-based metrics. 
""" 

import jax.numpy as jnp


class MeanMetric:
  """
  Base scalar metric designed to use on a per-epoch basis 
  and updated on per-batch basis. For getting average across
  the epoch. 
  """

  def __init__(self):
    self.vals = []

  def update(self, new_val):
    # if isinstance(new_val, jnp.array): # squeeze only when true? 
    #     pass 

    self.vals.append(new_val)

  def reset(self):
    self.vals = []

  def get(self):
    mean_value = sum(self.vals)/len(self.vals)
    return 


class Accuracy(MeanMetric):
    """
    Subclass of MeanMetric which defines 
    a standard accuracy update function. 
    """

    def update(self, logits, labels):
        new_val = jnp.mean(jnp.argmax(logits, -1) == labels)
        self.vals.append(new_val)



def compute_accuracy(logits, labels):
  """stand alone accuracy metric computation"""
  return jnp.mean(jnp.argmax(logits, -1) == labels)



def compute_binary_accuracy(logits, labels):
  """
  Computes stand alone accuracy for binary classification
  where the net outputs one element per prediction. 
  Source: https://github.com/tristandeleu/jax-meta-learning/blob/master/jax_meta/utils/metrics.py
  """
  return jnp.mean((logits > 0).astype(type(labels)) == labels)
  
  