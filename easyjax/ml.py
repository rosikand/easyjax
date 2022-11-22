"""
File: ml.py 
-------------- 
Contains useful functions and utilities for machine learning in Jax. 
""" 


import jax
from jax import grad, jit, vmap, tree_util
import jax.numpy as jnp


# --------------------------- (Losses) ------------------------------

bce_loss = lambda logits, y: -(y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(1-logits)) 
bce_batched = vmap(bce_loss)


# --------------------------- (Training) ------------------------------
# gradient_step = lambda p, g, lr: p - lr * g  # doesn't work because lr needs to be a pytree then 

gradient_step = lambda p, g: p - lr * g  # lr needs to be defined in user scope 


@jax.jit
def update_step(params, grads, lr):
    """
    Updates params via one step of gradient descent. 
    Arguments: 
    -----------
    - params: pytree of parameters for the net 
    - grads: gradient of loss wrt to params 
    - lr: learning rate (step size) 
    Returns: 
    -----------
    loss_value: updated params  
    """
    gradient_step = lambda p, g: p - lr * g   
    return tree_util.tree_map(gradient_step, params, grads)


def stateless_loss(params, net_apply, loss_function, x, y, return_logits=False):
    """
    Computes loss in single function using arguments in stateless (no OOP). 
    Useful to wrap this function in 'grad' to get the gradient with respect
    to params. 
    Arguments: 
    -----------
    - params: pytree of parameters for the net 
    - net_apply: model apply fn 
    - loss_function: function for computing the loss 
    - x: input
    - y: label 
    - return_logits: if True, returns logits as well (must specify hax_aux=True when taking grad then)
    Returns: 
    -----------
    loss_value: scalar loss value 
    """

    model_preds = net_apply(params, x)
    loss_value = loss_function(model_preds, y)
    loss_value = jnp.squeeze(loss_value)  # remove axes 
    if return_logits == False:
        return loss_value
    else:
        loss_value, model_preds
