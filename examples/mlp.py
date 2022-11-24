"""
File: mlp.py
------------------
Harness easyjax to train MLP. 
"""


import jax.numpy as jnp
from jax import random
import copy
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Dense, Relu, Flatten, Sigmoid)
from jax import grad, jit, vmap, tree_util, value_and_grad
import easyjax as ej
from easyjax import ml, metrics
import rsbox
from rsbox import misc
from rsbox import ml as rml
import pdb


dset = misc.load_dataset("https://stanford.edu/~rsikand/assets/datasets/mini_binary_mnist.pkl")


# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Flatten,
    Dense(1*28*28), Relu,
    Dense(128), Relu,
    Dense(64), Relu,
    Dense(1), Sigmoid
)

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)
in_shape = (-1, 1, 28, 28)
out_shape, net_params = net_init(rng, in_shape)



def evaluate():
    eval_acc = rml.MeanMetric()
    for batch in dset:
        x, y = batch
        predictions = net_apply(net_params, x)
        new_acc = jnp.mean((predictions > 0.5).astype(type(y)) == y)
        eval_acc.update(new_acc)

    acc = eval_acc.get()
    print("Accuracy: ", acc)
    return acc



def train(num_epochs=10):
    learning_rate = 0.01
    loss_metric = rml.MeanMetric()
    params = copy.deepcopy(net_params)

    # grad of loss
    loss_grad_fn = value_and_grad(ml.stateless_loss)


    for i in range(num_epochs):
        loss_metric.reset()
        for batch in dset:
            x, y = batch
            loss_val, grads = loss_grad_fn(params, net_apply, ml.bce_loss, x, y)
            loss_metric.update(loss_val)
            params = ml.update_step(params, grads, learning_rate)
        
        epoch_loss = loss_metric.get()
        print(f"Loss (epoch {i}): ", epoch_loss)
    
    return params



pre_training_accuracy = evaluate()
net_params = train(num_epochs=15)
post_training_accuracy = evaluate()
