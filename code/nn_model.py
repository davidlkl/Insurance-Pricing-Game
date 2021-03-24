# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:54:51 2021

@author: ling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate
from tensorflow.keras.layers import Embedding, Flatten, Activation, AlphaDropout, PReLU, add

import numpy as np

from tensorflow.keras import regularizers

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from functools import wraps


# Return:
# Categorical input: List of (n_sample, 1) arrays
# Numeric input: An (n_sample, n_numeric_features) array
def split_categorical_numeric_inputs(X, n_categorical_feat):
    return [X[:, i:i+1] for i in range(n_categorical_feat)], X[:, n_categorical_feat:]


def identity(Y):
    return Y

def transform_logits(logits):
    return zero_inflated_lognormal_pred(logits).numpy().flatten()

def transform_tweedie_logits(logits):
    return zero_inflated_tweedie_pred(logits).numpy().flatten()


def zero_inflated_lognormal_params(logits: tf.Tensor) -> tf.Tensor:
  """Calculates parameters of zero inflated lognormal logits.
  Arguments:
    logits: [batch_size, 3] tensor of logits.
  Returns:
    parameters: [batch_size, 3] tensor of frequency, log-normal location and scale.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = K.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = K.softplus(logits[..., 2:])
  
  parameters = K.concatenate([positive_probs, loc, scale], axis=-1)
  return parameters

def zero_inflated_lognormal_pred(logits: tf.Tensor) -> tf.Tensor:
  """Calculates predicted mean of zero inflated lognormal logits.
  Arguments:
    logits: [batch_size, 3] tensor of logits.
  Returns:
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = K.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = K.softplus(logits[..., 2:])
  preds = (
      positive_probs *
      K.exp(loc + 0.5 * K.square(scale))
  )
  return preds

def zero_inflated_lognormal_std(logits: tf.Tensor) -> tf.Tensor:
  """Calculates predicted std of zero inflated lognormal logits.
  Arguments:
    logits: [batch_size, 3] tensor of logits.
  Returns:
    preds: [batch_size, 1] tensor of predicted std.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = K.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = K.softplus(logits[..., 2:])
  
  log_normal_variance = (K.exp(K.square(scale)) - 1) * K.exp(2*loc + K.square(scale))
  log_normal_mean = K.exp(loc + 0.5 * K.square(scale))
  std = K.sqrt(
      positive_probs * (log_normal_variance + K.square(log_normal_mean)) 
      - K.square(positive_probs * log_normal_mean)
  )
  
  return std

def mse_from_logits(labels: tf.Tensor,
                     logits: tf.Tensor) -> tf.Tensor:
  """
  Calculates the rmse of the predicted mean of zero inflated lognormal logits.
  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: [batch_size, 3] tensor of logits.
  Returns:
    MSE loss value.
  """
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  
  positive_probs = K.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = K.softplus(logits[..., 2:])
  
  preds = (
      positive_probs *
      K.exp(loc + 0.5 * K.square(scale))
  )
  
  mse = K.mean(
      K.square(
          preds - labels  
      ),axis=-1
  )
  return mse   

def tweedie_loss_from_logits(labels: tf.Tensor,
                     logits: tf.Tensor) -> tf.Tensor:

  p = 1.25
  
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  
  positive_probs = K.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = K.softplus(logits[..., 2:])
  
  preds = (
      positive_probs *
      K.exp(loc + 0.5 * K.square(scale))
  )
      
  tweedie_loss = K.mean(
      2 * (K.pow(labels, 2-p)/((1-p) * (2-p)) -
          labels * K.pow(preds, 1-p)/(1-p) +
          K.pow(preds, 2-p)/(2-p)
      ),axis=-1
  )
  return tweedie_loss


def zero_inflated_lognormal_loss(labels: tf.Tensor,
                                 logits: tf.Tensor) -> tf.Tensor:
  """Computes the zero inflated lognormal loss.
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```
  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].
  Returns:
    Zero inflated lognormal loss value.
  """
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [3]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.losses.binary_crossentropy(
      y_true=positive, y_pred=positive_logits, from_logits=True)

  loc = logits[..., 1:2]
  scale = tf.math.maximum(
      K.softplus(logits[..., 2:]),
      tf.math.sqrt(K.epsilon()))
  safe_labels = positive * labels + (
      1 - positive) * K.ones_like(labels)
  regression_loss = -K.mean(
      positive * (
          tfd.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
      ),
      axis=-1)

  return 0.5 * classification_loss + 0.5 * regression_loss


def mse_from_tweedie_logits(labels: tf.Tensor,
                     logits: tf.Tensor) -> tf.Tensor:

  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  
  positive_probs = K.sigmoid(logits[..., :1])
  mean = K.exp(logits[..., 1:2])

  preds = (
      positive_probs * mean
  )
  
  mse = K.mean(
      K.square(
          preds - labels  
      ),axis=-1
  )
  return mse

def zero_inflated_tweedie_pred(logits: tf.Tensor) -> tf.Tensor:

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = K.sigmoid(logits[..., :1])
  mean = K.exp(logits[..., 1:2])

  preds = (
      positive_probs * mean
  )
  return preds

def zero_inflated_tweedie_loss(labels: tf.Tensor,
                                 logits: tf.Tensor) -> tf.Tensor:
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [2]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.losses.binary_crossentropy(
      y_true=positive, y_pred=positive_logits, from_logits=True)

  mean = K.exp(logits[..., 1:2])
  
  safe_labels = positive * labels + (1 - positive) * K.zeros_like(labels)
  p = 1.1
  regression_loss = 2 * K.mean(
      positive * (
          K.pow(safe_labels, 2-p)/((1-p) * (2-p)) - 
          safe_labels * K.pow(mean, 1-p)/(1-p) +
          K.pow(mean, 2-p)/(2-p)
      ),
      axis=-1
  )

  return 0.5 * classification_loss + 0.5 * regression_loss

def tweedieloss(y_true, y_pred):
    p = 1.25
    dev = 2 * (K.pow(y_true, 2-p)/((1-p) * (2-p)) -
                   y_true * K.pow(y_pred, 1-p)/(1-p) +
                   K.pow(y_pred, 2-p)/(2-p))
    return K.mean(dev)

def nn_model_fn(numeric_input_dim, cardinalities=[], embedding_dimensions=[]):

    opt = optimizers.Adam (lr=0.0008)
    
    numeric_input = Input((numeric_input_dim, ))
    model_inputs = numeric_input
    input_layer = numeric_input
    
    # If categorical embedding is used
    if len(cardinalities) > 0:
        categorical_inputs = []
        categorical_flatten_layers = []
        for i in range(len(cardinalities)):
            embedding_input = Input((1,))
            categorical_inputs.append(embedding_input)
            embedding_layer = Embedding(input_dim=cardinalities[i],
                                   output_dim=embedding_dimensions[i],
                                   embeddings_initializer="glorot_uniform",
                                   name='embedding'+str(i))(embedding_input)
    
            embedding_flatten = Flatten()(embedding_layer)
            categorical_flatten_layers.append(embedding_flatten)
        model_inputs = categorical_inputs + [numeric_input]
        input_layer = concatenate(categorical_flatten_layers + [numeric_input], axis=-1)
    
    common_dense1 = Dense(64, kernel_initializer='lecun_normal', kernel_constraint=MaxNorm(3))(input_layer)
    common_dropout1 = Dropout(0.3)(common_dense1)
    common_batchnorm1 = BatchNormalization()(common_dropout1)
    common_act1 = Activation(activation='elu')(common_batchnorm1)
    
    common_dense2 = Dense(128, kernel_initializer='lecun_normal', kernel_constraint=MaxNorm(3))(common_act1)
    common_dropout2 = Dropout(0.3)(common_dense2)
    common_batchnorm2 = BatchNormalization()(common_dropout2)
    common_act2 = Activation(activation='linear')(common_batchnorm2)
    
    final_output = Dense(1, activation='exponential')(common_act2)
    
    model = Model(inputs=model_inputs, outputs=final_output)
    
    model.compile(loss=tweedieloss, optimizer=opt)
    
    return model


def nn_model_bayesian_fn(numeric_input_dim, cardinalities=[], embedding_dimensions=[]):

    opt = optimizers.Adam (lr=0.0008)
    
    numeric_input = Input((numeric_input_dim, ))
    model_inputs = numeric_input
    input_layer = numeric_input
    
    # If categorical embedding is used
    if len(cardinalities) > 0:
        categorical_inputs = []
        categorical_flatten_layers = []
        for i in range(len(cardinalities)):
            embedding_input = Input((1,))
            categorical_inputs.append(embedding_input)
            embedding_layer = Embedding(input_dim=cardinalities[i],
                                   output_dim=embedding_dimensions[i],
                                   embeddings_initializer="glorot_uniform",
                                   name='embedding'+str(i))(embedding_input)
    
            embedding_flatten = Flatten()(embedding_layer)
            categorical_flatten_layers.append(embedding_flatten)
        model_inputs = categorical_inputs + [numeric_input]
        input_layer = concatenate(categorical_flatten_layers + [numeric_input], axis=-1)
    
    common_dense1 = Dense(64, kernel_initializer='lecun_normal', kernel_constraint=MaxNorm(3))(input_layer)
    common_dropout1 = Dropout(0.3)(common_dense1)
    common_batchnorm1 = BatchNormalization()(common_dropout1)
    common_act1 = Activation(activation='linear')(common_batchnorm1)
    
    p_dense1 = Dense(64, kernel_initializer='lecun_normal', activation='elu', kernel_constraint=MaxNorm(3))(common_act1)
    p_dropout1 = Dropout(0.4)(p_dense1)
    p_batchnorm1 = BatchNormalization()(p_dropout1)
    p_output = Dense(1)(p_dropout1)
    
    loc_scale_dense1 = Dense(64, kernel_initializer='lecun_normal', activation='elu', kernel_constraint=MaxNorm(3))(common_act1)
    loc_scale_dropout1 = Dropout(0.4)(loc_scale_dense1)
    loc_scale_batchnorm1 = BatchNormalization()(loc_scale_dropout1)
    loc_scale_output = Dense(2)(loc_scale_dropout1)
    
    final_output = concatenate([p_output, loc_scale_output], axis=-1)
    model = Model(inputs=model_inputs, outputs=final_output)
    
    model.compile(loss=zero_inflated_lognormal_loss, optimizer=opt, metrics=tweedie_loss_from_logits)
    
    return model
