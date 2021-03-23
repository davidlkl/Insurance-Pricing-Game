# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 23:15:54 2021

@author: ling
"""

from stacking_model import SubmissionModel, MyTransformedTargetRegressor
from joblib import dump, load
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from nn_model import  tweedieloss, zero_inflated_lognormal_pred
from nn_model import zero_inflated_lognormal_loss, mse_from_logits, tweedie_loss_from_logits


def load_stack_model(path):
    stack_model = load(path)
    # Load NN model in stacking_model
    for base_cv_model_list in stack_model.base_cv_models:
        base_cv_model = base_cv_model_list[0]
        for cv_idx, model in enumerate(base_cv_model):
            custom_objects = {'tweedieloss': tweedieloss}
            if isinstance(model, MyTransformedTargetRegressor):
                model = model.regressor
                custom_objects = {
                    'zero_inflated_lognormal_loss': zero_inflated_lognormal_loss,
                    'tweedie_loss_from_logits' : tweedie_loss_from_logits
                }
            if isinstance(model, KerasRegressor):
                model.model = load_model(model.model_path, custom_objects=custom_objects)    
    return stack_model