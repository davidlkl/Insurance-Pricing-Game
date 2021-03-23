# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:36:17 2020

@author: ling
"""
import numpy as np
import pandas as pd

from preprocessing import transform_columns, generate_agg_data, create_raw_features
from sklearn.compose import TransformedTargetRegressor
from nn_model import split_categorical_numeric_inputs, zero_inflated_lognormal_std

class MyTransformedTargetRegressor(TransformedTargetRegressor):
    def fit(self, X, Y):
        return self.regressor.fit(X, self.func(Y))
    
    def predict(self, X):
        return self.inverse_func(self.regressor.predict(X))
    
class StackingLargeClaimClassifier:
    def __init__(self, base_cv_models):
        self.base_cv_models = base_cv_models
    
    def predict(self, X, X_tree, X_nn):
        X_with_prediction = np.zeros((X.shape[0], len(self.base_cv_models)))
        
        for model_idx, base_cv_model_list in enumerate(self.base_cv_models):
            base_cv_model, model_type = base_cv_model_list[0], base_cv_model_list[1]
            prediction = np.zeros((X_with_prediction.shape[0], len(base_cv_model)))
            for cv_idx, model in enumerate(base_cv_model):
                if model_type == 'glm':
                    prediction[:, cv_idx] = model.predict_proba(X)[:, 1]
                elif model_type == 'tree':
                    prediction[:, cv_idx] = model.predict_proba(X_tree)[:, 1]
                elif model_type == 'nn':
                    prediction[:, cv_idx] = model.predict_proba(X_nn)[:, 1]
            prediction = (np.mean(prediction, axis=1) >= 0.5) * 1
            X_with_prediction[:, model_idx] = prediction
        
        final_prediction = np.max(X_with_prediction, axis=1)
        
        return final_prediction

class StackingModel:
    def __init__(self, base_cv_models, final_model, column_trans, tree_column_trans, nn_column_trans, large_claim_clf=None, target_encoder=None):
        self.base_cv_models = base_cv_models
        self.final_model = final_model
        self.column_trans = column_trans
        self.tree_column_trans = tree_column_trans
        self.nn_column_trans = nn_column_trans
        self.large_claim_clf = large_claim_clf
        self.target_encoder = target_encoder
        
    def predict(self, X, X_tree, X_nn):
        
        X_with_prediction = np.zeros((X.shape[0], len(self.base_cv_models)))
        
        for model_idx, base_cv_model_list in enumerate(self.base_cv_models):
            base_cv_model, model_type = base_cv_model_list[0], base_cv_model_list[1]
            prediction = np.zeros((X_with_prediction.shape[0], len(base_cv_model)))
            for cv_idx, model in enumerate(base_cv_model):
                if model_type == 'glm':
                    prediction[:, cv_idx] = model.predict(X)
                elif model_type == 'tree':
                    prediction[:, cv_idx] = model.predict(X_tree)
                elif model_type == 'nn':
                    prediction[:, cv_idx] = model.predict(X_nn)
            prediction = np.mean(prediction, axis=1)
            X_with_prediction[:, model_idx] = prediction
        
        final_prediction = np.clip(self.final_model.predict(X_with_prediction), 0, None)
        
        return final_prediction
    
    def predict_risk(self, X_nn):
        for model_idx, base_cv_model_list in enumerate(self.base_cv_models):
            base_cv_model, model_name = base_cv_model_list[0], base_cv_model_list[2]
            if model_name == 'bay_nn':
                prediction = np.zeros((X_nn[1].shape[0], len(base_cv_model)))
                for cv_idx, model in enumerate(base_cv_model):
                    prediction[:, cv_idx] = zero_inflated_lognormal_std(model.regressor.predict(X_nn)).numpy().flatten()
                prediction = np.mean(prediction, axis=1)
                return prediction
        
        return None
    
    def predict_large_claim(self, X, X_tree, X_nn):
        if self.large_claim_clf is not None:
            return self.large_claim_clf.predict(X, X_tree, X_nn)
        
        return None
        
        
class SubmissionModel:
    def __init__(self, generic_model, hist_model, df_data_path):
        self.generic_model = generic_model
        self.hist_model = hist_model
        self.df_data = pd.read_csv(df_data_path)
        
    def separate_hist_and_generic(self, X_raw):
        df_data_new = X_raw.copy()
        df_data_old = self.df_data.copy()
        df_data_old = df_data_old[df_data_old['id_policy'].isin(df_data_new['id_policy'])].reset_index(drop=True)
        
        df_data = (
            pd.concat([df_data_old, df_data_new])
            .drop_duplicates(('id_policy', 'year'))
            .sort_values(['year', 'id_policy'])
            .reset_index(drop=True)
        )
        
        df_agg_data = generate_agg_data(df_data[['id_policy', 'year', 'pol_no_claims_discount', 'claim_amount']])
        
        if df_agg_data.empty:
            print("empty agg data")
            df_data_hist = pd.DataFrame()
            df_data_generic = X_raw.reset_index(drop=True)
        else:
            df_combined = X_raw.merge(df_agg_data, on=['id_policy', 'year'], how='left')
            df_data_hist = df_combined[~df_combined['agg_claim_amount'].isna()].reset_index(drop=True)
            df_data_generic = df_combined[df_combined['agg_claim_amount'].isna()][list(X_raw.columns)].reset_index(drop=True)
            print("hist model: ", len(df_data_hist))
            print("generic model: ", len(df_data_generic))
            
        df_data_hist = create_raw_features(df_data_hist)
        df_data_generic = create_raw_features(df_data_generic)
        return df_data_hist, df_data_generic
    
    def predict(self, X_raw):
        
        df_data_hist, df_data_generic = self.separate_hist_and_generic(X_raw)
        
        if self.hist_model.target_encoder is not None:
            df_data_hist = self.hist_model.target_encoder.transform(df_data_hist)
        if self.generic_model.target_encoder is not None:
            df_data_generic = self.generic_model.target_encoder.transform(df_data_generic)
        
        if not df_data_hist.empty:
            X_raw_hist, _, _, _, _, _ = transform_columns(
                df_data_hist, 
                column_trans=self.hist_model.column_trans,
                use_agg=True,
                model_type='glm',
            )
            X_raw_hist_tree, _, _, _, _, _ = transform_columns(
                df_data_hist, 
                column_trans=self.hist_model.tree_column_trans,
                use_agg=True,
                model_type='tree',
            )
            X_raw_hist_nn, _, _, _, cardinalities, _ = transform_columns(
                df_data_hist, 
                column_trans=self.hist_model.nn_column_trans,
                use_agg=True,
                model_type='nn',
            )
            X_raw_hist_nn_cat, X_raw_hist_nn_numeric = split_categorical_numeric_inputs(X_raw_hist_nn, len(cardinalities))
            df_data_hist['prediction'] = self.hist_model.predict(X_raw_hist,
                                                                 X_raw_hist_tree,
                                                                 [X_raw_hist_nn_cat, X_raw_hist_nn_numeric])
            prediction_risk = self.hist_model.predict_risk([X_raw_hist_nn_cat, X_raw_hist_nn_numeric])
            prediction_is_large_claim = self.hist_model.predict_large_claim(X_raw_hist,
                                                                 X_raw_hist_tree,
                                                                 [X_raw_hist_nn_cat, X_raw_hist_nn_numeric])
            df_data_hist['prediction_risk'] = prediction_risk
            df_data_hist['prediction_is_large_claim'] = prediction_is_large_claim
                
        if not df_data_generic.empty:
            X_raw_generic, _, _, _, _, _ = transform_columns(
                df_data_generic,
                column_trans=self.generic_model.column_trans,
                use_agg=False,
                model_type='glm',
            )
            X_raw_generic_tree, _, _, _, _, _ = transform_columns(
                df_data_generic,
                column_trans=self.generic_model.tree_column_trans,
                use_agg=False,
                model_type='tree',
            )
            X_raw_generic_nn, _, _, _, cardinalities, _ = transform_columns(
                df_data_generic,
                column_trans=self.generic_model.nn_column_trans,
                use_agg=False,
                model_type='nn',
            )
            X_raw_generic_nn_cat, X_raw_generic_nn_numeric = split_categorical_numeric_inputs(X_raw_generic_nn, len(cardinalities))
            df_data_generic['prediction'] = self.generic_model.predict(X_raw_generic,
                                                                       X_raw_generic_tree,
                                                                       [X_raw_generic_nn_cat, X_raw_generic_nn_numeric])
            prediction_risk = self.generic_model.predict_risk([X_raw_generic_nn_cat, X_raw_generic_nn_numeric])
            prediction_is_large_claim = self.generic_model.predict_large_claim(X_raw_generic,
                                                                       X_raw_generic_tree,
                                                                       [X_raw_generic_nn_cat, X_raw_generic_nn_numeric])
            df_data_generic['prediction_risk'] = prediction_risk
            df_data_generic['prediction_is_large_claim'] = prediction_is_large_claim
                
        df_data_with_prediction = pd.concat([df_data_hist, df_data_generic])[['id_policy', 'year', 'prediction', 'prediction_risk', 'prediction_is_large_claim']]
        final_prediction = X_raw.merge(df_data_with_prediction, on=['id_policy', 'year'], how='left')[['prediction','prediction_risk','prediction_is_large_claim']].values
        
        return_prediction_claim = final_prediction[:, 0]
        return_prediction_risk = None
        return_prediction_is_large_claim = None
        if prediction_risk is not None:
            return_prediction_risk = final_prediction[:, 1]
        if prediction_is_large_claim is not None:
            return_prediction_is_large_claim = final_prediction[:, 2]
        
        return return_prediction_claim, return_prediction_risk, return_prediction_is_large_claim