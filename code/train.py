"""This script is used to train your model. You can modify it if you want."""

import numpy as np
import pandas as pd
import os
import sys

from joblib import dump, load

from xgboost import XGBRegressor
from deepforest import CascadeForestRegressor
from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import TweedieRegressor, LinearRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, Ridge, BayesianRidge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_tweedie_deviance, auc

import tensorflow
tensorflow.autograph.set_verbosity(0)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model
from nn_model import *

from stacking_model import MyTransformedTargetRegressor, StackingLargeClaimClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.metrics import confusion_matrix

from stacking_model import StackingModel
from preprocessing import transform_columns, generate_agg_data, transform_cat_col_for_catboost
from preprocessing import create_raw_features, add_target_encoding

import category_encoders as ce

from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def lorenz_curve(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount

def gini(y_true, y_pred):
    ordered_samples, cum_claims = lorenz_curve(y_true, y_pred)
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    return gini

def tweedie_deviance_residual(prediction, target, p=1.5):
    dev = 2 * (np.power(target, 2-p)/((1-p) * (2-p)) -
                   target * np.power(prediction, 1-p)/(1-p) +
                   np.power(prediction, 2-p)/(2-p))
    
    return (dev**0.5) * np.sign(prediction - target)

def rmse(prediction, target):
    return np.mean((prediction-target)**2)**0.5
def xgb_monotone_constraint_str(monotonicity_list):
    return '(' + ','.join(map(str, monotonicity_list)) + ')'
def lgbm_monotone_constraint_str(monotonicity_list):
    return ','.join(map(str, monotonicity_list))

input_dataset = 'training_imputed.csv'  # The default value.
if len(sys.argv) >= 2:
	input_dataset = sys.argv[1]

for use_agg in [False, True]:
    df_data = pd.read_csv(input_dataset)
    
    # Calculate aggregated data
    if use_agg:
        min_year = 1
        df_agg_data = generate_agg_data(df_data[['id_policy', 'year', 'pol_no_claims_discount', 'claim_amount']])
        df_data = df_data.merge(df_agg_data, on=['id_policy', 'year'])
        df_data = df_data[df_data['year'] >= min_year]
    
    # Clipp claim amount
    claim_upper_cap = 5000
    claim_lower_cap = 0
    
    df_data['claim_amount_original'] = df_data['claim_amount']
    df_data['claim_amount'] = df_data['claim_amount'].clip(upper=claim_upper_cap)
    df_data['claim_amount'] = df_data['claim_amount'].apply(lambda c: c if c >= claim_lower_cap else 0)
    
    # Remove outlier
    outlier_cap = 200000
    policy_outlier = df_data[df_data['claim_amount_original']>=outlier_cap]['id_policy'].unique()
    df_data = df_data[~df_data['id_policy'].isin(policy_outlier)].reset_index(drop=True)
    df_data = df_data.drop(['claim_amount_original'], axis=1)
    
    # Create features
    df_data = create_raw_features(df_data)
    
    Y = df_data['claim_amount']
    Y_is_large = df_data['claim_amount'] >= 3000
    
    df_data_X = df_data.drop(['claim_amount'], axis=1)
    
    target_encoder = None
    # df_data_X, target_encoder = add_target_encoding(df_data_X, Y)
    
    X, _, _, column_trans, _, _ = transform_columns(
        df_data_X,
        column_trans=None,
        use_agg=use_agg,
        model_type='glm',
    )
    
    X_for_tree, cat_feat_list, monotonicity_list, column_trans_for_tree, _, _ = transform_columns(
        df_data_X,
        column_trans=None,
        use_agg=use_agg,
        model_type='tree',
    )
    
    X_for_nn, _, _, column_trans_for_nn, cardinalities, embedding_dimensions = transform_columns(
        df_data_X,
        column_trans=None,
        use_agg=use_agg,
        model_type='nn',
    )
    
    X_for_nn_cat, X_for_nn_numeric = split_categorical_numeric_inputs(X_for_nn, len(cardinalities))
    
    # Catboost transformer to avoid error
    # As it cannot accept float (e.g. 0.0, 1.0) as categories
    cat_feat_list_catboost = np.where(cat_feat_list)[0].tolist()
    ft_catboost = FunctionTransformer(transform_cat_col_for_catboost, kw_args={'cat_feat_list': cat_feat_list_catboost}, validate=False)
    
    checkpoint_filepath = 'tmp/checkpoint'
    
    glm_rmse = []
    lgbm_rmse = []
    df_rmse= []
    xgb_rmse = []
    ridge_rmse = []
    ctb_rmse = []
    nn_rmse = []
    bay_nn_rmse = []
    
    glm_td = []
    lgbm_td = []
    df_td= []
    xgb_td = []
    ridge_td = []
    ctb_td = []
    nn_td = []
    bay_nn_td = []
    
    glm_gini = []
    lgbm_gini = []
    df_gini= []
    xgb_gini = []
    ridge_gini = []
    ctb_gini = []
    nn_gini = []
    bay_nn_gini = []    
    
    glm_cv_models = []
    lgbm_cv_models = []
    df_cv_models = []
    xgb_cv_models = []
    ridge_cv_models = []
    ctb_cv_models = []
    nn_cv_models = []
    bay_nn_cv_models = []
    
    xgb_cv_clfs = []
    lr_cv_clfs = []
    
    gkf = GroupKFold(n_splits=5)
    
    predictions_cv = []
    
    # 5-fold Cross validation
    for train_index, test_index in gkf.split(X, Y, groups=df_data['id_policy']):
        print("fitting cv")
        y_train_cv, y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
        y_is_large_train_cv, y_is_large_test_cv = Y_is_large.iloc[train_index], Y_is_large.iloc[test_index]
        
        df_data_train, df_data_test = df_data_X.iloc[train_index], df_data_X.iloc[test_index]
        
        # df_data_train, target_encoder_cv = add_target_encoding(df_data_train, y_train_cv)
        # df_data_test, _ = add_target_encoding(df_data_test, None, target_encoder=target_encoder_cv)
        
        X_train_cv, _, _, column_trans_cv, _, _ = transform_columns(
            df_data_train,
            column_trans=None,
            use_agg=use_agg,
            model_type='glm',
        )
        X_test_cv, _, _, _, _, _ = transform_columns(
            df_data_test,
            column_trans=column_trans_cv,
            use_agg=use_agg,
            model_type='glm',
        )        
            
        
        X_train_tree_cv, _, _, column_trans_tree_cv, _, _ = transform_columns(
            df_data_train,
            column_trans=None,
            use_agg=use_agg,
            model_type='tree',
        )
        X_test_tree_cv, _, _, _, _, _ = transform_columns(
            df_data_test,
            column_trans=column_trans_tree_cv,
            use_agg=use_agg,
            model_type='tree',
        )
        
        X_train_nn_cv, _, _, column_trans_nn_cv, cardinalities, embedding_dimensions = transform_columns(
            df_data_train,
            column_trans=None,
            use_agg=use_agg,
            model_type='nn',
        )
        X_train_nn_cv_cat, X_train_nn_cv_numeric = split_categorical_numeric_inputs(X_train_nn_cv, len(cardinalities))
        X_test_nn_cv, _, _, _, _, _ = transform_columns(
            df_data_test,
            column_trans=column_trans_nn_cv,
            use_agg=use_agg,
            model_type='nn',
        )
        X_test_nn_cv_cat, X_test_nn_cv_numeric = split_categorical_numeric_inputs(X_test_nn_cv, len(cardinalities))
        
        
        xgb_mc = xgb_monotone_constraint_str(monotonicity_list)
        lgb_mc = lgbm_monotone_constraint_str(monotonicity_list)
        

        glm_regressor = TweedieRegressor(power=1.25, alpha=1, max_iter=10000)
        lgbm_regressor = LGBMRegressor(max_depth=3, num_leaves=8, objective='tweedie', learning_rate=0.02, 
                                        n_estimators=600, tweedie_variance_power=1.25, subsample_freq=1, subsample=0.8, min_child_samples=50,
                                        monotone_constraints=lgb_mc
                                        )
        df_regressor = CascadeForestRegressor(max_depth=10, n_trees=250, n_estimators=4, n_jobs=-1, n_tolerant_rounds=1,
                                              min_samples_leaf=50, use_predictor=True,
                                              predictor='xgboost',
                                              predictor_kwargs={'objective': 'reg:tweedie', 'tweedie_variance_power':1.25,
                                                                'n_estimators': 250, 'min_child_weight':75, 'eta':0.03,
                                                                'max_depth':4, 'subsample':0.7, 'colsample_bytree': 0.5,
                                                                'monotone_constraints':xgb_mc,}
                                              )
        xgb_regressor = XGBRegressor(n_estimators=350, objective='reg:tweedie', 
                                      min_child_weight=75, subsample=0.7, colsample_bytree=0.5, 
                                      tree_method='hist', max_depth=4, eta=0.03, tweedie_variance_power=1.25,
                                      monotone_constraints=xgb_mc)
        ctb_regressor = make_pipeline(
            ft_catboost,
            CatBoostRegressor(n_estimators=800, loss_function='Tweedie:variance_power=1.25', 
                              colsample_bylevel=0.7, subsample=0.7, l2_leaf_reg=5, langevin=True,
                              silent=True, learning_rate=0.02, cat_features=cat_feat_list_catboost,)
        )
        # ridge_regressor = make_pipeline(StandardScaler(), Ridge())
        nn_regressor = KerasRegressor(
            build_fn=nn_model_fn,
            numeric_input_dim = X_train_nn_cv_numeric.shape[1],
            cardinalities=cardinalities,
            embedding_dimensions=embedding_dimensions,
            batch_size=256,
            epochs=75,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', patience=16),
                ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True
                ),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001)
            ],
        )
        
        bay_nn_regressor = MyTransformedTargetRegressor(
            regressor = KerasRegressor(
                build_fn=nn_model_bayesian_fn,
                numeric_input_dim = X_train_nn_cv_numeric.shape[1],
                cardinalities=cardinalities,
                embedding_dimensions=embedding_dimensions,
                callbacks=[
                    EarlyStopping(monitor='val_loss', mode='min', patience=21),
                    ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        save_weights_only=True,
                        monitor='val_tweedie_loss_from_logits',
                        mode='min',
                        save_best_only=True
                    ),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=0.00001)
                ],
                batch_size=256,
                epochs=100,
                verbose=0
            ),
            func=identity,
            inverse_func=transform_logits
        )
        
        nn_regressor.set_params(validation_data=([X_test_nn_cv_cat, X_test_nn_cv_numeric], y_test_cv))
        bay_nn_regressor.regressor.set_params(validation_data=([X_test_nn_cv_cat, X_test_nn_cv_numeric], y_test_cv))
        

        
        glm_regressor.fit(X_train_cv, y_train_cv)
        lgbm_regressor.fit(X_train_tree_cv, y_train_cv, eval_set=[(X_test_tree_cv, y_test_cv)], early_stopping_rounds=30, verbose=False)
        df_regressor.fit(X_train_tree_cv, y_train_cv)
        xgb_regressor.fit(X_train_tree_cv, y_train_cv,
                          eval_set=[(X_test_tree_cv, y_test_cv)],
                          eval_metric=['tweedie-nloglik@1.25'],
                          early_stopping_rounds=20,
                          verbose=False)
        ctb_regressor.fit(X_train_tree_cv, y_train_cv, catboostregressor__early_stopping_rounds=30,
                            catboostregressor__eval_set=(ft_catboost.transform(X_test_tree_cv), y_test_cv))
        # ridge_regressor.fit(X_train_cv, y_train_cv)
        nn_regressor.fit([X_train_nn_cv_cat, X_train_nn_cv_numeric], y_train_cv)
        nn_regressor.model.load_weights(checkpoint_filepath)
        bay_nn_regressor.fit([X_train_nn_cv_cat, X_train_nn_cv_numeric], y_train_cv)
        bay_nn_regressor.regressor.model.load_weights(checkpoint_filepath)
        
        prediction_glm = glm_regressor.predict(X_test_cv)
        prediction_lgbm = lgbm_regressor.predict(X_test_tree_cv)
        prediction_df = df_regressor.predict(X_test_tree_cv)
        prediction_xgb = xgb_regressor.predict(X_test_tree_cv)
        prediction_ctb = ctb_regressor.predict(X_test_tree_cv)
        # prediction_ridge = ridge_regressor.predict(X_test_cv)
        prediction_nn = nn_regressor.predict([X_test_nn_cv_cat, X_test_nn_cv_numeric])
        prediction_bay_nn = bay_nn_regressor.predict([X_test_nn_cv_cat, X_test_nn_cv_numeric])
        
        prediction_cv = np.hstack((
            prediction_glm.reshape(-1, 1),
            prediction_lgbm.reshape(-1, 1),
            prediction_df.reshape(-1, 1),
            prediction_xgb.reshape(-1, 1),
            prediction_ctb.reshape(-1, 1),
            # prediction_ridge.reshape(-1, 1),
            prediction_nn.reshape(-1, 1),
            prediction_bay_nn.reshape(-1, 1),
            y_test_cv.values.reshape(-1, 1)
        ))
        predictions_cv.append(prediction_cv)
        
        glm_td.append(mean_tweedie_deviance(y_test_cv, prediction_glm, power=1.5))
        lgbm_td.append(mean_tweedie_deviance(y_test_cv, prediction_lgbm, power=1.5))
        df_td.append(mean_tweedie_deviance(y_test_cv, prediction_df, power=1.5))
        xgb_td.append(mean_tweedie_deviance(y_test_cv, prediction_xgb, power=1.5))
        ctb_td.append(mean_tweedie_deviance(y_test_cv, prediction_ctb, power=1.5))
        # ridge_td.append(mean_tweedie_deviance(y_test_cv, prediction_ridge, power=1.5))
        nn_td.append(mean_tweedie_deviance(y_test_cv, prediction_nn, power=1.5))
        bay_nn_td.append(mean_tweedie_deviance(y_test_cv, prediction_bay_nn, power=1.5))
        
        glm_gini.append(gini(y_test_cv, prediction_glm))
        lgbm_gini.append(gini(y_test_cv, prediction_lgbm))
        df_gini.append(gini(y_test_cv, prediction_df))
        xgb_gini.append(gini(y_test_cv, prediction_xgb))
        ctb_gini.append(gini(y_test_cv, prediction_ctb))
        # ridge_gini.append(gini(y_test_cv, prediction_ridge))
        nn_gini.append(gini(y_test_cv, prediction_nn))
        bay_nn_gini.append(gini(y_test_cv, prediction_bay_nn))
        
        glm_rmse.append(rmse(prediction_glm, y_test_cv))
        lgbm_rmse.append(rmse(prediction_lgbm, y_test_cv))
        df_rmse.append(rmse(prediction_df, y_test_cv))
        xgb_rmse.append(rmse(prediction_xgb, y_test_cv))
        ctb_rmse.append(rmse(prediction_ctb, y_test_cv))
        # ridge_rmse.append(rmse(prediction_ridge, y_test_cv))
        nn_rmse.append(rmse(prediction_nn, y_test_cv))
        bay_nn_rmse.append(rmse(prediction_bay_nn, y_test_cv))
        
        glm_cv_models.append(glm_regressor)
        lgbm_cv_models.append(lgbm_regressor)
        df_cv_models.append(df_regressor)
        xgb_cv_models.append(xgb_regressor)
        ctb_cv_models.append(ctb_regressor)
        # ridge_cv_models.append((ridge_regressor)
        nn_cv_models.append(nn_regressor)
        bay_nn_cv_models.append(bay_nn_regressor)
        
        xgb_clf = BalancedBaggingClassifier(
            base_estimator=XGBClassifier(n_estimators=300,subsample=0.5, colsample_bytree=0.6, 
                                              tree_method='hist', max_depth=4, eta=0.03, random_state=42, use_label_encoder=False),
            n_estimators=20, random_state=42, n_jobs=-1
        )
        lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        xgb_clf.fit(X_train_tree_cv, y_is_large_train_cv)
        lr_clf.fit(X_train_cv, y_is_large_train_cv)
        xgb_cv_clfs.append(xgb_clf)
        lr_cv_clfs.append(lr_clf)
        
        # prediction_clf = np.any([lr_clf.predict(X_test_cv), xgb_clf.predict(X_test_tree_cv)], axis=0)
        # print(confusion_matrix(y_is_large_test_cv, prediction_clf))

        
    print(np.mean(glm_td))
    print(np.mean(lgbm_td))
    print(np.mean(df_td))
    print(np.mean(xgb_td))
    print(np.mean(ctb_td))
    # print(np.mean(ridge_td))
    print(np.mean(nn_td))
    print(np.mean(bay_nn_td))

    print(np.mean(glm_gini))
    print(np.mean(lgbm_gini))
    print(np.mean(df_gini))
    print(np.mean(xgb_gini))
    print(np.mean(ctb_gini))
    # print(np.mean(ridge_gini))
    print(np.mean(nn_gini))
    print(np.mean(bay_nn_gini))
    
    print(np.mean(glm_rmse))
    print(np.mean(lgbm_rmse))
    print(np.mean(df_rmse))
    print(np.mean(xgb_rmse))
    print(np.mean(ctb_rmse))
    # print(np.mean(ridge_rmse))
    print(np.mean(nn_rmse))
    print(np.mean(bay_nn_rmse))
    
    oof_result = []
    coefs = []
    intercepts = []
    
    # Fit meta model
    mtd = []
    for cv_idx, (_, _) in enumerate(gkf.split(X, Y, groups=df_data['id_policy'])):
        print("fitting stack cv")
        
        stack_train = np.vstack([predictions_cv[i] for i in set(range(len(predictions_cv))) - set([cv_idx])])
        stack_test = predictions_cv[cv_idx]
        
        X_stack_train_cv, X_stack_test_cv = stack_train[:, :-1], stack_test[:, :-1]
        Y_stack_train_cv, Y_stack_test_cv = stack_train[:, -1], stack_test[:, -1]
        
        # stack_reg_cv = LinearRegression(positive=True, n_jobs=-1).fit(X_stack_train_cv, Y_stack_train_cv)
        stack_reg_cv = make_pipeline(
            FunctionTransformer(np.log, validate=False),
            TweedieRegressor(power=1.5, alpha=0.1, max_iter=10000).fit(
                np.log(X_stack_train_cv), Y_stack_train_cv
            )
        ).fit(X_stack_train_cv, Y_stack_train_cv)
        
        coefs.append(stack_reg_cv[1].coef_.tolist())
        intercepts.append(stack_reg_cv[1].intercept_)
        
        prediction = stack_reg_cv.predict(X_stack_test_cv)
        
        oof_result.append((prediction, Y_stack_test_cv))
        
        mtd.append(mean_tweedie_deviance(Y_stack_test_cv, prediction, power=1.5))
    
    print(np.mean(mtd))
    stack_reg_cv[1].intercept_= np.mean(intercepts)
    stack_reg_cv[1].coef_ = np.mean(coefs, axis=0)
    
    stack_large_claim_clf = StackingLargeClaimClassifier(base_cv_models=[
        [xgb_cv_clfs, 'tree', 'xgb'],
        [lr_cv_clfs, 'glm', 'lr'],
    ])
    
    stack_model = StackingModel(
        base_cv_models=[
            [glm_cv_models, 'glm', 'glm'], 
            [lgbm_cv_models, 'tree', 'lgbm'],
            [df_cv_models, 'tree', 'df'],
            [xgb_cv_models, 'tree', 'xgb'],
            [ctb_cv_models, 'tree', 'ctb'],
            [nn_cv_models, 'nn', 'nn'],
            [bay_nn_cv_models, 'nn', 'bay_nn'],
        ], 
        final_model=stack_reg_cv,
        column_trans=column_trans,
        tree_column_trans=column_trans_for_tree,
        nn_column_trans=column_trans_for_nn,
        large_claim_clf=stack_large_claim_clf,
        #target_encoder=target_encoder,
    )
    
    print(rmse(stack_model.predict(X, X_for_tree, [X_for_nn_cat, X_for_nn_numeric]), Y))
    
    if use_agg:
        stack_model_name = 'stacked_hist_min_year_%d_model_final.joblib'%(min_year)
        nn_model_name = 'nn_hist_min_year_%d_model'%(min_year)
    else:
        stack_model_name = 'stacked_model_final.joblib'
        nn_model_name = 'nn_model'
        
    # Save stacking_model_nn
    
    for base_cv_model_list in stack_model.base_cv_models:
        base_cv_model = base_cv_model_list[0]
        for cv_idx, model in enumerate(base_cv_model):
            model_name = nn_model_name
            if isinstance(model, MyTransformedTargetRegressor):
                model = model.regressor
                model_name = 'bay_' + model_name
                
            if isinstance(model, KerasRegressor):
                model_path = model_name + '_cv_' + str(cv_idx) + '.h5'
                model.model.save(model_path)
                model.model_path = model_path
                del model.model
                del model.sk_params['callbacks']
                del model.sk_params['validation_data']
                
    # Save stack_model as a whole
    dump(stack_model, stack_model_name)
    
    # Load Stacking model as a whole
    stack_model = load(stack_model_name)
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
