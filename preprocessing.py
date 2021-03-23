# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 19:52:45 2020

@author: ling
"""

import numpy as np
import pandas as pd
import os

pd.options.mode.chained_assignment = None

from scipy.special import boxcox1p
from scipy.stats import boxcox
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from functools import reduce

import category_encoders as ce

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def add_target_encoding(df_data, Y, min_sample_leaf=350, smoothing=0, noise_level=0, target_encoder=None):
    is_train = (target_encoder is None)
    if is_train:
        target_encoder = ce.CatBoostEncoder(cols=['vh_make_model_grp'], a=200)
        target_encoder.fit(df_data, Y)
        
    df_data = target_encoder.transform(df_data, Y)
    if is_train:
        df_data['vh_make_model_grp'] = add_noise(df_data['vh_make_model_grp'], noise_level)   
    df_data['log_vh_make_model_grp'] = np.log(df_data['vh_make_model_grp'] + 1e-5)
    
    return df_data, target_encoder

def create_groups(df_data):
    
    # Group driver 1 age
    df_data['drv_age1_grp'] = pd.cut(
        df_data['drv_age1'].clip(lower=19, upper=120),
        bins=[17, 23, 29, 37, 42, 47, 52, 56, 60, 65, 70, 77, 120]
    )
    
    # Group driver 1 age_lic
    df_data['drv_age_lic1_grp'] = pd.cut(
        df_data['drv_age_lic1'].clip(lower=1, upper=80),
        bins=[0, 5, 10, 15, 21, 27, 31, 35, 41, 47, 55, 60, 80]
    )
    
    # Group driver 2 age
    df_data['drv_age2_grp'] = pd.cut(
        df_data['drv_age2'].fillna(0).clip(upper=150), 
        bins=[-1, 17, 19, 21, 23, 46, 59, 71, 150]
    )
    # Group driver 2 age_lic
    df_data['drv_age_lic2_grp'] = pd.cut(
        df_data['drv_age_lic2'].fillna(0).clip(upper=80), 
        bins=[-1, 0, 2, 5, 10, 15, 25, 35, 50, 80]
    )
    
    df_data['drv_age_lic1*2_grp'] = pd.cut(
        (df_data['drv_age_lic1'] * df_data['drv_age_lic2']).fillna(0).clip(upper=10000),
        bins=[-1, 0, 98, 160, 280, 416, 567, 751, 990, 1295, 1680, 2205, 3000, 10000]
    )
    
    # Group vehicle age 
    df_data['vh_age_grp'] = pd.cut(
        df_data['vh_age'].fillna(1),
        bins=[0, 3, 6, 8, 9, 10, 12, 14, 15, 30, 100]
    )
    
    # Group vehicle speed
    df_data['vh_speed_grp'] = pd.cut(
        df_data['vh_speed'].fillna(170).clip(lower=95, upper=300),
        bins=[0, 94.844, 105.4, 121, 126.2, 136.6, 152.2, 157.4, 167.8, 183.4, 204.2, 300]
    )
    
    # Group vehicle weight
    df_data['vh_weight_grp'] = pd.cut(
        df_data['vh_weight'].fillna(0).clip(lower=1, upper=3000),
        bins=[-1, 383.1, 700, 900, 1021.6, 1145, 1245, 1315, 1500, 3000]
    )
    
    df_data['vh_value_grp'] = pd.cut(
        df_data['vh_value'].fillna(16321).clip(lower=1000, upper=120000),
        bins=[-1, 8896, 11228, 12533, 13390, 16047, 19225, 21772, 23900, 29271, 35000, 120000]
    )
    
    # Group policy sit duration
    df_data['pol_sit_duration_grp'] = pd.cut(
        df_data['pol_sit_duration'].clip(lower=1, upper=30),
        bins=[-1, 2, 3, 4, 5, 6, 8, 11, 14, 30]
    )
    # Group policy duration
    df_data['pol_duration_grp'] = pd.cut(
        df_data['pol_duration'].clip(lower=1, upper=30),
        bins=[-1, 2, 3, 6, 7, 8, 9, 11, 15, 19, 25, 30]
    )
    
    df_data['pol_coverage_grp'] = df_data['pol_coverage'].map({
        'Med1': 'Med',
        'Med2': 'Med',
        'Max': 'Max',
        'Min': 'Min'
    })
    
    df_data['vh_fuel_grp'] = df_data['vh_fuel'].map({
        'Diesel' : 'Diesel',
        'Gasoline': 'Gasoline',
        'Hybrid' : 'Diesel'
    })
    df_data['pol_usage_grp'] = df_data['pol_usage'].map({
        'AllTrips': 'Professional',
        'Professional': 'Professional',
        'Retired': 'Retired',
        'WorkPrivate': 'WorkPrivate'
    })
    
    # Combine driver 1 and 2 gender
    df_data['drv_sex_1_2'] = df_data['drv_sex1'] + df_data['drv_sex2'].fillna(0)
    
    df_data["pol_coverage_usage_grp"] = df_data["pol_coverage_grp"] + '_' + df_data["pol_usage_grp"]
    df_data["pol_coverage_fuel_grp"] = df_data["pol_coverage_grp"] + '_' + df_data["vh_fuel_grp"]
    df_data['pol_usage_fuel_grp'] =  df_data["pol_usage_grp"] + '_' + df_data["vh_fuel_grp"]
    
    df_data['pol_coverage_usage_type'] = df_data["pol_coverage_grp"] + '_' + df_data["pol_usage_grp"] + '_' + df_data['vh_type']
    df_data['vh_type_fuel'] = df_data['vh_type'] + '_' + df_data['vh_fuel_grp']
    df_data['vh_type_pol_usage'] = df_data['vh_type'] + '_' + df_data['pol_usage_grp']
    return df_data

def impute_and_clip(df_data):
    
    df_data['population'] = df_data['population'].clip(lower=10)
    
    df_data['drv_age1'] = df_data['drv_age1'].clip(upper=90)
    df_data['drv_age2'] = df_data['drv_age2'].clip(upper=90)
    df_data['drv_age_lic1'] = df_data['drv_age_lic1'].clip(upper=75)
    df_data['drv_age_lic2'] = df_data['drv_age_lic2'].clip(upper=75)
    df_data['drv_sex2'] = df_data['drv_sex2'].fillna(0)
    
    df_data['vh_age'] = df_data['vh_age'].fillna(10).clip(upper=30)
    df_data['vh_value'] = df_data['vh_value'].fillna(16321)
    df_data['vh_speed'] = df_data['vh_speed'].fillna(170).clip(lower=95)
    df_data['vh_weight'] = df_data['vh_weight'].fillna(0).clip(lower=400)
    
    df_data['pol_duration'] = df_data['pol_duration'].clip(upper=35)
    
    return df_data

def create_raw_features(df_data):
    df_data = create_groups(df_data)
    
    df_data = impute_and_clip(df_data)
    
    # Group vh make model
    vh_make_model_grp = df_data.groupby('vh_make_model')['id_policy'].count()
    vh_make_model_above_1000 = vh_make_model_grp[vh_make_model_grp>=1000].index.tolist()
    df_data['vh_make_model_grp'] = df_data['vh_make_model'].apply(lambda v: v if v in vh_make_model_above_1000 else 'others')
    
    # Population, Town surface area
    df_data['population_per_area'] = df_data['population'] / df_data['town_surface_area']
    df_data['log_population_per_area'] = np.log10(df_data['population'] / df_data['town_surface_area'])
    df_data['population^2'] = df_data['population']**2
    df_data['town_surface_area^2'] = df_data['town_surface_area'] ** 2
    df_data['log_population^2'] = np.log10(df_data['population']) ** 2
    df_data['log_town_surface_area^2'] = np.log10(df_data['population']) ** 2
    
    # Prep Driver 1 age_lic
    df_data['drv_age_lic1^2'] = df_data['drv_age_lic1']
    df_data['drv_age_diff'] = (df_data['drv_age1'] - df_data['drv_age2']).fillna(0)
    df_data['drv_age_lic_diff'] = (df_data['drv_age_lic1'] - df_data['drv_age_lic2']).fillna(0)
    df_data['drv_age_lic2_imputed'] = df_data['drv_age_lic2'].fillna(0)
    df_data['drv_age2_imputed'] = df_data['drv_age2'].fillna(0)
    df_data['log_shift_drv_age1'] = np.log10(df_data['drv_age1']-18)
    
    df_data['log_drv_age_lic1'] = np.log10(df_data['drv_age_lic1'])
    df_data['log_drv_age_lic1^2'] = df_data['log_drv_age_lic1'] ** 2
    df_data['drv_age_lic1*2'] = (df_data['drv_age_lic1'] * df_data['drv_age_lic2']).fillna(750)
    df_data['drv_age1*2'] = (df_data['drv_age1'] * df_data['drv_age2']).fillna(2500)
    
    # Vehicle features
    df_data['vh_detail_missing'] = df_data['vh_value'].isna()

    df_data['log_vh_value'] = np.log10(df_data['vh_value'])
    df_data['log_vh_speed'] = np.log10(df_data['vh_speed'])
    df_data['log_vh_weight'] = np.log10(df_data['vh_weight'])
    df_data['log_vh_age'] = np.log10(df_data['vh_age'])
    
    df_data['log_vh_speed^2'] = df_data['log_vh_speed']**2
    
    df_data['log_vh_age*vh_weight'] = np.log10(df_data['vh_age'] * df_data['log_vh_weight'] + 1)
    df_data['vh_age*vh_weight'] = df_data['vh_age'] * df_data['vh_weight']
    df_data['vh_speed*vh_weight'] = df_data['vh_speed'] * df_data['vh_weight']
    df_data['vh_speed/vh_weight'] = df_data['vh_speed'] / df_data['vh_weight']
    df_data['vh_value*vh_weight'] = df_data['vh_value'] * df_data['vh_weight']
    df_data['present_vh_value'] = np.clip(
        df_data['vh_value'] * np.exp((-df_data['vh_age']+1)*0.2),
        500, None
    )
    df_data['vh_value/vh_age'] = df_data['vh_value'] / df_data['vh_age']
    df_data['vh_speed/vh_age'] = df_data['vh_speed'] / df_data['vh_age']
    df_data['vh_value*vh_speed'] = df_data['vh_value'] * df_data['vh_speed']
    
    # Policy features
    df_data['pol_no_claims_discount^2'] = df_data['pol_no_claims_discount'] ** 2
    df_data['log_pol_no_claims_discount'] = np.log1p(df_data['pol_no_claims_discount'])
    df_data['pol_sit_duration*pol_no_claims_discount'] = df_data['pol_sit_duration'] * df_data['pol_no_claims_discount']
    
    df_data['yearm1'] = df_data['year'] - 1

    
    return df_data


def select_columns(model_type='glm', use_agg=False):
    
    if model_type=='glm':
        ordinal_categorical_cols = []
        onehot_categorical_cols = ["pol_coverage_grp", "pol_usage", 'pol_coverage_usage_grp', 'pol_coverage_fuel_grp', 'pol_usage_fuel_grp', "pol_pay_freq", 
                                   'pol_coverage_usage_type', "pol_payd", "vh_fuel", "vh_type", 'vh_type_fuel', "drv_sex_1_2", 
                                   "vh_age_grp", 'drv_age1_grp', 'pol_duration_grp', 'pol_sit_duration_grp', 'vh_value_grp',
                                   'drv_age_lic2_grp', 'drv_age_lic1*2_grp', 'vh_weight_grp', 
                                  ]
        numeric_discretize_cols = []
        numeric_cols = ["log_pol_no_claims_discount", "pol_sit_duration", 
                        'log_vh_age*vh_weight', 'drv_age1', 'drv_age_lic1', 'log_vh_speed',
                        'log_drv_age_lic1', 'pol_no_claims_discount^2', 'drv_age_diff', 
                        "log_population_per_area", 'population^2', 'town_surface_area^2',
                        'vh_value*vh_weight', 'vh_speed/vh_age', 'present_vh_value', 
                        ]
        numeric_max_abs_cols = ['vh_age', 'pol_duration', "yearm1", ]
        numeric_monotonicity = []    
    elif model_type=='tree':
        ordinal_categorical_cols = []
        onehot_categorical_cols = ["pol_coverage_grp", "pol_usage", "pol_coverage_usage_grp", 'pol_coverage_fuel_grp',
                                   'pol_usage_fuel_grp', "pol_pay_freq", "pol_payd",
                                   "vh_fuel", "vh_type", "drv_sex1", 'drv_sex2', 'drv_age_lic2_grp',
                                  ]
        numeric_discretize_cols = []
        numeric_cols = ["yearm1", "pol_no_claims_discount", "pol_sit_duration", "pol_duration", 'vh_age',
                        "vh_speed", "vh_weight", 'vh_value', 'log_vh_age*vh_weight', 'drv_age_lic1',
                        "drv_age1", "population_per_area", 'population', 'town_surface_area', 'vh_speed*vh_weight',
                         'vh_value*vh_weight', 'vh_value*vh_speed', 'present_vh_value', 'vh_speed/vh_age', 'drv_age_diff',
                         'drv_age1*2', 
                        ]
        numeric_max_abs_cols = []
        numeric_monotonicity = [0, 0, -1, 0, -1,
                                1, 1, 1, 0, 0,
                                0, 1, 0, 0, 0,
                                0, 0, 1, 1, 1, 
                                0
                                ]         
    elif model_type == 'nn':
        ordinal_categorical_cols = ["pol_coverage_grp", "pol_usage_grp",  'drv_age_lic2_grp', 'vh_type_fuel',]
        onehot_categorical_cols = ['pol_coverage_usage_grp', 'pol_coverage_fuel_grp', 'pol_usage_fuel_grp', 'pol_coverage_usage_type',
                                   "drv_sex_1_2", "pol_payd", "vh_fuel_grp", "vh_type", "pol_pay_freq", 
                                   "vh_age_grp", 'drv_age1_grp', 'pol_duration_grp', 'vh_value_grp', ]
        numeric_discretize_cols = []
        numeric_cols = ["yearm1", "log_population_per_area", "log_pol_no_claims_discount",  
                            "pol_sit_duration", 'vh_age', 'drv_age_lic1', 'log_vh_age*vh_weight', 'vh_weight',
                            "drv_age1", "log_vh_speed", 'population^2', 'drv_age_lic1^2', 'town_surface_area^2', 'pol_duration',
                            'drv_age_diff', 'present_vh_value', 
                        ]
        numeric_max_abs_cols = []
        numeric_monotonicity = []
        
        
    elif model_type == 'tabnet':
        ordinal_categorical_cols = ["pol_coverage", "pol_usage_grp", "pol_pay_freq", "drv_sex_1_2",
                                    "pol_payd", "vh_fuel_grp", "vh_type"]
        onehot_categorical_cols = []
        numeric_discretize_cols = []
        numeric_cols = ["yearm1", "pol_no_claims_discount", 'vh_value', "pol_sit_duration", 'vh_age', 
                        'drv_age1', 'drv_age_lic1', 'pol_duration', 'vh_speed', 'vh_weight',
                        "log_population_per_area", 'population', 'town_surface_area',
                        ]
        numeric_max_abs_cols = []
        numeric_monotonicity = []
        
    if use_agg:
        onehot_categorical_cols += ['delta_sign_pol_no_claims_discount',]
        numeric_cols += ['agg_pos_delta_count_pol_no_claim_discount', 'agg_claim_amount', 'agg_claim_count', 'year_from_last_claim']
        if model_type == 'tree':
            numeric_monotonicity += [0, 1, 1, 0]
        
    return ordinal_categorical_cols, onehot_categorical_cols, numeric_discretize_cols, numeric_cols, numeric_max_abs_cols, numeric_monotonicity



def transform_columns(df_data, column_trans=None, use_agg=False, model_type='glm'):
    
    # Feature selection
    (ordinal_categorical_cols, onehot_categorical_cols,
     numeric_discretize_cols, numeric_cols, numeric_max_abs_cols, numeric_monotonicity) = select_columns(model_type=model_type, use_agg=use_agg)
    
    # Apply transform
    n_bins = 10
    
    if column_trans is None:
        column_trans = ColumnTransformer(
            [
                ("ordinal_categorical", OrdinalEncoder(),
                    ordinal_categorical_cols),
                ("onehot_categorical", OneHotEncoder(drop='if_binary'),
                    onehot_categorical_cols),
                ("binned_onehot_numeric_1", KBinsDiscretizer(n_bins=n_bins),
                    numeric_discretize_cols),
                ("numeric", StandardScaler(),
                    numeric_cols),
                ("nuumeric_maxabs", MaxAbsScaler(),
                    numeric_max_abs_cols),
            ],
            remainder="drop",
        )
        
        X = column_trans.fit_transform(df_data)
    else:
        X = column_trans.transform(df_data)
    
    try:
        X = X.toarray()
    except:
        pass
    
    cat_feat_list = ([True]  * len(ordinal_categorical_cols) +
                     [True]  * (X.shape[1] -
                                len(ordinal_categorical_cols)  -
                                len(numeric_discretize_cols) * n_bins -
                                len(numeric_cols) -
                                len(numeric_max_abs_cols)
                                ) +
                     [True] * len(numeric_discretize_cols) * n_bins +
                     [False] * len(numeric_cols) +
                     [False] * len(numeric_max_abs_cols)
                     )
    
    monotonicity_list = ([0]  * len(ordinal_categorical_cols) +
                     [0]  * (X.shape[1] -
                                len(ordinal_categorical_cols)  -
                                len(numeric_discretize_cols) * n_bins -
                                len(numeric_cols) -
                                len(numeric_max_abs_cols)
                                ) +
                     [0] * len(numeric_discretize_cols) * n_bins +
                     numeric_monotonicity 
                     )
    
    cardinalities = [df_data[cat_col].unique().size for cat_col in ordinal_categorical_cols]
    embedding_dimensions = [min(10, max(3, (cardinality+1) // 2)) for cardinality in cardinalities]
    
    return X, cat_feat_list, monotonicity_list, column_trans, cardinalities, embedding_dimensions

def get_year_from_last_claim(df):
    claim_year = np.where(df['claim_amount']>0)[0] + 1
    df['year_from_last_claim'] = df['year'].apply(lambda y: y - np.max(claim_year[claim_year<y], initial=-9)).clip(upper=10)
    return df

def transform_cat_col_for_catboost(X, cat_feat_list):
    df = pd.DataFrame(X)
    df[cat_feat_list] = df[cat_feat_list].astype(int)
    return df

def generate_agg_data(df):
    
    df_data = df.copy()
    
    is_hist_available = (df_data.groupby('id_policy')['year'].count() == df_data.groupby('id_policy')['year'].max())
    policy_with_hist = is_hist_available[is_hist_available==True].index.tolist()
    
    if len(policy_with_hist) == 0:
        return pd.DataFrame()
    
    df_data = df_data[df_data['id_policy'].isin(policy_with_hist)]
    
    df_data['agg_claim_amount'] = df_data.groupby('id_policy')['claim_amount'].transform(lambda x: x.cumsum().shift()).fillna(0)
    df_data['agg_claim_count'] = df_data.groupby('id_policy')['claim_amount'].transform(lambda x: (x>0).cumsum().shift()).fillna(0)
    
    # Add year_from_last_claim
    df_data_with_claim = df_data.groupby('id_policy').filter(lambda df: (df['claim_amount']>0).any())
    df_data_with_claim = df_data_with_claim.groupby('id_policy').apply(get_year_from_last_claim)

    df_data_without_claim = df_data.groupby('id_policy').filter(lambda df: not (df['claim_amount']>0).any())
    df_data_without_claim['year_from_last_claim'] = 10
    
    df_data = pd.concat([df_data_with_claim, df_data_without_claim]).sort_values(['year', 'id_policy'])
    
    df_data['delta_pol_no_claims_discount'] = df_data.groupby('id_policy')['pol_no_claims_discount'].diff().fillna(0)
    df_data['delta_sign_pol_no_claims_discount'] = np.sign(df_data['delta_pol_no_claims_discount'])
    df_data['agg_pos_delta_count_pol_no_claim_discount'] = df_data.groupby('id_policy')['delta_sign_pol_no_claims_discount'].cumsum()
        
    return df_data[['id_policy', 'year', 'agg_claim_amount', 'agg_claim_count', 'year_from_last_claim',
                    'delta_pol_no_claims_discount', 'delta_sign_pol_no_claims_discount',
                    'agg_pos_delta_count_pol_no_claim_discount']]
