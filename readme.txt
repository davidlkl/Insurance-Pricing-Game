How to run the model:
1. run train.py
2. run predict.py

Training
- By default will read training_imputed.csv (I have uploaded this file)
- Output:
    - 2 stacked models:
        - Historical (stacked_hist_min_year_1_model_final.joblib)
        	- On top of the baseline model, historical info like no_claim_discount and past claim_amount will be used as features
        - Generic (stacked_model_final.joblib)
        	- Baseline model
        	
    - Neural network base layer models: ((bay_)nn_model_cv_0-4.h5)
        - nn model do not support pickling so need to be saved as h5 format
        
Predicting
- By default will read training.csv
- Expected behavior in the final testing set (100k policies):
    - for the 60k policies with year 1-4 histories: Historical model is used
    - for the rest: Generic model is used
    - expected output:
        - "hist model: 60000"
        - "generic model: 40000"
