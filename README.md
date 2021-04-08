# AICrowd Insurance Pricing Game - 1st Place Solution

https://www.aicrowd.com/challenges/insurance-pricing-game

## Folder structure:
### Preprocessing.py
Feature engineering
  1. Binning<br>
     Separate continuous variable into segments (Clipping is implicitly done too)<br>
     Used for GLM to help capturing non-linear relationship<br>
  2. Interactions<br>
     a) Population density<br>
     b) Driver Gender combination<br>
     c) Vehicle feature interactions<br>
         vh_value * vh_weight<br>
         present_vh_value (exponential decay by vh_age)<br>
         and more... <br>
  3. Grouping<br>
     Grouped Med1 and Med2 together in policy type<br>
  4. Transformation<br>
     Log-transform, power transform of some continuous variables<br>
  5. History variable<br>
     Historical Claim amount, Historical claim count, year since last claim, change in NCD


### Training.py

#### Large Claim detection model:<br>
- A XGBoost and Logistic regression model to predict whether a claim would be >3k.<br>

#### Claim estimation model:<br>
- I stacked 7 base models using a Tweedie GLM as the meta-learner under 5 fold CV.<br>
  - Tweedie GLM<br>
  - Light GBM<br>
  - DeepForest<br>
  - XGBoost<br>
  - CatBoost<br>
  - Neural Network with Tweedie deviance as loss function<br>
  - Neural network with log-normal distribution likelihood as loss function (learning the mu and sigma of the loss)<br>

### Model.py
- The script that is used to produce prediction inside the AICrowd environment
- Pricing strategy is incorporated in the predict_premium function

The final presentation is also uploaded to this repository.
