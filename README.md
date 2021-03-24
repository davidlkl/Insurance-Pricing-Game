# Insurance-Pricing-Game - 1st place solution

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
Model training

Large Claim detection model:
A XGBoost and Logistic regression model to predict whether a claim would be >3k.

Claim estimation model:
I stacked 7 base models using a Tweedie GLM as the meta-learner under 5 fold CV.
Base models:

Tweedie GLM
Light GBM
DeepForest
XGBoost
CatBoost
Neural Network with Tweedie deviance as loss function
A neural network with log-normal distribution likelihood as loss function (learning the mu and sigma of the loss)

### Model.py
- The script that is used to produce prediction inside the AICrowd environment
- Pricing strategy is incorporated in the predict_premium function

The final presentation is also uploaded to this repository.
