"""This script is used to predict prices from your model.

DO NOT MODIFY."""


import numpy as np
import sys
import pandas as pd
import os


outputs_dir = os.getenv("OUTPUTS_DIR", ".") + "/"
# This script expects sys.args arguments for (1) the dataset and (2) the output file.
input_dataset = os.getenv("DATASET_PATH", "training.csv")  # The default value.
output_claims_file = outputs_dir + 'claims.csv'  # The file where the expected claims should be saved.
output_prices_file = outputs_dir + 'prices.csv'  # The file where the prices should be saved.

if len(sys.argv) >= 2:
	input_dataset = sys.argv[1]
if len(sys.argv) >= 3:
	output_claims_file = sys.argv[2]
if len(sys.argv) >= 4:
	output_prices_file = sys.argv[3]

# Load the dataset.
# Remove the claim_amount column if it is in the dataset.
Xraw = pd.read_csv(input_dataset)
if 'claim_amount' in Xraw.columns:
	Xraw = Xraw.drop(columns=['claim_amount'])


# Load the saved model, and run it.

import model

trained_model = model.load_model()

if os.getenv("WEEKLY_EVALUATION", "false") == "true":
	prices = model.predict_premium(trained_model, Xraw)
	np.savetxt(output_prices_file, prices, delimiter=',', fmt='%.5f')
else:
	claims = model.predict_expected_claim(trained_model, Xraw)
	np.savetxt(output_claims_file, claims, delimiter=',', fmt='%.5f')

