# Pickle for loading the model
import pickle

# Pandas reading 
import pandas as pd

# Argument parsing 
import argparse

# Numpy 
import numpy as np

# Mlflow logging
import mlflow

# Ploting 
import matplotlib.pyplot as plt

# Image wrangling 
from PIL import Image

# Defining the model training function 
def eval_model(
        input_data_path: str,
        input_model_path: str,
        output_dir: str
): 
    # Reading the data 
    data = pd.read_parquet(input_data_path)

    # Loading the model 
    model = pickle.load(open(input_model_path, 'rb'))

    # Saving the columns 
    columns = data.columns.tolist()

    # The X features are the last two columns
    # The remaining columns are the y features
    X_columns = columns[-2:]
    y_column = columns[:-2]

    # Splitting the data into X and y
    X = data[X_columns]
    y = data[y_column]

    # Predicting on the train set 
    y_pred = model.predict(X)
    
    # Calculating and printing the mae, mse and mape 
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean(np.square(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y))

    # Logging the metrics
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mape", mape)
    mlflow.log_metric("N_test", len(y))

if __name__ == '__main__':
    # Parsing the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to the input data",
    )

    parser.add_argument(
        "--input_model_path",
        type=str,
        help="Path to the input model",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output model",
    )

    args = parser.parse_args()

    # Printing the arguments
    print(f"input_data_path: {args.input_data}")
    print(f"input_model_path: {args.input_model_path}")
    print(f"output_model_path: {args.output_dir}")

    # Calling the model training function
    eval_model(
        input_data_path=args.input_data,
        input_model_path=args.input_model_path,
        output_dir=args.output_dir,
    )