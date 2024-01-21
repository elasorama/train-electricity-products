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
    
    # Logging the metrics for each of the columns
    for i, column in enumerate(y_column):
        # Calculating the metrics
        mae = np.mean(np.abs(y[column] - y_pred[:, i]))
        mse = np.mean(np.square(y[column] - y_pred[:, i]))
        mape = np.mean(np.abs((y[column] - y_pred[:, i]) / y[column]))

        # Logging the metrics 
        mlflow.log_metric(f"train_mae_{column}", mae)
        mlflow.log_metric(f"train_mse_{column}", mse)
        mlflow.log_metric(f"train_mape_{column}", mape)

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