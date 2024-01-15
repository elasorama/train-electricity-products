# Pickle for loading the model
import pickle

# Pandas reading 
import pandas as pd

# Argument parsing 
import argparse

# Numpy 
import numpy as np

# Json wrangling 
import json

# Defining the model training function 
def train_model(
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

    # The X features will be seconds_since_midnight_sin, seconds_since_midnight_cos
    # The remaining column will be the y feature
    X_columns = columns[1:]
    y_column = columns[0]

    # Splitting the data into X and y
    X = data[X_columns]
    y = data[y_column]

    # Predicting on the train set 
    y_pred = model.predict(X)

    # Calculating and printing the mae, mse and mape 
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean(np.square(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y))
    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"mape: {mape}")

    # Saving the acc metrics on the test set 
    acc = {
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "N": len(y),
    }

    # Saving the acc dict 
    with open(f"{output_dir}/acc.json", "w") as f:
        json.dump(acc, f)

    # Saving the model
    pickle.dump(model, open(f"{output_dir}/model.pkl", 'wb'))

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
    train_model(
        input_data_path=args.input_data,
        input_model_path=args.input_model_path,
        output_model_path=args.output_dir,
    )