# Importing linear regression
from sklearn.linear_model import LinearRegression

# Pickle for saving the model
import pickle

# Pandas reading 
import pandas as pd

# Argument parsing 
import argparse

# Numpy 
import numpy as np

# Defining the model training function 
def train_model(
        input_data_path: str,
        output_model_path: str
): 
    # Reading the data 
    data = pd.read_parquet(input_data_path)

    # Saving the columns 
    columns = data.columns.tolist()

    # The X features will be seconds_since_midnight_sin, seconds_since_midnight_cos
    # The remaining column will be the y feature
    X_columns = columns[1:]
    y_column = columns[0]

    # Splitting the data into X and y
    X = data[X_columns]
    y = data[y_column]

    # Creating the model
    model = LinearRegression()

    # Fitting the model
    model.fit(X, y)

    # Predicting on the train set 
    y_pred = model.predict(X)

    # Calculating and printing the mae and mape 
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y))
    print(f"mae: {mae}")
    print(f"mape: {mape}")

    # Saving the model
    pickle.dump(model, open(output_model_path, 'wb'))

if __name__ == '__main__':
    # Parsing the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to the input data",
    )

    parser.add_argument(
        "--output_model_path",
        type=str,
        help="Path to the output model",
    )

    args = parser.parse_args()

    # Printing the arguments
    print(f"input_data_path: {args.input_data}")
    print(f"output_model_path: {args.output_model_path}")

    # Calling the model training function
    train_model(
        input_data_path=args.input_data,
        output_model_path=args.output_model_path,
    )