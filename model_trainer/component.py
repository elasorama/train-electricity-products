# Importing linear regression
from sklearn.linear_model import LinearRegression

# Multioutput regressor
from sklearn.multioutput import MultiOutputRegressor

# Pickle for saving the model
import pickle

# Pandas reading 
import pandas as pd

# Argument parsing 
import argparse

# Numpy 
import numpy as np

# Importing mlflow 
import mlflow

# Model signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Defining the model training function 
def train_model(
        input_data_path: str,
        output_model_path: str
): 
    # Reading the data 
    data = pd.read_parquet(input_data_path)

    # Saving the columns 
    columns = data.columns.tolist()

    # The X features are the last two columns
    # The remaining columns are the y features
    X_columns = columns[-2:]
    y_column = columns[:-2]

    # Splitting the data into X and y
    X = data[X_columns]
    y = data[y_column]

    # Creating the model
    model = MultiOutputRegressor(LinearRegression())

    # Fitting the model
    model.fit(X, y)

    # Predicting on the train set 
    y_pred = model.predict(X)

    # Logging the train mse, mae and mape
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean(np.square(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y))

    # Logging the metrics
    mlflow.log_metric("train_mae", mae)
    mlflow.log_metric("train_mse", mse)
    mlflow.log_metric("train_mape", mape)

    # Creating the input schema for the model 
    input_schema = Schema([ColSpec("double", x) for x in X_columns])

    # Creating the output schema for the model
    output_schema = Schema([ColSpec("double", y_column)])

    # Creating the signature for the model
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Logging the model and the signature
    mlflow.sklearn.log_model(model, "model", signature=signature)

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