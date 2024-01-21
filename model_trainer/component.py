# Importing linear regression
from sklearn.linear_model import LinearRegression

# sklearn ensemble regressors
from sklearn.ensemble import GradientBoostingRegressor

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

# Ploting 
import numpy as np 
import matplotlib.pyplot as plt

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
    model = MultiOutputRegressor(GradientBoostingRegressor())

    # Fitting the model
    model.fit(X, y)

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

    # Creating linespaces for the X features
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)

    # Creating the meshgrid
    x1, x2 = np.meshgrid(x1, x2)

    # Creating the input features
    X = np.array([x1.flatten(), x2.flatten()]).T

    # Predicting on the meshgrid
    y_pred = model.predict(X)

    # Creating a heatmap picture where the X and Y axes are the input features
    # and the color is the output feature

    # Creating the sequences of X and Z features
    plt.imshow(y_pred[:, 0].reshape(100, 100), extent=[-1, 1, -1, 1], origin='lower')
    plt.colorbar()
    plt.xlabel(X_columns[0])
    plt.ylabel(X_columns[1])
    plt.title(y_column[0])

    # Logging the heatmap picture
    mlflow.log_figure(plt.gcf(), "heatmap.png")

    # Creating the input schema for the model 
    input_schema = Schema([ColSpec("double", x) for x in X_columns])

    # Creating the output schema for the model
    output_schema = Schema([ColSpec("double", y) for y in y_column])

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