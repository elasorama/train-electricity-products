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

    # The X features will be seconds_since_midnight_sin, seconds_since_midnight_cos
    # The remaining column will be the y feature
    X_columns = columns[1:]
    y_column = columns[0]

    # Setting the run id for mlflow tracking 
    mlflow.set_experiment("Initial_pipeline")

    # Splitting the data into X and y
    X = data[X_columns]
    y = data[y_column]

    # Predicting on the train set 
    y_pred = model.predict(X)

    # Saving the figure of the real vs predicted values
    plt.plot(y, label="real")
    plt.plot(y_pred, label="predicted")
    plt.legend()
    mlflow.log_figure(plt.gcf(), "real_vs_predicted.png")
    
    # Calculating and printing the mae, mse and mape 
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean(np.square(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y))
    print(f"mae: {mae}")
    print(f"mse: {mse}")
    print(f"mape: {mape}")

    # Logging the metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("N_test", len(y))

    # Logging the model 
    mlflow.sklearn.log_model(model, "model")

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
    eval_model(
        input_data_path=args.input_data,
        input_model_path=args.input_model_path,
        output_dir=args.output_dir,
    )