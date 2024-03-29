# Importing the necesary packages for the component
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Importing the mlflow tracking 
import mlflow

# Defining the component function
def prep_data(
    input_data: str,
    training_data_path: str,
    test_data_path: str,
    train_test_split_ratio: float = 0.75
):
    """
    Prepares the data for training and testing
    """
    # Reading the data from the input folder
    data = pd.read_parquet(input_data)

    # Logging the number of rows
    mlflow.log_metric("N_rows_total", len(data))
    
    # Splitting the data into train and test
    train, test = train_test_split(data, train_size=train_test_split_ratio, random_state=42)
    
    # Logging the number of rows
    mlflow.log_metric("N_rows_train", len(train))
    mlflow.log_metric("N_rows_test", len(test))

    # Saving the data into the output folder
    train.to_parquet(training_data_path, index=False)
    test.to_parquet(test_data_path, index=False)

if __name__ == '__main__': 
    # Parsing the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to the input data",
    )

    parser.add_argument(
        "--training_data_path",
        type=str,
        help="Path to the training data",
    )

    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to the test data",
    )

    parser.add_argument(
        "--train_test_split",
        type=float,
        help="Share of train data in the input data",
    )

    args = parser.parse_args()

    # Printing the arguments
    print(f"input_data: {args.input_data}")
    print(f"training_data_path: {args.training_data_path}")
    print(f"share of train data: {args.train_test_split}")
    print(f"test_data_path: {args.test_data_path}")

    # Calling the component function
    prep_data(
        args.input_data,
        args.training_data_path,
        args.test_data_path,
        args.train_test_split
    )