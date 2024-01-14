# Importing the necesary packages for the component
import pandas as pd
import argparse
import datetime
import math 

# Defining the constant for the seconds in day, week, month and year
SECONDS_IN_DAY = 86400

def get_seconds_in_day(timestamp: datetime) -> dict:
    """
    Creates the following features from the timestamp column: 
    - seconds passed since midnight
    
    And then creates the sin and cos of each of the previous features
    """
    # Extracting the hour, minute and second from the timestamp
    hour = timestamp.hour
    minute = timestamp.minute
    second = timestamp.second

    # Calculating the seconds passed since midnight
    seconds_since_midnight = hour * 3600 + minute * 60 + second

    # Converting to sin and cos features
    seconds_since_midnight_sin = math.sin(2 * math.pi * seconds_since_midnight / SECONDS_IN_DAY)
    seconds_since_midnight_cos = math.cos(2 * math.pi * seconds_since_midnight / SECONDS_IN_DAY)

    # Returning the created features
    return {
        "seconds_since_midnight_sin": seconds_since_midnight_sin,
        "seconds_since_midnight_cos": seconds_since_midnight_cos,
    }

# Defining the component function
def prep_data(
    input_data: str,
    dependant_variable: str,
    output_data: str
):
    """
    Creates the input format for model training; 

    The input data needs to have the following columns: 
    - timestamp
    - dependant_variable

    The output data will have the following columns:
    - dependant_variable
    - seconds_since_midnight_sin
    - seconds_since_midnight_cos
    """
    # Reading the data from the input folder
    data = pd.read_parquet(input_data)

    # Printing some stats about the data
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Extracting the timestamps 
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Creating the seconds in day features
    seconds_in_day = data["timestamp"].apply(get_seconds_in_day)

    # Creating the seconds in day features
    seconds_in_day = pd.DataFrame(seconds_in_day.tolist())

    # Concatenating the dataframes
    data = pd.concat([data, seconds_in_day], axis=1)

    # Only leaving the needed columns 
    data = data[[dependant_variable, "seconds_since_midnight_sin", "seconds_since_midnight_cos"]]

    # Saving the data into the output folder
    data.to_parquet(output_data, index=False)

if __name__ == '__main__': 
    # Parsing the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to the input data",
    )

    parser.add_argument(
        "--dependant_variable", 
        type=str,
        help="Name of the dependant variable",
    )

    parser.add_argument(
        "--output_data",
        type=str,
        help="Path to the output data",
    )

    args = parser.parse_args()

    # Printing the arguments
    print(f"input_data: {args.input_data}")
    print(f"dependant_variable: {args.dependant_variable}")
    print(f"output_data: {args.output_data}")

    # Calling the component function
    prep_data(
        args.input_data,
        args.dependant_variable,
        args.output_data
    )