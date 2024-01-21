# Importing the necesary packages for the component
import pandas as pd
import argparse
import math 

# Defining the constant for the minutes in day 
MINUTES_IN_DAY = 1440

def convert_hour_of_day(hour, minute) -> dict:
    """
    Given the hour and minute, returns the sin and cos of the hour of day
    """
    # Calculating the minutes passed since midnight
    minutes_since_midnight = hour * 60 + minute

    # Converting to sin and cos features
    minutes_since_midnight_sin = math.sin(2 * math.pi * minutes_since_midnight / MINUTES_IN_DAY)
    minutes_since_midnight_cos = math.cos(2 * math.pi * minutes_since_midnight / MINUTES_IN_DAY)

    # Returning the created features
    return {
        "minutes_since_midnight_sin": minutes_since_midnight_sin,
        "minutes_since_midnight_cos": minutes_since_midnight_cos,
    }

# Defining the component function
def prep_data(
    input_data: str,
    dependant_variable: str,
    time_range_ahead: str,
    output_data: str,
):
    """
    Creates the input format for model training; 

    The input data needs to have the following columns: 
    - timestamp
    - dependant_variable

    The output data will have the following columns:
    - dependant_variable
    - minutes_since_midnight_sin
    - minutes_since_midnight_cos
    """
    # Reading the data from the input folder
    data = pd.read_parquet(input_data)

    # Printing some stats about the data
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")

    # Creating the minutes in day features
    minutes_in_day = data.apply(lambda x: convert_hour_of_day(x['hour'], x['minute']), axis=1, result_type='expand')

    # Appending the two columns to the data
    data = pd.concat([data, minutes_in_day], axis=1)

    # Converting the time_range_ahead string to list of ints
    time_range_ahead = [int(time_range) for time_range in time_range_ahead.split(",")]

    # Creating a new column for each of the time ranges
    for time_range in time_range_ahead:
        # Creating the sum of dependant variable usage for the next time_range_ahead minutes
        data[f"{dependant_variable}_{time_range}"] = data[dependant_variable].rolling(time_range).sum()

    # Dropping the rows with nan values
    data.dropna(inplace=True)

    # Only leaving the needed columns 
    data = data[[f"{dependant_variable}_{time_range}" for time_range in time_range_ahead] + ["minutes_since_midnight_sin", "minutes_since_midnight_cos"]]

    # Saving the data into the output folder
    data.to_parquet(output_data, index=False)

    # Returning from the function
    return

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
        '--time_range_ahead',
        type=str,
        help="Time range ahead to predict",
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
    print(f"time_range_ahead: {args.time_range_ahead}")
    print(f"output_data: {args.output_data}")

    # Calling the component function
    prep_data(
        input_data=args.input_data,
        dependant_variable=args.dependant_variable,
        time_range_ahead=args.time_range_ahead,
        output_data=args.output_data,
    )