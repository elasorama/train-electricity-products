# Importing all the components from the pipeline 
from feature_engineer_forecast_ahead.component import prep_data as feature_engineer_forecast_ahead
from model_trainer.component import train_model
from train_test_split.component import prep_data as train_test_split
from model_eval.component import eval_model

# OS packages
import os 
import shutil

# Tempfile creation 
import tempfile

# Defining the test function 
def test_pipeline() -> None:
    #-- Assembling the pipeline
    
    # Getting the current dir 
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Creating the temporary folder
    tmp_dir = tempfile.mkdtemp()

    # Defining the path to the input data
    input_data_path = os.path.join(cur_dir, "..", "..", "test_data", "2024-01-20-14-02_2024-01-21-13-57.parquet")

    # Defining the output train and test data path 
    train_data_path = os.path.join(tmp_dir, "train.parquet")
    test_data_path = os.path.join(tmp_dir, "test.parquet")

    # Defining the time ranges ahead to create models 
    time_ranges_ahead = "5, 15, 60"

    # Defining the dependant variable 
    dependant_variable = "power_usage"

    # Defining the output path for the model 
    model_path = os.path.join(tmp_dir, "model.pkl")

    #-- Acting 
    train_test_split(
        input_data=input_data_path, 
        training_data_path=train_data_path, 
        test_data_path=test_data_path
        )
    feature_engineer_forecast_ahead(
        input_data=train_data_path, 
        dependant_variable=dependant_variable, 
        time_range_ahead=time_ranges_ahead, 
        output_data=train_data_path
        )
    feature_engineer_forecast_ahead(
        input_data=test_data_path, 
        dependant_variable=dependant_variable, 
        time_range_ahead=time_ranges_ahead, 
        output_data=test_data_path
        )
    train_model(
        input_data_path=train_data_path, 
        output_model_path=model_path
        )
    eval_model(
        input_data_path=test_data_path, 
        input_model_path=model_path, 
        output_dir=tmp_dir
    )
    
    #-- Asserting 
    assert os.path.exists(model_path)

    # Deleting the temporary folder
    shutil.rmtree(tmp_dir)