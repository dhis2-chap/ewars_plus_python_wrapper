import json
import csv
from pathlib import Path
import subprocess
import sys
import logging
import pandas as pd
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run_command(command):
    logger.info(f"Running command {command}") 
    subprocess.run(command, shell=True)
    # get output of command
    output = subprocess.check_output(command, shell=True)
    logger.info(f"Output: {output}")
    return output


def standardize_rainfall(train_data: pd.DataFrame):
    rainfall = train_data['rainfall'].values

    # standardize by subtracting the mean and dividing by the standard deviation
    mean = rainfall.mean()
    std = rainfall.std()
    rainfall_standardized = (rainfall - mean) / std 

    return mean, std, rainfall_standardized



def train(historic_data, config_file, geojson_file, mode_file_name):
    # historic_data should be a csv that follows the chap format
    data = pd.read_csv(historic_data)
    required_columns = ["location", "mean_temperature", "rainfall", "disease_cases"]
    for col in required_columns:
        assert col in data.columns, f"Missing required column: {col}"

    mean, std, rainfall_standardized = standardize_rainfall(data)
    # write mean, std to model_file_name, we need these to standardize future data in predict
    mode_file = Path(mode_file_name)
    with open(mode_file, "w") as f:
        f.write(json.dumps({"rainfall_mean": mean, "rainfall_std": std}))
    
    data["rainfall_std"] = rainfall_standardized
    new_historic_data_file_name = historic_data.replace(".csv", "_std.csv")

    # change location to district (which is the name ewars uses)
    data = data.rename(columns={"location": "district"})

    data.to_csv(new_historic_data_file_name, index=False)
    
    logger.info(f"Training model with historic data: {historic_data}")
    curl_command = f"""curl -X POST \
        http://127.0.0.1:3288/Ewars_run \
        -H 'accept: */*' \
        -F "csv_File=@{new_historic_data_file_name}" \
        -F "shape_File=@{geojson_file}" \
        -F "config_File=@{config_file}"
    """

    logger.info(f"Executing command: {curl_command}")
    run_command(curl_command)


def predict(model_file_name, future_data, config_file, out_file):
    # future_data should be a csv that follows the chap format"""
    required_columns = ["location", "mean_temperature", "rainfall", "disease_cases"]
    # standardize rainfall
    data = pd.read_csv(future_data)
    rainfall = data['rainfall'].values
    model = json.loads(open(model_file_name).read())
    rainfall_std = rainfall - model["rainfall_mean"] / model["rainfall_std"]
    # add this column to data
    data["rainfall_std"] = rainfall_std
    new_future_data_file_name = future_data.replace(".csv", "_std.csv")

    # change location to district (which is the name ewars uses)
    data = data.rename(columns={"location": "district"})

    data.to_csv(new_future_data_file_name, index=False)


    logger.info(f"Predicting cases for future data: {future_data}")
    curl_command = f"""curl -X POST http://127.0.0.1:3288/Ewars_predict \
        -H 'accept: */*' \
        -F "pros_csv_File=@{new_future_data_file_name}" \
        -F "config_File=@{config_file}" \
    """
    logger.info(f"Executing command: {curl_command}")
    run_command(curl_command)
    
    # get predictions
    out_file = Path(out_file)
    # check that file type is csv
    assert out_file.suffix == ".csv"
    out_file_json = str(out_file).replace(".csv", ".json")
    curl_command = f"curl -o {out_file_json} http://127.0.0.1:3288/retrieve_predicted_cases"
    predictions = run_command(curl_command)
    change_prediction_format_to_chap(out_file_json, out_file)


def test_train():
    historic_data = "demo_data/laos_dengue_and_diarrhea_surv_data_2015_2023.csv"
    historic_data = "demo_data/subset_train_chap.csv"
    geojson = "demo_data/laos_province_shapefile.GEOJSON"
    config_file = "demo_data/ewars_config.json"
    model_file_name = "demo_data/model.json"
    train(historic_data, config_file, geojson, model_file_name)


def test_predict():
    future_data = "laos_dengue_and_diarrhea_pros_data_2024.csv"
    future_data = "demo_data/subset_predict_chap.csv"
    config_file = "demo_data/ewars_config.json"
    out_file = "demo_data/predictions.csv"
    model_file_name = "demo_data/model.json"
    predict(model_file_name, future_data, config_file, out_file)

 
def change_prediction_format_to_chap(predictions_json, out_csv):
    json_data = json.loads(open(predictions_json).read())

    # Extract relevant data
    rows = []
    for item in json_data:
        prospective_predictions = item.get('Prospective_prediction', [])
        for prediction in prospective_predictions:
            # Check if 'predicted_cases' exists, as not all entries have this field
            if 'predicted_cases' in prediction:
                rows.append({
                    'time_period': f"{prediction.get('year')}-{prediction.get('week')}",
                    'sample_0': prediction.get('predicted_cases'),
                    'sample_1': prediction.get('predicted_cases_lci'),
                    'sample_2': prediction.get('predicted_cases_uci'),
                    #'location': prediction.get('district'),
                    #'predicted_cases': prediction.get('predicted_cases'),
                    #'year': prediction.get('year'),
                    #'week': prediction.get('week')
                })

    # Write to CSV
    csv_file = out_csv
    with open(csv_file, mode='w', newline='') as file:
        #writer = csv.DictWriter(file, fieldnames=['district', 'predicted_cases', 'year', 'week'])
        writer = csv.DictWriter(file, fieldnames=['time_period', 'sample_0', 'sample_1', 'sample_2', 'location'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Data has been written to {csv_file}")


if __name__ == "__main__":
    #test_train()
    test_predict()
    """
    command = sys.argv[1]
    if command == "train":
        train(sys.argv[2], "demo_data/ewars_config.json", sys.argv[3])
    elif command == "predict":
        predict(sys.argv[2], "demo_data/ewars_config.json", sys.argv[3])
   train(historic_data, config_file, geojson)
    """
