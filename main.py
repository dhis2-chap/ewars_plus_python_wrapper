import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# print logger level
print(logger.level)

import json
import csv
from pathlib import Path
import subprocess
import sys
import pandas as pd


import json

def add_district_to_geojson(filename):
    with open(filename) as f:
        data = json.load(f)

    for feat in data.get("features", []):
        props = feat.setdefault("properties", {})
        if "id" in props and "district" not in props:
            props["district"] = props["id"]

    with open(filename, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def run_command(command):
    logger.info(f"Running command {command}") 
    print("----------------------------------")
    print(f"Running command {command}")
    print("----------------------------------")
    subprocess.run(command, shell=True)
    # get output of command
    output = subprocess.check_output(command, shell=True)
    logger.info(f"Output: {output}")
    return output


def standardize_rainfall(train_data: pd.DataFrame):
    rainfall = train_data['rainfall'].values
    print("Rainfall")
    print(rainfall)
    # count number of nans in rainfall
    n_nans = sum(pd.isna(rainfall))
    print(f"Number of nans in rainfall: {n_nans}")
    # set nans to 0 in order to compute mean/std
    subset_for_mean_std = rainfall[~pd.isna(rainfall)]

    # standardize by subtracting the mean and dividing by the standard deviation
    mean = subset_for_mean_std.mean()
    std = subset_for_mean_std.std()
    print("Mean, std")
    print(mean, std)
    rainfall_standardized = (rainfall - mean) / std 

    return mean, std, rainfall_standardized



def train(historic_data, config_file, geojson_file, mode_file_name):
    add_district_to_geojson(geojson_file)
    # historic_data should be a csv that follows the chap format
    data = pd.read_csv(historic_data)
    required_columns = ["location", "mean_temperature", "rainfall", "disease_cases"]
    for col in required_columns:
        assert col in data.columns, f"Missing required column: {col}"

    mean, std, rainfall_standardized = standardize_rainfall(data)
    print("Rainfall standardized")
    print(rainfall_standardized)
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
         http://ewars_plus:3288/Ewars_run \
        -H 'accept: */*' \
        -F "csv_File=@{new_historic_data_file_name}" \
        -F "shape_File=@{geojson_file}" \
        -F "config_File=@{config_file}"
    """
    print(curl_command)

    logger.info(f"Executing command: {curl_command}")
    run_command(curl_command)


def _add_year_week_columns(data):
    # time_period is in format 2014-12-29/2015-01-04
    pass


def predict_wrapper(model_file_name, historic_data_file_name, future_data, config_file, out_file):
    """
    Since the Ewars model has "arbitrary" offset for each region depending on the lag found,
    we first run one predict to see what lag it uses and then adjust the data so
    that the it is offseted correctly so that predictions actually start where we want them to
    for each region.

    In the first run, we always send in more historic data than the maximum lag we think 
    the model will use. This means the predictions will start somewhere in the historic data (since the model
    always starts the the amount of weeks into the data that corresponds the the lag it found in step 1.
    Based on where this is, we adjust the historic data in the next run so that the predictions
    start exactly where the future data starts (which is what we want).

    Note: historic data has values for covariates and disease cases, future data has NaN for these.
    """

    historic_data = pd.read_csv(historic_data_file_name)
    historic_data = historic_data.groupby("location").tail(15)  # only keep the last 12 weeks, that is the maximum lag that can be found

    # write to csv
    historic_data_file_name = historic_data_file_name.replace(".csv", "_last_12_weeks.csv")
    historic_data.to_csv(historic_data_file_name, index=False)

    print("--- historic data ---")
    print(historic_data)

    print("--- future data ---")
    print(future_data)

    first_prediction = predict(model_file_name, historic_data_file_name, future_data, config_file, out_file)
    
    # for each region, find which weeks it actually gave predictions for
    regions = first_prediction["location"].unique()
    # for each region, find the first week it gave predictions for
    first_weeks = {}
    offsets = {}
    for region in regions:
        # the predictions are sorted, so that the first entry for that region will be the first week
        first_entry = first_prediction[first_prediction["location"] == region].iloc[0]
        historic_data_for_region = historic_data[historic_data["location"] == region]
        # find the row index of the historic_data_for_region that matches the first entry
        matching_row = historic_data_for_region[
            (historic_data_for_region["week"] == first_entry["week"]) &
            (historic_data_for_region["year"] == first_entry["year"])
        ]
        row_index = matching_row.index[0] if not matching_row.empty else None
        if row_index == None:
            print(f"Warning: No matching row found for region {region} in historic data for week {first_entry['week']} and year {first_entry['year']}")
            print("Historic data:")
            print(historic_data_for_region)
            continue
        # get row number, not index instead
        row_number = historic_data_for_region.index.get_loc(row_index)
        first_weeks[region] = row_number
        # find the offset, which is how many weeks are before this in the historic data (this is the lag the model found for this region)
        #index_of_first_week = historic_data_for_region.index[0]
        offsets[region] = row_number  # row_index - index_of_first_week

    print("Offsets for each region:", offsets)
    
    # change historic data so that each region starts at the offset
    new_historic_data = []
    for region in regions:
        region_data = historic_data[historic_data["location"] == region]
        # get the offset for this region
        offset = offsets[region]
        # get the data offset back from the end
        region_data = region_data.iloc[-offset:]
        # add the region data to the new_historic_data
        new_historic_data.append(region_data)
    
    # concat the new_historic_data
    new_historic_data = pd.concat(new_historic_data)
    # write the new historic data to a file
    new_historic_data_file_name = historic_data_file_name.replace(".csv", "_adjusted.csv")
    new_historic_data.to_csv(new_historic_data_file_name, index=False) 

    # now call predict with this adjusted historic data
    results = predict(model_file_name, new_historic_data_file_name, future_data, config_file, out_file)


def predict(model_file_name, historic_data, future_data, config_file, out_file):
    # future_data should be a csv that follows the chap format"""
    required_columns = ["location", "mean_temperature", "rainfall", "disease_cases"]
    # standardize rainfall
    d1 = pd.read_csv(historic_data)
    # the model needs the n_lags last rows from the historic data, where n_lags is configure in the config file
    #d1 = d1.iloc[-9:]
    logger.info("Historic data")
    logger.info(d1)

    d2 = pd.read_csv(future_data)
    # get number of weeks/months to predict as the number of entries in the future data for a single location
    d2_first_location = d2[d2["location"] == d2["location"].unique()[0]]
    logging.info(f"There are {len(d2_first_location)} weeks to predict for the first location. Assuming the same for all.")
    n_to_predict = len(d2_first_location)
    # concat historic and future data
    data = pd.concat([d1, d2])
    # sort data on location, year, week
    data = data.sort_values(["location", "year", "week"])
    print("Full data")
    print(data)

    rainfall = data['rainfall'].values
    model = json.loads(open(model_file_name).read())
    rainfall_std = rainfall - model["rainfall_mean"] / model["rainfall_std"]
    # add this column to data
    data["rainfall_std"] = rainfall_std
    new_future_data_file_name = future_data.replace(".csv", "_std.csv")

    # add disease_cases column if not in data
    if "disease_cases" not in data.columns:
        data["disease_cases"] = 0

    # change location to district (which is the name ewars uses)
    data = data.rename(columns={"location": "district"})

    logger.info("Writing new future data to " + new_future_data_file_name)
    data.to_csv(new_future_data_file_name, index=False)

    print("--- data sent to ewars ---")
    print(data)

    curl_command = f"""curl -X POST http://ewars_plus:3288/Ewars_predict \
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
    curl_command = f"curl -o {out_file_json} http://ewars_plus:3288/retrieve_predicted_cases"
    output = run_command(curl_command)
    df = change_prediction_format_to_chap(out_file_json, out_file, n_to_predict=n_to_predict)
    df.to_csv(out_file, index=False)
    
    print("--- df returned from ewars ---")
    print(df)

    return df
    
    # keep only those week and years that are in the future data
    # the model gives more
    print("d2")
    print(d2)
    print("---- df")
    print(df)
    df = df.merge(d2[['year', 'week']].drop_duplicates(), on=['year', 'week'], how='inner')
    assert len(df) == len(d2), f"Length of predictions {len(df)} does not match length of future data {len(d2)}"
    df.to_csv(out_file, index=False)





def test_train():
    historic_data = "demo_data/laos_dengue_and_diarrhea_surv_data_2015_2023.csv"
    historic_data = "demo_data/subset_train_chap.csv"
    geojson = "demo_data/laos_province_shapefile.GEOJSON"
    config_file = "demo_data/ewars_config.json"
    model_file_name = "demo_data/model.json"
    train(historic_data, config_file, geojson, model_file_name)


def test_predict():
    #future_data = "laos_dengue_and_diarrhea_pros_data_2024.csv"
    #future_data = "demo_data/subset_predict_chap.csv"
    historic_data = "demo_data/historic_data.csv"
    future_data = "demo_data/future_data.csv"
    config_file = "demo_data/ewars_config.json"
    out_file = "demo_data/predictions.csv"
    model_file_name = "demo_data/model.json"
    predict(model_file_name, historic_data, future_data, config_file, out_file)

 
def change_prediction_format_to_chap(predictions_json, out_csv, n_to_predict):
    assert type(predictions_json) == str, "predictions_json should be a string"
    json_data = json.loads(open(predictions_json).read())

    print(type(json_data))
    print(json_data)

    # Extract relevant data
    rows = []
    for item in json_data:
        prospective_predictions = item.get('Prospective_prediction', [])
        for prediction in prospective_predictions:
            # Check if 'predicted_cases' exists, as not all entries have this field
            if 'predicted_cases' in prediction:
                rows.append({
                    'time_period': f"{prediction.get('year')}W{prediction.get('week')}",
                    'sample_0': prediction.get('predicted_cases'),
                    'sample_1': prediction.get('predicted_cases_lci'),
                    'sample_2': prediction.get('predicted_cases_uci'),
                    'location': prediction['district'],
                    #'predicted_cases': prediction.get('predicted_cases'),
                    'year': prediction.get('year'),
                    'week': prediction.get('week')
                })

    df = pd.DataFrame(rows)
    # only keep the first n_to_predict rows for each location 
    filtered = df.groupby('location').head(n_to_predict)

    # return as dataframe
    return filtered

    # Write to CSV
    csv_file = out_csv
    with open(csv_file, mode='w', newline='') as file:
        #writer = csv.DictWriter(file, fieldnames=['district', 'predicted_cases', 'year', 'week'])
        writer = csv.DictWriter(file, fieldnames=['time_period', 'sample_0', 'sample_1', 'sample_2', 'location'])
        writer.writeheader()
        writer.writerows(rows[:n_to_predict])
        #writer.writerows(rows)

    print(f"Data has been written to {csv_file}")


if __name__ == "__main__":
    #test_train()
    
    #test_predict()
    #sys.exit()

    command = sys.argv[1]
    if command == "train":
        print("Training")
        #train(sys.argv[2], "demo_data/ewars_config.json", sys.argv[3], sys.argv[4])
        pass
    elif command == "predict":
        train(sys.argv[3], "demo_data/ewars_config.json", sys.argv[6], sys.argv[2])
        predict_wrapper(sys.argv[2], sys.argv[3], sys.argv[4], "demo_data/ewars_config.json", sys.argv[5])
    else:
        print(f"Command {command} not recognized")
        assert False

   #train(historic_data, config_file, geojson)
