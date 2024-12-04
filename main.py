import subprocess
import sys
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run_command(command):
    subprocess.run(command, shell=True)
    # get output of command
    output = subprocess.check_output(command, shell=True)
    logger.info(f"Output: {output}")
    return output


def train(historic_data, config_file, geojson_file):
    curl_command = f"""curl -X POST \
        http://127.0.0.1:3288/Ewars_run \
        -H 'accept: */*' \
        -F "csv_File=@{historic_data}" \
        -F "shape_File=@{geojson_file}" \
        -F "config_File=@{config_file}"
    """

    logger.info(f"Executing command: {curl_command}")
    run_command(curl_command)


def predict(future_data, config_file, out_file):
    curl_command = f"""curl -X POST http://127.0.0.1:3288/Ewars_predict \
        -H 'accept: */*' \
        -F "pros_csv_File=@{future_data} \
        -F "config_File=@{config_file}" \
    """
    logger.info(f"Executing command: {curl_command}")
    
    # get predictions
    curl_command = f"curl http://127.0.0.1:3288/retrieve_predicted_cases"
    predictions = run_command(curl_command)

    with open(out_file, "w") as f:
        f.write(predictions)
    
