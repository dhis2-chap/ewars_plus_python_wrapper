"""
"""
from main import train, predict


def test_train():
    historic_data = "demo_data/laos_dengue_and_diarrhea_surv_data_2015_2023.csv"
    geojson = "demo_data/laos_province_shapefile.GEOJSON"
    config_file = "demo_data/ewars_config.json"
    train(historic_data, config_file, geojson)



def test_predict():
    future_data = "laos_dengue_and_diarrhea_pros_data_2024.csv"
    geojson = "demo_data/laos_province_shapefile.GEOJSON"
    config_file = "demo_data/ewars_config.json"
    out_file = "demo_data/predictions.csv"
    predict(future_data, config_file, out_file)

if __name__ == "__main__":
    test_train()

