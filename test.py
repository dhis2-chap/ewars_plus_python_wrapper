"""
"""
from main import predict_wrapper, train, predict


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
    #predict_wrapper(model_file_name, historic_data, future_data, config_file, out_file)
    predict(model_file_name, historic_data, future_data, config_file, out_file)



if __name__ == "__main__":
    #test_train()
    test_predict()

