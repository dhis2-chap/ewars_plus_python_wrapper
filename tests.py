from main import change_prediction_format_to_chap


def test_change_prediction_format_to_chap():
    predictions = "demo_data/test_predictions.csv"
    change_prediction_format_to_chap(predictions, "test.csv")
 

if __name__ == "__main__":
    test_change_prediction_format_to_chap()