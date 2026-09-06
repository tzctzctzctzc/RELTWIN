from score_interval_predictions import select_prediction


def test_select_prediction_accepts_nova_router_output():
    row = {
        "incumbent_prediction": [[1.0, 2.0]],
        "selected_prediction": [[1.2, 2.2]],
    }
    assert select_prediction(row, 10.0) == [(1.2, 2.2)]


def test_select_prediction_keeps_base_prediction_precedence():
    row = {
        "prediction": [[1.0, 2.0]],
        "selected_prediction": [[3.0, 4.0]],
    }
    assert select_prediction(row, 10.0) == [(1.0, 2.0)]
