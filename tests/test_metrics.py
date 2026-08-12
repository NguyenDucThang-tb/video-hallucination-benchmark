from src.evaluation.metrics import accuracy, macro_average


def test_macro_average_ignores_missing_tasks():
    assert macro_average({"a": 0.5, "b": None, "c": 1.0}) == 0.75
    assert macro_average({"a": None}) is None


def test_missing_prediction_is_not_valid():
    score, n, correct = accuracy([True, False, None])
    assert (score, n, correct) == (0.5, 2, 1)
