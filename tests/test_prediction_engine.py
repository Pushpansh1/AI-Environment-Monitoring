import pytest
from modules.prediction_engine import PredictionEngine

# Sample test data
sample_data = {
    'features': [[1, 2], [2, 3], [3, 4], [4, 5]],
    'labels': [0, 1, 0, 1]
}

@pytest.fixture
def prediction_engine():
    return PredictionEngine()

# Test cases

def test_predict(prediction_engine):
    predictions = prediction_engine.predict(sample_data['features'])
    assert predictions is not None
    assert len(predictions) == len(sample_data['features'])


def test_accuracy(prediction_engine):
    accuracy = prediction_engine.calculate_accuracy(sample_data['features'], sample_data['labels'])
    assert accuracy is not None
    assert 0 <= accuracy <= 1