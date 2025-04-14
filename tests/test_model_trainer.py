import pytest
from modules.model_trainer import ModelTrainer

# Sample test data
sample_data = {
    'features': [[1, 2], [2, 3], [3, 4], [4, 5]],
    'labels': [0, 1, 0, 1]
}

@pytest.fixture
def model_trainer():
    return ModelTrainer()

# Test cases

def test_train_model(model_trainer):
    trained_model = model_trainer.train(sample_data['features'], sample_data['labels'])
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')


def test_evaluate_model(model_trainer):
    evaluation_result = model_trainer.evaluate(sample_data['features'], sample_data['labels'])
    assert evaluation_result is not None
    assert 'accuracy' in evaluation_result