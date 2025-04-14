import pytest
from modules.data_processor import DataProcessor

# Sample test data
sample_data = {
    'temperature': [20, 21, 19, 22],
    'humidity': [30, 35, 33, 32],
    'pressure': [1012, 1013, 1011, 1014],
    'wind_speed': [5, 6, 5, 7]
}

@pytest.fixture
def data_processor():
    return DataProcessor()

# Test cases

def test_process_temperature(data_processor):
    processed_data = data_processor.process_temperature(sample_data)
    assert processed_data is not None
    assert 'temperature' in processed_data


def test_process_humidity(data_processor):
    processed_data = data_processor.process_humidity(sample_data)
    assert processed_data is not None
    assert 'humidity' in processed_data


def test_process_pressure(data_processor):
    processed_data = data_processor.process_pressure(sample_data)
    assert processed_data is not None
    assert 'pressure' in processed_data


def test_process_wind_speed(data_processor):
    processed_data = data_processor.process_wind_speed(sample_data)
    assert processed_data is not None
    assert 'wind_speed' in processed_data