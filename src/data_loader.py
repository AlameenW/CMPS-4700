import pandas as pd
from pathlib import Path

# Base Directory Config
BASE_INPUT_PATH = Path("INPUT")
TRAIN_PATH = BASE_INPUT_PATH / "TRAIN" / "train.csv"
TEST_PATH = BASE_INPUT_PATH / "TEST" / "test.csv"

# Train Data Loader
def load_train_data():
    """
    Loads training dataset from INPUT/TRAIN directory
    """

    # Read train file
    train_dataframe = pd.read_csv(TRAIN_PATH)
    return train_dataframe

# #

# Test Data Loader
def load_test_data():
    """
    Loads test dataset from INPUT/TEST directory
    """

    # Read test file
    test_dataframe = pd.read_csv(TEST_PATH)
    return test_dataframe

# #