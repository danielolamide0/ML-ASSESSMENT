import pandas as pd
import pytest

from severity_triage import config as C


@pytest.fixture(scope="session")
def raw_train() -> pd.DataFrame:
    return pd.read_csv(C.TRAIN_PATH)


@pytest.fixture(scope="session")
def raw_holdback() -> pd.DataFrame:
    return pd.read_csv(C.HOLDBACK_PATH)
