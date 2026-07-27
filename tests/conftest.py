from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(name: str) -> pd.DataFrame:
        return pd.read_csv(FIXTURES_DIR / name)

    return _load
