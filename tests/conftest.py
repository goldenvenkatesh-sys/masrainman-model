import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from masrainman.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """The real config on a tiny 9 x 9 domain, with data in a temp folder."""
    c = copy.deepcopy(load_config())
    c["data_dir"] = str(tmp_path / "data")
    c["run_dir"] = str(tmp_path / "runs")
    c["domain"] = {"north": 2.0, "south": 0.0, "west": 0.0, "east": 2.0, "grid": 0.25, "boundary_width": 2}
    c["focus"] = {"north": 2.0, "south": 0.0, "west": 0.0, "east": 2.0}
    c["years"] = {"train": [2019, 2019], "val": [2019, 2019], "test": [2019, 2019]}
    return c
