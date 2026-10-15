"""Use inherited Windows workspace ACLs for test scratch directories.

Python 3.12's mode=0700 tempfile directories exclude the desktop sandbox SID on
this Windows host. Ordinary mkdir inherits the writable workspace permissions.
"""
from pathlib import Path
import shutil
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path():
    root = (Path(__file__).parent / ".tmp").resolve()
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError("Refusing cleanup outside test scratch root")
        shutil.rmtree(path)
