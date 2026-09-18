"""Archived results remain locatable; new writes never recreate old roots."""
from pathlib import Path
import pytest
from src.framework.run_paths import run_directories

def test_archive_lookup(tmp_path):
    old = tmp_path / "legacy/log/old"
    old.mkdir(parents=True)
    assert run_directories("old", tmp_path) == (old, tmp_path / "legacy/output/old")

def test_new_lookup(tmp_path):
    assert run_directories("new", tmp_path) == (tmp_path/"runs/new", tmp_path/"runs/new")
    assert not (tmp_path/"log").exists()

def test_ambiguous_history_rejected(tmp_path):
    (tmp_path/"runs/duplicate").mkdir(parents=True)
    (tmp_path/"legacy/log/duplicate").mkdir(parents=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        run_directories("duplicate", tmp_path)

@pytest.mark.parametrize("value", ["", ".", "..", "../escape", "a/b", "a\\b"])
def test_invalid_run_id(value, tmp_path):
    with pytest.raises(ValueError):
        run_directories(value, tmp_path)
