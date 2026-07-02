"""Data paths must be absolute (project-root based) so any CWD works."""
import os

from backend.config import settings


def test_data_paths_are_absolute():
    assert os.path.isabs(settings.DATA_DIR)
    assert os.path.isabs(settings.PERSONAS_DIR)
    assert os.path.isabs(settings.SEEDS_DIR)


def test_data_dir_points_at_the_real_project_data_folder():
    assert os.path.isdir(settings.DATA_DIR)
    assert os.path.isdir(settings.PERSONAS_DIR)
