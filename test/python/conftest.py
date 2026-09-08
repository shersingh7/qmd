import os
import pytest

# Enforce offline Hugging Face / Transformers operation during test runs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-models",
        action="store_true",
        default=False,
        help="Run tests that instantiate real MLX models / Metal weights",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_model: Mark test as requiring real MLX model downloads or resident Metal weights",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-real-models"):
        # Explicit opt-in: run real model tests
        return

    skip_real = pytest.mark.skip(
        reason="Real model integration test skipped by default. Pass --run-real-models to run."
    )
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip_real)
