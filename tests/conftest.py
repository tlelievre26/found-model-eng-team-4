import pytest


def pytest_addoption(parser):
    parser.addoption("--run-real-data", action="store_true", default=False,
                     help="also run tests marked real_data (they need the built splits and network access, and are slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-real-data"):
        return
    skip = pytest.mark.skip(reason="needs the real data; run with --run-real-data")
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip)
