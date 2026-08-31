import os
import socket

import pytest
from utils import *


def pytest_configure(config):
    if "PORT" in os.environ or "DEBUG_EXTERNAL" in os.environ:
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["PORT"] = str(sock.getsockname()[1])


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
