"""One module per demo: a ``Spec`` class the take runner and exporters drive."""

import importlib


def load(demo: str):
    return importlib.import_module(f"demos.{demo}").Spec()
