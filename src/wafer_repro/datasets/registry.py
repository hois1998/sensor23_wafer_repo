from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wafer_repro.core.registry import Registry
from wafer_repro.datasets.base import DataBundle


DataModuleBuilder = Callable[[dict[str, Any]], DataBundle]

DATA_MODULE_REGISTRY: Registry[DataModuleBuilder] = Registry("data_module")


def _import_builtin_data_modules() -> None:
    # Import side effects register the built-in data module builders.
    import wafer_repro.datasets.image_folder.datamodule  # noqa: F401
    import wafer_repro.datasets.wm811k.datamodule  # noqa: F401


def create_data_bundle(module_name: str, config: dict[str, Any]) -> DataBundle:
    _import_builtin_data_modules()
    builder = DATA_MODULE_REGISTRY.get(module_name)
    return builder(config)


def registered_data_module_names() -> tuple[str, ...]:
    _import_builtin_data_modules()
    return DATA_MODULE_REGISTRY.keys()
