import sys
import os
from types import ModuleType
import importlib
from itertools import chain
import copy


# local modules
from .utils import (
    error_or_print,
    get_nested_attr
)


PACKAGE_SEPARATOR = "."


FROM_CLASS_KEY = "from_class"


class LazyModuleWrapper(ModuleType):

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        spec = importlib.util.find_spec(self.module_name)
        self.default_attrs = {
            "__spec__": spec,
            "__name__": spec.name
        }

    def try_to_import(self):
        if self.module is None:
            sys.modules.pop(self.module_name)
            self.module = importlib.import_module(self.module_name)
            sys.modules[self.module_name] = self

    def __getattr__(self, name: str):

        if name in self.default_attrs:
            return self.default_attrs[name]

        if not is_bulitin_name(name):
            self.try_to_import()

        if self.module is not None:
            return getattr(self.module, name)

        return None


def is_bulitin_name(name):
    return len(name) > 4 and name[:2] == "__" and name[-2:] == "__"


def importlib_lazy_import(name):
    spec = importlib.util.find_spec(name)
    lazy_loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = lazy_loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    return module


def lazy_import(name):
    module = LazyModuleWrapper(name)
    sys.modules[name] = module
    return module


# taken from: https://github.com/huggingface/transformers/blob/e218249b02465ec8b6029f201f2503b9e3b61feb/src/transformers/file_utils.py#L1945
class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str):
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


def make_lazy_module(
    name,
    module_file,
    import_structure,
    module_spec=None,
    extra_objects=None
):
    return _LazyModule(
        name,
        module_file,
        import_structure,
        module_spec,
        extra_objects
    )


def pop_all_modules_by_filter(filter_condition, filter_name=None, logger=None):
    imported_modules = sorted(list(filter(filter_condition, sys.modules.keys())))
    if filter_name is None:
        filter_name = " AND ".join(
            set.intersection(
                *[set(name.split(PACKAGE_SEPARATOR)) for name in imported_modules]
            )
        )
    if len(imported_modules) > 0:
        error_or_print(
            f"Removing all modules satisfying filter \"{filter_name}\" "
            f"from sys.modules",
            logger
        )
    for m in imported_modules:
        sys.modules.pop(m)
    return imported_modules


# https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L88
def import_from_string(string, reload=False, nested_attrs_depth=1):

    module_path_and_attrs = string.rsplit(PACKAGE_SEPARATOR, nested_attrs_depth)

    module_path = module_path_and_attrs[0]

    nested_attrs = module_path_and_attrs[1:]

    module = importlib.import_module(module_path)

    if reload:
        importlib.reload(module)

    if len(nested_attrs) > 0:
        module = get_nested_attr(
            module,
            nested_attrs
        )

    return module


def make_from_class_ctor(from_class_config, pos_args_list=[]):

    assert "class" in from_class_config
    class_ctor = import_from_string(from_class_config["class"])
    kwargs = from_class_config.get("kwargs", {})
    importable_kwargs = from_class_config.get("kwargs_to_import", {})

    importable_kwargs = update_enums_in_config(
        importable_kwargs,
        importable_kwargs.keys()
    )

    return class_ctor(
        *pos_args_list,
        **(kwargs | importable_kwargs)
    )


def update_enums_in_config(config, enums, nested_attrs_depth=2):
    config = copy.deepcopy(config)
    for enum in enums:
        if enum in config:
            assert isinstance(config[enum], str), \
                f"Please convert \"{enum}\" to string in config:\n{config}"
            config[enum] = import_from_string(
                config[enum],
                nested_attrs_depth=nested_attrs_depth
            )
    return config
