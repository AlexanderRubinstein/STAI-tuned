import sys
import os
from types import ModuleType
import importlib
from itertools import chain


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
