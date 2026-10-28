import ast
import __future__
import importlib
import inspect
import logging
import os
from pathlib import Path
import pkgutil
import sys
from types import ModuleType, SimpleNamespace
import pytest


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sglang_double
    from k2hv_sglang.check_runtime import check_source
    from k2hv_sglang.runtime import MODEL_ARCHITECTURE, REGISTRATION_ENVIRONMENT

    source = Path(sys.argv[1])
    check_source(source)
    patch = pytest.MonkeyPatch()
    sglang_double.install(patch, load_adapters=False)
    assert 'k2hv_sglang.k2_horizon_v' not in sys.modules
    assert 'k2hv_sglang.processor' not in sys.modules
    assert all(os.environ[name] == value for name, value in REGISTRATION_ENVIRONMENT.items())
    environ = ModuleType('sglang.srt.environ')
    environ.envs = SimpleNamespace(
        SGLANG_EXTERNAL_MODEL_PACKAGE=SimpleNamespace(get=lambda: os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']),
        SGLANG_DISABLED_MODEL_ARCHS=SimpleNamespace(get=lambda: []),
    )
    sys.modules[environ.__name__] = environ
    registry = ModuleType('sglang.srt.models.registry')
    sys.modules[registry.__name__] = registry
    filename = source / 'models/registry.py'
    exec(compile(filename.read_text(), str(filename), 'exec'), registry.__dict__)
    model = registry.ModelRegistry.models[MODEL_ARCHITECTURE]
    assert model.__module__ == 'k2hv_sglang.k2_horizon_v'

    filename = source / 'managers/multimodal_processor.py'
    tree = ast.parse(filename.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'import_processors')
    namespace = {'importlib': importlib, 'inspect': inspect, 'pkgutil': pkgutil,
                 'BaseMultimodalProcessor': sglang_double.BaseProcessor, 'PROCESSOR_MAPPING': {},
                 'logger': logging.getLogger('discovery')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    namespace['import_processors'](os.environ['SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE'])
    processor = namespace['PROCESSOR_MAPPING'][model]
    assert processor.__module__ == 'k2hv_sglang.processor'
    print('Pinned upstream discovered the model and processor from environment settings')


if __name__ == "__main__":
    main()
