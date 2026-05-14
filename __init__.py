import sys, os, importlib.util, importlib.machinery

# Inject PyTorch nvdiffrast shim before any node imports
_shim_pkg = os.path.join(os.path.dirname(__file__), '_bruno_nvdiffrast')
if 'nvdiffrast' not in sys.modules:
    _nvdiffrast_mod = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec('nvdiffrast', None, is_package=True)
    )
    _nvdiffrast_mod.__path__ = [_shim_pkg]
    _nvdiffrast_mod.__package__ = 'nvdiffrast'
    sys.modules['nvdiffrast'] = _nvdiffrast_mod

    _shim_torch_spec = importlib.util.spec_from_file_location(
        'nvdiffrast.torch',
        os.path.join(_shim_pkg, 'torch.py'),
    )
    _nvdiffrast_torch = importlib.util.module_from_spec(_shim_torch_spec)
    _shim_torch_spec.loader.exec_module(_nvdiffrast_torch)
    _nvdiffrast_mod.torch = _nvdiffrast_torch
    sys.modules['nvdiffrast.torch'] = _nvdiffrast_torch

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]