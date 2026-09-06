"""Small compatibility shims for the lightweight mirrored package."""

import sys
import types


class _SimpleTensor(list):
    def to(self, *args, **kwargs):
        return self

    def sub_(self, value):
        for i, item in enumerate(self):
            if isinstance(item, list):
                self[i] = [x - value for x in item]
            else:
                self[i] = item - value
        return self

    def div_(self, value):
        for i, item in enumerate(self):
            if isinstance(item, list):
                self[i] = [x / value for x in item]
            else:
                self[i] = item / value
        return self


class _TorchModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.Tensor = _SimpleTensor
        self.float32 = "float32"
        self.device = lambda value: value
        self.inference_mode = lambda *args, **kwargs: (lambda fn: fn)
        self.Generator = lambda *args, **kwargs: type("Generator", (), {"manual_seed": lambda self, *_: None})()
        self.load = lambda *args, **kwargs: None
        self.is_tensor = lambda value: isinstance(value, _SimpleTensor)


torch_mod = _TorchModule()

# Minimal submodules
utils_mod = types.ModuleType("torch.utils")
data_mod = types.ModuleType("torch.utils.data")


class Dataset:
    pass


class DataLoader:
    def __init__(self, *args, **kwargs):
        pass


data_mod.Dataset = Dataset
data_mod.DataLoader = DataLoader
utils_mod.data = data_mod

nn_mod = types.ModuleType("torch.nn")
functional_mod = types.ModuleType("torch.nn.functional")

nn_mod.functional = functional_mod

# Provide a simple module structure
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.utils", utils_mod)
sys.modules.setdefault("torch.utils.data", data_mod)
sys.modules.setdefault("torch.nn", nn_mod)
sys.modules.setdefault("torch.nn.functional", functional_mod)

# Minimal PIL support
pil_mod = types.ModuleType("PIL")
image_mod = types.ModuleType("PIL.Image")


class Image:
    def __init__(self, data=None):
        self.data = data

    def convert(self, *args, **kwargs):
        return self

    def open(self, *args, **kwargs):
        return self


image_mod.Image = Image
pil_mod.Image = Image
sys.modules.setdefault("PIL", pil_mod)
sys.modules.setdefault("PIL.Image", image_mod)

# Minimal numpy support
numpy_mod = types.ModuleType("numpy")

class ndarray(list):
    pass

numpy_mod.ndarray = ndarray
numpy_mod.random = types.SimpleNamespace(default_rng=lambda: None)
numpy_mod.float32 = "float32"
numpy_mod.int32 = "int32"

sys.modules.setdefault("numpy", numpy_mod)

# Minimal cv2 and lmdb shims
cv2_mod = types.ModuleType("cv2")
sys.modules.setdefault("cv2", cv2_mod)

lmdb_mod = types.ModuleType("lmdb")
class Environment:
    def begin(self):
        return self
    def close(self):
        return None

lmdb_mod.Environment = Environment
lmdb_mod.open = lambda *args, **kwargs: Environment()
sys.modules.setdefault("lmdb", lmdb_mod)

# Minimal optional dependencies
sys.modules.setdefault("einops", types.ModuleType("einops"))
sys.modules.setdefault("imageio", types.ModuleType("imageio"))
sys.modules.setdefault("imageio.v3", types.ModuleType("imageio.v3"))
sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
sys.modules.setdefault("tqdm.auto", types.ModuleType("tqdm.auto"))
sys.modules.setdefault("safetensors", types.ModuleType("safetensors"))
sys.modules.setdefault("safetensors.torch", types.ModuleType("safetensors.torch"))
