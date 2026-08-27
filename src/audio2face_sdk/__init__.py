"""Audio file to Audio2Face vertex-animation cache."""
import os
import sys

_dll_directories = []
if sys.platform == "win32":
    for _root, _child in ((os.environ.get("CUDA_PATH"), "bin"),
                          (os.environ.get("TENSORRT_ROOT_DIR"), "bin")):
        if _root and os.path.isdir(_path := os.path.join(_root, _child)):
            _dll_directories.append(os.add_dll_directory(_path))

from .api import Audio2FaceSession, audio_to_cache, audio_to_vertices, find_default_model, load_audio
from .models import (
    IdentityNeutralModel, IdentitySkinModel, load_identity_neutral,
    load_identity_skin_model, prepare_identity_model,
)

__all__ = [
    "Audio2FaceSession", "IdentityNeutralModel", "IdentitySkinModel", "audio_to_cache",
    "audio_to_vertices", "find_default_model",
    "load_audio", "load_identity_neutral", "load_identity_skin_model",
    "prepare_identity_model",
]
__version__ = "0.1.0"
