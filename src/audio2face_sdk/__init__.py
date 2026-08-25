"""Audio file to Audio2Face vertex-animation cache."""
import os
import sys

_dll_directories = []
if sys.platform == "win32":
    for _root, _child in ((os.environ.get("CUDA_PATH"), "bin"),
                          (os.environ.get("TENSORRT_ROOT_DIR"), "bin")):
        if _root and os.path.isdir(_path := os.path.join(_root, _child)):
            _dll_directories.append(os.add_dll_directory(_path))

from .api import audio_to_cache, find_default_model, load_audio

__all__ = ["audio_to_cache", "find_default_model", "load_audio"]
__version__ = "0.1.0"
