from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

DEFAULT_MODEL_REPO = "nvidia/Audio2Face-3D-v3.0"
_REQUIRED_FILES = (
    "model.json", "network.onnx", "network_info.json", "trt_info.json",
    "model_config_Claire.json", "model_config_James.json", "model_config_Mark.json",
    "model_data_Claire.npz", "model_data_James.npz", "model_data_Mark.npz",
)


def _model_cache_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "audio2face_sdk" / "Audio2Face-3D-v3.0"


def _find_trtexec() -> str:
    executable = "trtexec.exe" if os.name == "nt" else "trtexec"
    root = os.environ.get("TENSORRT_ROOT_DIR")
    if root:
        candidate = Path(root) / "bin" / executable
        if candidate.is_file():
            return str(candidate)
    candidate = shutil.which("trtexec")
    if candidate:
        return candidate
    raise RuntimeError(
        "TensorRT trtexec was not found. Set TENSORRT_ROOT_DIR or add trtexec to PATH."
    )


def _trt_arguments(info_path: Path) -> list[str]:
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    defaults = info.get("defaults", {})
    return [argument.format(**defaults)
            for arguments in info["trt_build_param"].values()
            for argument in arguments]


def _download_model(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=DEFAULT_MODEL_REPO,
            local_dir=destination,
            allow_patterns=list(_REQUIRED_FILES),
        )
    except (HfHubHTTPError, EntryNotFoundError) as error:
        raise RuntimeError(
            f"Could not download {DEFAULT_MODEL_REPO} from Hugging Face. "
            "Check network access and HF_TOKEN if authentication is required."
        ) from error


def _build_engine(model_dir: Path, device_id: int) -> None:
    temporary_dir = Path(tempfile.mkdtemp(prefix="a2f-trt-"))
    temporary_engine = temporary_dir / "network.trt"
    command = [
        _find_trtexec(), f"--onnx={model_dir / 'network.onnx'}",
        f"--saveEngine={temporary_engine}", f"--device={device_id}",
        *_trt_arguments(model_dir / "trt_info.json"),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary_engine.is_file():
            raise RuntimeError("trtexec completed without producing network.trt")
        shutil.move(str(temporary_engine), str(model_dir / "network.trt"))
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"TensorRT engine generation failed with exit code {error.returncode}") from error
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def prepare_default_model(*, device_id: int = 0) -> Path:
    """Download the default geometry model and build its TensorRT engine once."""
    model_dir = _model_cache_dir()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(model_dir.parent / ".Audio2Face-3D-v3.0.lock"):
        if not all((model_dir / name).is_file() for name in _REQUIRED_FILES):
            _download_model(model_dir)
        if not (model_dir / "network.trt").is_file():
            _build_engine(model_dir, device_id)
    return model_dir / "model.json"
