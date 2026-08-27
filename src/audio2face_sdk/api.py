from __future__ import annotations

import os
import json
import threading
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def find_default_model() -> Path:
    """Resolve an explicit model, otherwise prepare the default Hugging Face model."""
    configured = os.environ.get("A2F_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"A2F_MODEL_PATH does not point to a file: {path}")
    from .models import prepare_default_model
    return prepare_default_model()


def load_audio(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a WAV file as 16 kHz mono float32 samples."""
    sample_rate, samples = wavfile.read(Path(path))
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        samples = samples.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        samples = samples.astype(np.float32)
    if sample_rate != 16_000:
        divisor = np.gcd(sample_rate, 16_000)
        samples = resample_poly(samples, 16_000 // divisor, sample_rate // divisor)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _resolve_identity(model_path: Path, identity: str) -> tuple[int, str]:
    if not isinstance(identity, str) or not identity.strip():
        raise TypeError("identity must be a non-empty actor name")
    with model_path.open(encoding="utf-8") as stream:
        model_config = json.load(stream)
    network_info_path = model_path.parent / model_config["networkInfoPath"]
    with network_info_path.open(encoding="utf-8") as stream:
        identities = json.load(stream)["params"]["identities"]
    requested = identity.strip().casefold()
    for index, actor in enumerate(identities):
        if actor.casefold() == requested:
            return index, actor
    choices = ", ".join(identities)
    raise ValueError(f"Unknown identity {identity!r}. Available identities: {choices}")


class Audio2FaceSession:
    """Reusable Audio2Face executor for multiple clips with one model and identity."""

    def __init__(self, *, model_path: str | os.PathLike[str] | None = None,
                 device_id: int = 0, identity: str = "Claire",
                 constant_noise: bool = False) -> None:
        from ._native import Executor

        if model_path:
            self.model_path = Path(model_path).expanduser().resolve()
        else:
            from .models import prepare_default_model
            self.model_path = prepare_default_model(device_id=device_id)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model does not exist: {self.model_path}")
        self.identity_index, self.identity = _resolve_identity(self.model_path, identity)
        self.device_id = device_id
        self.constant_noise = constant_noise
        self._lock = threading.Lock()
        self._frames: dict[str, list[np.ndarray]] | None = None
        self._timestamps: list[int] | None = None
        self._next_timestamps: list[int] | None = None
        self._executor = Executor(self._collect, str(self.model_path), device_id,
                                  self.identity_index, constant_noise)

    def _collect(self, result) -> None:
        if self._frames is None or self._timestamps is None or self._next_timestamps is None:
            raise RuntimeError("Audio2Face returned a result outside an active inference")
        self._timestamps.append(result.timestamp_current_frame)
        self._next_timestamps.append(result.timestamp_next_frame)
        for name in self._frames:
            self._frames[name].append(np.array(getattr(result, name), copy=True))

    def infer_audio(self, audio_path: str | os.PathLike[str], *,
                    chunk_size: int = 16_000) -> dict[str, np.ndarray]:
        """Infer one WAV clip while retaining the initialized native executor."""
        return self.infer_samples(load_audio(audio_path), chunk_size=chunk_size)

    def infer_samples(self, samples: np.ndarray, *, chunk_size: int = 16_000) -> dict[str, np.ndarray]:
        """Infer one 16 kHz mono float32 clip."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Audio2FaceSession does not support concurrent inference")
        self._frames = {name: [] for name in (
            "skin_vertices", "tongue_vertices", "jaw_transform", "eyes_rotation"
        )}
        self._timestamps, self._next_timestamps = [], []
        try:
            self._executor.start_execution()
            for start in range(0, values.size, chunk_size):
                self._executor.accumulate_audio(values[start:start + chunk_size])
            self._executor.end_execution()
            if not self._timestamps:
                raise RuntimeError("Audio2Face produced no animation frames")
            timestamps = np.asarray(self._timestamps, dtype=np.int64)
            next_timestamps = np.asarray(self._next_timestamps, dtype=np.int64)
            frame_step = float(np.median(next_timestamps - timestamps))
            if frame_step <= 0:
                raise RuntimeError("Audio2Face produced invalid frame timestamps")
            return {
                "timestamps": timestamps, "next_timestamps": next_timestamps,
                "timestamp_sample_rate": np.asarray(16_000, dtype=np.int32),
                "fps": np.asarray(round(16_000 / frame_step), dtype=np.int32),
                "identity": np.asarray(self.identity),
                "identity_index": np.asarray(self.identity_index, dtype=np.int32),
                **{name: np.stack(frames) for name, frames in self._frames.items()},
            }
        finally:
            if self._executor.is_active():
                self._executor.end_execution()
            self._frames = self._timestamps = self._next_timestamps = None
            self._lock.release()

    def infer_batch(self, audio_paths: Iterable[str | os.PathLike[str]], *,
                    chunk_size: int = 16_000) -> Iterator[dict[str, np.ndarray]]:
        """Infer WAV clips sequentially with this session's initialized executor."""
        for path in audio_paths:
            yield self.infer_audio(path, chunk_size=chunk_size)


def audio_to_cache(
    audio_path: str | os.PathLike[str], output_path: str | os.PathLike[str], *,
    model_path: str | os.PathLike[str] | None = None, device_id: int = 0,
    identity: str = "Claire", constant_noise: bool = False, chunk_size: int = 16_000,
) -> Path:
    """Run Audio2Face and write a compressed NumPy vertex cache (.npz)."""
    result = audio_to_vertices(
        audio_path, model_path=model_path, device_id=device_id, identity=identity,
        constant_noise=constant_noise, chunk_size=chunk_size,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    return destination


def audio_to_vertices(
    audio_path: str | os.PathLike[str], *,
    model_path: str | os.PathLike[str] | None = None, device_id: int = 0,
    identity: str = "Claire", constant_noise: bool = False, chunk_size: int = 16_000,
) -> dict[str, np.ndarray]:
    """Run Audio2Face and return animation arrays in memory without cache I/O."""
    session = Audio2FaceSession(model_path=model_path, device_id=device_id,
                                identity=identity, constant_noise=constant_noise)
    return session.infer_audio(audio_path, chunk_size=chunk_size)
