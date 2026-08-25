from __future__ import annotations

import os
from pathlib import Path
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


def audio_to_cache(
    audio_path: str | os.PathLike[str], output_path: str | os.PathLike[str], *,
    model_path: str | os.PathLike[str] | None = None, device_id: int = 0,
    identity_index: int = 0, constant_noise: bool = False, chunk_size: int = 16_000,
) -> Path:
    """Run Audio2Face and write a compressed NumPy vertex cache (.npz)."""
    from ._native import Executor

    if model_path:
        model = Path(model_path).expanduser().resolve()
    else:
        from .models import prepare_default_model
        model = prepare_default_model(device_id=device_id)
    if not model.is_file():
        raise FileNotFoundError(f"Model does not exist: {model}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    frames: dict[str, list[np.ndarray]] = {
        "skin_vertices": [], "tongue_vertices": [], "jaw_transform": [], "eyes_rotation": []
    }
    timestamps: list[int] = []
    next_timestamps: list[int] = []

    def collect(result) -> None:
        timestamps.append(result.timestamp_current_frame)
        next_timestamps.append(result.timestamp_next_frame)
        for name in frames:
            frames[name].append(np.array(getattr(result, name), copy=True))

    executor = Executor(collect, str(model), device_id, identity_index, constant_noise)
    samples = load_audio(audio_path)
    executor.start_execution()
    for start in range(0, samples.size, chunk_size):
        executor.accumulate_audio(samples[start:start + chunk_size])
    executor.end_execution()
    if not timestamps:
        raise RuntimeError("Audio2Face produced no animation frames")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, timestamps=np.asarray(timestamps, dtype=np.int64),
        next_timestamps=np.asarray(next_timestamps, dtype=np.int64), fps=np.asarray(30, dtype=np.int32),
        **{name: np.stack(values) for name, values in frames.items()},
    )
    return destination
