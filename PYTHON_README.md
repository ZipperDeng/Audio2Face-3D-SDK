# NVIDIA Audio2Face SDK Python package

After CUDA 12.8+, TensorRT, a C++ compiler, SDK dependencies, and models are available:

```powershell
pip install -e .
```

Convert a WAV file to a compressed vertex cache:

```python
from audio2face_sdk import audio_to_cache
cache = audio_to_cache("voice.wav", "voice_vertices.npz")
```

Or run `audio2face-cache voice.wav voice_vertices.npz`.

On first use the package downloads `nvidia/Audio2Face-3D-v3.0` into the
Hugging Face cache and builds `network.trt` for the selected GPU. Later calls
reuse both caches. Set `HF_HOME` to choose the cache root. The model is covered
by the NVIDIA Open Model License.

The `.npz` cache contains `skin_vertices` and `tongue_vertices` shaped
`(frames, vertices, 3)`, plus `jaw_transform`, `eyes_rotation`, `timestamps`,
`next_timestamps`, and `fps`. Pass `model_path=...`, use `--model`, or set
`A2F_MODEL_PATH` to override the automatically managed model.

TensorRT runtime libraries must be discoverable by the OS (`TENSORRT_ROOT_DIR/bin`
on Windows or the corresponding library path on Linux).
