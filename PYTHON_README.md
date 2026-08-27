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

Select an actor by name with `identity="Claire"`, `"James"`, or `"Mark"`:

```python
cache = audio_to_cache("voice.wav", "voice_vertices.npz", identity="James")
```

For processing pipelines, avoid the large compressed cache and keep vertices in memory:

```python
from audio2face_sdk import audio_to_vertices
animation = audio_to_vertices("voice.wav", identity="Claire", device_id=0)
```

For a batch, initialize the native model once and reuse the session:

```python
from audio2face_sdk import Audio2FaceSession

session = Audio2FaceSession(identity="Claire", device_id=0)
for animation in session.infer_batch(["one.wav", "two.wav", "three.wav"]):
    consume(animation)
```

An `Audio2FaceSession` is bound to one model, GPU and identity. Calls are sequential;
create separate sessions if different actors are needed.

Neutral identity geometry used by downstream mesh fitting is exposed without
requiring callers to inspect Hugging Face snapshot directories:

```python
from audio2face_sdk import load_identity_neutral

neutral = load_identity_neutral("James")
vertices = neutral.vertices
frontal_mask = neutral.frontal_mask
```

Calibration tools that also need the actor blendshapes can call
`load_identity_skin_model("James")`; it returns `neutral`, `frontal_mask`,
`pose_names`, and absolute `pose_vertices` without exposing cache layout.

Or run `audio2face-cache voice.wav voice_vertices.npz --identity James`.

On first use the package downloads `nvidia/Audio2Face-3D-v3.0` into the
Hugging Face cache and builds `network.trt` for the selected GPU. Later calls
reuse both caches. Set `HF_HOME` to choose the cache root. The model is covered
by the NVIDIA Open Model License.

The `.npz` cache contains `skin_vertices` and `tongue_vertices` shaped
`(frames, vertices, 3)`, plus `jaw_transform`, `eyes_rotation`, `timestamps`,
`next_timestamps`, and `fps`. Pass `model_path=...`, use `--model`, or set
`A2F_MODEL_PATH` to override the automatically managed model.

`timestamps` are sample positions at `timestamp_sample_rate` (16 kHz). `fps`
is derived from consecutive SDK timestamps rather than hard-coded.

TensorRT runtime libraries must be discoverable by the OS (`TENSORRT_ROOT_DIR/bin`
on Windows or the corresponding library path on Linux).
