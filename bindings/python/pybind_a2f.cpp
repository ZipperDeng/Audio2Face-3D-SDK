// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include <cuda_runtime.h>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct PyGeometryResults {
  std::size_t track_index{};
  std::int64_t timestamp_current_frame{}, timestamp_next_frame{};
  py::array_t<float> skin_vertices, tongue_vertices, jaw_transform, eyes_rotation;
};

class PyAudio2FaceExecutor {
 public:
  PyAudio2FaceExecutor(std::function<void(const PyGeometryResults&)> callback,
                       const std::string& model_path, int device_id = 0,
                       std::size_t identity_index = 0, bool constant_noise = false)
      : callback_(std::move(callback)) {
    check(nva2x::SetCudaDeviceIfNeeded(device_id), "set CUDA device");
    bundle_.reset(nva2f::ReadDiffusionGeometryExecutorBundle(
        1, model_path.c_str(), nva2f::IGeometryExecutor::ExecutionOption::All,
        identity_index, constant_noise, nullptr));
    if (!bundle_) throw std::runtime_error("Failed to load Audio2Face model: " + model_path);
    auto cb = [](void* userdata, const nva2f::IGeometryExecutor::Results& value) -> bool {
      auto* self = static_cast<PyAudio2FaceExecutor*>(userdata);
      auto skin = copy(value.skinGeometry, value.skinCudaStream);
      auto tongue = copy(value.tongueGeometry, value.tongueCudaStream);
      auto jaw = copy(value.jawTransform, value.jawCudaStream);
      auto eyes = copy(value.eyesRotation, value.eyesCudaStream);
      py::gil_scoped_acquire acquire;
      PyGeometryResults result;
      result.track_index = value.trackIndex;
      result.timestamp_current_frame = value.timeStampCurrentFrame;
      result.timestamp_next_frame = value.timeStampNextFrame;
      result.skin_vertices = vertices(skin);
      result.tongue_vertices = vertices(tongue);
      result.jaw_transform = array(jaw);
      result.eyes_rotation = array(eyes);
      self->callback_(result);
      return true;
    };
    check(bundle_->GetExecutor().SetResultsCallback(cb, this), "set results callback");
  }

  ~PyAudio2FaceExecutor() { if (active_) bundle_->GetAudioAccumulator(0).Close(); }

  void start_execution() {
    if (active_) throw std::runtime_error("Execution is already active");
    auto& accumulator = bundle_->GetEmotionAccumulator(0);
    std::vector<float> neutral(accumulator.GetEmotionSize(), 0.0f);
    check(accumulator.Accumulate(
              0, nva2x::HostTensorFloatConstView{neutral.data(), neutral.size()},
              bundle_->GetCudaStream().Data()),
          "accumulate neutral emotion");
    check(accumulator.Close(), "close emotion accumulator");
    active_ = true;
  }

  void accumulate_audio(py::array_t<float, py::array::c_style | py::array::forcecast> audio) {
    if (!active_) throw std::runtime_error("Call start_execution() first");
    auto buffer = audio.request();
    if (buffer.ndim != 1) throw std::runtime_error("Audio must be one-dimensional");
    check(bundle_->GetAudioAccumulator(0).Accumulate(
              nva2x::HostTensorFloatConstView{
                  static_cast<float*>(buffer.ptr), static_cast<std::size_t>(buffer.size)},
              bundle_->GetCudaStream().Data()), "accumulate audio");
    process();
  }

  void end_execution() {
    if (!active_) throw std::runtime_error("No active execution");
    check(bundle_->GetAudioAccumulator(0).Close(), "close audio accumulator");
    process();
    check(bundle_->GetExecutor().Reset(0), "reset executor");
    bundle_->GetAudioAccumulator(0).Reset();
    bundle_->GetEmotionAccumulator(0).Reset();
    active_ = false;
  }
  bool is_active() const { return active_; }
  std::size_t skin_vertex_count() const { return bundle_->GetExecutor().GetSkinGeometrySize() / 3; }
  std::size_t tongue_vertex_count() const { return bundle_->GetExecutor().GetTongueGeometrySize() / 3; }

 private:
  struct Destroy { void operator()(nva2f::IGeometryExecutorBundle* p) const { if (p) p->Destroy(); } };
  static void check(std::error_code error, const char* action) {
    if (error) throw std::runtime_error(std::string("Failed to ") + action + ": " + error.message());
  }
  static std::vector<float> copy(nva2x::DeviceTensorFloatConstView view, cudaStream_t stream) {
    std::vector<float> host(view.Size());
    if (host.empty()) return host;
    auto error = cudaMemcpyAsync(host.data(), view.Data(), host.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return host;
  }
  static py::array_t<float> array(const std::vector<float>& value) {
    py::array_t<float> result(value.size());
    if (!value.empty()) std::memcpy(result.mutable_data(), value.data(), value.size() * sizeof(float));
    return result;
  }
  static py::array_t<float> vertices(const std::vector<float>& value) {
    if (value.size() % 3) throw std::runtime_error("Geometry is not XYZ data");
    py::array_t<float> result({static_cast<py::ssize_t>(value.size() / 3), py::ssize_t(3)});
    if (!value.empty()) std::memcpy(result.mutable_data(), value.data(), value.size() * sizeof(float));
    return result;
  }
  void process() {
    py::gil_scoped_release release;
    while (nva2x::GetNbReadyTracks(bundle_->GetExecutor()) > 0)
      check(bundle_->GetExecutor().Execute(nullptr), "execute geometry model");
  }
  std::function<void(const PyGeometryResults&)> callback_;
  bool active_{};
  std::unique_ptr<nva2f::IGeometryExecutorBundle, Destroy> bundle_;
};

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native NVIDIA Audio2Face audio-to-vertex bindings";
  py::class_<PyGeometryResults>(m, "GeometryResults")
      .def_readonly("track_index", &PyGeometryResults::track_index)
      .def_readonly("timestamp_current_frame", &PyGeometryResults::timestamp_current_frame)
      .def_readonly("timestamp_next_frame", &PyGeometryResults::timestamp_next_frame)
      .def_readonly("skin_vertices", &PyGeometryResults::skin_vertices)
      .def_readonly("tongue_vertices", &PyGeometryResults::tongue_vertices)
      .def_readonly("jaw_transform", &PyGeometryResults::jaw_transform)
      .def_readonly("eyes_rotation", &PyGeometryResults::eyes_rotation);
  py::class_<PyAudio2FaceExecutor>(m, "Executor")
      .def(py::init<std::function<void(const PyGeometryResults&)>, const std::string&, int,
                    std::size_t, bool>(), py::arg("callback"), py::arg("model_path"),
           py::arg("device_id") = 0, py::arg("identity_index") = 0,
           py::arg("constant_noise") = false)
      .def("start_execution", &PyAudio2FaceExecutor::start_execution)
      .def("accumulate_audio", &PyAudio2FaceExecutor::accumulate_audio)
      .def("end_execution", &PyAudio2FaceExecutor::end_execution)
      .def("is_active", &PyAudio2FaceExecutor::is_active)
      .def_property_readonly("skin_vertex_count", &PyAudio2FaceExecutor::skin_vertex_count)
      .def_property_readonly("tongue_vertex_count", &PyAudio2FaceExecutor::tongue_vertex_count);
  m.attr("__version__") = "0.1.0";
}
