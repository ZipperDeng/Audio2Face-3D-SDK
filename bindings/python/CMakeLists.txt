find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(audio2face_python MODULE pybind_a2f.cpp)
set_target_properties(audio2face_python PROPERTIES OUTPUT_NAME "_native" INSTALL_RPATH "$ORIGIN")
target_link_libraries(audio2face_python PRIVATE audio2x CUDA::cudart)

install(TARGETS audio2face_python LIBRARY DESTINATION audio2face_sdk RUNTIME DESTINATION audio2face_sdk)
install(FILES $<TARGET_FILE:audio2x> DESTINATION audio2face_sdk)
