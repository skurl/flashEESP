cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(flashee_tflm)
# idf.py flash also writes the model, so firmware and weights never drift apart
esptool_py_flash_to_partition(flash "model" "${CMAKE_SOURCE_DIR}/../flashee_int8.tflite")
