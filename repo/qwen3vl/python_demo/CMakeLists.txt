cmake_minimum_required(VERSION 3.10)
project(Qwen3VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(--std=c++17 -fPIC -Wall -Werror)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat_qwen3 chat.cpp)
target_link_libraries(chat_qwen3 PUBLIC bmrt bmlib)
install(TARGETS chat_qwen3 DESTINATION python)

