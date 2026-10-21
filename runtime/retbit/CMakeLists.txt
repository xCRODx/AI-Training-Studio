cmake_minimum_required(VERSION 3.22)
project(retbit_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(retbit_core STATIC
    src/gguf.cpp
    src/model.cpp
    src/operators.cpp
)
target_include_directories(retbit_core PUBLIC include)

if(MSVC)
    target_compile_options(retbit_core PRIVATE /W4 /permissive- /arch:AVX2)
else()
    target_compile_options(retbit_core PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(retbit_core_tests tests/test_operators.cpp)
    target_link_libraries(retbit_core_tests PRIVATE retbit_core)
    add_test(NAME retbit_core_tests COMMAND retbit_core_tests)
endif()

add_executable(retbit_cli_build src/cli.cpp)
target_link_libraries(retbit_cli_build PRIVATE retbit_core)
