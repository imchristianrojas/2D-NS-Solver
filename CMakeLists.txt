cmake_minimum_required(VERSION 3.20)

project(
    ns_solver_2d
    VERSION 0.1.0
    DESCRIPTION "2D incompressible Navier-Stokes solver with real-time visualization"
    LANGUAGES CXX
)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NS_SOLVER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Allow FetchContent dependencies that use older CMake versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ── Dependencies ────────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
# Eigen is header-only — populate without processing its CMakeLists.txt
# to avoid Fortran/BLAS/LAPACK discovery issues.
FetchContent_GetProperties(Eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(Eigen3)
endif()
add_library(Eigen3_HeaderOnly INTERFACE)
target_include_directories(Eigen3_HeaderOnly INTERFACE "${eigen3_SOURCE_DIR}")
add_library(Eigen3::Eigen ALIAS Eigen3_HeaderOnly)

FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG        3.0.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(SFML)

# ── Core library ────────────────────────────────────────────────────────────────
add_library(ns_solver_core
    src/FluidField.cpp
)

target_include_directories(ns_solver_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ns_solver_core
    PUBLIC
        Eigen3::Eigen
)

target_compile_features(ns_solver_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(ns_solver_core PRIVATE /W4)
else()
    target_compile_options(ns_solver_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NS_SOLVER_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(ns_solver_core PRIVATE /WX)
    else()
        target_compile_options(ns_solver_core PRIVATE -Werror)
    endif()
endif()

# ── App executable ──────────────────────────────────────────────────────────────
add_executable(ns_solver_app
    src/main.cpp
)

target_link_libraries(ns_solver_app
    PRIVATE
        ns_solver_core
        SFML::Graphics
        SFML::Window
        SFML::System
)

# ── Test executable ─────────────────────────────────────────────────────────────
add_executable(ns_solver_tests
    tests/FluidFieldTests.cpp
)

target_link_libraries(ns_solver_tests
    PRIVATE
        ns_solver_core
)

add_test(NAME fluid_field_tests COMMAND ns_solver_tests)

# ── CUDA targets (optional — only built when CUDA toolkit is available) ────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Detect GPU architecture — default to sm_86 for RTX 3070
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    # ── CUDA core library ──────────────────────────────────────────────────
    add_library(ns_solver_cuda
        src/FluidFieldCuda.cu
    )

    target_include_directories(ns_solver_cuda
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(ns_solver_cuda
        PUBLIC
            CUDA::cudart
    )

    set_target_properties(ns_solver_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # ── CUDA app executable ────────────────────────────────────────────────
    add_executable(ns_solver_app_cuda
        src/main_cuda.cpp
    )

    target_link_libraries(ns_solver_app_cuda
        PRIVATE
            ns_solver_cuda
            SFML::Graphics
            SFML::Window
            SFML::System
    )

    # ── CUDA test executable ───────────────────────────────────────────────
    add_executable(ns_solver_cuda_tests
        tests/FluidFieldCudaTests.cpp
    )

    target_link_libraries(ns_solver_cuda_tests
        PRIVATE
            ns_solver_cuda
    )

    add_test(NAME fluid_field_cuda_tests COMMAND ns_solver_cuda_tests)

    message(STATUS "CUDA enabled — building GPU targets (arch: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA not found — GPU targets skipped")
endif()