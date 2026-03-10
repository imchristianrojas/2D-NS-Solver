cmake_minimum_required(VERSION 3.20)

project(
    ns_solver_2d
    VERSION 0.1.0
    DESCRIPTION "2D incompressible Navier-Stokes solver with real-time visualization"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NS_SOLVER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

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
