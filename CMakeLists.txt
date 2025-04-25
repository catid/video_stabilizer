cmake_minimum_required(VERSION 3.15)
project(halide_test CXX)


################################################################################
# Dependencies

# Must specify cmake -DHALIDE_VERSION=14.0.0
if(DEFINED HALIDE_VERSION)
    set(HALIDE_VERSION ${HALIDE_VERSION} CACHE STRING "Halide version")
elseif(DEFINED ENV{HALIDE_VERSION})
    set(HALIDE_VERSION $ENV{HALIDE_VERSION} CACHE STRING "Halide version")
else()
    set(HALIDE_VERSION "15.0.1" CACHE STRING "Halide version")
endif()

string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" HALIDE_VERSION_MAJOR ${HALIDE_VERSION})
string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\..*" "\\1" HALIDE_VERSION_MINOR ${HALIDE_VERSION})
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" HALIDE_VERSION_PATCH ${HALIDE_VERSION})

add_definitions(-DHALIDE_VERSION_MAJOR=${HALIDE_VERSION_MAJOR})
add_definitions(-DHALIDE_VERSION_MINOR=${HALIDE_VERSION_MINOR})
add_definitions(-DHALIDE_VERSION_PATCH=${HALIDE_VERSION_PATCH})

message("Halide version specified: ${HALIDE_VERSION}: ${HALIDE_VERSION_MAJOR}.${HALIDE_VERSION_MINOR}.${HALIDE_VERSION_PATCH}")

# Expect to find halide under ./distros/
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-x86-64-linux/lib/cmake/Halide")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-x86-64-linux/lib/cmake/HalideHelpers")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-arm-64-linux/lib/cmake/Halide")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-arm-64-linux/lib/cmake/HalideHelpers")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-x86-64-windows/lib/cmake/Halide")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-x86-64-windows/lib/cmake/HalideHelpers")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-arm-64-osx/lib/cmake/Halide")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/distros/Halide-${HALIDE_VERSION}-arm-64-osx/lib/cmake/HalideHelpers")

find_package(Halide REQUIRED)

find_package(OpenCV REQUIRED)

# Check if OpenCV was found
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and ensure it's in your CMAKE_PREFIX_PATH.")
endif()

message("OpenCV version: ${OpenCV_VERSION}")
message("OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV libraries: ${OpenCV_LIBS}")


################################################################################
# Configuration

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# C++ language version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug/Release Modes
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g ${AIK_DEBUG_OPT} -march=native -fstack-protector")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# In debug mode, add -DDEBUG
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    # Static library: -fPIC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


################################################################################
# Library: imgproc

add_library(imgproc SHARED
    imgproc.cpp
    imgproc.hpp
    alignment.cpp
    alignment.hpp
    tools.cpp
    tools.hpp
    stabilizer.cpp
    stabilizer.hpp
    smoother.cpp
    smoother.hpp
)

target_include_directories(imgproc
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(imgproc
    PUBLIC
        Halide::Halide
        Threads::Threads
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")

    # Make DLL work like on Linux
    set_target_properties(imgproc PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()


################################################################################
# Generators

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64-bit build")
    set(HALIDE_BITS 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(STATUS "32-bit build")
    set(HALIDE_BITS 32)
else()
    message(FATAL_ERROR "Unknown architecture")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(STATUS "Configuring on Linux")
    set(HALIDE_OS "linux")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(STATUS "Configuring on macOS")
    set(HALIDE_OS "osx")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    message(STATUS "Configuring on Windows")
    set(HALIDE_OS "windows")
else()
    message(FATAL_ERROR "Unknown OS ${CMAKE_SYSTEM_NAME}")
endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm|aarch64")
    message(STATUS "ARM architecture detected")
    set(HALIDE_TARGET_ARCH "arm-${HALIDE_BITS}-${HALIDE_OS}-enable_llvm_loop_opt")
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "amd64|AMD64|x86_64|X86_64")
    message(STATUS "Intel architecture detected")
    set(HALIDE_TARGET_ARCH "x86-64-${HALIDE_OS}-avx-avx2-f16c-fma-sse41-enable_llvm_loop_opt")
else()
    message(FATAL_ERROR "Unknown architecture is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

function(generate_target)

    # Pass through arguments to usual function from Halide
    add_halide_library(${ARGV}
        # Configure it for the target architecture
        TARGETS
            ${HALIDE_TARGET_ARCH}
    )

    message("Configuring generator: ${ARGV0}")

    set(GENERATED_HEADERS ${GENERATED_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${ARGV0}.h PARENT_SCOPE)

    # "whole-archive" is not supported on OSX.
    if (APPLE)
        set(WHOLE_ARCHIVE_FLAG "-all_load")
        set(NO_WHOLE_ARCHIVE_FLAG "-noall_load")
    else()
        set(WHOLE_ARCHIVE_FLAG "--whole-archive")
        set(NO_WHOLE_ARCHIVE_FLAG "--no-whole-archive")
    endif()

    # This causes the generated library to get sucked up into the imgproc library
    # and the symbol persists even if it's not used by anything
    target_link_libraries(imgproc
        PUBLIC
            "-Wl,${WHOLE_ARCHIVE_FLAG}"
            ${ARGV0}
            "-Wl,${NO_WHOLE_ARCHIVE_FLAG}"
    )
endfunction()

add_executable(generators
    generators.cpp
    generators_tools.cpp
)
target_link_libraries(generators PRIVATE Halide::Generator)

generate_target(pyr_down FROM generators
    GENERATOR pyr_down
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)

generate_target(grad_xy FROM generators
    GENERATOR grad_xy
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)

generate_target(image_warp FROM generators
    GENERATOR image_warp
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)

generate_target(grad_argmax_2 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=2
)
generate_target(grad_argmax_4 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=4
)
generate_target(grad_argmax_6 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=6
)
generate_target(grad_argmax_8 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=8
)
generate_target(grad_argmax_10 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=10
)
generate_target(grad_argmax_12 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=12
)
generate_target(grad_argmax_14 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=14
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)
generate_target(grad_argmax_16 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=16
)
generate_target(grad_argmax_18 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=18
)
generate_target(grad_argmax_20 FROM generators
    GENERATOR grad_argmax
    PARAMS tile_size=20
)

generate_target(sparse_jac FROM generators
    GENERATOR sparse_jac
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)

generate_target(sparse_ica FROM generators
    GENERATOR sparse_ica
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)

generate_target(sparse_warpdiff FROM generators
    GENERATOR sparse_warpdiff
    #AUTOSCHEDULER Halide::Adams2019
    #SCHEDULE ON
)


################################################################################
# Apps

# App: align_test

add_executable(align_test
    align_test.cpp
)
target_link_libraries(align_test
    imgproc
)
install(TARGETS align_test DESTINATION bin)

# App: video_test

add_executable(video_test
video_test.cpp
)
target_link_libraries(video_test
    imgproc
)
install(TARGETS video_test DESTINATION bin)

# -----------------------------------------------------------------------------
# Tool: eval_jitter  (computes frame-to-frame jitter of a video)
# -----------------------------------------------------------------------------

add_executable(eval_jitter
    eval_jitter.cpp
)
target_link_libraries(eval_jitter
    imgproc
)
install(TARGETS eval_jitter DESTINATION bin)

# -----------------------------------------------------------------------------
# Tool: grid_search_smoother  (hyper-parameter tuner)
# -----------------------------------------------------------------------------

add_executable(grid_search_smoother
    grid_search_smoother.cpp
)
target_link_libraries(grid_search_smoother
    imgproc
)
install(TARGETS grid_search_smoother DESTINATION bin)

# Tool: grid_search_align (aligner parameter tuner)
add_executable(grid_search_align
    grid_search_align.cpp
)
target_link_libraries(grid_search_align
    imgproc
)
install(TARGETS grid_search_align DESTINATION bin)
