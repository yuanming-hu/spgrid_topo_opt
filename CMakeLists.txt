set(TAICHI_PROJECT_NAME "topo_opt")

file(GLOB_RECURSE PROJECT_SOURCES "topo_opt/*.cpp" "topo_opt/*.h" "topo_opt/*/*.h" "topo_opt/*/*.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHASWELL -DNANOGUI_GLAD")

list(APPEND SHARED_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/solver/libSPGridCPUSolver.so)
list(APPEND SHARED_LIBS $ENV{TC_MKL_PATH}/libmkl_rt.so)
file(GLOB SPGridSource "external/SPGrid/*/*.cpp")

include_directories(include)
include_directories(external/)
include_directories(external/glfw/include)

add_subdirectory(external/glfw)

set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

if ($ENV{CUDA_ARCH})
    message("CUDA Arch: $ENV{CUDA_ARCH}")
    find_package(CUDA 8.0 REQUIRED)
    set (CUDA_HOST_COMPILER g++)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=compute_$ENV{CUDA_ARCH} -code=sm_$ENV{CUDA_ARCH} -Xcompiler "-fPIC" --maxrregcount 64 --use_fast_math --ptxas-options=-allow-expensive-optimizations=true,-O3)
    if ($ENV{CUDA_ARCH} LESS 60)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
    else()
        # __CUDA_ARCH__ is 0 here, and cannot be redefined...
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D__CUDA_ARCH___=$ENV{CUDA_ARCH}0)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(external/cuda_samples/)
    list(APPEND PROJECT_SOURCES misc/volume_renderer.cpp)
    cuda_add_library(volume_renderer_kernel misc/volume_rendering_kernel.cu)
else()
    message("Environment variable CUDA_ARCH not found. Not using CUDA.")
endif()

add_library(taichi_${TAICHI_PROJECT_NAME} SHARED ${PROJECT_SOURCES} ${SPGridSource})
target_link_libraries(taichi_${TAICHI_PROJECT_NAME} ${CORE_LIBRARY_NAME} ${SHARED_LIBS} nlopt)
if ($ENV{CUDA_ARCH})
    target_link_libraries(taichi_${TAICHI_PROJECT_NAME} volume_renderer_kernel GL GLU stdc++fs rt dl X11 xcb Xau Xdmcp glfw)
endif()

if ($ENV{TC_NEED_ZLIB})
    target_link_libraries(taichi_${TAICHI_PROJECT_NAME} z)
endif()

