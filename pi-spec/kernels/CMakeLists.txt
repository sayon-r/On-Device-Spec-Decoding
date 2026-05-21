cmake_minimum_required(VERSION 3.16)
project(pispec_kernels CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ARM DOTPROD + NEON flags.
set(ARCH_FLAGS "-march=armv8-a+dotprod -mfpu=neon")

add_library(pispec SHARED
    matmul_int8.cpp
    quantize.cpp
    layernorm.cpp
)

target_compile_options(pispec PRIVATE
    -O3
    ${ARCH_FLAGS}
    -ffast-math
    -fPIC
)

set_target_properties(pispec PROPERTIES
    OUTPUT_NAME "pispec"
    PREFIX      "lib"
    SUFFIX      ".so"
)

install(TARGETS pispec DESTINATION lib)
install(FILES matmul_int8.h quantize.h layernorm.h DESTINATION include)
