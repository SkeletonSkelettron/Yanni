cmake_minimum_required(VERSION 3.18)
project(Yanni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optional CUDA ──────────────────────────────────────────────────────────────
option(USE_CUDA "Build with CUDA support" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(USE_CUDA)
endif()

# ── nlohmann/json ──────────────────────────────────────────────────────────────
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── Sources ────────────────────────────────────────────────────────────────────
set(SOURCES
    Yanni.cpp
    cliOptions.cpp
    core/neuralNetwork.cpp
    core/layer.cpp
    core/workerThread.cpp
    data/readMnist.cpp
    functions/activationFunctions.cpp
    functions/lossFunctions.cpp
    functions/statisticFunctions.cpp
)

if(USE_CUDA)
    list(APPEND SOURCES
        cuda/activationFunctionsCuda.cu
        cuda/lossFunctionsCuda.cu
        cuda/neuNetCuda.cu
        cuda/random.cu
    )
endif()

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(Yanni ${SOURCES})

target_include_directories(Yanni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/functions
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_link_libraries(Yanni PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
find_package(Threads REQUIRED)

# მცურავი წერტილის თავისუფლება ვექტორიზაციისთვის.
# -fassociative-math (და მისი წინაპირობები) რედუქციების ვექტორიზაციას ხსნის:
#   for (i) sum += a[i]*b[i]   ->   8 ნაწილობრივი ჯამი AVX2 რეგისტრში
# განზრახ არ არის -ffinite-math-only (ანუ სრული -ffast-math): ის
# GetMaxIndex-ის std::isnan შემოწმებას ჩუმად false-ად აქცევდა.
set(FASTMATH -fno-math-errno -fassociative-math -freciprocal-math
             -fno-signed-zeros -fno-trapping-math -ffp-contract=fast)

target_compile_options(Yanni PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wno-unused-parameter ${FASTMATH}>
)

if(USE_CUDA)
    set_target_properties(Yanni PROPERTIES CUDA_ARCHITECTURES "89")
    target_link_libraries(Yanni PRIVATE CUDA::cuda_driver)
endif()

# ── Runtime assets ─────────────────────────────────────────────────────────────
# Copy netConfig.json next to the binary so relative-path loading works
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/netConfig.json
    ${CMAKE_CURRENT_BINARY_DIR}/netConfig.json COPYONLY
)

# Create an empty mnist/ folder in the build directory.
# Place the four MNIST binary files there before running:
#   train-images.idx3-ubyte   train-labels.idx1-ubyte
#   t10k-images.idx3-ubyte    t10k-labels.idx1-ubyte
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mnist)
