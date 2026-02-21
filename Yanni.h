#pragma once
#include <iostream>
#include <chrono>
#include <nlohmann/json.hpp>
#include <thread>
#include "core/neuralNetwork.h"
#include "data/mnistData.h"
#include "data/readMnist.h"
#ifdef USE_CUDA
#  include <cuda.h>
#endif
#ifdef _WIN32
#  include <windows.h>
#endif