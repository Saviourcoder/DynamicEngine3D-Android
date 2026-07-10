cmake_minimum_required(VERSION 3.20)

project(DynamicEngine3DAndroid CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fvectorize")

add_library(DynamicEngine3DAndroid SHARED
    src/Solver.cpp
    src/Collisions.cpp
    src/Deformer.cpp
    src/SoftBodyInstance.cpp
    src/RigidBodyInstance.cpp
 "include/ThreadPool.h" "include/GTP.h")

target_include_directories(DynamicEngine3DAndroid PUBLIC include)

# Force Clang to link the shared C++ library manually, overriding the bad toolchain defaults
target_link_options(DynamicEngine3DAndroid PRIVATE "-Wl,-z,max-page-size=16384" "-lc++_shared")

# Link Android base APIs
target_link_libraries(DynamicEngine3DAndroid PUBLIC log android m)