cmake_minimum_required(VERSION 3.20)

project(DynamicEngine3DAndroid CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-math-errno -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-decl-specifier")

add_library(DynamicEngine3DAndroid SHARED
    src/Solver.cpp
    src/Collisions.cpp
    src/Deformer.cpp
    src/SoftBodyInstance.cpp
    src/RigidBodyInstance.cpp
    src/Registry.cpp
    src/Error.cpp
    "include/ThreadPool.h")

target_include_directories(DynamicEngine3DAndroid PUBLIC include)

# Force Clang to link the shared C++ library manually, overriding the bad toolchain defaults
target_link_options(DynamicEngine3DAndroid PRIVATE "-Wl,-z,max-page-size=16384" "-lc++_shared")

# Link Android base APIs. dl lives in libc on Android, but naming it keeps the
# dladdr() dependency in Error.cpp explicit.
target_link_libraries(DynamicEngine3DAndroid PUBLIC log android m dl)