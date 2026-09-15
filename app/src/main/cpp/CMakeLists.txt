cmake_minimum_required(VERSION 3.22.1)
project("mobilescan3d")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mobilescan3d SHARED
    native_engine.cpp
    gaussian_engine.cpp
    vio_engine.cpp
    keyframe_engine.cpp
    depth_fusion.cpp
    ai_quality.cpp
    ai_backend.cpp
    tsdf_engine.cpp
    vins_prelude.cpp
    vins_ceres_check.cpp
    vins_opencv_check.cpp
    third_party/camodocal/src/camera_models/Camera.cc
    third_party/camodocal/src/camera_models/CameraFactory.cc
    third_party/camodocal/src/camera_models/CataCamera.cc
    third_party/camodocal/src/camera_models/CostFunctionFactory.cc
    third_party/camodocal/src/camera_models/EquidistantCamera.cc
    third_party/camodocal/src/camera_models/PinholeCamera.cc
    third_party/camodocal/src/camera_models/ScaramuzzaCamera.cc
    third_party/camodocal/src/gpl/gpl.cc
    vins/vins_params.cpp
    vins/feature_tracker/feature_tracker.cpp
    vins/estimator/estimator.cpp
    vins/estimator/feature_manager.cpp
    vins/estimator/parameters.cpp
    vins/estimator/factor/marginalization_factor.cpp
    vins/estimator/factor/pose_local_parameterization.cpp
    vins/estimator/factor/projection_factor.cpp
    vins/estimator/factor/projection_td_factor.cpp
    vins/estimator/initial/initial_aligment.cpp
    vins/estimator/initial/initial_ex_rotation.cpp
    vins/estimator/initial/initial_sfm.cpp
    vins/estimator/initial/solve_5pts.cpp
    vins/estimator/utility/utility.cpp
    vins/vins_bridge.cpp
)
find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)
target_include_directories(mobilescan3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mobilescan3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vins)
target_include_directories(mobilescan3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
target_include_directories(mobilescan3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/camodocal/include)
target_include_directories(mobilescan3d PRIVATE
    "D:/WorkBuddy_3DGS/.toolchain/ceres_src/ceres-solver-2.2.0/include"
    "D:/WorkBuddy_3DGS/.toolchain/ceres_build/include"
    "D:/WorkBuddy_3DGS/.toolchain/ceres_src/ceres-solver-2.2.0/internal/ceres/miniglog")
target_include_directories(mobilescan3d PRIVATE
    "D:/WorkBuddy_3DGS/.toolchain/opencv-sdk/OpenCV-android-sdk/sdk/native/jni/include")
target_compile_options(mobilescan3d PRIVATE -O3 -ffast-math)
find_package(Threads REQUIRED)
add_library(opencv_java SHARED IMPORTED)
set_target_properties(opencv_java PROPERTIES
    IMPORTED_LOCATION "D:/WorkBuddy_3DGS/.toolchain/opencv-sdk/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java4.so")
target_link_libraries(mobilescan3d
    ${log-lib}
    ${android-lib}
    ${vulkan-lib}
    Threads::Threads
    opencv_java
    "D:/WorkBuddy_3DGS/.toolchain/ceres_build/lib/libceres.a")
