cmake_minimum_required(VERSION 3.22.1)
project("mobilescan3d")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mobilescan3d SHARED
    native_engine.cpp
    surfel_engine.cpp
    vio_engine.cpp
    keyframe_engine.cpp
    depth_fusion.cpp
    ai_quality.cpp
    ai_backend.cpp
    tsdf_engine.cpp
    object_tracker.cpp
    target_mask_engine.cpp
    appearance_tracker.cpp
    depth_calib.cpp
    mesh/mesh_engine.cpp
    export/gltf_exporter.cpp
    v06/mesh_postprocess.cpp
    v06/uv_unwrap.cpp
    v06/texture_baker.cpp
    v06/textured_glb_exporter.cpp
    ar_textured_asset.cpp
    persistent_relocalizer.cpp
    hq_fusion.cpp
    multicam_fusion.cpp
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
find_package(Threads REQUIRED)

# ------------------------------------------------------------
# External dependency paths
# ------------------------------------------------------------

set(CERES_SOURCE_DIR "" CACHE PATH "Ceres source directory")
set(CERES_BUILD_DIR "" CACHE PATH "Ceres Android build directory")
set(OPENCV_ANDROID_SDK "" CACHE PATH "OpenCV Android SDK directory")

if(NOT CERES_SOURCE_DIR)
    message(FATAL_ERROR "CERES_SOURCE_DIR is not configured. Set ceres.sourceDir in local.properties.")
endif()

if(NOT CERES_BUILD_DIR)
    message(FATAL_ERROR "CERES_BUILD_DIR is not configured. Set ceres.buildDir in local.properties.")
endif()

if(NOT OPENCV_ANDROID_SDK)
    message(FATAL_ERROR "OPENCV_ANDROID_SDK is not configured. Set opencv.sdkDir in local.properties.")
endif()

set(CERES_STATIC_LIB "${CERES_BUILD_DIR}/lib/libceres.a")
set(OPENCV_JAVA_LIB "${OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so")

if(NOT EXISTS "${CERES_SOURCE_DIR}/include/ceres/ceres.h")
    message(FATAL_ERROR "Ceres headers not found: ${CERES_SOURCE_DIR}/include/ceres/ceres.h")
endif()

if(NOT EXISTS "${CERES_STATIC_LIB}")
    message(FATAL_ERROR "Ceres library not found: ${CERES_STATIC_LIB}")
endif()

if(NOT EXISTS "${OPENCV_ANDROID_SDK}/sdk/native/jni/include/opencv2/core.hpp")
    message(FATAL_ERROR "OpenCV headers not found: ${OPENCV_ANDROID_SDK}")
endif()

if(NOT EXISTS "${OPENCV_JAVA_LIB}")
    message(FATAL_ERROR "OpenCV library not found: ${OPENCV_JAVA_LIB}")
endif()

target_include_directories(
    mobilescan3d
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vins
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/camodocal/include

    ${CERES_SOURCE_DIR}/include
    ${CERES_BUILD_DIR}/include
    ${CERES_SOURCE_DIR}/internal/ceres/miniglog

    ${OPENCV_ANDROID_SDK}/sdk/native/jni/include
)

# 评审第七点：去掉全局 -ffast-math。
#
# -ffast-math 会打开 -ffinite-math-only / -funsafe-math-optimizations，
# 也就是允许编译器**假定不存在 NaN / Inf**，并重排浮点运算顺序。本工程
# 大量依赖 std::isfinite() 做 guard（深度、VINS 残差、Ceres 收敛判据），
# 在 fast-math 下这些检查会被优化掉 —— 一个 NaN 就能静默污染整条 TSDF。
# 真正需要激进数学的 compute kernel（将来 Vulkan/NEON）再单独开。
target_compile_options(
    mobilescan3d
    PRIVATE
    -O3
    -fno-finite-math-only
)

add_library(opencv_java SHARED IMPORTED)
set_target_properties(
    opencv_java
    PROPERTIES
    IMPORTED_LOCATION "${OPENCV_JAVA_LIB}"
)

add_library(ceres STATIC IMPORTED)
set_target_properties(
    ceres
    PROPERTIES
    IMPORTED_LOCATION "${CERES_STATIC_LIB}"
)

target_link_libraries(
    mobilescan3d

    ${log-lib}
    ${android-lib}
    ${vulkan-lib}

    Threads::Threads

    opencv_java
    ceres
)


# ---------------------------------------------------------------------------
# V0.6 UV atlas 后端（可选）
#
# xatlas 官方说明集成只需要 xatlas.cpp + xatlas.h 两个源文件，MIT License。
# 源码已 vendor 在 third_party/xatlas/。即使目录缺失也**不会**中断构建 ——
# uv_unwrap.cpp 自带 dependency-free 的 triangle-atlas 回退路径，只是图谱
# 利用率不如 xatlas（每三角形一个 chart）。
# ---------------------------------------------------------------------------
set(MOBILESCAN_XATLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas")
if(EXISTS "${MOBILESCAN_XATLAS_DIR}/xatlas.cpp"
   AND EXISTS "${MOBILESCAN_XATLAS_DIR}/xatlas.h")
    target_sources(mobilescan3d PRIVATE "${MOBILESCAN_XATLAS_DIR}/xatlas.cpp")
    target_include_directories(mobilescan3d PRIVATE "${MOBILESCAN_XATLAS_DIR}")
    target_compile_definitions(mobilescan3d PRIVATE MOBILESCAN_HAS_XATLAS=1)
    message(STATUS "MobileScan3D V0.6: xatlas enabled")
else()
    message(WARNING "MobileScan3D V0.6: xatlas missing; using UV fallback")
endif()
