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
    mesh/mesh_registration.cpp
    mesh/hard_surface.cpp
    export/gltf_exporter.cpp
    v06/mesh_postprocess.cpp
    v06/uv_unwrap.cpp
    v06/texture_baker.cpp
    v06/textured_glb_exporter.cpp
    ar_textured_asset.cpp
    persistent_relocalizer.cpp
    hq_fusion.cpp
    multicam_fusion.cpp
    stereo_anchor_bridge.cpp
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
set(OPENCV_ANDROID_SDK "" CACHE PATH "OpenCV Android SDK directory (optional; if empty, use in-repo vendored OpenCV)")

# --- Ceres：仍为外部预编译依赖，必须配置 ---
if(NOT CERES_SOURCE_DIR)
    message(FATAL_ERROR "CERES_SOURCE_DIR is not configured. Set ceres.sourceDir in local.properties or CERES_SOURCE_DIR in the environment.")
endif()

if(NOT CERES_BUILD_DIR)
    message(FATAL_ERROR "CERES_BUILD_DIR is not configured. Set ceres.buildDir in local.properties or CERES_BUILD_DIR in the environment.")
endif()

set(CERES_STATIC_LIB "${CERES_BUILD_DIR}/lib/libceres.a")

# --- OpenCV：仓库内 vendored 4.12 为默认；外部 SDK 仅作可选覆盖 ---
# 默认路径下，头用 third_party/opencv4android 里的 4.12 头，链接库直接复用
# app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so（同一份、已 git 跟踪、4.12）。
# 这样 clone 后无需任何外部 OpenCV 配置，且「链接库 == 运行库」是结构性保证，
# 不会再出现 4.10 头链接 / 4.12 库运行 的静默 ABI 错配闪退。
if(OPENCV_ANDROID_SDK)
    set(MOBILESCAN_OPENCV_INCLUDE "${OPENCV_ANDROID_SDK}/sdk/native/jni/include")
    set(OPENCV_JAVA_LIB "${OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so")
    message(STATUS "MobileScan3D: using EXTERNAL OpenCV SDK = ${OPENCV_ANDROID_SDK}")
else()
    set(MOBILESCAN_OPENCV_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv4android/sdk/native/jni/include")
    set(OPENCV_JAVA_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so")
    message(STATUS "MobileScan3D: using IN-REPO vendored OpenCV 4.12")
endif()

# OpenCV 版本校验：头必须声明 4.12，否则链接/运行 ABI 必然错配。
set(_opencv_ver_h "${MOBILESCAN_OPENCV_INCLUDE}/opencv2/core/version.hpp")
if(NOT EXISTS "${_opencv_ver_h}")
    message(FATAL_ERROR "OpenCV headers not found: ${_opencv_ver_h}")
endif()
file(READ "${_opencv_ver_h}" _opencv_ver_content)
string(REGEX MATCH "#define CV_VERSION_MAJOR[ \t]+([0-9]+)" _opencv_major_match "${_opencv_ver_content}")
set(_opencv_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define CV_VERSION_MINOR[ \t]+([0-9]+)" _opencv_minor_match "${_opencv_ver_content}")
set(_opencv_minor "${CMAKE_MATCH_1}")
if(NOT _opencv_major OR NOT _opencv_minor)
    message(FATAL_ERROR "Cannot parse OpenCV version from ${_opencv_ver_h}")
endif()
if(NOT (_opencv_major STREQUAL "4" AND _opencv_minor STREQUAL "12"))
    message(FATAL_ERROR
        "OpenCV must be version 4.12 (link==runtime ABI guarantee). "
        "Found ${_opencv_major}.${_opencv_minor}. Headers: ${MOBILESCAN_OPENCV_INCLUDE}")
endif()

if(NOT EXISTS "${CERES_SOURCE_DIR}/include/ceres/ceres.h")
    message(FATAL_ERROR "Ceres headers not found: ${CERES_SOURCE_DIR}/include/ceres/ceres.h")
endif()

if(NOT EXISTS "${CERES_STATIC_LIB}")
    message(FATAL_ERROR "Ceres library not found: ${CERES_STATIC_LIB}")
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

    ${MOBILESCAN_OPENCV_INCLUDE}
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
    # V0.13.2 修复：真机实测 xatlas 多线程 TaskScheduler 存在组槽位复用竞态
    # （wait() 置 group.free 后慢 worker 仍在旧任务 ref-- 上，污染新组计数），
    # 表现为 nativeBakeTexturedGlb 在 ComputeCharts 里 sched_yield 永久自旋，
    # MeshExport 线程死锁、GLB 永不产出（bugreport 线程栈实锤）。
    # 强制单线程：无 worker 线程即无该竞态；2~4 万三角形网格单线程展开
    # 1~3 秒，仅发生在停止扫描导出时，可接受。
    set_source_files_properties(
        "${MOBILESCAN_XATLAS_DIR}/xatlas.cpp" PROPERTIES
        COMPILE_DEFINITIONS "XA_MULTITHREADED=0")
    message(STATUS "MobileScan3D V0.6: xatlas enabled (single-threaded)")
else()
    message(WARNING "MobileScan3D V0.6: xatlas missing; using UV fallback")
endif()

# Ceres 用 miniglog（没有真正的 glog 可用），它的 VLOG/LOG(INFO) 会直接
# 写 logcat。真机实测（OnePlus PLK110）：2 分钟里 tag=native 打了 37501 行
# （detect_structure / schur_eliminator / trust_region 迭代表），占全部
# APP 日志的 43%，把 LOG_FLOWCTRL 配额打爆。
#
# miniglog 的宏在定义了 MAX_LOG_LEVEL 后会变成 LOG_IF(..., n <= MAX_LOG_LEVEL)，
# 严重级别：FATAL=-3 / ERROR=-2 / WARNING=-1 / INFO=0 / VLOG(n)=n(n>=1)。
# 取 -1（WARNING）即只保留 WARNING 及以上，INFO 与所有 VLOG 静默。
# libceres.a 必须用同一个值编译（见构建脚本），这里同步定义以保证
# 内联进本目标的代码行为一致。
target_compile_definitions(mobilescan3d PRIVATE MAX_LOG_LEVEL=-1)
