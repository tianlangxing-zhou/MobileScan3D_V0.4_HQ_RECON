# SETUP — 从 GitHub 克隆后在一台新机器上构建

> **GitHub 是本工程的主干（single source of truth）。** 任何修改都会 commit + push 到
> `origin/main`。本文件说明如何在一台只装了标准 Android 开发环境的新电脑上 clone 并构建，
> 无需任何外部 OpenCV 安装。

## 1. 前置环境（标准 Android 开发栈，机器相关）

| 组件 | 版本 / 要求 | 说明 |
| --- | --- | --- |
| Android SDK | `compileSdk = 36`，`targetSdk = 36`，`minSdk = 26` | 用 Android Studio 或 `sdkmanager` 装好 |
| NDK | **27.1.12297006**（与 `app/build.gradle.kts` 的 `ndkVersion` 完全一致） | NDK 版本不一致会导致 Ceres 预编译库链接失败 |
| JDK | 17 | CMake/Kotlin 需要 |
| CMake | 3.22.1+ | Gradle 会按需下载，或系统自带 |
| Gradle | 8.13（仓库 `.toolchain/gradle` 已固定，或用 Android Studio 的） | 见下方说明 |

> 仅构建 **arm64-v8a**（`abiFilters += arm64-v8a`）。

## 2. 克隆

```bash
git clone https://github.com/tianlangxing-zhou/MobileScan3D_V0.4_HQ_RECON.git
cd MobileScan3D_V0.4_HQ_RECON
```

## 3. 配置 `local.properties`（机器相关，不入库，已被 .gitignore 忽略）

在仓库根目录新建 `local.properties`，填入你本机的路径：

```properties
# Android SDK 根目录（必填，标准）
sdk.dir=/path/to/your/android-sdk

# Ceres 2.2.0：仍为外部预编译依赖（必填，见第 5 节）
ceres.sourceDir=/path/to/ceres-solver-2.2.0
ceres.buildDir=/path/to/ceres_android_build

# OpenCV：可选！不设则自动使用仓库内 vendored 的 OpenCV 4.12（推荐）
# 只有当你想覆盖为本地另一个 OpenCV 4.12 SDK 时才填这一行
# opencv.sdkDir=/path/to/OpenCV-4.12.0-android-sdk/OpenCV-android-sdk
```

> 不要提交 local.properties —— 它只包含本机绝对路径。换机器时各自生成自己的即可。

## 4. OpenCV 已 vendoring（无需任何外部 OpenCV）

- 头文件：`app/src/main/cpp/third_party/opencv4android/sdk/native/jni/include`（OpenCV 4.12.0，已随仓库提交）。
- 运行/链接库：`app/src/main/jniLibs/arm64-v8a/libopencv_java4.so`（同一份、已 git 跟踪、4.12）。

构建时 CMake 默认用这两处，而且链接库就是打包进 APK 的那个 .so —— 因此
链接库 == 运行库 是结构性保证，不会再出现 4.10 头链接 / 4.12 库运行 那种
`dlopen: cannot locate symbol` 静默 ABI 错配闪退。CMake 还会在配置期读取
version.hpp 校验 OpenCV 必须是 4.12，版本不对直接 FATAL_ERROR 给出明确提示。

请勿把 `jniLibs/arm64-v8a/libopencv_java4.so` 替换成其他 OpenCV 版本，否则校验会失败。

## 5. Ceres（仍为外部预编译，需本机准备）

Ceres 的预编译静态库 `libceres.a` 约 1 GB（把 Eigen / SuiteSparse 等都静态打进去了），
无法合理入库。因此每台机器需要自行准备一份 Android arm64 的 Ceres 2.2.0 构建：

- `ceres.sourceDir/` 下需存在：
  - `include/ceres/ceres.h`
  - `internal/ceres/miniglog/`（miniglog 头，仓库代码直接 include）
- `ceres.buildDir/` 下需存在：
  - `lib/libceres.a`（NDK r27 编出的 arm64 静态库）
  - `include/ceres/internal/config.h`、`include/ceres/internal/export.h`（构建生成的配置头）

编译 Ceres 时必须对齐以下 flags（否则与仓库代码行为/链接不一致）：

- 用 NDK r27 的 `toolchain.cmake`，`ANDROID_ABI=arm64-v8a`，`ANDROID_STL=c++_shared`。
- 不要开 `-ffast-math`（会让 `std::isfinite` guard 被优化掉，污染 TSDF）。
- 定义 `MAX_LOG_LEVEL=-1`（只保留 WARNING 及以上，否则 miniglog 会把 logcat 打爆）。
- 开启 `MINIGLOG`（用自带 miniglog 而非系统 glog）。

参考构建命令（按需改路径）：

```bash
cd /path/to/ceres-solver-2.2.0
mkdir -p build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIGLOG=ON -DCXX11=ON \
  -DCMAKE_CXX_FLAGS="-O3 -fno-finite-math-only -DMAX_LOG_LEVEL=-1" \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j$(nproc)
# 产物：build-android/lib/libceres.a 与 build-android/include/ceres/internal/config.h
```

## 6. 构建

```bash
# 方式 A：用仓库固定的 Gradle（离线，已配置好）
/path/to/.toolchain/gradle/gradle-8.13/bin/gradle --offline :app:clean :app:assembleDebug

# 方式 B：Android Studio 打开后 Build -> Build Bundle(s) / APK(s) -> Build APK
```

产物：`app/build/outputs/apk/debug/app-debug.apk`（arm64-v8a debug）。

> 首次克隆后若报 `CERES_SOURCE_DIR is not configured`，说明 local.properties 没配
> ceres.sourceDir / ceres.buildDir；按第 3、5 节补上即可。若报 OpenCV 版本错误，
> 说明 jniLibs 里的 libopencv_java4.so 被换成了非 4.12 版本，恢复为仓库原文件即可。
