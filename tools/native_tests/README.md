# centroid_ab_test — 目标掩膜质心读取 A/B 对照测试

用于复现并回归验证 `app/src/main/cpp/target_mask_engine.cpp` 中
`TargetMaskEngine::build()` 读取质心的那处闪退缺陷。

## 缺陷回顾

`cv::connectedComponentsWithStats()` 的 `centroids` 输出是 **CV_64F、
nlabels x 2 的两列单通道**矩阵，每个元素只有一个 `double`（`elemSize()==8`）。

旧代码写成：

```cpp
const cv::Vec2d c = centroids_.at<cv::Vec2d>(chosen);   // 错：sizeof(Vec2d)==16
```

`cv::Mat::at<T>()` 是内联在 `opencv2/core/mat.inl.hpp` 里的模板，会带着
**调用方自己的编译选项**展开。`assembleDebug` 的原生构建不定义 `NDEBUG`，
于是 `CV_DbgAssert(elemSize() == sizeof(_Tp))` 生效 —— 一旦锁定目标、
深度线程走到这一行就抛 `cv::Exception`。该异常穿过 JNI 边界没有任何 C++
catch，直接 `std::terminate` 杀掉进程，Kotlin 侧 `try/catch` 完全兜不住。

Release 下断言被编译掉，读到的地址恰好是行首（行长 `2*8=16` 字节，与
`Vec2d` 同宽），所以结果"碰巧正确" —— 这正是它长期没被发现的原因。
但按类型规则这是越界读取的未定义行为。

修复后按 `double` 逐列取，并加了形状守卫。

## 编译（arm64-v8a，不定义 NDEBUG，与 app Debug 构建一致）

```bash
NDK=<android-sdk>/ndk/27.1.12297006
OPENCV=<opencv-android-sdk>     # sdk/native/jni/include 与 sdk/native/libs

$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe \
  --target=aarch64-linux-android26 -std=c++20 -g -O0 \
  -I $OPENCV/sdk/native/jni/include \
  -o centroid_ab_test centroid_ab_test.cpp \
  -L $OPENCV/sdk/native/libs/arm64-v8a -lopencv_java4
```

## 在真机上运行

```bash
adb push centroid_ab_test                          /data/local/tmp/cab
adb push $OPENCV/sdk/native/libs/arm64-v8a/libopencv_java4.so /data/local/tmp/
adb push $NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/
adb shell chmod 755 /data/local/tmp/cab
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./cab'
```

> 注意：`adb push` 的目标必须写**完整文件名**，只给目录名会静默失败。

## 期望输出

- `OLD at<cv::Vec2d>(chosen)` —— 每个场景都 `THREW cv::Exception`
  （`Assertion failed: elemSize() == sizeof(_Tp)`）
- `NEW at<double>(chosen,0/1)` —— 每个场景都 `-> values OK`
- 场景 5（centroids 形状异常）—— 走守卫分支，**不抛异常**

已在 OnePlus PLK110（Android 16 / arm64-v8a）+ OpenCV 4.12.0 上验证通过。
