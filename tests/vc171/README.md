# VC171 定向回归

仅测试本轮改动及其几何保护边界，不等同于 Android 集成/相机实机测试。
从工程根目录运行；需 Python 3、G++ (C++20)、OpenCV 4.10+（包含 TrackerNano）、Kotlin/JDK。

```bash
g++ -std=c++20 -O2 -Iapp/src/main/cpp tests/vc171/core_regression.cpp app/src/main/cpp/depth_calib.cpp -o core_regression
./core_regression
python3 tests/vc171/run_epoch_regression.py app/src/main/cpp
g++ -std=c++20 -O2 -Iapp/src/main/cpp tests/vc171/vision_regression.cpp app/src/main/cpp/object_tracker.cpp app/src/main/cpp/appearance_tracker.cpp $(pkg-config --cflags --libs opencv4) -o vision_regression
./vision_regression
kotlinc app/src/main/java/com/mobilescan3d/depth/DepthPreprocessor.kt tests/vc171/DepthPreprocessorRegression.kt -include-runtime -d preprocessor_test.jar
java -cp preprocessor_test.jar com.mobilescan3d.depth.DepthPreprocessorRegressionKt
```

- `core_regression.cpp`：缺样/重复/过期/矛盾证据、整个输入域的映射一致性、源置信度、深度边缘、重复壳保护。
- `run_epoch_regression.py`：提取当前 `native_engine.cpp` 中实际的标定 epoch 状态机，以受控标定输入执行；覆盖负/大原始 q、空数据、冷启动、禁止混尺度重锚、缓存不推进 EMA、间歇证据恢复。临时生成物在当前目录创建并自动删除。不是 JNI 运行测试。
- `vision_regression.cpp`：平移/缩放补偿、异常 mask 拦截、身份模板精确复测/贴边、无关纹理、80 ms 候选复核、长时间出画返回、细长目标旋转、LOST → TRACKING → KLT。通过测试专用访问方式调用真实身份打分，不更改生产类可见性。没有运行 NanoTrack 模型推理。
- `DepthPreprocessorRegression.kt`：含 VC170 测试专用参考实现，比较完整 FLOAT32 tensor 位模式；覆盖奇数尺寸、YUV 行填充、同宽度切换色度像素步长、截断输入及重复调用。附桌面 JVM 微基准，不能解释为 Android 整体帧率。

本次实际环境：G++ 13、OpenCV 4.10 主机运行；仓库 OpenCV 4.12 头文件原生语法检查；Kotlin 2.0.21 + JDK 17。core 另以 UndefinedBehaviorSanitizer 运行通过。AddressSanitizer/LeakSanitizer 受当前主机 `/proc` 访问限制未完成，未记为通过。
