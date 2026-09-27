#pragma once

#include <android/log.h>

/**
 * ROS -> Android 日志桥接。
 *
 * VINS 是从 ROS 移植过来的，源码里到处是 ROS_DEBUG —— 在 ROS 里它表示
 * 「默认不输出」的调试级日志，但最初的移植把它直接映射成
 * __android_log_print(ANDROID_LOG_DEBUG, ...)，于是 feature_tracker
 * 每帧打 5~8 行；30Hz 下就是每秒 150~240 行。
 *
 * 真机实测（OnePlus PLK110 / Android 16）：2 分钟里 MobileScan3D-VINS
 * 打了 31135 行，直接触发系统的 LOG_FLOWCTRL 配额限流
 * （"LOGS OVER PROC QUOTA(300) ... DROPPED"），而相机线程、VINS 线程、
 * 渲染线程同时都在跑 —— 持续占用 logd 写入带宽会实实在在拖慢实时链路。
 *
 * 现在 ROS_DEBUG 默认关闭（与 ROS 的默认行为一致）。写法上用
 * `if (false)` 而不是直接删掉：参数仍参与编译，格式串和变量引用会继续
 * 被检查，也不会产生 unused-variable 警告；运行时分支不进入，零开销。
 *
 * 需要临时打开跟踪流时，编译时定义 MOBILESCAN3D_VINS_TRACE 即可。
 */
#ifdef MOBILESCAN3D_VINS_TRACE
#define ROS_DEBUG(...) \
    __android_log_print(ANDROID_LOG_DEBUG, "MobileScan3D-VINS", __VA_ARGS__)
#else
#define ROS_DEBUG(...)                                                    \
    do {                                                                  \
        if (false) {                                                      \
            __android_log_print(ANDROID_LOG_DEBUG, "MobileScan3D-VINS",   \
                                __VA_ARGS__);                             \
        }                                                                 \
    } while (0)
#endif

#define ROS_INFO(...) \
    __android_log_print(ANDROID_LOG_INFO, "MobileScan3D-VINS", __VA_ARGS__)

#define ROS_WARN(...) \
    __android_log_print(ANDROID_LOG_WARN, "MobileScan3D-VINS", __VA_ARGS__)

#define ROS_ERROR(...) \
    __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-VINS", __VA_ARGS__)
