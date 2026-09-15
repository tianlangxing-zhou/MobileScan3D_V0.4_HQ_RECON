#pragma once

#include <android/log.h>

#define ROS_DEBUG(...) \
    __android_log_print(ANDROID_LOG_DEBUG, "MobileScan3D-VINS", __VA_ARGS__)

#define ROS_INFO(...) \
    __android_log_print(ANDROID_LOG_INFO, "MobileScan3D-VINS", __VA_ARGS__)

#define ROS_WARN(...) \
    __android_log_print(ANDROID_LOG_WARN, "MobileScan3D-VINS", __VA_ARGS__)

#define ROS_ERROR(...) \
    __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-VINS", __VA_ARGS__)
