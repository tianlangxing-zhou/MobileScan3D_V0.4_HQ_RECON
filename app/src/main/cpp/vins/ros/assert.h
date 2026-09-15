#pragma once

#include <cassert>

#include "ros/console.h"

#define ROS_ASSERT(x) assert(x)
#define ROS_ASSERT_MSG(x, ...) assert(x)
#define ROS_BREAK() do {} while (0)
