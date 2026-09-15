#pragma once
#include <cstdint>
struct AiQuality { float sharpness=0, exposure=0, motion=0, geometry=0, completion=0; };
class AiQualityEngine { public: void reset(); void analyze(const uint8_t* y,int w,int h,int stride=0); AiQuality q()const{return q_;} private: AiQuality q_; };
