#pragma once
#include <cstdint>
#include <vector>
struct DepthStats { uint64_t samples=0, valid=0, refined=0; float confidence=0; };
class DepthFusion {
public:
 void reset();
 void ingestLuma(const uint8_t* y,int w,int h,int stride=0);
 void ingestExternalDepth(const float* depth,int w,int h,float confidence,uint64_t ts);
 const std::vector<float>& depth() const{return depth_;}
 int width()const{return w_;} int height()const{return h_;}
 uint64_t timestamp()const{return ts_;}
 DepthStats stats()const{return s_;}
private: DepthStats s_; std::vector<float> depth_; int w_=0,h_=0; uint64_t ts_=0;
};
