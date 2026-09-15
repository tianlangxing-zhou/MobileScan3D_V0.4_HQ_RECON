#include "depth_fusion.h"
#include <algorithm>
void DepthFusion::reset(){s_={};depth_.clear();w_=h_=0;}
void DepthFusion::ingestLuma(const uint8_t* y,int w,int h,int stride){if(!y||w<=0||h<=0)return;if(stride<w)stride=w;w_=w;h_=h;if(depth_.size()!=(size_t)w*h)depth_.assign((size_t)w*h,0.f);s_.samples+=(uint64_t)w*h;uint64_t v=0;for(int yy=0;yy<h;yy++){const uint8_t* row=y+(size_t)yy*stride;for(int x=0;x<w;x+=16)if(row[x]>3&&row[x]<252)v++;}s_.valid+=v;s_.refined+=v;s_.confidence=s_.samples?float(s_.valid)/float(s_.samples/16+1):0;}
void DepthFusion::ingestExternalDepth(const float* d,int w,int h,float confidence,uint64_t ts){if(!d||w<=0||h<=0)return;w_=w;h_=h;ts_=ts;depth_.assign(d,d+(size_t)w*h);s_.refined+=(uint64_t)w*h;s_.confidence=std::max(s_.confidence,confidence);}
