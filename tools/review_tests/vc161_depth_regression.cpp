#include "depth_refinement.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <chrono>
int main(){
    constexpr int w=256,h=256;std::vector<float> d(w*h),out;
    std::mt19937 rng(7);std::normal_distribution<float> noise(0,.008f);
    double before=0,after=0;
    for(auto& v:d)v=1.f+noise(rng);
    depth_refinement::spatial(d.data(),w,h,false,out);
    for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){int i=y*w+x;before+=std::pow(d[i]-1,2);after+=std::pow(out[i]-1,2);}
    assert(after<before*.65);
    std::fill(d.begin(),d.end(),1.f);
    for(int y=0;y<h;++y)for(int x=w/2;x<w;++x)d[y*w+x]=2.f;
    d[20*w+20]=0;d[30*w+30]=std::numeric_limits<float>::quiet_NaN();
    depth_refinement::spatial(d.data(),w,h,false,out);
    assert(out[20*w+20]==0 && std::isnan(out[30*w+30]));
    for(int y=1;y<h-1;++y){assert(out[y*w+w/2-1]==1.f);assert(out[y*w+w/2]==2.f);}
    float sample=0;assert(!depth_refinement::calibrationSample(d.data(),w,h,w/2,80,false,sample));
    assert(depth_refinement::calibrationSample(d.data(),w,h,80,80,false,sample)&&sample==1);
    assert(!depth_refinement::calibrationSample(d.data(),w,h,-1,0,false,sample));
    std::fill(d.begin(),d.end(),-.25f);
    depth_refinement::spatial(d.data(),w,h,true,out);assert(out[1000]==-.25f);
    assert(depth_refinement::calibrationSample(d.data(),w,h,80,80,true,sample)&&sample==-.25f);
    depth_refinement::spatial(nullptr,w,h,true,out);assert(out.empty());
    std::fill(d.begin(),d.end(),1.f);
    auto t=std::chrono::steady_clock::now();
    for(int i=0;i<100;++i)depth_refinement::spatial(d.data(),w,h,false,out);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/100;
    std::cout<<"PASS depth: noise MSE ratio="<<after/before<<", sharp edges, holes, negative inverse values, sample gates; host ms/frame="<<ms<<"\n";
}
