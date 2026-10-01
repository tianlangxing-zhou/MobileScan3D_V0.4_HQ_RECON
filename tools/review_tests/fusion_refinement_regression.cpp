#include "surfel_engine.h"
#include "depth_calib.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <chrono>
#include <vector>
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }
static bool measure=false;
static void accuracy() {
    SurfelEngine e;
    // Independent samples at the same physical surface within one fine voxel.
    std::mt19937 rng(723);std::uniform_real_distribution<float> noise(-.003f,.003f);
    double squared=0;size_t n=0;
    for(int i=0;i<10000;++i){e.ingestPoint(.005f,.005f,1.005f+noise(rng),101,137,193,1);
        if(i>100){float x,y,z;e.centroid(&x,&y,&z);squared+=(z-1.005)*(z-1.005);++n;}}
    float point[6];assert(e.copyPoints(point,1,3)==1);
    const double rms=std::sqrt(squared/n);
    std::cout<<"stationary_point_rms_m="<<rms<<'\n';
    if(!measure)assert(rms<.00035);
    // A one-level color change must not stall due to repeated integer truncation.
    for(int i=0;i<2000;++i)e.ingestPoint(.005f,.005f,1.005f,102,138,194,1);
    e.copyPoints(point,1,3);
    std::cout<<"converged_red="<<std::lround(point[3]*255)<<'\n';
    if(!measure)assert(std::lround(point[3]*255)==102 && std::lround(point[4]*255)==138);
    e.reset();
    e.ingestPoint(.001f,.001f,1.001f,0,0,0,.25f);
    e.ingestPoint(.009f,.001f,1.001f,200,100,50,.75f);
    e.copyPoints(point,1,2);
    if(!measure)assert(std::fabs(point[0]-.007f)<1e-7f && std::lround(point[3]*255)==150);
    // Confident history reduces variance but must eventually follow a changed observation.
    for(int i=0;i<2000;++i)e.ingestPoint(.009f,.001f,1.001f,200,100,50,1);
    e.copyPoints(point,1,2);if(!measure)assert(std::fabs(point[0]-.009f)<1e-6f);
}
static void budget() {
    SurfelEngine e;
    for(int i=0;i<1001;++i)for(int hit=0;hit<3;++hit)e.ingestPoint(i*.02f,.005f,1.005f,100,100,100,1);
    std::vector<float> p(1000*6,-1);
    const auto written=e.copyPoints(p.data(),1000,3);
    std::cout<<"preview_budget_written="<<written<<'\n';
    if(!measure)assert(written==1000);
    assert(e.confirmedCount(2)==1001 && e.confirmedCount(3)==1001 && e.confirmedCount(4)==0);
    e.reset();assert(e.confirmedCount(2)==0 && e.confirmedCount(3)==0);
}
static void calibrationBudget() {
    for (int n : {512,513,777,1024,1025}) {
        std::vector<float> d(n),z(n);
        for(int i=0;i<n;++i){d[i]=.1f+i*.001f;z[i]=.7f*d[i]+.4f;}
        DepthCalibrator cal;assert(cal.update(d,z));
        std::cout<<"input_samples="<<n<<" calibration_samples="<<cal.calibration().samples<<'\n';
        if(!measure)assert(cal.calibration().samples==512);
        for(int i=0;i<n;++i)assert(std::fabs(cal.calibration().toMetric(d[i],0)-z[i])<1e-5f);
    }
}
static void hudTiming() {
    SurfelEngine e;
    for(int i=0;i<100000;++i)for(int h=0;h<3;++h)e.ingestPoint(i*.02f,.005f,1.005f,100,100,100,1);
    size_t count=0;auto start=std::chrono::steady_clock::now();
    for(int i=0;i<2000;++i)count+=e.confirmedCount(2)+e.confirmedCount(3);
    std::cout<<"hud_2000_queries_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" count="<<count<<'\n';
    assert(count==400000000);
}
int main(int argc,char** argv){measure=argc>1&&std::string(argv[1])=="--measure";accuracy();budget();calibrationBudget();if(measure)hudTiming();std::cout<<"PASS fusion refinement\n";}
