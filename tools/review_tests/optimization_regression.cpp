#include "depth_calib.h"
#include "surfel_engine.h"
#include "tsdf_engine.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }

static void accuracy(bool check) {
    std::vector<float> d, z;
    for (int i=0; i<80; ++i) {
        d.push_back(1000000.f + (i%20)*.125f);
        z.push_back(1.f + (i%20)*.015625f);
    }
    auto fit=fitDepthRobust(d,z,false);
    double err=0;
    for(size_t i=0;i<d.size();++i) err=std::max(err,double(std::fabs(fit.toMetric(d[i],0)-z[i])));
    std::cout << "offset_fit_valid=" << fit.valid << " offset_max_error_m=" << err << '\n';
    if(check) assert(fit.valid && err<1e-5);
    auto constant=fitDepthRobust(std::vector<float>(80,2),z,false);
    if(check) assert(!constant.valid);
    DepthCalibration huge;
    huge.valid=true;huge.scale=std::numeric_limits<float>::max();
    if(check) {
        assert(huge.toMetric(std::numeric_limits<float>::max(),-1)==-1);
        const float before=huge.scale;
        assert(!huge.reparameterizeLinearInput(2,0,1,0));
        assert(huge.scale==before && huge.shift==0);
    }

    // Seeded sensor noise, plus 10% gross outliers, for both supported models.
    std::mt19937 rng(153);
    std::normal_distribution<float> noise(0,.004f);
    for(bool inverse : {false,true}) {
        d.clear();z.clear();
        for(int i=0;i<200;++i) {
            const float x=.2f+i*.01f;
            const float clean=inverse ? 1.f/(.4f*x+.1f) : .7f*x+.2f;
            d.push_back(x); z.push_back(clean*(1+noise(rng))+(i%10==0?.7f:0));
        }
        fit=fitDepthRobust(d,z,inverse,10,inverse);
        err=0;
        for(float x:d) {
            const float expected=inverse?1.f/(.4f*x+.1f):.7f*x+.2f;
            err=std::max(err,double(std::fabs(fit.toMetric(x,0)-expected)/expected));
        }
        std::cout << "inverse=" << inverse << " noisy_max_relative_error=" << err << '\n';
        if(check) assert(fit.valid && err<.015);
    }
    // Exact multiples of the sampling budget must keep all 512 requested samples.
    d.resize(1024);z.resize(1024);
    for(int i=0;i<1024;++i) { d[i]=.1f+i*.001f;z[i]=.5f*d[i]+1; }
    DepthCalibrator cal; cal.update(d,z);
    std::cout << "calibration_samples=" << cal.calibration().samples << '\n';
    if(check) assert(cal.calibration().samples==512);

    SurfelEngine cloud;
    double sx=0,sy=0,sz=0;
    for(int i=0;i<100000;++i) {
        float x=1000.125f, y=(i%1000)*.02f, zz=(i/1000)*.02f;
        cloud.ingestPoint(x,y,zz,1,2,3,1);sx+=x;sy+=y;sz+=zz;
    }
    float x,y,zz;cloud.centroid(&x,&y,&zz);
    err=std::max({std::abs(x-sx/cloud.count()),std::abs(y-sy/cloud.count()),std::abs(zz-sz/cloud.count())});
    std::cout << "centroid_error_m=" << err << '\n';
    if(check) assert(cloud.count()==100000 && err<.0001);
    float out[6];
    const auto copied=cloud.copyPoints(out,1,65536);
    std::cout << "oversized_min_hits_copied=" << copied << '\n';
    if(check) assert(copied==0);
}

static uint64_t digest(const TsdfEngine& tsdf) {
    // Commutative block hash; independent of unordered_map iteration order.
    uint64_t all=0;
    tsdf.forEachBlock([&](int bx,int by,int bz,const TsdfBlock* b) {
        uint64_t h=1469598103934665603ull;
        auto add=[&](uint32_t v){ h=(h^v)*1099511628211ull; };
        add(bx);add(by);add(bz);
        for(const auto& v:b->voxels) {add(uint16_t(v.tsdf));add(v.weight);add(v.color565);add(v.colorWeight);}
        all+=h;
    });
    return all;
}

static void fusion(bool bench) {
    constexpr int n=128;
    std::vector<float> depth(n*n);
    std::vector<uint8_t> rgb(n*n*3);
    for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
        depth[y*n+x]=1.f+.001f*x+.0005f*y;
        if((x+17*y)%37==0) depth[y*n+x]=0;
        for(int c=0;c<3;++c) rgb[(y*n+x)*3+c]=uint8_t(x+2*y+50*c);
    }
    const float a=.3f, R[]={std::cos(a),0,std::sin(a),0,1,0,-std::sin(a),0,std::cos(a)};
    std::vector<double> times;
    uint64_t result=0;
    for(int rep=0;rep<(bench?7:1);++rep) {
        TsdfEngine tsdf;tsdf.setVoxelSize(.012f);tsdf.setMaxBlocks(128);
        const auto start=std::chrono::steady_clock::now();
        for(int frame=0;frame<16;++frame) {
            const float t[]={-.3f+frame*.001f,-.1f,-.05f};
            tsdf.integrateDepth(depth.data(),n,n,rgb.data(),n,n,140,145,64,64,R,t,.4f);
        }
        const auto end=std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(end-start).count()/16);
        const auto hash=digest(tsdf);
        if(rep) assert(hash==result);
        result=hash;
        assert(tsdf.voxels()>0 && tsdf.blocks()<=128);
        if(!rep) std::cout << "fusion_digest=" << result << " voxels=" << tsdf.voxels() << " blocks=" << tsdf.blocks() << '\n';
        tsdf.reset();assert(tsdf.voxels()==0 && tsdf.blocks()==0);
        const float t[]={0,0,0};
        tsdf.integrateDepth(depth.data(),n,n,nullptr,0,0,140,145,64,64,R,t,1);
        assert(tsdf.voxels()>0);
    }
    std::sort(times.begin(),times.end());
    std::cout << "fusion_median_ms_per_frame=" << times[times.size()/2] << '\n';
}

int main(int argc,char** argv) {
    const bool measure=argc>1 && std::string(argv[1])=="--measure";
    accuracy(!measure);fusion(measure);
    if(!measure) std::cout << "PASS calibration stability, noisy/outlier fits, sampling budget, centroid, hit threshold, fusion/reset\n";
}
