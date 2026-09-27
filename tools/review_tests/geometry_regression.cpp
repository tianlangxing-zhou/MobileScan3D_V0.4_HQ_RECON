#include <algorithm>
#include "depth_geometry.h"
#include "depth_calib.h"
#include "tsdf_engine.h"
#include "surfel_engine.h"
#include "mesh/mesh_engine.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <map>

extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }

int main() {
    const auto k = depthIntrinsics(1000, 1000, 640, 360, 1280, 720, 256, 256);
    assert(std::abs(k.fx - 200) < 1e-5 && std::abs(k.cy - 128) < 1e-5);
    // Same camera ray at source and depth resolutions, including nonsquare source.
    assert(std::abs((192-k.cx)/k.fx - (960.f-640)/1000) < 1e-6);
    assert(std::abs((192-k.cy)/k.fy - (540.f-360)/1000) < 1e-6);

    std::vector<float> raw, metric;
    for (int i=0; i<100; ++i) { raw.push_back(.2f+i*.02f); metric.push_back(1.f/(.4f*raw.back()+.1f)); }
    const auto fit = fitDepthRobust(raw, metric, true, 10, true);
    assert(fit.valid && fit.inverseDepthModel);
    assert(std::abs(fit.toMetric(1,0)-2) < .001f);
    DepthCalibrator c;
    auto cfg=c.config(); cfg.forceInverseDepth=true; c.setConfig(cfg);
    assert(c.update(raw,metric));
    auto invalid=metric;
    for (int i=0;i<90;++i) invalid[i]=std::numeric_limits<float>::quiet_NaN();
    assert(!c.update(raw,invalid));
    assert(!c.update(raw, std::vector<float>(5,1)));
    cfg.maxSamples=0;c.setConfig(cfg);assert(!c.update(raw,metric));
    cfg.maxSamples=512;c.setConfig(cfg);
    auto opposite=metric;
    for(size_t i=0;i<raw.size();++i) opposite[i]=1.f/(1.5f-.4f*raw[i]);
    assert(!c.update(raw,opposite)); // Never EMA across a slope sign reversal.

    SurfelEngine surfels;
    surfels.ingestPoint(-1,-2,-3,255,0,0,1);
    surfels.ingestPoint(0,0,12,255,0,0,1);
    assert(surfels.count()==2);
    surfels.ingestPoint(std::numeric_limits<float>::quiet_NaN(),0,1,0,0,0,1);
    surfels.ingestPoint(0,0,1,0,0,0,std::numeric_limits<float>::quiet_NaN());
    assert(surfels.count()==2);

    constexpr int N=64;
    std::vector<float> plane(N*N,1.007f);
    const float R[]={1,0,0,0,1,0,0,0,1}, t[]={0,0,0};
    TsdfEngine tsdf;
    tsdf.setVoxelSize(.02f);tsdf.setMaxBlocks(64);
    assert(tsdf.pixelStep()==1);
    tsdf.integrateDepth(plane.data(),N,N,nullptr,0,0,80,80,32,32,R,t,1);
    assert(tsdf.surfacePresent());
    // Every observed lattice position must encode signed projective distance.
    int checked=0;
    tsdf.forEachBlock([&](int bx,int by,int bz,const TsdfBlock* block){
        for(int z=0;z<8;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x) {
            const auto& v=block->voxels[(z*8+y)*8+x];
            if(!v.weight) continue;
            const float expected=std::clamp((1.007f-(bz*8+z)*.02f)/.08f,-1.f,1.f);
            assert(std::abs(v.tsdf/kTsdfValueScale-expected)<.0001f); ++checked;
        }
    });
    assert(checked>1000);
    // Fill the allocation budget by scanning widely separated patches.
    for(int i=0; i<100 && !tsdf.atCapacity(); ++i){
        const float moved[]={float(i+1),0,0};
        tsdf.integrateDepth(plane.data(),N,N,nullptr,0,0,80,80,32,32,R,moved,.25f);
    }
    assert(tsdf.atCapacity());
    auto weights=[&](){uint64_t s=0;tsdf.forEachBlock([&](int,int,int,const TsdfBlock* b){for(auto& v:b->voxels)s+=v.weight;});return s;};
    auto before=weights();
    tsdf.integrateDepth(plane.data(),N,N,nullptr,0,0,80,80,32,32,R,t,1);
    assert(tsdf.blocks()==64 && weights()>before);
    before=weights();
    tsdf.integrateDepth(plane.data(),N,N,nullptr,0,0,80,80,32,32,R,t,std::numeric_limits<float>::quiet_NaN());
    assert(weights()==before);

    // End-to-end planar reconstruction: extraction must retain the actual surface depth.
    TsdfEngine dense;dense.setVoxelSize(.02f);
    for(int i=0;i<3;++i) dense.integrateDepth(plane.data(),N,N,nullptr,0,0,80,80,32,32,R,t,1);
    MeshEngine mesh;MeshOptions options;MeshBuildStats stats;
    options.enableSmoothing=false;options.enableDecimation=false;options.enableComponentFilter=false;
    assert(mesh.build(dense,options,stats));assert(mesh.mesh().triangleCount()>100);
    std::vector<float> depths;
    for(size_t i=2;i<mesh.mesh().positions.size();i+=3)depths.push_back(mesh.mesh().positions[i]);
    std::cerr << "plane z median=" << medianOf(depths) << " count=" << depths.size() << "\n";
    for (float z : depths) assert(std::abs(z-1.007f)<.002f);
    std::map<std::pair<unsigned,unsigned>,int> edges;
    const auto& indices=mesh.mesh().indices;
    for(size_t i=0;i<indices.size();i+=3) for(int e=0;e<3;++e) {
        auto a=indices[i+e], b=indices[i+(e+1)%3];
        if(a>b) std::swap(a,b);
        assert((++edges[{a,b}]<=2)); // No branching surfaces from diagonal-key collisions.
    }
    for(const auto& [edge,count]:edges) {
        const auto& p=mesh.mesh().positions;
        if(std::abs(p[edge.first*3])<.2f && std::abs(p[edge.first*3+1])<.2f &&
           std::abs(p[edge.second*3])<.2f && std::abs(p[edge.second*3+1])<.2f)
            assert(count==2); // Interior edges must have two neighboring faces.
    }
    std::cout << "PASS projection, fixed inverse calibration, invalid samples, sign reversal, world coordinates, TSDF budget and plane mesh\n";
}
