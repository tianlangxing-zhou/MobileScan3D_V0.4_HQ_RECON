#include "scan_policy.h"
#include "tsdf_engine.h"
#include "mesh/mesh_engine.h"
#include <cassert>
#include <iostream>
#include <limits>
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }
int main() {
    using scan_policy::inRange;
    assert(inRange(.5f,0,0,1));
    assert(inRange(1,0,0,1));
    assert(!inRange(1.001f,0,0,1));
    assert(!inRange(.8f,1,0,1)); // Z is close, Euclidean distance is far.
    assert(inRange(2,0,0,1,2)); // physical-meter/world conversion
    assert(!inRange(2.01f,0,0,1,2));
    for(float v: {0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        assert(!inRange(v,0,0,1));
    assert(!inRange(.5f,0,0,1,0));
    assert(!inRange(.5f,std::numeric_limits<float>::quiet_NaN(),0,1));

    DepthCalibration frozen; frozen.valid=true; frozen.inverseDepthModel=true;
    frozen.scale=.4f; frozen.shift=.1f;
    std::vector<float> raw, world;
    for(int i=0;i<30;++i){raw.push_back(1.f+i*.05f);world.push_back(frozen.toMetric(raw.back(),0));}
    assert(scan_policy::validatesFrozen(frozen,raw,world));
    auto wrong=world;for(auto& z:wrong)z*=1.3f;
    assert(!scan_policy::validatesFrozen(frozen,raw,wrong));
    assert(!scan_policy::validatesFrozen(frozen,raw,std::vector<float>(19,1)));
    auto noisy=world;for(int i=0;i<6;++i)noisy[i]*=2;
    assert(scan_policy::validatesFrozen(frozen,raw,noisy));
    noisy[6]*=2;assert(!scan_policy::validatesFrozen(frozen,raw,noisy));

    // Far geometry must never enter the reconstructed mesh. Two adjacent planes,
    // same frame: left at 0.5m, right at 2m; only left survives a 1m limit.
    constexpr int N=64;
    std::vector<float> depth(N*N);
    for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        const float z=x<N/2?.503f:2.f;
        depth[y*N+x]=inRange(z,(x-32.f)/80,(y-32.f)/80,1)?z:0;
    }
    const float R[]={1,0,0,0,1,0,0,0,1},t[]={0,0,0};
    TsdfEngine volume;volume.setVoxelSize(.01f);
    for(int i=0;i<8;++i)volume.integrateDepth(depth.data(),N,N,nullptr,0,0,80,80,32,32,R,t,1);
    MeshEngine mesh;MeshOptions options;MeshBuildStats stats;
    options.enableSmoothing=false;options.enableComponentFilter=false;options.enableDecimation=false;
    assert(mesh.build(volume,options,stats));assert(mesh.mesh().triangleCount()>20);
    for(size_t i=2;i<mesh.mesh().positions.size();i+=3)
        assert(std::fabs(mesh.mesh().positions[i]-.503f)<.002f);
    std::cout<<"PASS radial range, meter conversion, invalid inputs, frozen evidence, near-only TSDF mesh\n";
}
