#include "adaptive_sampling.h"
#include "surfel_engine.h"
#include "tsdf_engine.h"
#include "mesh/mesh_engine.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }
const float I[]={1,0,0,0,1,0,0,0,1}, T[]={0,0,0};
const adaptive::Geometry plane{0,0,1,false};
static void classification() {
    constexpr int n=32; std::vector<float> d(n*n,1);
    assert(!adaptive::geometry(d.data(),n,n,16,16,80,80,16,16,I).protectedDetail);
    for(int y=0;y<n;++y)for(int x=0;x<n;++x)d[y*n+x]=1.f/(1+.3f*(x-16)/120+.2f*(y-16)/120);
    auto normal=adaptive::geometry(d.data(),n,n,16,16,120,120,16,16,I);
    assert(!normal.protectedDetail && std::fabs(normal.nx/normal.nz-.3f)<1e-4f);
    d.assign(n*n,1);d[16*n+18]=0;
    assert(adaptive::geometry(d.data(),n,n,16,16,80,80,16,16,I).protectedDetail);
    d[16*n+18]=std::numeric_limits<float>::quiet_NaN();
    assert(adaptive::geometry(d.data(),n,n,16,16,80,80,16,16,I).protectedDetail);
    d.assign(n*n,1);for(int y=0;y<n;++y)d[y*n+16]=.98f;
    assert(adaptive::geometry(d.data(),n,n,15,16,80,80,16,16,I).protectedDetail);
    // Smoothly curved bump: no discontinuity but non-planar second derivative.
    d.assign(n*n,1);d[16*n+16]=.997f;
    assert(adaptive::geometry(d.data(),n,n,16,16,80,80,16,16,I).protectedDetail);
    std::cout<<"PASS planar normals, mask/hole halo, NaN, thin ridge and curvature protection\n";
}
static void history() {
    adaptive::Sampler s(8);s.beginFrame(.04f);
    for(int i=0;i<10000;++i)assert(s.select(-.011f,.001f,1.001f,plane,1,i%32,i/32));
    assert(s.stats().skipped==0); // same-frame pixel counts do not prove stability
    for(int f=0;f<16;++f){s.beginFrame(.04f);for(int i=0;i<64;++i)s.select(-.011f,.001f,1.001f,plane,1,i%8,i/8);}
    assert(s.stats().skipped>0);
    assert(s.select(-.011f,.001f,1.009f,plane,1,0,0));assert(s.stats().reactivated>0);
    for(int i=0;i<100;++i)assert(s.select(float(i),0,1,plane,1,0,0));
    assert(s.cells()==8 && s.stats().evicted>0);
    s.reset();assert(s.cells()==0 && s.stats().candidates==0);
    std::cout<<"PASS multi-frame stability, world-space novelty, reactivation, history budget/reset\n";
}
static void cloud() {
    SurfelEngine g;size_t initial=0,bytes=0;
    for(int f=0;f<64;++f){g.beginFrame();for(int y=-20;y<20;++y)for(int x=-20;x<20;++x){
        g.ingestAdaptivePoint(x*.01f+.004f,y*.01f+.004f,1.001f,120,130,140,1,plane,x+20,y+20);}
        if(f==0){initial=g.count();bytes=g.storageBytes();}g.endFrame();}
    std::cout<<"preview_points_before="<<initial<<" after="<<g.count()<<" bytes_before="<<bytes<<" after="<<g.storageBytes()<<" patches="<<g.compressedCount()<<'\n';
    assert(initial==1600 && g.count()<initial*.6 && g.storageBytes()<bytes);
    assert(g.stableCount()==g.count() && g.confirmedCount(3)==g.count());
    const size_t before=g.count();g.beginFrame();
    // A newly observed detail breaks a compacted patch and consumes freed quota.
    g.ingestAdaptivePoint(.004f,.004f,1.007f,255,0,0,1,adaptive::Geometry{},1,1);
    assert(g.reactivatedCount()>0 && g.count()>=before);
    for(int y=0;y<10;++y)for(int x=0;x<10;++x)
        g.ingestAdaptivePoint(2+x*.01f,2+y*.01f,1,0,0,0,1,plane,x,y);
    assert(g.count()>=before+100);
    float minx,miny,minz,maxx,maxy,maxz;g.boundingBox(&minx,&miny,&minz,&maxx,&maxy,&maxz);assert(minx<0&&maxx>2);
    g.reset();assert(g.count()==0&&g.compressedCount()==0&&g.reclaimedCount()==0);
    // Protected edges and conflicting normals in a coarse cell never compress.
    for(int f=0;f<48;++f){g.beginFrame();for(int y=0;y<10;++y)for(int x=0;x<10;++x){
        g.ingestAdaptivePoint(x*.01f+.004f,y*.01f+.004f,1.001f,0,0,0,1,adaptive::Geometry{},x,y);}g.endFrame();}
    assert(g.count()==100&&g.compressedCount()==0);
    g.reset();
    for(int f=0;f<48;++f){g.beginFrame();for(int y=0;y<10;++y)for(int x=0;x<10;++x){
        auto normal=plane;if(x%2)normal.nz=-1;
        g.ingestAdaptivePoint(x*.01f+.004f,y*.01f+.004f,1.001f,0,0,0,1,normal,x,y);}g.endFrame();}
    assert(g.count()==100&&g.compressedCount()==0);
    std::cout<<"PASS planar point/memory reclamation, new region, compressed patch reactivation, negative grid, edge/normal protection\n";
}
static void fusion(bool measure) {
    constexpr int n=96;std::vector<float> depth(n*n,1.013f),weights(n*n,1);
    // A step, a hole and a one-pixel thin structure exercise protection in the real integrator.
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        if(x>=60)depth[y*n+x]=1.113f;
        if(x==23)depth[y*n+x]=.983f;
        if(x>35&&x<42&&y>35&&y<42)depth[y*n+x]=0;
    }
    TsdfEngine dense,adapt;
    dense.setVoxelSize(.01f);adapt.setVoxelSize(.01f);
    double timings[2]={};
    for(int mode=0;mode<2;++mode){auto& volume=mode?adapt:dense;
        const auto start=std::chrono::steady_clock::now();
        for(int f=0;f<48;++f)volume.integrateDepth(depth.data(),n,n,nullptr,0,0,120,120,n/2,n/2,I,T,.8f,1,0,weights.data(),mode);
        timings[mode]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/48;
    }
    const auto s=adapt.adaptiveStats();
    assert(s.skipped>0 && s.protectedSamples>0 && adapt.voxels()==dense.voxels());
    double maxSdf=0;size_t missing=0;
    dense.forEachBlock([&](int bx,int by,int bz,const TsdfBlock* block){for(int z=0;z<8;++z)for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        const auto& v=block->voxels[(z*8+y)*8+x];if(!v.weight)continue;
        const int vx=bx*8+x,vy=by*8+y,vz=bz*8+z;
        if(!adapt.weightAt(vx,vy,vz))++missing;
        maxSdf=std::max(maxSdf,double(std::fabs(dense.valueAt(vx,vy,vz)-adapt.valueAt(vx,vy,vz)))*dense.truncation());
    }});
    assert(missing==0 && maxSdf<.003);
    MeshOptions opt;opt.enableComponentFilter=opt.enableSmoothing=opt.enableDecimation=false;
    MeshEngine m1,m2;MeshBuildStats a,b;assert(m1.build(dense,opt,a)&&m2.build(adapt,opt,b));
    assert(m2.mesh().triangleCount()>m1.mesh().triangleCount()*.98);
    std::cout<<"tsdf_candidates="<<s.candidates<<" skipped="<<s.skipped<<" max_field_difference_m="<<maxSdf
             <<" triangles_dense="<<m1.mesh().triangleCount()<<" triangles_adaptive="<<m2.mesh().triangleCount()<<'\n';
    if(measure)std::cout<<"dense_ms_per_frame="<<timings[0]<<" adaptive_ms_per_frame="<<timings[1]<<'\n';
    const auto selected=s.selected;const auto react=s.reactivated;
    for(auto& d:depth)if(d>0)d+=.009f;
    adapt.integrateDepth(depth.data(),n,n,nullptr,0,0,120,120,n/2,n/2,I,T,.8f,1,0,weights.data(),true);
    assert(adapt.adaptiveStats().selected>selected && adapt.adaptiveStats().reactivated>react);
    const auto before=adapt.voxels();std::fill(weights.begin(),weights.end(),0);
    const float moved[]={4,0,0};adapt.integrateDepth(depth.data(),n,n,nullptr,0,0,120,120,n/2,n/2,I,moved,.8f,1,0,weights.data(),true);
    assert(adapt.voxels()==before);adapt.reset();assert(adapt.adaptiveCells()==0);
    std::cout<<"PASS TSDF coverage/mesh retention, contradiction reactivation, zero-confidence rejection/reset\n";
}
static void movingPlane() {
    constexpr int n=64;std::vector<float> d(n*n),weight(n*n,1);
    TsdfEngine dense,adapt;dense.setVoxelSize(.008f);adapt.setVoxelSize(.008f);
    // World plane z=1.019 + .15*x; varying camera translation, seeded sub-mm noise.
    uint32_t rng=9;
    for(int f=0;f<32;++f){
        const float t[]={f*.0007f,0,0};
        for(int y=0;y<n;++y)for(int x=0;x<n;++x){
            rng=1664525u*rng+1013904223u;
            d[y*n+x]=(1.019f+.15f*t[0])/(1-.15f*(x-32)/120.f)+(int(rng>>24)-128)*.000002f;
        }
        for(int mode=0;mode<2;++mode)(mode?adapt:dense).integrateDepth(d.data(),n,n,nullptr,0,0,120,120,32,32,I,t,.8f,1,0,weight.data(),mode);
    }
    double maxDifference=0;size_t points=0,missing=0;
    for(int y=-20;y<20;++y)for(int x=-20;x<20;++x)for(int z=123;z<132;++z){
        if(!dense.weightAt(x,y,z) || std::fabs(dense.valueAt(x,y,z))>.5f)continue;
        ++points; if(!adapt.weightAt(x,y,z)){++missing;continue;}
        maxDifference=std::max(maxDifference,double(std::fabs(dense.valueAt(x,y,z)-adapt.valueAt(x,y,z)))*dense.truncation());
    }
    // Ray subsampling may leave fringe lattice samples unobserved; bound the
    // near-surface loss and separately validate extracted geometry below.
    std::cout<<"moving coverage missing="<<missing<<" / "<<points<<" diff="<<maxDifference<<"\n";
    assert(points>1000 && missing<points/100 && adapt.adaptiveStats().skipped>0 && maxDifference<.002);
    std::cout<<"moving_noisy_plane_max_field_difference_m="<<maxDifference<<" checked_voxels="<<points<<'\n';
    MeshOptions opt;opt.enableSmoothing=opt.enableComponentFilter=opt.enableDecimation=false;
    MeshEngine denseMesh,adaptiveMesh;MeshBuildStats ds,as;
    assert(denseMesh.build(dense,opt,ds) && adaptiveMesh.build(adapt,opt,as));
    assert(adaptiveMesh.mesh().triangleCount()>denseMesh.mesh().triangleCount()*.98);
    float maxPlaneError=0;size_t interior=0;
    const auto& positions=adaptiveMesh.mesh().positions;
    for(size_t i=0;i<positions.size();i+=3) {
        if(std::fabs(positions[i])>.15f || std::fabs(positions[i+1])>.15f)continue;
        const float error=std::fabs(positions[i+2]-1.019f-.15f*positions[i]);
        maxPlaneError=std::max(maxPlaneError,error);++interior;
    }
    assert(interior>100 && maxPlaneError<.003f);
    std::cout<<"moving_plane_mesh_max_error_m="<<maxPlaneError<<" triangles_dense="<<denseMesh.mesh().triangleCount()
             <<" triangles_adaptive="<<adaptiveMesh.mesh().triangleCount()<<'\n';
    // A translated new view must add blocks even after old regions become stable.
    const auto before=adapt.blocks();const float t[]={2,0,0};
    adapt.integrateDepth(d.data(),n,n,nullptr,0,0,120,120,32,32,I,t,.8f,1,0,weight.data(),true);
    assert(adapt.blocks()>before);
    // Sparse stereo/default calls bypass adaptation, including after a dense call.
    const auto c=adapt.adaptiveStats().candidates;
    adapt.integrateDepth(d.data(),n,n,nullptr,0,0,120,120,32,32,I,t,.8f);
    assert(adapt.adaptiveStats().candidates==c);
    std::cout<<"PASS moving noisy slanted plane, novel view and sparse/default bypass\n";
}
int main(int argc,char** argv){classification();history();cloud();movingPlane();fusion(argc>1&&std::string(argv[1])=="--measure");}
