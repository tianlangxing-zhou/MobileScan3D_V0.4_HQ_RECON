#include "color_contours.h"
#include "fusion_guard.h"
#include "surfel_engine.h"
#include "tsdf_engine.h"
#include "mesh/mesh_engine.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
extern "C" const char* __asan_default_options(){return "detect_leaks=0";}
static const float I[9]={1,0,0,0,1,0,0,0,1},T[3]={0,0,0};
static constexpr int W=64;
static const float K[4]={80,80,32,32};
static void contours(){
    ColorContours c;std::vector<float>d(W*W,1),weight(W*W,1);
    std::vector<uint8_t>rgb(W*W*3,100);
    c.build(rgb.data(),W,W,d.data(),W,W);assert(c.priorityPixels==0);
    for(int y=0;y<W;++y)for(int x=0;x<W;++x){size_t i=(y*W+x)*3;
        // Near-equal grayscale luminance, strongly different chroma.
        rgb[i]=x<W/2?200:0;rgb[i+1]=x<W/2?0:102;rgb[i+2]=0;}
    c.build(rgb.data(),W,W,d.data(),W,W,weight.data());
    assert(c.contourPixels>W/2 && c.priorityPixels<W*6);
    for(int y=4;y<W-4;++y){bool found=false;for(int x=30;x<=33;++x)found|=c.band[y*W+x]!=0;assert(found);}
    std::cout<<"chroma_contour_pixels="<<c.contourPixels<<" band="<<c.priorityPixels<<'\n';
    std::fill(weight.begin(),weight.end(),0);c.build(rgb.data(),W,W,d.data(),W,W,weight.data());assert(!c.priorityPixels);
    weight.assign(W*W,1);d.assign(W*W,std::numeric_limits<float>::quiet_NaN());
    c.build(rgb.data(),W,W,d.data(),W,W,weight.data());assert(!c.priorityPixels);
    d.assign(W*W,1);rgb.assign(W*W*3,100);rgb[(32*W+32)*3]=255;
    c.build(rgb.data(),W,W,d.data(),W,W);assert(c.priorityPixels==0);
    // Curved colored disk must produce a connected outline, not fill its interior.
    for(int y=0;y<W;++y)for(int x=0;x<W;++x)for(int ch=0;ch<3;++ch)
        rgb[(y*W+x)*3+ch]=(x-32)*(x-32)+(y-32)*(y-32)<18*18?230:20;
    c.build(rgb.data(),W,W,d.data(),W,W);assert(c.contourPixels>60 && !c.band[32*W+32]);
    assert(c.priorityPixels<900);c.build(nullptr,W,W,d.data(),W,W);assert(c.band.empty());
    std::cout<<"PASS flat color, isoluminant blocks, curved outlines, noise, confidence/NaN gates\n";
}
static void guard(){
    FusionGuard g;std::vector<float>d(W*W,1.003f);uint64_t ts=1000000000;
    assert(g.accept(d.data(),W,W,K,I,T,ts));g.commit(d.data(),W,W,K,I,T,ts);
    assert(g.references()==1);
    TsdfEngine volume;volume.setVoxelSize(.01f);
    volume.integrateDepth(d.data(),W,W,nullptr,0,0,K[0],K[1],K[2],K[3],I,T,1);
    const auto before=volume.voxels();
    // A persistent wrong depth layer must remain rejected even minutes later.
    d.assign(W*W,1.08f);
    for(int frame=1;frame<=8;++frame){bool accepted=g.accept(d.data(),W,W,K,I,T,ts+frame*3000000000ULL);
        assert(!accepted);if(accepted)volume.integrateDepth(d.data(),W,W,nullptr,0,0,K[0],K[1],K[2],K[3],I,T,1);}
    assert(volume.voxels()==before && g.references()==1 && g.rejected==8);
    d.assign(W*W,1.003f);assert(g.accept(d.data(),W,W,K,I,T,30000000000ULL));
    assert(!g.accept(d.data(),W,W,K,I,T,ts));
    const float translated[3]={.12f,0,0};assert(g.accept(d.data(),W,W,K,I,translated,31000000000ULL));
    g.commit(d.data(),W,W,K,I,translated,31000000000ULL);assert(g.references()==2);
    // Camera motion along Z should agree when the true camera depth changes.
    const float forward[3]={0,0,.08f};d.assign(W*W,.923f);
    assert(g.accept(d.data(),W,W,K,I,forward,32000000000ULL));
    // Novel view with no overlap is permitted; it does not prove registration.
    const float novel[3]={3,0,0};d.assign(W*W,1.003f);
    assert(g.accept(d.data(),W,W,K,I,novel,33000000000ULL));
    float changedK[4]={81,80,32,32};assert(!g.accept(d.data(),W,W,changedK,I,T,34000000000ULL));
    for(int j=0;j<8;++j){float at[3]={.2f*j,0,0};g.commit(d.data(),W,W,K,I,at,35000000000ULL+j*1000000000ULL);}
    assert(g.references()==4);d.assign(W*W,1.08f);
    assert(!g.accept(d.data(),W,W,K,I,T,60000000000ULL));g.reset();assert(!g.references()&&!g.rejected);
    std::cout<<"PASS persistent second-shell rejection, accepted-only history, recovery, camera motion, novel views, bounded anchors/reset\n";
}
static void rendering(){
    SurfelEngine g;const adaptive::Geometry ordinary{0,0,1,false,false},edge{0,0,1,true,true};
    for(int f=0;f<3;++f){g.beginFrame();for(int i=0;i<100;++i)
        g.ingestAdaptivePoint(i*.03f+.005f,.005f,1.003f,i<10?255:0,0,0,1,i<10?edge:ordinary,i,1);}
    float out[20*6];assert(g.copyPoints(out,20,3)==20);
    int red=0;for(int i=0;i<20;++i)if(out[i*6+3]>.5f)++red;assert(red==10);
    float one[6];assert(g.copyPoints(one,1,3)==1 && one[3]>.5f);
    std::vector<float>all(100*6);assert(g.copyPoints(all.data(),100,3)==100);
    assert(g.copyPoints(all.data(),100,4)==0);
    std::cout<<"PASS contour display quota and ordinary surface coverage; no duplicate points\n";
}
static void actualFusion(){
    std::vector<float>d(W*W,1.003f),weights(W*W,1);std::vector<uint8_t>rgb(W*W*3);
    for(int y=0;y<W;++y)for(int x=0;x<W;++x)for(int c=0;c<3;++c)rgb[(y*W+x)*3+c]=x<32?20:230;
    ColorContours contours;contours.build(rgb.data(),W,W,d.data(),W,W,weights.data());
    TsdfEngine e;e.setVoxelSize(.01f);
    for(int i=0;i<16;++i)e.integrateDepth(d.data(),W,W,rgb.data(),W,W,80,80,32,32,I,T,1,1,0,weights.data(),true,contours.band.data());
    MeshEngine mesh;MeshOptions options;options.enableSmoothing=options.enableDecimation=options.enableComponentFilter=false;
    MeshBuildStats stats;assert(mesh.build(e,options,stats));
    float error=0;const auto& pos=mesh.mesh().positions;
    for(size_t i=0;i<pos.size();i+=3)error=std::max(error,std::fabs(pos[i+2]-1.003f));
    assert(error<.001f);auto vox=e.voxels();std::fill(weights.begin(),weights.end(),0);float moved[3]={4,0,0};
    e.integrateDepth(d.data(),W,W,rgb.data(),W,W,80,80,32,32,I,moved,1,1,0,weights.data(),true,contours.band.data());
    assert(e.voxels()==vox);
    std::cout<<"painted_plane_mesh_max_error_m="<<error<<" PASS no artificial ridge, zero-confidence rejection\n";
}
int main(){contours();guard();rendering();actualFusion();}
