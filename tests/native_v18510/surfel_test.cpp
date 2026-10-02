
#include <cassert>
#include <cmath>
#include <iostream>
#include "surfel_engine.h"
int main(){
    SurfelEngine s;
    adaptive::Geometry g;
    g.nx=0;g.ny=0;g.nz=1;g.protectedDetail=false;
    for(int f=0;f<16;++f){
        s.beginFrame();
        s.ingestAdaptivePoint(.001f,.001f,1.f,120,130,140,.9f,g,10,10);
        s.ingestAdaptivePoint(.011f,.001f,1.f,120,130,140,.9f,g,20,10);
        s.ingestAdaptivePoint(.001f,.011f,1.f,120,130,140,.9f,g,10,20);
        s.endFrame();
    }
    float x=0,y=0,z=0,nx=0,ny=0,nz=0;uint16_t hits=0;
    bool ok=s.nearestStableSurfel(.056f,.004f,1.f,.060f,
                                  &x,&y,&z,&nx,&ny,&nz,&hits);
    std::cout<<"count="<<s.count()<<" coarse="<<s.compressedCount()
             <<" found="<<ok<<" q="<<x<<","<<y<<","<<z
             <<" n="<<nx<<","<<ny<<","<<nz<<" hits="<<hits<<"\n";
    assert(s.compressedCount()>=1);
    assert(ok);
    assert(std::fabs(nz-1.f)<1e-3f);
    assert(hits>=3);
    return 0;
}
