
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include "frame_icp.h"

struct P { float x,y,z,nx,ny,nz; };

static float surfaceZ(float x,float y){
    return .82f + .10f*x - .07f*y + .55f*x*x + .30f*x*y + .38f*y*y
         + .018f*std::sin(18.f*x) * std::cos(13.f*y);
}
static void surfaceNormal(float x,float y,float* nx,float* ny,float* nz){
    const float dfdx=.10f+1.10f*x+.30f*y
        +.018f*18.f*std::cos(18.f*x)*std::cos(13.f*y);
    const float dfdy=-.07f+.76f*y+.30f*x
        -.018f*13.f*std::sin(18.f*x)*std::sin(13.f*y);
    float a=-dfdx,b=-dfdy,c=1.f;
    const float l=std::sqrt(a*a+b*b+c*c);
    *nx=a/l;*ny=b/l;*nz=c/l;
}

int main(){
    constexpr int W=96,H=72;
    constexpr float fx=100.f,fy=100.f,cx=(W-1)*.5f,cy=(H-1)*.5f;
    std::vector<float> depth(W*H);
    std::vector<P> model;
    model.reserve(W*H);

    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        const float ux=(x-cx)/fx, uy=(y-cy)/fy;
        float z=.82f;
        for(int it=0;it<8;++it) z=surfaceZ(ux*z,uy*z);
        depth[y*W+x]=z;
        const float px=ux*z,py=uy*z,pz=z;
        float nx,ny,nz;surfaceNormal(px,py,&nx,&ny,&nz);
        model.push_back({px,py,pz,nx,ny,nz});
    }

    const float deg=1.2f*3.14159265358979323846f/180.f;
    const float cs=std::cos(deg),sn=std::sin(deg);
    float R[9]={cs,-sn,0, sn,cs,0, 0,0,1};
    float t[3]={.020f,-.014f,.009f};
    float outR[9]{},outT[3]{};

    auto nearest=[&](float x,float y,float z,float maxD,
                     float* ox,float* oy,float* oz,
                     float* nx,float* ny,float* nz,
                     uint16_t* hits)->bool{
        float best=maxD*maxD;const P* q=nullptr;
        for(const auto& p:model){
            const float dx=p.x-x,dy=p.y-y,dz=p.z-z;
            const float d2=dx*dx+dy*dy+dz*dz;
            if(d2<best){best=d2;q=&p;}
        }
        if(!q)return false;
        *ox=q->x;*oy=q->y;*oz=q->z;
        *nx=q->nx;*ny=q->ny;*nz=q->nz;*hits=8;
        return true;
    };

    auto d=frame_icp::frameToModelIcp(
        depth.data(),W,H,fx,fy,cx,cy,R,t,nearest,outR,outT);

    std::cout<<"attempted="<<d.attempted<<" applied="<<d.applied
             <<" p2l="<<d.pointToPlaneIterations
             <<" p2p="<<d.pointToPointFallbacks
             <<" inliers="<<d.inliers<<" rmse="<<d.rmseM
             <<" corrT="<<d.transM<<" corrR="<<d.rotDeg
             <<" outT="<<outT[0]<<","<<outT[1]<<","<<outT[2]<<"\n";
    assert(d.attempted && d.applied);
    assert(d.inliers>=frame_icp::kIcpMinInliers);
    assert(d.pointToPlaneIterations>0);
    assert(d.rmseM>=0.f && d.rmseM<.018f);
    const float finalT=std::sqrt(outT[0]*outT[0]+outT[1]*outT[1]+outT[2]*outT[2]);
    assert(finalT<.018f);
    return 0;
}
