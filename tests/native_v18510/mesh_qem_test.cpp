
#include <cassert>
#include <cmath>
#include <iostream>
#include "mesh/mesh_engine.h"
int main(){
    Mesh m;
    constexpr int N=22;
    for(int y=0;y<N;++y) for(int x=0;x<N;++x){
        float fx=(x-(N-1)*.5f)*.01f,fy=(y-(N-1)*.5f)*.01f;
        float fz=.003f*std::sin(fx*25.f)*std::cos(fy*22.f);
        m.positions.insert(m.positions.end(),{fx,fy,fz});
        m.colors.insert(m.colors.end(),{.5f,.6f,.7f});
    }
    for(int y=0;y<N-1;++y) for(int x=0;x<N-1;++x){
        uint32_t a=y*N+x,b=a+1,c=(y+1)*N+x+1,d=(y+1)*N+x;
        m.indices.insert(m.indices.end(),{a,b,c,a,c,d});
    }
    const auto before=m.triangleCount();
    MeshBuildStats st;
    bool ok=meshDecimate(m,before/2,&st);
    std::cout<<"before="<<before<<" after="<<m.triangleCount()
             <<" verts="<<m.vertexCount()<<"\n";
    assert(ok && m.triangleCount()<before && m.triangleCount()>before/4);
    for(size_t t=0;t<m.indices.size();t+=3){
        const float* a=&m.positions[m.indices[t]*3];
        const float* b=&m.positions[m.indices[t+1]*3];
        const float* c=&m.positions[m.indices[t+2]*3];
        float ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2];
        float vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
        float nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        float area2=std::sqrt(nx*nx+ny*ny+nz*nz);
        assert(area2>1e-8f);
        assert(nz>=-1e-7f);
    }
    return 0;
}
