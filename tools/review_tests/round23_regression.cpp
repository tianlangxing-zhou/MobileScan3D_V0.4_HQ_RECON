#include "depth_confidence.h"
#include "mesh/hard_surface.h"
#include "v06/uv_unwrap.h"
#include "v06/textured_glb_exporter.h"
#include "export/gltf_exporter.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <chrono>
#include <algorithm>

struct V{float x,y,z;};
V rot(V p){float c=.8253356f,s=.5646425f;float y=.921061f*p.y-.389418f*p.z,z=.389418f*p.y+.921061f*p.z;return {c*p.x-s*y+1.2f,s*p.x+c*y-.7f,z+2.3f};}
Mesh box(float x,float y,float z,int faces=6,float noise=0){
 Mesh m;std::mt19937 rng(42);std::normal_distribution<float> jitter(0,noise);
 for(int f=0;f<faces;++f){int axis=f/2,u=(axis+1)%3,v=(axis+2)%3;float dims[3]={x,y,z};bool side=f%2;
  // Three adjacent faces are selected explicitly for incomplete scans.
  if(faces==3){axis=f;u=(axis+1)%3;v=(axis+2)%3;side=true;}
  int base=m.vertexCount();int grid=12;
  for(int b=0;b<=grid;++b)for(int a=0;a<=grid;++a){float p[3]{};p[axis]=(side?.5f:-.5f)*dims[axis];p[u]=(float(a)/grid-.5f)*dims[u];p[v]=(float(b)/grid-.5f)*dims[v];
   p[axis]+=jitter(rng);V q=rot({p[0],p[1],p[2]});m.positions.insert(m.positions.end(),{q.x,q.y,q.z});m.colors.insert(m.colors.end(),{.2f,.5f,.8f});}
  for(int b=0;b<grid;++b)for(int a=0;a<grid;++a){uint32_t k=base+b*(grid+1)+a,n=grid+1;
   if(side)m.indices.insert(m.indices.end(),{k,k+1,k+n+1,k,k+n+1,k+n});else m.indices.insert(m.indices.end(),{k,k+n+1,k+1,k,k+n,k+n+1});}
 }
 meshRecomputeNormals(m);return m;
}
Mesh sphere(){Mesh m;const int rows=40,cols=80;for(int y=0;y<=rows;++y)for(int x=0;x<=cols;++x){float a=3.14159265f*y/rows,b=6.2831853f*x/cols;m.positions.insert(m.positions.end(),{std::sin(a)*std::cos(b),std::sin(a)*std::sin(b),std::cos(a)});}
 for(int y=0;y<rows;++y)for(int x=0;x<cols;++x){uint32_t k=y*(cols+1)+x,n=cols+1;if(y>0)m.indices.insert(m.indices.end(),{k,k+1,k+n});if(y<rows-1)m.indices.insert(m.indices.end(),{k+1,k+n+1,k+n});}meshRecomputeNormals(m);return m;}
void temporal(){int w=32,h=24,n=w*h;std::vector<float>a(n,1),b(n,1),scratch,weights;std::vector<uint8_t>mask(n);float R[9]={1,0,0,0,1,0,0,0,1},t[3]={};int tested=0,agree=0;
 temporalConsistencyMask(a.data(),b.data(),w,h,30,30,16,12,R,t,mask.data(),1,.015f,.025f,&tested,&agree,&scratch);assert(tested==n && agree==n);
 auto capacity=scratch.capacity();
 for(int y=0;y<h;++y)for(int x=0;x<w;++x)b[y*w+x]=(x<w/2?1.2f:1.f);
 temporalConsistencyMask(a.data(),b.data(),w,h,30,30,16,12,R,t,mask.data(),1,.015f,.025f,&tested,&agree,&scratch);
 assert(mask[12*w+4]==kTemporalDisagree);assert(mask[12*w+15]==kTemporalDisoccluded);assert(mask[12*w+24]==kTemporalAgree);assert(capacity==scratch.capacity());
 buildPixelWeight(b.data(),w,h,mask.data(),weights);assert(weights[12*w+4]==0);assert(weights[12*w+24]>.99f);assert(weights[12*w+15]>.1f);
 std::vector<float>source(n,.4f);source[12*w+24]=std::numeric_limits<float>::quiet_NaN();buildPixelWeight(b.data(),w,h,nullptr,weights,source.data());assert(weights[12*w+24]==0);assert(weights[12*w+25]>.3f && weights[12*w+25]<.4f);
 std::fill(a.begin(),a.end(),0);temporalConsistencyMask(a.data(),b.data(),w,h,30,30,16,12,R,t,mask.data(),1,.015f,.025f,&tested,&agree,&scratch);assert(tested==0 && std::all_of(mask.begin(),mask.end(),[](int v){return v==kTemporalUntested;}));
 std::fill(a.begin(),a.end(),1);std::fill(b.begin(),b.end(),1);t[0]=1e30f;temporalConsistencyMask(a.data(),b.data(),w,h,30,30,16,12,R,t,mask.data(),1,.015f,.025f,&tested,&agree,&scratch);assert(tested==0);
 // Collision: two old surfaces project to one pixel; closest one wins regardless of order.
 std::fill(a.begin(),a.end(),0);std::fill(b.begin(),b.end(),1);t[0]=1; a[10*w+5]=1; a[10*w+10]=2;
 temporalConsistencyMask(a.data(),b.data(),w,h,10,10,0,10,R,t,mask.data(),1,.015f,.025f,&tested,&agree,&scratch);assert(mask[10*w+15]==kTemporalAgree);assert(scratch[10*w+15]==1);
 buildPixelWeight(nullptr,-1,4,nullptr,weights);assert(weights.empty());
 buildPixelWeight(nullptr,4,4,nullptr,weights);assert(std::all_of(weights.begin(),weights.end(),[](float v){return v==0;}));
 // Invalid camera and resized output always clear masks.
 mask.assign(16,kTemporalAgree);temporalConsistencyMask(a.data(),b.data(),4,4,0,10,0,0,R,t,mask.data());assert(std::all_of(mask.begin(),mask.end(),[](int v){return v==0;}));
 std::cout<<"PASS temporal identity, conflicts, visibility changes, z-buffer, huge projection, source confidence, resize, buffer reuse\n";
}
void fusion(){int w=32,h=32;std::vector<float>d(w*h,1),pw(w*h,0);float R[9]={1,0,0,0,1,0,0,0,1},t[3]={};TsdfEngine a,b;
 a.integrateDepth(d.data(),w,h,nullptr,0,0,30,30,16,16,R,t,.5f,1,0,pw.data());assert(a.voxels()==0);
 std::fill(pw.begin(),pw.end(),1);a.integrateDepth(d.data(),w,h,nullptr,0,0,30,30,16,16,R,t,.5f,1,0,pw.data());b.integrateDepth(d.data(),w,h,nullptr,0,0,30,30,16,16,R,t,.5f);assert(a.voxels()>0 && a.voxels()==b.voxels());
 std::cout<<"PASS zero weights prevent TSDF allocation; unit weights preserve voxel count\n";
}
void geometry(const char* dir){Mesh out;HardSurfaceStats st;auto in=box(.8f,.6f,.4f);auto start=std::chrono::steady_clock::now();
 assert(fitHardSurface(in,HardSurfaceMode::Cuboid,.004f,out,st));assert(out.vertexCount()==24 && out.triangleCount()==12 && st.inferredFaces==0);assert(st.rms<.002f);
 std::cout<<"cuboid support="<<st.support<<" rms="<<st.rms<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<"\n";
 for(size_t i=0;i<out.indices.size();i+=3){auto a=out.indices[i],b=out.indices[i+1],c=out.indices[i+2];float u[3],v[3];for(int j=0;j<3;++j){u[j]=out.positions[3*b+j]-out.positions[3*a+j];v[j]=out.positions[3*c+j]-out.positions[3*a+j];}float cross[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};float d=0;for(int j=0;j<3;++j)d+=cross[j]*out.normals[3*a+j];assert(d>0);}
 assert(exportGlb(out,std::string(dir)+"/cuboid.glb","cuboid",nullptr));
 AosMesh aos;for(size_t i=0;i<out.vertexCount();++i){AosVertex v;v.px=out.positions[3*i];v.py=out.positions[3*i+1];v.pz=out.positions[3*i+2];v.nx=out.normals[3*i];v.ny=out.normals[3*i+1];v.nz=out.normals[3*i+2];aos.vertices.push_back(v);}aos.indices=out.indices;UvMesh uv;UvUnwrapStats uvStats;assert(UvUnwrapper::unwrap(aos,512,8,uv,&uvStats));assert(uv.indices.size()==36);std::cout<<"PASS box UV unwrap indices="<<uv.indices.size()<<"\n";
 auto good=out.positions;assert(!fitHardSurface(in,HardSurfaceMode::Cube,.004f,out,st));assert(out.positions==good);
 assert(fitHardSurface(box(.6f,.6f,.6f),HardSurfaceMode::Cube,.004f,out,st));
 assert(fitHardSurface(box(.8f,.6f,.4f,3),HardSurfaceMode::Cuboid,.004f,out,st));assert(st.inferredFaces==3);
 assert(fitHardSurface(box(.8f,.6f,.4f,6,.001f),HardSurfaceMode::Cuboid,.004f,out,st));
 assert(fitHardSurface(in,HardSurfaceMode::Planar,.004f,out,st));
 assert(!fitHardSurface(box(.8f,.6f,.005f),HardSurfaceMode::Cuboid,.004f,out,st));
 assert(!fitHardSurface(box(.8f,.6f,.4f,2),HardSurfaceMode::Cuboid,.004f,out,st));
 assert(!fitHardSurface(sphere(),HardSurfaceMode::Cuboid,.004f,out,st));
 assert(!fitHardSurface(sphere(),HardSurfaceMode::Planar,.004f,out,st));
 Mesh separated=box(.3f,.3f,.3f),second=separated;uint32_t offset=separated.vertexCount();
 for(size_t i=0;i<second.positions.size();i+=3)second.positions[i]+=2.f;
 separated.positions.insert(separated.positions.end(),second.positions.begin(),second.positions.end());
 separated.colors.insert(separated.colors.end(),second.colors.begin(),second.colors.end());
 for(auto i:second.indices)separated.indices.push_back(i+offset);
 assert(!fitHardSurface(separated,HardSurfaceMode::Cuboid,.004f,out,st));
 Mesh dent=box(.8f,.6f,.4f);
 // Remove a central patch and replace it with a recessed patch: must not seal a large dent as a box.
 for(size_t i=0;i<169;++i)if(i%13>2 && i%13<10 && i/13>2 && i/13<10)dent.positions[3*i]+=.12f;
 assert(!fitHardSurface(dent,HardSurfaceMode::Cuboid,.004f,out,st));
 Mesh invalid=in;invalid.indices[0]=999999;assert(!fitHardSurface(invalid,HardSurfaceMode::Cuboid,.004f,out,st));invalid=in;invalid.positions[0]=NAN;assert(!fitHardSurface(invalid,HardSurfaceMode::Cuboid,.004f,out,st));
 std::cout<<"PASS rotated/noisy/partial cuboid, cube constraint, hard normals/winding, planar, sphere/thin/two-face/separated/dented/invalid rejection\n";
}
extern "C" const char* __asan_default_options(){return "detect_leaks=0";}
int main(int argc,char**argv){std::cout << std::unitbuf;assert(argc==2);temporal();fusion();geometry(argv[1]);}
