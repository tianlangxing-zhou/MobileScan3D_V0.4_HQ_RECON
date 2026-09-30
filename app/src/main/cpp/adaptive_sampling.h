#pragma once
// Round 4: conservative geometry classification and bounded world-space history.
// No RGB edges, confidence inflation, or removal of the final TSDF geometry.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include "grid_index.h"

namespace adaptive {
struct Geometry {
    float nx=0, ny=0, nz=0; // world normal, oriented consistently toward camera +Z
    bool protectedDetail=true;
};
inline Geometry classify(const float* d, int w, int h, int x, int y,
                         float fx, float fy, const float* R,
                         float scale=1.f, float shift=0.f) {
    Geometry out;
    // Two-pixel halo protects holes, silhouettes, mask borders and depth edges.
    if (!d || !R || x<2 || y<2 || x>=w-2 || y>=h-2 ||
        !(fx>0 && fy>0)) return out;
    const auto zAt=[&](int a,int b) { const float raw=d[size_t(b)*w+a];
        return raw>0 && std::isfinite(raw) ? raw*scale+shift : 0.f; };
    const float z=zAt(x,y);
    if (!(z>.08f && z<8.f)) return out;
    const float edge=std::max(.006f,z*.012f);
    for (int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
        const float n=zAt(x+dx,y+dy);
        if (!std::isfinite(n) || n<=.08f || std::fabs(n-z)>edge) return out;
    }
    const float l=zAt(x-1,y),r=zAt(x+1,y),u=zAt(x,y-1),b=zAt(x,y+1);
    // Curvature in inverse depth: exactly zero for a perspective plane.
    const float curve=std::max(std::fabs(1/l+1/r-2/z),std::fabs(1/u+1/b-2/z))*z*z;
    if (curve>std::max(.001f,z*.0015f)) return out;
    // Perspective plane normal. x/y principal point terms only affect nz;
    // use the overload below to account for them.
    float nx=(1/r-1/l)*.5f*fx, ny=(1/b-1/u)*.5f*fy, nz=1/z;
    const float len=std::sqrt(nx*nx+ny*ny+nz*nz);
    if (!(len>1e-6f) || !std::isfinite(len)) return out;
    nx/=len;ny/=len;nz/=len;
    out.nx=R[0]*nx+R[1]*ny+R[2]*nz;
    out.ny=R[3]*nx+R[4]*ny+R[5]*nz;
    out.nz=R[6]*nx+R[7]*ny+R[8]*nz;
    out.protectedDetail=false;
    return out;
}
// Full principal-point-correct version: build a local coordinate rotation-free
// normal first, then correct its perspective offset before world rotation.
inline Geometry geometry(const float* d,int w,int h,int x,int y,float fx,float fy,
                         float cx,float cy,const float* R,float scale=1,float shift=0) {
    const float I[9]={1,0,0,0,1,0,0,0,1};
    Geometry q=classify(d,w,h,x,y,fx,fy,I,scale,shift);
    if(q.protectedDetail) return q;
    q.nz-=q.nx*(x-cx)/fx+q.ny*(y-cy)/fy;
    const float len=std::sqrt(q.nx*q.nx+q.ny*q.ny+q.nz*q.nz);
    if (!(len>1e-6f) || !std::isfinite(len)) return Geometry{};
    const float nx=q.nx/len,ny=q.ny/len,nz=q.nz/len;
    q.nx=R[0]*nx+R[1]*ny+R[2]*nz;
    q.ny=R[3]*nx+R[4]*ny+R[5]*nz;
    q.nz=R[6]*nx+R[7]*ny+R[8]*nz;
    return q;
}
struct Key { int x,y,z; bool operator==(const Key& b)const{return x==b.x&&y==b.y&&z==b.z;} };
struct Hash { size_t operator()(const Key& k) const {
    return size_t(uint32_t(k.x)*73856093u ^ uint32_t(k.y)*19349663u ^ uint32_t(k.z)*83492791u);
}};
struct Stats { uint64_t candidates=0,selected=0,skipped=0,protectedSamples=0,reactivated=0,evicted=0; };
class Sampler {
    struct Cell { float x,y,z,nx,ny,nz; uint64_t last=0,blocked=0; unsigned age=1; };
    std::unordered_map<Key,Cell,Hash> cells_;
    std::vector<Key> fifo_;
    size_t cursor_=0,limit_=32768;
    uint64_t frame_=0;
    float cell_=0;
    Stats stats_;
public:
    explicit Sampler(size_t limit=32768):limit_(std::max(size_t(1),limit)){}
    void reset(){cells_.clear();fifo_.clear();cursor_=0;frame_=0;cell_=0;stats_={};}
    void beginFrame(float cell) { if(cell!=cell_) {reset();cell_=cell;} ++frame_; }
    const Stats& stats()const{return stats_;}
    // Called immediately after a skipped decision when the TSDF lacks surface evidence.
    void keepUnseenSample(){--stats_.skipped;++stats_.selected;}
    size_t cells()const{return cells_.size();}
    size_t storageBytes() const {
        return cells_.size()*(sizeof(Key)+sizeof(Cell))+cells_.bucket_count()*sizeof(void*)+fifo_.capacity()*sizeof(Key);
    }
    bool select(float x,float y,float z,const Geometry& g,float confidence,int px,int py) {
        ++stats_.candidates;
        auto take=[&](){++stats_.selected;return true;};
        Key k;
        if(!checkedGridIndex(x/cell_,k.x)||!checkedGridIndex(y/cell_,k.y)||!checkedGridIndex(z/cell_,k.z)) return take();
        auto it=cells_.find(k);
        const bool reliable=std::isfinite(confidence)&&confidence>=.12f&&!g.protectedDetail;
        if(it==cells_.end()) {
            // Only trustworthy planes consume history. Unseen/detail always fuses.
            if(!reliable) {++stats_.protectedSamples;return take();}
            if(cells_.size()>=limit_) {cells_.erase(fifo_[cursor_]);fifo_[cursor_]=k;cursor_=(cursor_+1)%limit_;++stats_.evicted;}
            else fifo_.push_back(k);
            cells_.emplace(k,Cell{x,y,z,g.nx,g.ny,g.nz,frame_,0,1});
            return take();
        }
        Cell& c=it->second;
        const float residual=std::fabs((x-c.x)*c.nx+(y-c.y)*c.ny+(z-c.z)*c.nz);
        const float dot=g.nx*c.nx+g.ny*c.ny+g.nz*c.nz;
        if(!reliable || residual>std::min(.003f,cell_*.12f) || dot<.996f || frame_-c.last>120) {
            if(c.age>=6) ++stats_.reactivated;
            c.age=0;c.blocked=frame_;c.last=frame_;
            if(reliable){c.x=x;c.y=y;c.z=z;c.nx=g.nx;c.ny=g.ny;c.nz=g.nz;}
            ++stats_.protectedSamples;return take();
        }
        // At most one stability vote per depth frame, never per pixel/ray hit.
        if(c.last!=frame_){c.last=frame_;c.age=std::min(255u,c.age+1);}
        if(c.blocked==frame_ || c.age<6) return take();
        // Full refresh every eighth frame; rotating 1/4 checks otherwise.
        // Every candidate is still classified and checked for contradictions.
        const unsigned phase=unsigned((px&1)|((py&1)<<1));
        if((frame_&7)==0 || phase==(frame_&3)) return take();
        ++stats_.skipped;return false;
    }
};
} // namespace adaptive
