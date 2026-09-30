#include "surfel_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

void SurfelEngine::reset() {
    sampler_.reset(); coarseIndex_.clear(); frame_=0; reclaimed_=reactivated_=0;
    g_.clear();
    index_.clear();
    stable_ = 0;
    merged_ = 0;
}

void SurfelEngine::ingestPoint(
        float x,
        float y,
        float z,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        float confidence) {
    // These are WORLD coordinates. Camera depth has already been gated by callers.
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        std::fabs(x) > 100000.f || std::fabs(y) > 100000.f || std::fabs(z) > 100000.f ||
        !std::isfinite(confidence) || confidence < 0.05f) {
        return;
    }

    confidence = std::min(confidence, 1.f);
    const float cell = 0.01f;
    Key k{
        (int)std::floor(x / cell),
        (int)std::floor(y / cell),
        (int)std::floor(z / cell)
    };

    auto it = index_.find(k);
    if (it != index_.end()) {
        Surfel& a = g_[it->second];
        updatePoint(a,x,y,z,r,g,b,confidence,true);
        return;
    }

    if (g_.size() >= 600000) {
        return;
    }

    Surfel a{};
    a.px = x;
    a.py = y;
    a.pz = z;
    a.sx = a.sy = a.sz = cell * 0.8f;
    a.qx = 0.f;
    a.qy = 0.f;
    a.qz = 0.f;
    a.qw = 1.f;
    a.r = r;
    a.g = g;
    a.b = b;
    a.opacity = (uint8_t)(confidence * 255.f);
    a.hits = 1;
    a.state = 0;
    index_[k] = g_.size();
    g_.push_back(a);
}

size_t SurfelEngine::count() const { return g_.size(); }
size_t SurfelEngine::stableCount() const { return stable_; }
size_t SurfelEngine::mergedCount() const { return merged_; }

size_t SurfelEngine::confirmedCount(int minHits) const {
    if (minHits <= 1) {
        return g_.size();
    }
    const uint16_t need = static_cast<uint16_t>(minHits > 65535 ? 65535 : minHits);
    size_t n = 0;
    for (const Surfel& a : g_) {
        if (a.hits >= need) {
            n++;
        }
    }
    return n;
}

size_t SurfelEngine::copyPoints(float* out, size_t maxPoints, int minHits) const {
    if (out == nullptr || maxPoints == 0 || g_.empty()) {
        return 0;
    }
    const uint16_t need = static_cast<uint16_t>(std::clamp(minHits, 1, 65535));

    // 先数一遍「满足 minHits 的点」，再按它算采样步长。
    // 旧实现直接对 g_ 全体均匀抽样并全部画出来，于是未验证点也进了渲染 ——
    // 实机统计 600000 总点里 stable 只有 28，屏幕自然是一层噪声。
    size_t enabled = 0;
    for (const Surfel& a : g_) {
        if (a.hits >= need) {
            enabled++;
        }
    }
    if (enabled == 0) {
        return 0;
    }

    const size_t step =
        enabled > maxPoints ? (enabled + maxPoints - 1) / maxPoints : 1;
    size_t seen = 0;
    size_t written = 0;
    for (size_t i = 0; i < g_.size() && written < maxPoints; ++i) {
        const Surfel& a = g_[i];
        if (a.hits < need) {
            continue;
        }
        const bool take = (seen % step) == 0;
        seen++;
        if (!take) {
            continue;
        }
        float* p = out + written * 6;
        p[0] = a.px;
        p[1] = a.py;
        p[2] = a.pz;
        p[3] = a.r / 255.0f;
        p[4] = a.g / 255.0f;
        p[5] = a.b / 255.0f;
        written++;
    }
    return written;
}

void SurfelEngine::boundingBox(float* minX, float* minY, float* minZ,
                                 float* maxX, float* maxY, float* maxZ) const {
    if (!minX || !minY || !minZ || !maxX || !maxY || !maxZ) {
        return;
    }
    *minX = *minY = *minZ = std::numeric_limits<float>::max();
    *maxX = *maxY = *maxZ = -std::numeric_limits<float>::max();
    for (const Surfel& a : g_) {
        *minX = std::min(*minX, a.px);
        *minY = std::min(*minY, a.py);
        *minZ = std::min(*minZ, a.pz);
        *maxX = std::max(*maxX, a.px);
        *maxY = std::max(*maxY, a.py);
        *maxZ = std::max(*maxZ, a.pz);
    }
    if (g_.empty()) {
        *minX = *minY = *minZ = 0.f;
        *maxX = *maxY = *maxZ = 0.f;
    }
}

void SurfelEngine::centroid(float* x, float* y, float* z) const {
    if (!x || !y || !z) {
        return;
    }
    *x = *y = *z = 0.f;
    if (g_.empty()) {
        return;
    }
    // Up to 600k world-space points: float accumulation can lose centimetres
    // even when each individual coordinate is represented accurately.
    double sx = 0, sy = 0, sz = 0;
    for (const Surfel& a : g_) {
        sx += a.px;
        sy += a.py;
        sz += a.pz;
    }
    const double inv = 1.0 / static_cast<double>(g_.size());
    *x = static_cast<float>(sx * inv);
    *y = static_cast<float>(sy * inv);
    *z = static_cast<float>(sz * inv);
}

void SurfelEngine::updatePoint(Surfel& a,float x,float y,float z,uint8_t r,uint8_t g,uint8_t b,float confidence,bool countHit) {
    const float w=std::min(.5f,confidence), iw=1.f/(1.f+w);
    a.px=(a.px+x*w)*iw;a.py=(a.py+y*w)*iw;a.pz=(a.pz+z*w)*iw;
    a.r=uint8_t((a.r+r*w)*iw);a.g=uint8_t((a.g+g*w)*iw);a.b=uint8_t((a.b+b*w)*iw);
    a.opacity=uint8_t(std::clamp(confidence,0.f,1.f)*255);
    if(countHit && a.hits<65535) ++a.hits;
    if(a.hits>=3 && a.state<2){a.state=2;++stable_;}
    ++merged_;
}
void SurfelEngine::beginFrame() {
    // Wrap cannot leave a stale compacted representative permanently protected.
    if(++frame_==0) {frame_=1;for(auto& a:g_){a.lastFrame=0;a.protectedUntil=0;}}
    sampler_.beginFrame(.04f);
}
SurfelEngine::Key SurfelEngine::keyFor(const Surfel& a) const {
    const float cell=a.coarse?.02f:.01f;
    return {int(std::floor(a.px/cell)),int(std::floor(a.py/cell)),int(std::floor(a.pz/cell))};
}
void SurfelEngine::erasePoint(size_t i) {
    const Surfel old=g_[i];
    (old.coarse?coarseIndex_:index_).erase(keyFor(old));
    if(old.state>=2)--stable_;
    if(i!=g_.size()-1){g_[i]=g_.back();(g_[i].coarse?coarseIndex_:index_)[keyFor(g_[i])]=i;}
    g_.pop_back();
}
void SurfelEngine::rebuildIndex() {
    // Swap releases old hash nodes/buckets after compaction, unlike clear alone.
    decltype(index_) fine,coarse;stable_=0;
    for(size_t i=0;i<g_.size();++i){auto& a=g_[i];(a.coarse?coarse:fine)[keyFor(a)]=i;if(a.state>=2)++stable_;}
    index_.swap(fine);coarseIndex_.swap(coarse);
}
void SurfelEngine::ingestAdaptivePoint(float x,float y,float z,uint8_t r,uint8_t g,uint8_t b,
                                      float confidence,const adaptive::Geometry& geo,int px,int py) {
    if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||std::fabs(x)>100000||std::fabs(y)>100000||std::fabs(z)>100000||
       !std::isfinite(confidence)||confidence<.05f)return;
    const Key parent{int(std::floor(x/.02f)),int(std::floor(y/.02f)),int(std::floor(z/.02f))};
    auto coarse=coarseIndex_.find(parent);
    if(coarse!=coarseIndex_.end()) {
        Surfel& a=g_[coarse->second];
        const float dot=a.nx*geo.nx+a.ny*geo.ny+a.nz*geo.nz;
        const float error=std::fabs((x-a.px)*a.nx+(y-a.py)*a.ny+(z-a.pz)*a.nz);
        if(geo.protectedDetail || dot<.996f || error>.002f) {
            // Keep a representative while restoring fine cells from fresh observations.
            Surfel old=a;erasePoint(coarse->second);old.coarse=false;old.hits=1;old.state=0;
            old.protectedUntil=frame_+32;old.detail=true;old.sx=old.sy=old.sz=.008f;
            index_[keyFor(old)]=g_.size();g_.push_back(old);++reactivated_;
        } else {
            if(sampler_.select(x,y,z,geo,confidence,px,py)) {
                // Geometry remains at its representative location; never drag an
                // entire patch toward whichever fine pixel happened to be last.
                if(a.lastFrame!=frame_) { updatePoint(a,a.px,a.py,a.pz,r,g,b,confidence,true);a.lastFrame=frame_; }
            }
            return;
        }
    }
    if(!sampler_.select(x,y,z,geo,confidence,px,py))return;
    const Key key{int(std::floor(x/.01f)),int(std::floor(y/.01f)),int(std::floor(z/.01f))};
    auto it=index_.find(key);
    const bool fresh=it==index_.end();
    if(fresh) {
        const size_t before=g_.size();ingestPoint(x,y,z,r,g,b,confidence);
        if(g_.size()==before)return;
        it=index_.find(key);
    } else updatePoint(g_[it->second],x,y,z,r,g,b,confidence,g_[it->second].lastFrame!=frame_);
    Surfel& a=g_[it->second];
    const float dot=a.nx*geo.nx+a.ny*geo.ny+a.nz*geo.nz;
    if(geo.protectedDetail || confidence<.12f || (!fresh && dot<.996f)) a.protectedUntil=frame_+32;
    a.detail=geo.protectedDetail;a.nx=geo.nx;a.ny=geo.ny;a.nz=geo.nz;a.lastFrame=frame_;
}
void SurfelEngine::endFrame() {
    if(!frame_ || frame_%16 || g_.size()<3)return;
    struct Group {size_t first=0,count=0;bool valid=true;double x=0,y=0,z=0,r=0,g=0,b=0;};
    std::unordered_map<Key,Group,Hash> groups;
    for(size_t i=0;i<g_.size();++i) {
        const auto& a=g_[i];
        const Key k{int(std::floor(a.px/.02f)),int(std::floor(a.py/.02f)),int(std::floor(a.pz/.02f))};
        auto& q=groups[k];if(q.count==0)q.first=i;
        const auto& ref=g_[q.first];
        q.valid=q.valid && !a.coarse && !a.detail && a.hits>=6 && frame_>=a.protectedUntil &&
            frame_-a.lastFrame<=32 && (a.nx*ref.nx+a.ny*ref.ny+a.nz*ref.nz)>.996f &&
            std::fabs((a.px-ref.px)*ref.nx+(a.py-ref.py)*ref.ny+(a.pz-ref.pz)*ref.nz)<.002f;
        ++q.count;q.x+=a.px;q.y+=a.py;q.z+=a.pz;q.r+=a.r;q.g+=a.g;q.b+=a.b;
    }
    size_t removed=0;
    for(const auto& kv:groups)if(kv.second.valid&&kv.second.count>=3)removed+=kv.second.count-1;
    if(!removed)return;
    std::vector<Surfel> compact;compact.reserve(g_.size()-removed);
    for(size_t i=0;i<g_.size();++i) {
        auto a=g_[i];
        const Key k{int(std::floor(a.px/.02f)),int(std::floor(a.py/.02f)),int(std::floor(a.pz/.02f))};
        const auto& q=groups.at(k);
        if(q.valid && q.count>=3) {
            if(i!=q.first)continue;
            a.px=float(q.x/q.count);a.py=float(q.y/q.count);a.pz=float(q.z/q.count);
            a.r=uint8_t(q.r/q.count);a.g=uint8_t(q.g/q.count);a.b=uint8_t(q.b/q.count);
            a.coarse=true;a.sx=a.sy=a.sz=.016f;
        }
        compact.push_back(a);
    }
    g_.swap(compact);reclaimed_+=removed;rebuildIndex();
}
size_t SurfelEngine::storageBytes() const {
    // Payload + bucket estimate; allocator node overhead is implementation-specific.
    return sampler_.storageBytes()+g_.capacity()*sizeof(Surfel)+(index_.size()+coarseIndex_.size())*(sizeof(Key)+sizeof(size_t))+
        (index_.bucket_count()+coarseIndex_.bucket_count())*sizeof(void*);
}
