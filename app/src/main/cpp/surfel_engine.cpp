#include "surfel_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

void SurfelEngine::reset() {
    sampler_.reset(); coarseIndex_.clear(); frame_=0; reclaimed_=reactivated_=0;
    g_.clear();
    index_.clear();
    confirmed_ = stable_ = 0;
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
    a.red = r; a.green = g; a.blue = b; a.fusionWeight = confidence;
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
    if (minHits == 2) return confirmed_;
    if (minHits == 3) return stable_;
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

    const size_t enabled = confirmedCount(need);
    if (enabled == 0) {
        return 0;
    }

    const size_t target = std::min(enabled, maxPoints);
    size_t edges=0;
    if(enabled>target)for(const auto& a:g_)if(a.hits>=need && a.colorBoundary)++edges;
    // Reserve up to half the display budget for measured contour points; retain
    // surface coverage, and redistribute unused capacity to either partition.
    const size_t edgeTarget=edges ? std::max(std::min(edges,(target+1)/2),
                                             target-std::min(enabled-edges,target)) : 0;
    size_t written=0;
    for(int pass=0;pass<(edges?2:1);++pass) {
        const bool edgePass=edges && pass==0;
        const size_t population=edges ? (edgePass?edges:enabled-edges) : enabled;
        const size_t budget=edges ? (edgePass?edgeTarget:target-edgeTarget) : target;
        if(!budget)continue;
        size_t phase=population-budget;
        for(const auto& a:g_) {
            if(a.hits<need || (edges && a.colorBoundary!=edgePass))continue;
            phase+=budget;if(phase<population)continue;phase-=population;
            float* p=out+written*6;
            p[0]=a.px;p[1]=a.py;p[2]=a.pz;
            p[3]=a.r/255.f;p[4]=a.g/255.f;p[5]=a.b/255.f;
            ++written;
        }
    }
    return written;
}

bool SurfelEngine::nearestStableSurfel(float x, float y, float z, float maxDistM,
                                       float* ox, float* oy, float* oz,
                                       float* onx, float* ony, float* onz,
                                       uint16_t* ohits) const {
    if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) ||
        !(std::isfinite(maxDistM) && maxDistM > 0.f) ||
        coarseIndex_.empty()) {
        return false;
    }

    constexpr float kCell = 0.02f;
    constexpr float kInvCell = 50.f;
    const int kx = int(std::floor(x * kInvCell));
    const int ky = int(std::floor(y * kInvCell));
    const int kz = int(std::floor(z * kInvCell));
    // maxDist=6cm -> radius=3 cells. Cap protects accidental huge queries.
    const int maxRadius = std::clamp(
        int(std::ceil(maxDistM / kCell)), 1, 4);

    const float limit2 = maxDistM * maxDistM;
    float best2 = limit2;
    size_t bestIndex = size_t(-1);

    // Search Chebyshev shells. r=1 is the legacy 3×3×3 fast path.
    // Expanding only the outer shell avoids re-querying earlier cells.
    for (int r = 1; r <= maxRadius; ++r) {
        for (int dz = -r; dz <= r; ++dz) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    if (r > 1 &&
                        std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) {
                        continue;
                    }
                    auto it = coarseIndex_.find(Key{kx + dx, ky + dy, kz + dz});
                    if (it == coarseIndex_.end()) continue;
                    const size_t idx = it->second;
                    if (idx >= g_.size()) continue;
                    const Surfel& a = g_[idx];
                    const float ex = a.px - x;
                    const float ey = a.py - y;
                    const float ez = a.pz - z;
                    const float d2 = ex * ex + ey * ey + ez * ez;
                    if (d2 <= best2) {
                        best2 = d2;
                        bestIndex = idx;
                    }
                }
            }
        }

        // If the nearest match is already well inside the first 2cm cell shell,
        // a farther shell cannot provide a materially better ICP anchor.
        if (bestIndex != size_t(-1) && best2 <= 0.000324f) { // 1.8cm²; shell r=2 cannot beat this materially
            break;
        }
    }

    if (bestIndex == size_t(-1)) return false;
    const Surfel& best = g_[bestIndex];
    if (ox) *ox = best.px;
    if (oy) *oy = best.py;
    if (oz) *oz = best.pz;

    float nx = best.nx, ny = best.ny, nz = best.nz;
    const float n2 = nx * nx + ny * ny + nz * nz;
    if (n2 > 1e-10f && std::isfinite(n2)) {
        const float inv = 1.f / std::sqrt(n2);
        nx *= inv; ny *= inv; nz *= inv;
    } else {
        nx = ny = nz = 0.f;
    }
    if (onx) *onx = nx;
    if (ony) *ony = ny;
    if (onz) *onz = nz;
    if (ohits) *ohits = best.hits;
    return true;
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
    const float w=std::clamp(confidence,0.f,1.f);
    const float sum=a.fusionWeight+w;
    const double alpha=sum>0 ? double(w)/sum : 0;
    // Double intermediates keep a convex update inside the original fine cell.
    a.px=float(double(a.px)+(double(x)-a.px)*alpha);
    a.py=float(double(a.py)+(double(y)-a.py)*alpha);
    a.pz=float(double(a.pz)+(double(z)-a.pz)*alpha);
    a.red+=float((double(r)-a.red)*alpha);
    a.green+=float((double(g)-a.green)*alpha);
    a.blue+=float((double(b)-a.blue)*alpha);
    a.r=uint8_t(std::clamp(std::lround(a.red),0L,255L));
    a.g=uint8_t(std::clamp(std::lround(a.green),0L,255L));
    a.b=uint8_t(std::clamp(std::lround(a.blue),0L,255L));
    // Bounded history still adapts to real changes after the map becomes stable.
    a.fusionWeight=std::min(sum,32.f);
    a.opacity=uint8_t(std::clamp(confidence,0.f,1.f)*255);
    if(countHit && a.hits<65535) { if(a.hits==1) ++confirmed_; ++a.hits; }
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
    if(old.hits>=2)--confirmed_;
    if(old.state>=2)--stable_;
    if(i!=g_.size()-1){g_[i]=g_.back();(g_[i].coarse?coarseIndex_:index_)[keyFor(g_[i])]=i;}
    g_.pop_back();
}
void SurfelEngine::rebuildIndex() {
    // Swap releases old hash nodes/buckets after compaction, unlike clear alone.
    decltype(index_) fine,coarse;
    fine.reserve(g_.size());
    coarse.reserve(std::max<size_t>(64, g_.size()/4));
    confirmed_=stable_=0;
    for(size_t i=0;i<g_.size();++i){auto& a=g_[i];(a.coarse?coarse:fine)[keyFor(a)]=i;if(a.hits>=2)++confirmed_;if(a.state>=2)++stable_;}
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
        // Compression accepts dot>.98 / plane error<5mm. Reactivation must use
        // hysteresis rather than the old .996/2mm thresholds, otherwise a patch
        // is compressed into an ICP anchor and immediately expanded again on
        // ordinary sensor noise. Still reactivate real detail/geometry changes.
        if(geo.protectedDetail || dot<.985f || error>.004f) {
            // Keep a representative while restoring fine cells from fresh observations.
            Surfel old=a;erasePoint(coarse->second);old.coarse=false;old.hits=1;old.state=0;
            old.fusionWeight=std::max(.05f,old.opacity/255.f);
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
    // Interior pixels later in the same frame must not erase an edge vote.
    a.colorBoundary=geo.colorBoundary || (a.lastFrame==frame_ && a.colorBoundary);
    a.detail=geo.protectedDetail;a.nx=geo.nx;a.ny=geo.ny;a.nz=geo.nz;a.lastFrame=frame_;
}
void SurfelEngine::endFrame() {
    // V0.13.42（vc185 重叠修复）：压缩频率 16→8 帧，让粗索引（ICP 锚）
    // 在模型刚有一小块稳定几何时就尽快建立，避免环绕漂移下迟迟无锚可修偏。
    if(!frame_ || frame_%8 || g_.size()<3)return;
    struct Group {size_t first=0,count=0;bool valid=true;double x=0,y=0,z=0,r=0,g=0,b=0,nx=0,ny=0,nz=0;};
    std::unordered_map<Key,Group,Hash> groups;
    groups.reserve(std::max<size_t>(64, g_.size()/2));
    for(size_t i=0;i<g_.size();++i) {
        const auto& a=g_[i];
        const Key k{int(std::floor(a.px/.02f)),int(std::floor(a.py/.02f)),int(std::floor(a.pz/.02f))};
        auto& q=groups[k];if(q.count==0)q.first=i;
        const auto& ref=g_[q.first];
        q.valid=q.valid && !a.coarse && !a.detail && a.hits>=3 && frame_>=a.protectedUntil &&
            frame_-a.lastFrame<=40 && (a.nx*ref.nx+a.ny*ref.ny+a.nz*ref.nz)>.98f &&
            std::fabs((a.px-ref.px)*ref.nx+(a.py-ref.py)*ref.ny+(a.pz-ref.pz)*ref.nz)<.005f;
        ++q.count;q.x+=a.px;q.y+=a.py;q.z+=a.pz;q.r+=a.red;q.g+=a.green;q.b+=a.blue;q.nx+=a.nx;q.ny+=a.ny;q.nz+=a.nz;
    }
    size_t removed=0;
    for(const auto& kv:groups)if(kv.second.valid&&kv.second.count>=2)removed+=kv.second.count-1;
    if(!removed)return;
    std::vector<Surfel> compact;compact.reserve(g_.size()-removed);
    for(size_t i=0;i<g_.size();++i) {
        auto a=g_[i];
        const Key k{int(std::floor(a.px/.02f)),int(std::floor(a.py/.02f)),int(std::floor(a.pz/.02f))};
        const auto& q=groups.at(k);
        if(q.valid && q.count>=2) {
            if(i!=q.first)continue;
            a.px=float(q.x/q.count);a.py=float(q.y/q.count);a.pz=float(q.z/q.count);
            a.red=float(q.r/q.count);a.green=float(q.g/q.count);a.blue=float(q.b/q.count);
            a.r=uint8_t(std::lround(a.red));a.g=uint8_t(std::lround(a.green));a.b=uint8_t(std::lround(a.blue));
            const float nn=float(std::sqrt(q.nx*q.nx+q.ny*q.ny+q.nz*q.nz));
            if(nn>1e-6f && std::isfinite(nn)){a.nx=float(q.nx)/nn;a.ny=float(q.ny)/nn;a.nz=float(q.nz)/nn;}
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
