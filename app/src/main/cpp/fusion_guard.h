#pragma once
#include "depth_calib.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// V0.13.22：视角门限。相对位移 / 转角超过它就判定「已不是同一个观察位置」，
// 该参考帧不再参与重复壳一致性检查 —— 否则移动扫描会与旧 anchor 永远对不上、
// 每一帧都被整帧否决，网格再也长不出来（详见 accept() 内注释）。
static constexpr float kGuardMaxTranslateM = 0.15f;   // 15 cm
static constexpr float kGuardMaxAngleDeg = 20.f;      // 20 deg

// Accepted-keyframe consistency, not pose estimation. Fixed first anchor plus
// three spatially spaced recent anchors prevents a rejected frame from becoming
// its own reference and keeps the original surface testable on a return visit.
class FusionGuard {
    struct Reference { std::vector<float> depth; std::array<float,9> R;
        std::array<float,3> t; std::array<float,4> K; int w=0,h=0; uint64_t ts=0; };
    std::vector<Reference> refs_;
    std::vector<uint8_t> mask_;
    std::vector<float> projection_;
public:
    uint64_t rejected=0,checks=0;
    int lastTested=0;float lastRatio=1;
    void reset(){refs_.clear();mask_.clear();projection_.clear();rejected=checks=0;lastTested=0;lastRatio=1;}
    size_t references()const{return refs_.size();}
    bool accept(const float* depth,int w,int h,const float K[4],const float R[9],
                const float t[3],uint64_t ts) {
        lastTested=0;lastRatio=1;
        if(!depth||!K||!R||!t||w<4||h<4||w>4096||h>4096||!ts)return false;
        if(!std::all_of(K,K+4,[](float v){return std::isfinite(v);})||K[0]<=0||K[1]<=0||
           !std::all_of(R,R+9,[](float v){return std::isfinite(v);})||
           !std::all_of(t,t+3,[](float v){return std::isfinite(v);}))return false;
        mask_.resize(size_t(w)*h);
        for(const auto& ref:refs_){
            if(ref.w!=w||ref.h!=h||!std::equal(ref.K.begin(),ref.K.end(),K)) {
                ++rejected;return false; // do not mix incompatible camera grids in a map
            }
            if(ts<=ref.ts){++rejected;return false;}
            float relativeR[9],relativeT[3];
            for(int i=0;i<3;++i){
                relativeT[i]=0;
                for(int k=0;k<3;++k)relativeT[i]+=R[k*3+i]*(ref.t[k]-t[k]);
                for(int j=0;j<3;++j){relativeR[i*3+j]=0;
                    for(int k=0;k<3;++k)relativeR[i*3+j]+=R[k*3+i]*ref.R[k*3+j];}
            }
            // V0.13.22：视角差异过大时，这个 ref 不可能构成「重复壳」判据。
            // 原实现只要任一 ref 的重叠一致率 < 0.60 就整帧拒绝；用户移动摄像头后
            // 与旧 anchor 的重叠本就稀疏、单目深度噪声又大，于是每一帧都被拒 ->
            // 再也长不出新几何（真机实测网格冻结在 verts=4257 长达 2min21s）。
            // 位移 >15cm 或转角 >20° 时跳过该 ref，只在「回到接近的旧视角」时
            // 才真正执行重复壳一致性检查。全部 ref 都被跳过 => 循环空转返回 true
            // （新区域放行），这正是移动扫描该有的行为。
            {
                float tr = relativeR[0]+relativeR[4]+relativeR[8];
                float cosA = (tr-1.f)*0.5f;
                cosA = cosA<-1.f?-1.f:(cosA>1.f?1.f:cosA);
                const float angleDeg = std::acos(cosA)*57.2957795f;
                const float dx=relativeT[0],dy=relativeT[1],dz=relativeT[2];
                const float distM = std::sqrt(dx*dx+dy*dy+dz*dz);
                if(distM>kGuardMaxTranslateM || angleDeg>kGuardMaxAngleDeg)continue;
            }
            int tested=0,agree=0;
            temporalConsistencyMask(ref.depth.data(),depth,w,h,K[0],K[1],K[2],K[3],
                relativeR,relativeT,mask_.data(),4,.012f,.018f,&tested,&agree,&projection_);
            ++checks;
            // A small but supported overlap matters even if most of the image is new.
            if(tested>=std::max(32,w*h/256)) {
                const float ratio=float(agree)/tested;
                if(ratio<lastRatio){lastRatio=ratio;lastTested=tested;}
                if(ratio<.60f){++rejected;return false;}
            }
        }
        return true; // no overlap supplies no evidence; never invent alignment
    }
    void commit(const float* depth,int w,int h,const float K[4],const float R[9],
                const float t[3],uint64_t ts,const float* weights=nullptr) {
        // Only call after all dense fusion gates pass. Suppressed pixels cannot
        // contaminate future references with geometry that was never integrated.
        if(!refs_.empty()) {
            const auto& prev=refs_.back();
            if(ts<=prev.ts)return;
            double distance=0,trace=0;
            for(int i=0;i<3;++i){const double d=t[i]-prev.t[i];distance+=d*d;}
            for(int i=0;i<9;++i)trace+=double(R[i])*prev.R[i];
            if(distance<.01 && trace>2.93)return; // <10 cm and <~15 degrees
        }
        Reference ref;ref.w=w;ref.h=h;ref.ts=ts;
        std::copy(K,K+4,ref.K.begin());std::copy(R,R+9,ref.R.begin());std::copy(t,t+3,ref.t.begin());
        ref.depth.assign(depth,depth+size_t(w)*h);
        for(size_t i=0;i<ref.depth.size();++i)if(!std::isfinite(ref.depth[i])||
            (weights&&(!std::isfinite(weights[i])||weights[i]<=0)))ref.depth[i]=0;
        if(refs_.size()==4)refs_.erase(refs_.begin()+1);
        refs_.push_back(std::move(ref));
    }
};
