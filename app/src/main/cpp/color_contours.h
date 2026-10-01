#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// RGB contours guide sampling only. Never move a measured 3D point or increase
// its depth confidence: a painted stripe is not evidence of a geometric crease.
class ColorContours {
    std::vector<uint8_t> rgb_, smooth_, candidate_;
    std::vector<uint16_t> strength_;
    std::vector<int> queue_;
public:
    std::vector<uint8_t> band;
    size_t contourPixels=0, priorityPixels=0;
    void reset() { band.clear(); contourPixels=priorityPixels=0; }
    void build(const uint8_t* rgb,int rw,int rh,const float* depth,int w,int h,
               const float* weights=nullptr) {
        reset();
        if(!rgb || !depth || rw<1 || rh<1 || w<5 || h<5 ||
           w>1024 || h>1024) return;
        const size_t n=size_t(w)*h;
        rgb_.resize(n*3); smooth_.resize(n*3);
        strength_.assign(n,0);candidate_.assign(n,0);band.assign(n,0);
        auto valid=[&](size_t i){return std::isfinite(depth[i]) && depth[i]>.08f &&
            (!weights || (std::isfinite(weights[i]) && weights[i]>0));};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            const size_t src=(size_t(int64_t(y)*rh/h)*rw+int64_t(x)*rw/w)*3;
            for(int c=0;c<3;++c)rgb_[(size_t(y)*w+x)*3+c]=rgb[src+c];
        }
        // Small separable box blur suppresses sensor noise; chromatic as well as
        // luminance differences survive (unlike a grayscale-only edge detector).
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c){
            int sum=0;for(int dx=-1;dx<=1;++dx)sum+=rgb_[(size_t(y)*w+std::clamp(x+dx,0,w-1))*3+c];
            smooth_[(size_t(y)*w+x)*3+c]=uint8_t((sum+1)/3);
        }
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c){
            int sum=0;for(int dy=-1;dy<=1;++dy)sum+=smooth_[(size_t(std::clamp(y+dy,0,h-1))*w+x)*3+c];
            rgb_[(size_t(y)*w+x)*3+c]=uint8_t((sum+1)/3);
        }
        for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){
            const size_t i=size_t(y)*w+x;int gx=0,gy=0;
            for(int c=0;c<3;++c){
                gx+=std::abs(int(rgb_[(i+1)*3+c])-int(rgb_[(i-1)*3+c]));
                gy+=std::abs(int(rgb_[(i+w)*3+c])-int(rgb_[(i-w)*3+c]));
            }
            strength_[i]=uint16_t(std::max(gx,gy)/3);
            candidate_[i]=gx>=gy ? 1 : 2; // gradient direction until thinning
        }
        // Nonmaximum suppression. Tie breaking keeps one side of a plateau.
        for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){
            const size_t i=size_t(y)*w+x;const size_t delta=candidate_[i]==1?1:size_t(w);
            candidate_[i]=uint8_t(valid(i) && strength_[i]>=24 &&
                strength_[i]>strength_[i-delta] && strength_[i]>=strength_[i+delta]);
        }
        // Reject isolated responses; retain connected outlines of at least 5 pixels.
        for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x){
            const int start=y*w+x;if(candidate_[start]!=1)continue;
            queue_.clear();queue_.push_back(start);candidate_[start]=2;
            for(size_t q=0;q<queue_.size();++q){
                const int cy=queue_[q]/w,cx=queue_[q]%w;
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                    const int xx=cx+dx,yy=cy+dy;
                    if(xx<1||yy<1||xx>=w-1||yy>=h-1)continue;
                    const int j=yy*w+xx;if(candidate_[j]==1){candidate_[j]=2;queue_.push_back(j);}
                }
            }
            if(queue_.size()<5)continue;
            contourPixels+=queue_.size();
            for(int index:queue_){const int cy=index/w,cx=index%w;
                for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                    const size_t j=size_t(cy+dy)*w+cx+dx;
                    if(valid(j) && !band[j]){band[j]=1;++priorityPixels;}
                }
            }
        }
    }
};
