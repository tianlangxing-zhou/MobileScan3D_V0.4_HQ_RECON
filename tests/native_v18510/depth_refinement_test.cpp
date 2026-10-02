
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "depth_refinement.h"
int main(){
    constexpr int W=7,H=7;
    float d[W*H],c[W*H];
    for(int i=0;i<W*H;++i){d[i]=1.f;c[i]=1.f;}
    d[3*W+3]=1.012f;c[3*W+3]=.25f;
    d[3*W+4]=1.20f;
    std::vector<float> out;
    depth_refinement::spatial(d,W,H,false,c,out);
    std::cout<<"center="<<out[3*W+3]<<" discontinuity="<<out[3*W+4]<<"\n";
    assert(out.size()==W*H);
    assert(out[3*W+3] < 1.012f && out[3*W+3] > .999f);
    assert(std::fabs(out[3*W+4]-1.20f)<1e-4f);
    return 0;
}
