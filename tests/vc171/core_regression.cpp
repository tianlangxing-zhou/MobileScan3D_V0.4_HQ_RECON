#include "fusion_evidence.h"
#include "scan_policy.h"
#include "depth_refinement.h"
#include "fusion_guard.h"
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    scan_policy::FusionEvidence evidence;
    uint64_t t = 1'000'000'000ULL;
    for (int i=0; i<6; ++i) {
        assert(evidence.observe(t, true, false) == (i==5));
        assert(!evidence.observe(t, true, false)); // duplicate exposure
        assert(!evidence.observe(t+100'000'000, false, false));
        t += 500'000'000;
    }
    assert(evidence.count()==6);
    assert(!evidence.observe(t, false, true));
    assert(evidence.count()==0);
    assert(!evidence.observe(t+1, true, false));
    assert(!evidence.observe(t+3'000'000'000, true, false));
    assert(evidence.count()==1); // old support expired
    assert(!evidence.observe(t, true, false));
    assert(evidence.count()==1);

    DepthCalibration a; a.valid=true; a.scale=.5f; a.shift=.5f; a.inverseDepthModel=true;
    auto b=a;
    std::vector<float> raw(256);
    for (int i=0;i<256;++i) raw[i]=.1f+1.8f*i/255;
    assert(scan_policy::mappingsAgree(a,b,raw.data(),raw.size(),.08f));
    b.scale=.2f; b.shift=.8f; // same z at q=1, different elsewhere
    assert(a.toMetric(1,0)==b.toMetric(1,0));
    assert(!scan_policy::mappingsAgree(a,b,raw.data(),raw.size(),.08f));
    std::vector<float> z; for(float q:raw)z.push_back(a.toMetric(q,0));
    assert(scan_policy::validatesFrozen(a,raw,z));
    assert(!scan_policy::validatesFrozen(b,raw,z));

    std::vector<float> d(49,1.f),c(49,1.f),out;
    d[24]=0.f; // valid inverse q=0 must survive with positive confidence
    c[25]=0.f; c[26]=std::numeric_limits<float>::quiet_NaN(); c[27]=2.f;
    depth_refinement::applySourceConfidence(d.data(),c.data(),d.size());
    assert(d[24]==0 && std::isnan(d[25]) && std::isnan(d[26]) && c[27]==1);
    depth_refinement::spatial(d.data(),7,7,true,out);
    assert(out[24]==0 && std::isnan(out[25]) && std::isnan(out[26]));
    d.assign(64,1.f);for(int y=0;y<8;++y)for(int x=4;x<8;++x)d[y*8+x]=2.f;
    depth_refinement::spatial(d.data(),8,8,false,out);
    assert(d==out); // sharp depth boundary is preserved
    float sample=0;assert(!depth_refinement::calibrationSample(d.data(),8,8,3,3,false,sample));
    assert(depth_refinement::calibrationSample(d.data(),8,8,1,3,false,sample) && sample==1.f);
    // Reprojection and accepted-reference shell guard remain strict.
    constexpr int w=64,h=48;
    std::vector<float> plane(w*h,1.f), shifted(w*h,1.25f);
    float K[4]={70,70,32,24},R[9]={1,0,0,0,1,0,0,0,1},T[3]={0,0,0};
    FusionGuard guard;
    assert(guard.accept(plane.data(),w,h,K,R,T,1));
    guard.commit(plane.data(),w,h,K,R,T,1);
    assert(guard.accept(plane.data(),w,h,K,R,T,2));
    assert(!guard.accept(shifted.data(),w,h,K,R,T,3));
    std::cout << "PASS core: evidence gaps/expiry/contradiction, mapping domain, confidence, edges, shell guard\n";
}
