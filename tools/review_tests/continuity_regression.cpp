#include "vins/pose_prediction.h"
#include "surfel_engine.h"
#include "tsdf_engine.h"
#include "mesh/mesh_engine.h"
#include <cassert>
#include <iostream>
#include <limits>

extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }
using namespace scan_pose;
int main() {
    State base;
    base.t = 1.; base.gravity = {0, 0, 9.81}; base.acc = base.gravity;
    base.v = {.3, 0, 0}; base.gyr = {0, 0, 2.};
    std::deque<ImuSample> samples;
    for (int i=1; i<=30; ++i) samples.push_back({1.+i*.005, base.acc, base.gyr});
    State predicted;
    assert(predict(base, samples, 1.073, predicted)); // fractional sample interpolation
    assert((predicted.p-Eigen::Vector3d(.3*.073,0,0)).norm()<1e-10);
    const Eigen::Matrix3d expected = Eigen::AngleAxisd(2.*.073,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    assert((predicted.R-expected).norm()<1e-10);
    assert(base.t==1. && samples.size()==30); // read-only prediction
    assert(!predict(base, samples, 1.2, predicted)); // stale visual state
    assert(!predict(base, {}, 1.02, predicted)); // missing future IMU
    assert(!predict(base, {{1.04,base.acc,base.gyr}}, 1.02, predicted)); // IMU gap
    assert(!predict(base, samples, .99, predicted)); // backwards clock
    assert(predict(base, {}, 1., predicted)); // exact optimized exposure
    State bad=base; bad.acc.x()=std::numeric_limits<double>::quiet_NaN();
    assert(!predict(bad,samples,1.02,predicted));
    State biased=base; biased.ba={.1,.2,.3}; biased.bg={.01,.02,.03};
    biased.acc=biased.gravity+biased.ba; biased.gyr=biased.bg; biased.v.setZero();
    std::deque<ImuSample> stationary;
    for(int i=1;i<=20;++i) stationary.push_back({1.+i*.005,biased.acc,biased.gyr});
    assert(predict(biased,stationary,1.05,predicted));
    assert(predicted.p.norm()<1e-10 && (predicted.R-Eigen::Matrix3d::Identity()).norm()<1e-10);

    CameraSample ca, cb; ca.ts=1'000'000'000; cb.ts=1'100'000'000;
    const Eigen::Matrix3d rotated=Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)cb.R[i*3+j]=rotated(i,j);
    cb.t[0]=.2f;
    float interpolated[12];
    assert(lookup({ca,cb},1'050'000'000,80'000'000,interpolated));
    assert(std::abs(interpolated[9]-.1)<1e-6);
    assert(std::abs(interpolated[0]-std::cos(.2))<1e-6);
    assert(!lookup({ca,cb},1'200'000'000,80'000'000,interpolated));
    assert(!lookup({},1'050'000'000,80'000'000,interpolated));
    cb.ts=1'300'000'000;
    assert(!lookup({ca,cb},1'150'000'000,80'000'000,interpolated));

    // Nonidentity camera extrinsics and many moving views must map the same
    // physical points into one world map, not a new camera-local cloud each time.
    SurfelEngine cloud;
    const Eigen::Matrix3d ric=Eigen::AngleAxisd(.5,Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Vector3d tic(.02,-.01,.03);
    std::vector<Eigen::Vector3d> world;
    for(int y=0;y<8;++y) for(int x=0;x<8;++x) world.emplace_back(.013+x*.02,.017+y*.02,1.003);
    for(int view=0;view<20;++view) {
        assert(predict(base,samples,1.+view*.005,predicted));
        Eigen::Matrix3d Rwc; Eigen::Vector3d twc;
        cameraPose(predicted,ric,tic,Rwc,twc);
        assert((twc-(predicted.p+predicted.R*tic)).norm()<1e-12);
        for(const auto& pw:world) {
            const Eigen::Vector3d pc=Rwc.transpose()*(pw-twc);
            const Eigen::Vector3d back=Rwc*pc+twc;
            cloud.ingestPoint(back.x(),back.y(),back.z(),100,150,200,1);
        }
    }
    assert(cloud.count()==world.size());
    assert(cloud.confirmedCount(20)==world.size());
    assert(cloud.mergedCount()==19*world.size());

    // Render analytic depth of a fixed plane from translated/yawed cameras,
    // integrate the production TSDF and extract the production mesh.
    TsdfEngine tsdf; tsdf.setVoxelSize(.02f); tsdf.setMaxBlocks(4096);
    constexpr int N=48; constexpr float f=60, c=24, planeZ=1.007f;
    std::vector<float> depth(N*N);
    size_t firstCount=0;
    for(int view=0;view<7;++view) {
        const double angle=(view-3)*.045;
        Eigen::Matrix3d R=Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Vector3d t((view-3)*.04,0,0);
        float rr[9],tt[3];
        for(int i=0;i<3;++i){ tt[i]=t[i];for(int j=0;j<3;++j) rr[i*3+j]=R(i,j); }
        for(int y=0;y<N;++y)for(int x=0;x<N;++x){
            Eigen::Vector3d ray=R*Eigen::Vector3d((x-c)/f,(y-c)/f,1);
            depth[y*N+x]=(planeZ-t.z())/ray.z();
        }
        tsdf.integrateDepth(depth.data(),N,N,nullptr,0,0,f,f,c,c,rr,tt,1);
        if(view==0) firstCount=tsdf.blocks();
        assert(tsdf.blocks()>=firstCount && tsdf.surfacePresent());
    }
    MeshEngine mesh; MeshBuildStats stats;
    assert(mesh.build(tsdf,MeshOptions{},stats));
    assert(!mesh.mesh().positions.empty());
    double maxError=0.;
    for(size_t i=2;i<mesh.mesh().positions.size();i+=3) maxError=std::max(maxError,std::abs(double(mesh.mesh().positions[i])-planeZ));
    assert(maxError<.03); // no duplicated moving shell
    std::cout << "PASS: timestamp prediction, IMU gaps/bias, camera extrinsics, 20-view surfel merge, 7-view TSDF mesh; max plane error=" << maxError << " m\n";
}
