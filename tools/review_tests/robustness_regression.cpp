#include "ar_textured_asset.h"
#include "tsdf_engine.h"
#include "mesh/mesh_engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

// Match the existing host suites: LeakSanitizer cannot inspect sandbox /proc.
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }

static void numeric() {
    float depth[16] = {}; depth[15] = 1.f;
    const float R[] = {1,0,0,0,1,0,0,0,1};
    TsdfEngine tsdf;
    for (float x : {1e20f, -1e20f, std::numeric_limits<float>::max()}) {
        const float t[] = {x,0,0};
        tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,R,t,1);
        assert(tsdf.voxels() == 0 && tsdf.blocks() == 0);
    }
    const float t[] = {0,0,0};
    float badR[] = {1,0,0,0,1,0,0,0,1};
    badR[0] = std::numeric_limits<float>::quiet_NaN();
    tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,badR,t,1);
    tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,R,t,1,
                        std::numeric_limits<float>::infinity(),0);
    assert(tsdf.blocks() == 0);
    tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,R,t,1);
    assert(tsdf.voxels() > 0); // Rejection must leave the engine usable.
}

static void ray() {
    float depth[16] = {}; depth[15] = 1.f;
    const float R[] = {1,0,0,0,1,0,0,0,1}, t[] = {0,0,0};
    TsdfEngine tsdf;
    tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,R,t,1);
    assert(tsdf.voxels() > 0);
    tsdf.forEachBlock([](int,int,int,const TsdfBlock* b) {
        for (const auto& v : b->voxels) assert(v.weight <= kTsdfWeightScale);
    }); // A single pixel contributes at most once to each traversed voxel.
    const auto count = tsdf.voxels();
    tsdf.integrateDepth(depth,4,4,nullptr,0,0,3,3,0,0,R,t,1);
    assert(tsdf.voxels()==count);
    tsdf.forEachBlock([](int,int,int,const TsdfBlock* b) {
        for (const auto& v : b->voxels) if (v.weight) assert(v.weight == 2*kTsdfWeightScale);
    }); // Later frames must still refine all existing samples.
}

static void cluster() {
    // These two cells have identical old XOR hashes, despite being far apart.
    Mesh mesh;
    mesh.positions = {3.9425f,.2825f,2.1125f, 3.0025f,2.9825f,3.3525f, .0025f,.0025f,.0025f};
    mesh.indices = {0,1,2};
    const auto original = mesh.positions;
    meshClusterSimplify(mesh,.01f);
    assert(mesh.vertexCount() == 3 && mesh.triangleCount() == 1);
    assert(mesh.positions == original);
    mesh.indices[2]=99;
    meshClusterSimplify(mesh,.01f);
    assert(mesh.indices[2]==99 && mesh.positions==original);
    mesh.indices[2]=2;
    meshClusterSimplify(mesh,std::numeric_limits<float>::quiet_NaN());
    assert(mesh.positions==original);
    mesh.positions[0] = std::numeric_limits<float>::max();
    const auto huge = mesh.positions;
    meshClusterSimplify(mesh,.001f);
    assert(mesh.positions == huge); // Invalid coordinates leave the input intact.
    Mesh close;
    close.positions={.1f,.1f,.1f, .2f,.2f,.2f, 1.1f,0,0, 0,1.1f,0};
    close.indices={0,2,3, 1,2,3};
    meshClusterSimplify(close,1.f);
    assert(close.vertexCount()==3); // Actual same-cell vertices still merge.
}

template<class T> static void put(std::vector<char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data()+offset, &value, sizeof(value));
}

static void asset(const std::filesystem::path& root) {
    UvMesh mesh; mesh.vertices.resize(3); mesh.indices = {0,1,2};
    mesh.vertices[1].base.px=1; mesh.vertices[2].base.py=1;
    const std::vector<uint8_t> jpeg = {0xff,0xd8,0xff,0xd9};
    ArTexturedAsset a;
    assert(a.set(mesh,jpeg));
    const auto original = a.vertices();
    auto bad = mesh; bad.indices[2]=3;
    assert(!a.set(bad,jpeg));
    bad=mesh; bad.indices.push_back(0); assert(!a.set(bad,jpeg));
    bad=mesh; bad.vertices[0].base.px=std::numeric_limits<float>::quiet_NaN();
    assert(!a.set(bad,jpeg));
    bad=mesh; bad.vertices[0].u=std::numeric_limits<float>::infinity();
    assert(!a.set(bad,jpeg));
    assert(a.vertices()==original && a.indices()==mesh.indices && a.jpeg()==jpeg);
    const auto valid=root/"valid.msar", broken=root/"broken.msar";
    assert(a.save(valid.string()));
    std::ifstream in(valid,std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});
    auto reject=[&](const std::vector<char>& data) {
        { std::ofstream out(broken,std::ios::binary); out.write(data.data(),data.size()); }
        assert(!a.load(broken.string()));
        assert(a.vertices()==original && a.indices()==mesh.indices && a.jpeg()==jpeg);
    };
    auto data=bytes; put(data,28,std::numeric_limits<float>::quiet_NaN()); reject(data);
    data=bytes; put(data,28+3*8*sizeof(float),uint32_t(3)); reject(data);
    data=bytes; data.pop_back(); reject(data);
    data=bytes; data.push_back(0); reject(data);
    // A tiny file must not allocate the maximum declared ~98 MiB payload.
    data.resize(28); put(data,16,uint32_t(800000)); put(data,20,uint32_t(2400000));
    put(data,24,uint32_t(64*1024*1024)); reject(data);
    ArTexturedAsset restored; assert(restored.load(valid.string()));
    assert(restored.vertices()==original);
    if (std::filesystem::exists("/dev/full")) assert(!a.save("/dev/full"));
}

int main(int argc, char** argv) {
    assert(argc==3);
    const std::string mode=argv[1];
    if(mode=="numeric") numeric();
    else if(mode=="ray") ray();
    else if(mode=="cluster") cluster();
    else if(mode=="asset") asset(argv[2]);
    else return 2;
    std::cout << "PASS " << mode << '\n';
}
