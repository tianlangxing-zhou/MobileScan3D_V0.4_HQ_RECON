#include "export/gltf_exporter.h"
#include "v06/textured_glb_exporter.h"
#include "depth_calib.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <string>

// LeakSanitizer needs /proc access unavailable in the review sandbox.
extern "C" const char* __asan_default_options() { return "detect_leaks=0"; }

struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root = argv[1];
    std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
    Mesh mesh;
    mesh.positions = {0.123456789f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    mesh.indices = {0, 1, 2};
    for (int flags = 0; flags < 4; ++flags) {
        mesh.normals = flags & 1 ? std::vector<float>{0,0,1,0,0,1,0,0,1} : std::vector<float>{};
        mesh.colors = flags & 2 ? std::vector<float>{1,0,0,0,1,0,0,0,1} : std::vector<float>{};
        assert(exportGlb(mesh, root + "/plain" + std::to_string(flags) + ".glb", "test", nullptr));
    }
    const std::string bad = root + "/invalid.glb";
    mesh.indices[2] = 3;
    assert(!exportGlb(mesh, bad, "test", nullptr));
    mesh.indices[2] = 2;
    mesh.positions[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!exportGlb(mesh, bad, "test", nullptr));
    mesh.positions[0] = 0.f;
    mesh.indices.push_back(0);
    assert(!exportGlb(mesh, bad, "test", nullptr));
    mesh.indices.pop_back();
    mesh.colors.pop_back();
    assert(!exportGlb(mesh, bad, "test", nullptr));

    UvMesh uv;
    uv.vertices.resize(3);
    for (int i = 0; i < 3; ++i) {
        uv.vertices[i].base.px = i == 1 ? 1.f : 0.123456789f;
        uv.vertices[i].base.py = i == 2 ? 1.f : 0.f;
        uv.vertices[i].u = i == 1 ? 1.f : 0.f;
        uv.vertices[i].v = i == 2 ? 1.f : 0.f;
    }
    uv.indices = {0,1,2};
    // Real JPEG written by the Python harness.
    std::ifstream jpegFile(root + "/atlas.jpg", std::ios::binary);
    std::vector<std::uint8_t> jpeg((std::istreambuf_iterator<char>(jpegFile)), {});
    assert(TexturedGlbExporter::write(root + "/textured.glb", uv, jpeg));
    uv.indices[2] = 3;
    assert(!TexturedGlbExporter::write(bad, uv, jpeg));
    uv.indices[2] = 2;
    uv.vertices[0].u = std::numeric_limits<float>::infinity();
    assert(!TexturedGlbExporter::write(bad, uv, jpeg));
    uv.vertices[0].u = 0.f;
    uv.indices.push_back(0);
    assert(!TexturedGlbExporter::write(bad, uv, jpeg));

    std::vector<float> d, z;
    for (int i = 0; i < 30; ++i) {
        d.push_back(0.2f + i * 0.05f);
        z.push_back(2.f * d.back() + 0.5f);
    }
    // Previously scoreModel walked d.size() and read past z.end().
    d.insert(d.end(), 100, 100.f);
    const auto fit = fitDepthRobust(d, z, false);
    assert(fit.valid);
    assert(std::abs(fit.toMetric(1.f, 0.f) - 2.5f) < 0.001f);
}
