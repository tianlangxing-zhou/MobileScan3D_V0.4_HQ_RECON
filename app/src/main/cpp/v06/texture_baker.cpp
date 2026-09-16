#include "texture_baker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z};
}
Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}
Vec3 operator*(const Vec3& a, float s) {
    return {a.x*s, a.y*s, a.z*s};
}
Vec3 operator/(const Vec3& a, float s) {
    return s == 0.0f ? Vec3{} : Vec3{a.x/s, a.y/s, a.z/s};
}
float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}
float norm(const Vec3& a) {
    return std::sqrt(dot(a,a));
}
Vec3 normalize(const Vec3& a) {
    const float n = norm(a);
    if (n <= 1e-12f) return {0,0,1};
    return a / n;
}
Vec3 pos(const UvVertex& v) {
    return {v.base.px, v.base.py, v.base.pz};
}
Vec3 normal(const UvVertex& v) {
    return {v.base.nx, v.base.ny, v.base.nz};
}
Vec3 colorRgb(const UvVertex& v) {
    return {
        std::clamp(v.base.r, 0.0f, 1.0f) * 255.0f,
        std::clamp(v.base.g, 0.0f, 1.0f) * 255.0f,
        std::clamp(v.base.b, 0.0f, 1.0f) * 255.0f
    };
}

float edge2(const Vec2& a, const Vec2& b, const Vec2& p) {
    return (p.x-a.x)*(b.y-a.y) - (p.y-a.y)*(b.x-a.x);
}

bool barycentric(
        const Vec2& a,
        const Vec2& b,
        const Vec2& c,
        const Vec2& p,
        float* w0,
        float* w1,
        float* w2) {
    const float area = edge2(a,b,c);
    if (std::abs(area) < 1e-10f) return false;

    const float a0 = edge2(b,c,p) / area;
    const float a1 = edge2(c,a,p) / area;
    const float a2 = 1.0f - a0 - a1;

    constexpr float eps = -1e-4f;
    if (a0 < eps || a1 < eps || a2 < eps) return false;

    *w0 = a0;
    *w1 = a1;
    *w2 = a2;
    return true;
}

struct LoadedFrame {
    TextureKeyframe meta;
    cv::Mat bgr;
    cv::Mat depth;  // CV_32FC1, coarse camera z
    float depthScaleX = 1.0f;
    float depthScaleY = 1.0f;
    float luminanceGain = 1.0f;
};

Vec3 cameraPosition(const TextureKeyframe& k) {
    return {k.twc[0], k.twc[1], k.twc[2]};
}

Vec3 worldToCamera(const TextureKeyframe& k, const Vec3& pw) {
    const Vec3 d{
        pw.x - k.twc[0],
        pw.y - k.twc[1],
        pw.z - k.twc[2]
    };

    // Rwc(camera -> world); pc = Rwc^T * (pw - twc)
    return {
        k.Rwc[0]*d.x + k.Rwc[3]*d.y + k.Rwc[6]*d.z,
        k.Rwc[1]*d.x + k.Rwc[4]*d.y + k.Rwc[7]*d.z,
        k.Rwc[2]*d.x + k.Rwc[5]*d.y + k.Rwc[8]*d.z
    };
}

bool project(
        const TextureKeyframe& k,
        const Vec3& pw,
        float* u,
        float* v,
        float* z) {
    const Vec3 pc = worldToCamera(k, pw);
    if (pc.z <= 0.02f) return false;

    const float px = k.fx * pc.x / pc.z + k.cx;
    const float py = k.fy * pc.y / pc.z + k.cy;

    if (!std::isfinite(px) || !std::isfinite(py)) return false;

    *u = px;
    *v = py;
    *z = pc.z;
    return true;
}

cv::Vec3f bilinearBgr(const cv::Mat& image, float x, float y) {
    if (image.empty()) return {0,0,0};

    x = std::clamp(x, 0.0f, static_cast<float>(image.cols - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(image.rows - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(image.cols - 1, x0 + 1);
    const int y1 = std::min(image.rows - 1, y0 + 1);

    const float tx = x - x0;
    const float ty = y - y0;

    const cv::Vec3b a = image.at<cv::Vec3b>(y0,x0);
    const cv::Vec3b b = image.at<cv::Vec3b>(y0,x1);
    const cv::Vec3b c = image.at<cv::Vec3b>(y1,x0);
    const cv::Vec3b d = image.at<cv::Vec3b>(y1,x1);

    cv::Vec3f top, bot;
    for (int ch = 0; ch < 3; ++ch) {
        top[ch] = a[ch] + tx*(b[ch]-a[ch]);
        bot[ch] = c[ch] + tx*(d[ch]-c[ch]);
    }

    cv::Vec3f out;
    for (int ch = 0; ch < 3; ++ch) {
        out[ch] = top[ch] + ty*(bot[ch]-top[ch]);
    }
    return out;
}

float approximateMedianLuma(const cv::Mat& bgr) {
    if (bgr.empty()) return 128.0f;

    std::array<int, 256> hist{};
    int total = 0;
    const int stepX = std::max(1, bgr.cols / 128);
    const int stepY = std::max(1, bgr.rows / 96);

    for (int y = 0; y < bgr.rows; y += stepY) {
        const auto* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; x += stepX) {
            const auto p = row[x];
            const int lum = std::clamp(
                static_cast<int>(
                    0.114f*p[0] +
                    0.587f*p[1] +
                    0.299f*p[2]),
                0, 255);
            ++hist[lum];
            ++total;
        }
    }

    const int target = std::max(1, total/2);
    int acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        if (acc >= target) return static_cast<float>(i);
    }
    return 128.0f;
}

void rasterizeDepth(
        const UvMesh& mesh,
        LoadedFrame& frame,
        int maxSide) {
    const int srcW = frame.bgr.cols;
    const int srcH = frame.bgr.rows;
    if (srcW <= 0 || srcH <= 0) return;

    const float scale =
        static_cast<float>(std::max(64, maxSide)) /
        static_cast<float>(std::max(srcW, srcH));

    const int dw =
        std::max(64, static_cast<int>(std::round(srcW * scale)));
    const int dh =
        std::max(64, static_cast<int>(std::round(srcH * scale)));

    frame.depth = cv::Mat(
        dh, dw, CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::infinity()));

    frame.depthScaleX =
        static_cast<float>(dw) / static_cast<float>(srcW);
    frame.depthScaleY =
        static_cast<float>(dh) / static_cast<float>(srcH);

    const std::size_t triCount = mesh.triangleCount();

    for (std::size_t t = 0; t < triCount; ++t) {
        const auto ia = mesh.indices[t*3u+0u];
        const auto ib = mesh.indices[t*3u+1u];
        const auto ic = mesh.indices[t*3u+2u];

        if (ia >= mesh.vertices.size() ||
            ib >= mesh.vertices.size() ||
            ic >= mesh.vertices.size()) {
            continue;
        }

        float u0,v0,z0,u1,v1,z1,u2,v2,z2;
        if (!project(frame.meta, pos(mesh.vertices[ia]), &u0,&v0,&z0) ||
            !project(frame.meta, pos(mesh.vertices[ib]), &u1,&v1,&z1) ||
            !project(frame.meta, pos(mesh.vertices[ic]), &u2,&v2,&z2)) {
            continue;
        }

        Vec2 a{u0*frame.depthScaleX, v0*frame.depthScaleY};
        Vec2 b{u1*frame.depthScaleX, v1*frame.depthScaleY};
        Vec2 c{u2*frame.depthScaleX, v2*frame.depthScaleY};

        int minX = std::max(
            0,
            static_cast<int>(
                std::floor(std::min({a.x,b.x,c.x}))));
        int minY = std::max(
            0,
            static_cast<int>(
                std::floor(std::min({a.y,b.y,c.y}))));
        int maxX = std::min(
            dw-1,
            static_cast<int>(
                std::ceil(std::max({a.x,b.x,c.x}))));
        int maxY = std::min(
            dh-1,
            static_cast<int>(
                std::ceil(std::max({a.y,b.y,c.y}))));

        if (minX > maxX || minY > maxY) continue;

        for (int y = minY; y <= maxY; ++y) {
            float* row = frame.depth.ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                float w0,w1,w2;
                if (!barycentric(
                        a,b,c,
                        Vec2{x+0.5f,y+0.5f},
                        &w0,&w1,&w2)) {
                    continue;
                }

                // Coarse visibility only; linear camera-Z interpolation is
                // sufficient at this resolution.
                const float z = w0*z0 + w1*z1 + w2*z2;
                if (z < row[x]) row[x] = z;
            }
        }
    }
}

bool visible(
        const LoadedFrame& f,
        float u,
        float v,
        float z,
        const TextureBakeOptions& options) {
    if (f.depth.empty()) return true;

    const int x = std::clamp(
        static_cast<int>(u * f.depthScaleX),
        0,
        f.depth.cols - 1);
    const int y = std::clamp(
        static_cast<int>(v * f.depthScaleY),
        0,
        f.depth.rows - 1);

    const float ref = f.depth.at<float>(y,x);
    if (!std::isfinite(ref)) return true;

    const float tol =
        options.visibilityAbsToleranceMeters +
        options.visibilityRelativeTolerance * z;

    return z <= ref + tol;
}

float triangleFrameScore(
        const UvVertex& a,
        const UvVertex& b,
        const UvVertex& c,
        const LoadedFrame& f,
        const TextureBakeOptions& options) {
    const Vec3 pa = pos(a);
    const Vec3 pb = pos(b);
    const Vec3 pc = pos(c);
    const Vec3 centroid = (pa + pb + pc) / 3.0f;
    const Vec3 n = normalize(cross(pb-pa, pc-pa));

    float u,v,z;
    if (!project(f.meta, centroid, &u,&v,&z)) return 0.0f;

    const int border = std::max(0, options.imageBorderPixels);
    if (u < border || v < border ||
        u >= f.bgr.cols - border ||
        v >= f.bgr.rows - border) {
        return 0.0f;
    }

    if (!visible(f,u,v,z,options)) return 0.0f;

    const Vec3 view =
        normalize(cameraPosition(f.meta) - centroid);
    const float facing = dot(n, view);
    if (facing < options.minViewCos) return 0.0f;

    float u0,v0,z0,u1,v1,z1,u2,v2,z2;
    if (!project(f.meta, pa, &u0,&v0,&z0) ||
        !project(f.meta, pb, &u1,&v1,&z1) ||
        !project(f.meta, pc, &u2,&v2,&z2)) {
        return 0.0f;
    }

    const float projectedArea =
        std::abs(
            (u1-u0)*(v2-v0) -
            (v1-v0)*(u2-u0)) * 0.5f;

    if (projectedArea < 2.0f) return 0.0f;

    const float resolutionScore =
        std::sqrt(projectedArea);
    const float quality =
        std::clamp(f.meta.quality, 0.10f, 3.0f);

    return quality *
           std::pow(std::max(0.0f, facing), 4.0f) *
           resolutionScore /
           (0.25f + z*z);
}

void dilateGutter(
        cv::Mat& atlas,
        cv::Mat& mask,
        int iterations) {
    if (iterations <= 0) return;

    for (int it = 0; it < iterations; ++it) {
        cv::Mat next = atlas.clone();
        cv::Mat nextMask = mask.clone();

        for (int y = 1; y + 1 < atlas.rows; ++y) {
            for (int x = 1; x + 1 < atlas.cols; ++x) {
                if (mask.at<std::uint8_t>(y,x) != 0) continue;

                cv::Vec3f sum(0,0,0);
                int n = 0;

                static constexpr int dx[4] = {-1,1,0,0};
                static constexpr int dy[4] = {0,0,-1,1};

                for (int k = 0; k < 4; ++k) {
                    const int xx = x + dx[k];
                    const int yy = y + dy[k];
                    if (mask.at<std::uint8_t>(yy,xx) == 0) continue;

                    const cv::Vec3b p =
                        atlas.at<cv::Vec3b>(yy,xx);
                    sum += cv::Vec3f(p[0],p[1],p[2]);
                    ++n;
                }

                if (n > 0) {
                    next.at<cv::Vec3b>(y,x) =
                        cv::Vec3b(
                            cv::saturate_cast<std::uint8_t>(sum[0]/n),
                            cv::saturate_cast<std::uint8_t>(sum[1]/n),
                            cv::saturate_cast<std::uint8_t>(sum[2]/n));
                    nextMask.at<std::uint8_t>(y,x) = 255;
                }
            }
        }

        atlas = std::move(next);
        mask = std::move(nextMask);
    }
}

}  // namespace

bool TextureBaker::bake(
        const UvMesh& mesh,
        const std::vector<TextureKeyframe>& keyframes,
        const TextureBakeOptions& options,
        cv::Mat& atlasBgr,
        TextureBakeStats* outStats) {
    TextureBakeStats stats;
    stats.requestedKeyframes =
        static_cast<int>(keyframes.size());

    if (mesh.empty() ||
        mesh.atlasWidth <= 0 ||
        mesh.atlasHeight <= 0 ||
        keyframes.empty()) {
        if (outStats) *outStats = stats;
        return false;
    }

    // Select by BOTH quality and viewpoint diversity. Taking the 12 sharpest
    // frames alone tends to cluster around one easy angle, leaving the back
    // of the object without texture support.
    std::vector<TextureKeyframe> pool = keyframes;
    std::sort(
        pool.begin(),
        pool.end(),
        [](const TextureKeyframe& a, const TextureKeyframe& b) {
            return a.quality > b.quality;
        });

    const int budget =
        options.maxKeyframes > 0
            ? std::min(
                  options.maxKeyframes,
                  static_cast<int>(pool.size()))
            : static_cast<int>(pool.size());

    std::vector<TextureKeyframe> selected;
    selected.reserve(static_cast<std::size_t>(budget));

    if (!pool.empty() && budget > 0) {
        selected.push_back(pool.front());
        pool.erase(pool.begin());
    }

    auto forward = [](const TextureKeyframe& k) -> Vec3 {
        // Rwc is row-major camera->world; camera +Z is column 2.
        return normalize(
            Vec3{k.Rwc[2], k.Rwc[5], k.Rwc[8]});
    };

    while (static_cast<int>(selected.size()) < budget &&
           !pool.empty()) {
        std::size_t bestIndex = 0;
        float bestScore = -1.0f;

        for (std::size_t i = 0; i < pool.size(); ++i) {
            const auto& candidate = pool[i];
            const Vec3 cp = cameraPosition(candidate);
            const Vec3 cf = forward(candidate);

            float minDiversity = 1.0f;
            for (const auto& chosen : selected) {
                const float translation =
                    norm(cp - cameraPosition(chosen));

                const float directionDelta =
                    std::sqrt(
                        std::max(
                            0.0f,
                            1.0f -
                            std::clamp(
                                dot(cf, forward(chosen)),
                                -1.0f,
                                1.0f)));

                // 8 cm is already a meaningful baseline for object capture.
                const float translationTerm =
                    std::clamp(
                        translation / 0.08f,
                        0.0f,
                        1.0f);

                const float diversity =
                    std::max(
                        translationTerm,
                        std::clamp(directionDelta, 0.0f, 1.0f));

                minDiversity =
                    std::min(minDiversity, diversity);
            }

            const float score =
                std::clamp(candidate.quality, 0.10f, 3.0f) *
                (0.25f + 0.75f * minDiversity);

            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

        selected.push_back(pool[bestIndex]);
        pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(bestIndex));
    }

    std::vector<LoadedFrame> frames;
    frames.reserve(selected.size());
    std::vector<float> medians;
    medians.reserve(selected.size());

    for (const auto& k : selected) {
        cv::Mat img =
            cv::imread(k.imagePath, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        TextureKeyframe meta = k;

        // Registration records the expected native-orientation dimensions.
        // If a vendor ignores JPEG_ORIENTATION=0 and produces a rotated file,
        // reject it instead of projecting with the wrong intrinsics.
        if (meta.width > 0 && meta.height > 0 &&
            (img.cols != meta.width ||
             img.rows != meta.height)) {
            continue;
        }

        const int originalW = img.cols;
        const int originalH = img.rows;

        if (options.sourceMaxSide > 0 &&
            std::max(originalW, originalH) > options.sourceMaxSide) {
            const float scale =
                static_cast<float>(options.sourceMaxSide) /
                static_cast<float>(std::max(originalW, originalH));

            const int resizedW =
                std::max(2, static_cast<int>(std::round(originalW * scale)));
            const int resizedH =
                std::max(2, static_cast<int>(std::round(originalH * scale)));

            cv::Mat resized;
            cv::resize(
                img,
                resized,
                cv::Size(resizedW, resizedH),
                0.0, 0.0,
                cv::INTER_AREA);
            img = std::move(resized);

            const float sx =
                static_cast<float>(resizedW) /
                static_cast<float>(originalW);
            const float sy =
                static_cast<float>(resizedH) /
                static_cast<float>(originalH);

            meta.fx *= sx;
            meta.fy *= sy;
            meta.cx *= sx;
            meta.cy *= sy;
            meta.width = resizedW;
            meta.height = resizedH;
        }

        LoadedFrame f;
        f.meta = meta;
        f.bgr = std::move(img);

        const float med = approximateMedianLuma(f.bgr);
        medians.push_back(med);

        rasterizeDepth(
            mesh,
            f,
            options.visibilityMaxSide);

        frames.push_back(std::move(f));
    }

    stats.loadedKeyframes =
        static_cast<int>(frames.size());

    if (frames.empty()) {
        if (outStats) *outStats = stats;
        return false;
    }

    if (options.exposureNormalize && !medians.empty()) {
        std::vector<float> tmp = medians;
        const auto mid =
            tmp.begin() + tmp.size()/2u;
        std::nth_element(tmp.begin(), mid, tmp.end());
        const float target =
            std::max(20.0f, *mid);

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const float med =
                std::max(8.0f, medians[i]);
            frames[i].luminanceGain =
                std::clamp(target / med, 0.70f, 1.35f);
        }
    }

    stats.usedKeyframes =
        static_cast<int>(frames.size());

    atlasBgr = cv::Mat(
        mesh.atlasHeight,
        mesh.atlasWidth,
        CV_8UC3,
        cv::Scalar(0,0,0));

    cv::Mat painted(
        mesh.atlasHeight,
        mesh.atlasWidth,
        CV_8UC1,
        cv::Scalar(0));

    const std::size_t triCount = mesh.triangleCount();
    const int blendCount =
        std::max(1, options.maxBlendFrames);

    for (std::size_t t = 0; t < triCount; ++t) {
        const auto ia = mesh.indices[t*3u+0u];
        const auto ib = mesh.indices[t*3u+1u];
        const auto ic = mesh.indices[t*3u+2u];

        if (ia >= mesh.vertices.size() ||
            ib >= mesh.vertices.size() ||
            ic >= mesh.vertices.size()) {
            continue;
        }

        const UvVertex& a = mesh.vertices[ia];
        const UvVertex& b = mesh.vertices[ib];
        const UvVertex& c = mesh.vertices[ic];

        std::vector<std::pair<float,int>> scored;
        scored.reserve(frames.size());

        for (int fi = 0;
             fi < static_cast<int>(frames.size());
             ++fi) {
            const float s =
                triangleFrameScore(
                    a,b,c,
                    frames[fi],
                    options);
            if (s > 0.0f) {
                scored.emplace_back(s, fi);
            }
        }

        std::sort(
            scored.begin(),
            scored.end(),
            [](const auto& x, const auto& y) {
                return x.first > y.first;
            });

        if (scored.size() >
            static_cast<std::size_t>(blendCount)) {
            scored.resize(
                static_cast<std::size_t>(blendCount));
        }

        if (!scored.empty()) ++stats.texturedTriangles;
        else ++stats.fallbackTriangles;

        const Vec2 ua{
            a.u * (mesh.atlasWidth - 1),
            a.v * (mesh.atlasHeight - 1)
        };
        const Vec2 ub{
            b.u * (mesh.atlasWidth - 1),
            b.v * (mesh.atlasHeight - 1)
        };
        const Vec2 uc{
            c.u * (mesh.atlasWidth - 1),
            c.v * (mesh.atlasHeight - 1)
        };

        int minX = std::max(
            0,
            static_cast<int>(
                std::floor(std::min({ua.x,ub.x,uc.x}))));
        int minY = std::max(
            0,
            static_cast<int>(
                std::floor(std::min({ua.y,ub.y,uc.y}))));
        int maxX = std::min(
            mesh.atlasWidth-1,
            static_cast<int>(
                std::ceil(std::max({ua.x,ub.x,uc.x}))));
        int maxY = std::min(
            mesh.atlasHeight-1,
            static_cast<int>(
                std::ceil(std::max({ua.y,ub.y,uc.y}))));

        if (minX > maxX || minY > maxY) continue;

        const Vec3 pa = pos(a);
        const Vec3 pb = pos(b);
        const Vec3 pc = pos(c);

        for (int y = minY; y <= maxY; ++y) {
            auto* dst = atlasBgr.ptr<cv::Vec3b>(y);
            auto* mask = painted.ptr<std::uint8_t>(y);

            for (int x = minX; x <= maxX; ++x) {
                float w0,w1,w2;
                if (!barycentric(
                        ua,ub,uc,
                        Vec2{x+0.5f,y+0.5f},
                        &w0,&w1,&w2)) {
                    continue;
                }

                const Vec3 pw =
                    pa*w0 + pb*w1 + pc*w2;
                const Vec3 n =
                    normalize(
                        normal(a)*w0 +
                        normal(b)*w1 +
                        normal(c)*w2);

                cv::Vec3f sum(0,0,0);
                float weightSum = 0.0f;

                for (const auto& item : scored) {
                    const LoadedFrame& f =
                        frames[item.second];

                    float u,v,z;
                    if (!project(f.meta, pw, &u,&v,&z)) continue;

                    const int border =
                        std::max(0, options.imageBorderPixels);
                    if (u < border || v < border ||
                        u >= f.bgr.cols-border ||
                        v >= f.bgr.rows-border) {
                        continue;
                    }

                    if (!visible(f,u,v,z,options)) continue;

                    const Vec3 view =
                        normalize(cameraPosition(f.meta) - pw);
                    const float facing =
                        dot(n, view);
                    if (facing < options.minViewCos) continue;

                    float weight =
                        std::clamp(f.meta.quality, 0.10f, 3.0f) *
                        std::pow(std::max(0.0f, facing), 4.0f) /
                        (0.20f + z*z);

                    // The triangle-level score also captures projected detail.
                    weight *= std::max(0.01f, item.first);

                    cv::Vec3f sample =
                        bilinearBgr(f.bgr, u, v);
                    sample *= f.luminanceGain;

                    sum += sample * weight;
                    weightSum += weight;
                }

                const bool haveHqSample =
                    weightSum > 1e-8f;

                if (haveHqSample) {
                    sum *= 1.0f / weightSum;
                    dst[x] = cv::Vec3b(
                        cv::saturate_cast<std::uint8_t>(sum[0]),
                        cv::saturate_cast<std::uint8_t>(sum[1]),
                        cv::saturate_cast<std::uint8_t>(sum[2]));
                } else {
                    // Never leave a chart interior black. When no camera sees
                    // this texel reliably, bake the already-fused vertex color.
                    const Vec3 ca = colorRgb(a);
                    const Vec3 cb = colorRgb(b);
                    const Vec3 cc = colorRgb(c);
                    const Vec3 rgb =
                        ca*w0 + cb*w1 + cc*w2;
                    dst[x] = cv::Vec3b(
                        cv::saturate_cast<std::uint8_t>(rgb.z),
                        cv::saturate_cast<std::uint8_t>(rgb.y),
                        cv::saturate_cast<std::uint8_t>(rgb.x));
                }

                if (mask[x] == 0) {
                    mask[x] = 255;
                    ++stats.paintedTexels;
                    if (haveHqSample) {
                        ++stats.hqTexels;
                    }
                }
            }
        }
    }

    dilateGutter(
        atlasBgr,
        painted,
        std::max(0, options.gutterDilationPixels));

    if (stats.paintedTexels > 0) {
        stats.coveragePercent =
            static_cast<float>(
                100.0 *
                static_cast<double>(stats.hqTexels) /
                static_cast<double>(stats.paintedTexels));
    }

    if (outStats) *outStats = stats;
    return !atlasBgr.empty() &&
           stats.paintedTexels > 0;
}
