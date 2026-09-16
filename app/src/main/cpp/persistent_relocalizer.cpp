#include "persistent_relocalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace {

constexpr std::array<char, 8> kMapMagic{
    'M','S','V','M','A','P','7','1'
};
constexpr std::array<char, 8> kAlignMagic{
    'M','S','A','L','I','G','N','7'
};
constexpr std::uint32_t kMapVersion = 1u;
constexpr std::uint32_t kMaxMapPoints = 12000u;

constexpr std::uint64_t kCaptureMaxGapNs =
    2'000'000'000ULL;
constexpr float kCaptureMinTranslation = 0.12f;
constexpr float kCaptureMinRotationDeg = 10.0f;

constexpr int kMinCaptureObservations = 12;
constexpr int kMinRelocMatches = 20;
constexpr int kMinRelocInliers = 16;
constexpr int kMinSavedMapPoints = 80;
constexpr int kMinSavedKeyframes = 3;


template <typename T>
bool writePod(
        std::ofstream& out,
        const T& value) {
    out.write(
        reinterpret_cast<const char*>(
            &value),
        static_cast<std::streamsize>(
            sizeof(T)));
    return static_cast<bool>(out);
}

template <typename T>
bool readPod(
        std::ifstream& in,
        T* value) {
    if (!value) return false;
    in.read(
        reinterpret_cast<char*>(
            value),
        static_cast<std::streamsize>(
            sizeof(T)));
    return static_cast<bool>(in);
}

struct DVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct DMat3 {
    double m[9] = {
        1,0,0,
        0,1,0,
        0,0,1
    };
};

struct DQuat {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

DVec3 operator+(
        const DVec3& a,
        const DVec3& b) {
    return {
        a.x+b.x,
        a.y+b.y,
        a.z+b.z
    };
}

DVec3 operator-(
        const DVec3& a,
        const DVec3& b) {
    return {
        a.x-b.x,
        a.y-b.y,
        a.z-b.z
    };
}

DVec3 operator*(
        const DVec3& a,
        double s) {
    return {
        a.x*s,
        a.y*s,
        a.z*s
    };
}

DVec3 operator*(
        double s,
        const DVec3& a) {
    return a*s;
}

double dot(
        const DVec3& a,
        const DVec3& b) {
    return
        a.x*b.x +
        a.y*b.y +
        a.z*b.z;
}

double norm(
        const DVec3& a) {
    return std::sqrt(
        dot(a,a));
}

bool finite(
        const DVec3& a) {
    return
        std::isfinite(a.x) &&
        std::isfinite(a.y) &&
        std::isfinite(a.z);
}

DMat3 matFromArray(
        const float r[9]) {
    DMat3 out;
    for (int i=0;i<9;++i) {
        out.m[i]=r[i];
    }
    return out;
}

DVec3 vecFromArray(
        const float t[3]) {
    return {t[0],t[1],t[2]};
}

void matToArray(
        const DMat3& R,
        float out[9]) {
    for (int i=0;i<9;++i) {
        out[i]=
            static_cast<float>(
                R.m[i]);
    }
}

void vecToArray(
        const DVec3& t,
        float out[3]) {
    out[0]=static_cast<float>(t.x);
    out[1]=static_cast<float>(t.y);
    out[2]=static_cast<float>(t.z);
}

DMat3 transpose(
        const DMat3& a) {
    DMat3 r;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            r.m[i*3+j]=
                a.m[j*3+i];
        }
    }
    return r;
}

DMat3 operator*(
        const DMat3& a,
        const DMat3& b) {
    DMat3 r;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            double v=0.0;
            for (int k=0;k<3;++k) {
                v +=
                    a.m[i*3+k] *
                    b.m[k*3+j];
            }
            r.m[i*3+j]=v;
        }
    }
    return r;
}

DVec3 operator*(
        const DMat3& a,
        const DVec3& b) {
    return {
        a.m[0]*b.x +
            a.m[1]*b.y +
            a.m[2]*b.z,
        a.m[3]*b.x +
            a.m[4]*b.y +
            a.m[5]*b.z,
        a.m[6]*b.x +
            a.m[7]*b.y +
            a.m[8]*b.z
    };
}

double determinant(
        const DMat3& a) {
    return
        a.m[0]*
            (a.m[4]*a.m[8]-
             a.m[5]*a.m[7]) -
        a.m[1]*
            (a.m[3]*a.m[8]-
             a.m[5]*a.m[6]) +
        a.m[2]*
            (a.m[3]*a.m[7]-
             a.m[4]*a.m[6]);
}

double trace(
        const DMat3& a) {
    return
        a.m[0]+a.m[4]+a.m[8];
}

bool finite(
        const DMat3& a) {
    for (double v:a.m) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

double rotationAngleDeg(
        const DMat3& a,
        const DMat3& b) {
    const DMat3 d =
        transpose(a)*b;
    const double c =
        std::clamp(
            (trace(d)-1.0)*0.5,
            -1.0,
            1.0);
    return
        std::acos(c) *
        180.0 /
        3.14159265358979323846;
}

DQuat normalized(
        DQuat q) {
    const double n =
        std::sqrt(
            q.w*q.w+
            q.x*q.x+
            q.y*q.y+
            q.z*q.z);

    if (n<=1e-15) {
        return {};
    }

    q.w/=n;
    q.x/=n;
    q.y/=n;
    q.z/=n;
    return q;
}

DQuat quatFromMat(
        const DMat3& R) {
    DQuat q;
    const double tr=trace(R);

    if (tr>0.0) {
        const double s=
            std::sqrt(tr+1.0)*2.0;
        q.w=0.25*s;
        q.x=(R.m[7]-R.m[5])/s;
        q.y=(R.m[2]-R.m[6])/s;
        q.z=(R.m[3]-R.m[1])/s;
    } else if (
        R.m[0]>R.m[4] &&
        R.m[0]>R.m[8]) {
        const double s=
            std::sqrt(
                1.0+
                R.m[0]-
                R.m[4]-
                R.m[8])*2.0;
        q.w=(R.m[7]-R.m[5])/s;
        q.x=0.25*s;
        q.y=(R.m[1]+R.m[3])/s;
        q.z=(R.m[2]+R.m[6])/s;
    } else if (R.m[4]>R.m[8]) {
        const double s=
            std::sqrt(
                1.0+
                R.m[4]-
                R.m[0]-
                R.m[8])*2.0;
        q.w=(R.m[2]-R.m[6])/s;
        q.x=(R.m[1]+R.m[3])/s;
        q.y=0.25*s;
        q.z=(R.m[5]+R.m[7])/s;
    } else {
        const double s=
            std::sqrt(
                1.0+
                R.m[8]-
                R.m[0]-
                R.m[4])*2.0;
        q.w=(R.m[3]-R.m[1])/s;
        q.x=(R.m[2]+R.m[6])/s;
        q.y=(R.m[5]+R.m[7])/s;
        q.z=0.25*s;
    }

    return normalized(q);
}

DMat3 matFromQuat(
        DQuat q) {
    q=normalized(q);

    const double xx=q.x*q.x;
    const double yy=q.y*q.y;
    const double zz=q.z*q.z;
    const double xy=q.x*q.y;
    const double xz=q.x*q.z;
    const double yz=q.y*q.z;
    const double wx=q.w*q.x;
    const double wy=q.w*q.y;
    const double wz=q.w*q.z;

    DMat3 R;
    R.m[0]=1.0-2.0*(yy+zz);
    R.m[1]=2.0*(xy-wz);
    R.m[2]=2.0*(xz+wy);

    R.m[3]=2.0*(xy+wz);
    R.m[4]=1.0-2.0*(xx+zz);
    R.m[5]=2.0*(yz-wx);

    R.m[6]=2.0*(xz-wy);
    R.m[7]=2.0*(yz+wx);
    R.m[8]=1.0-2.0*(xx+yy);

    return R;
}

DQuat slerp(
        DQuat a,
        DQuat b,
        double t) {
    a=normalized(a);
    b=normalized(b);

    double c=
        a.w*b.w+
        a.x*b.x+
        a.y*b.y+
        a.z*b.z;

    if (c<0.0) {
        b.w=-b.w;
        b.x=-b.x;
        b.y=-b.y;
        b.z=-b.z;
        c=-c;
    }

    c=std::clamp(c,-1.0,1.0);

    if (c>0.9995) {
        DQuat q{
            a.w+t*(b.w-a.w),
            a.x+t*(b.x-a.x),
            a.y+t*(b.y-a.y),
            a.z+t*(b.z-a.z)
        };
        return normalized(q);
    }

    const double theta=
        std::acos(c);
    const double sinTheta=
        std::sin(theta);

    if (std::abs(sinTheta)<1e-12) {
        return a;
    }

    const double wa=
        std::sin(
            (1.0-t)*theta)/
        sinTheta;
    const double wb=
        std::sin(
            t*theta)/
        sinTheta;

    return normalized({
        wa*a.w+wb*b.w,
        wa*a.x+wb*b.x,
        wa*a.y+wb*b.y,
        wa*a.z+wb*b.z
    });
}

double median(
        std::vector<double> v) {
    if (v.empty()) return 0.0;

    const std::size_t mid=
        v.size()/2u;

    std::nth_element(
        v.begin(),
        v.begin()+
            static_cast<std::ptrdiff_t>(
                mid),
        v.end());

    double m=v[mid];

    if ((v.size()&1u)==0u) {
        const auto it=
            std::max_element(
                v.begin(),
                v.begin()+
                    static_cast<std::ptrdiff_t>(
                        mid));

        if (it!=
            v.begin()+
                static_cast<std::ptrdiff_t>(
                    mid)) {
            m=0.5*(m+*it);
        }
    }

    return m;
}

cv::Ptr<cv::ORB> makeOrb(
        int maxFeatures) {
    return cv::ORB::create(
        maxFeatures,
        1.2f,
        8,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        12);
}

} // namespace

PersistentRelocalizer::
PersistentRelocalizer() {
    resetAll();
}

void PersistentRelocalizer::resetAll() {
    std::lock_guard<std::mutex> lk(
        mutex_);

    mapPoints_.clear();
    descriptorBytes_.clear();
    capturedFeatureIndex_.clear();

    captureEnabled_=false;
    capturedKeyframes_=0;
    haveLastCapturePose_=false;
    lastCaptureTimestampNs_=0;

    mapLoaded_=false;
    localized_=false;
    havePendingAlignment_=false;
    pendingHits_=0;
    lastAttemptNs_=0;

    std::fill(
        alignR_,
        alignR_+9,
        0.0f);
    alignR_[0]=
        alignR_[4]=
        alignR_[8]=1.0f;

    alignT_[0]=
        alignT_[1]=
        alignT_[2]=0.0f;

    stats_=
        PersistentRelocalizationStats{};
}

void PersistentRelocalizer::
resetLiveAlignment() {
    std::lock_guard<std::mutex> lk(
        mutex_);

    localized_=false;
    havePendingAlignment_=false;
    pendingHits_=0;
    lastAttemptNs_=0;

    std::fill(
        alignR_,
        alignR_+9,
        0.0f);
    alignR_[0]=
        alignR_[4]=
        alignR_[8]=1.0f;

    alignT_[0]=
        alignT_[1]=
        alignT_[2]=0.0f;

    stats_.localized=false;
    stats_.state=
        mapLoaded_ ? 1 : 0;
    stats_.lastDetected=0;
    stats_.lastMatches=0;
    stats_.lastInliers=0;
    stats_.lastInlierRatio=0.0f;
    stats_.lastMedianReprojectionPx=0.0f;
}

void PersistentRelocalizer::
setCaptureEnabled(
        bool enabled) {
    std::lock_guard<std::mutex> lk(
        mutex_);

    captureEnabled_=enabled;
    stats_.captureEnabled=enabled;

    if (enabled) {
        mapPoints_.clear();
        descriptorBytes_.clear();
        capturedFeatureIndex_.clear();

        capturedKeyframes_=0;
        haveLastCapturePose_=false;
        lastCaptureTimestampNs_=0;

        mapLoaded_=false;
        localized_=false;
        havePendingAlignment_=false;
        pendingHits_=0;

        stats_=
            PersistentRelocalizationStats{};
        stats_.captureEnabled=true;
    }
}

bool PersistentRelocalizer::
captureEnabled() const {
    std::lock_guard<std::mutex> lk(
        mutex_);
    return captureEnabled_;
}

bool PersistentRelocalizer::
shouldCaptureKeyframeLocked(
        std::uint64_t timestampNs,
        const float Rwc[9],
        const float twc[3]) const {
    if (!haveLastCapturePose_) {
        return true;
    }

    if (timestampNs>
        lastCaptureTimestampNs_+
            kCaptureMaxGapNs) {
        return true;
    }

    const DVec3 t=
        vecFromArray(twc);
    const DVec3 lastT=
        vecFromArray(
            lastCaptureTwc_);

    if (norm(t-lastT)>=
        kCaptureMinTranslation) {
        return true;
    }

    return
        rotationAngleDeg(
            matFromArray(
                lastCaptureRwc_),
            matFromArray(Rwc)) >=
        kCaptureMinRotationDeg;
}

bool PersistentRelocalizer::
captureFrame(
        const std::uint8_t* gray,
        int width,
        int height,
        int stride,
        std::uint64_t timestampNs,
        const float liveCameraRwc[9],
        const float liveCameraTwc[3],
        const PersistentWorldObservation*
            observations,
        int observationCount) {
    if (!gray ||
        width<=64 ||
        height<=64 ||
        stride<width ||
        !liveCameraRwc ||
        !liveCameraTwc ||
        !observations ||
        observationCount<
            kMinCaptureObservations) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        if (!captureEnabled_) {
            return false;
        }

        if (mapPoints_.size()>=
            kMaxMapPoints) {
            return false;
        }

        if (!shouldCaptureKeyframeLocked(
                timestampNs,
                liveCameraRwc,
                liveCameraTwc)) {
            return false;
        }
    }

    cv::Mat image(
        height,
        width,
        CV_8UC1,
        const_cast<std::uint8_t*>(gray),
        static_cast<std::size_t>(
            stride));

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(
        static_cast<std::size_t>(
            observationCount));

    for (int i=0;
         i<observationCount;
         ++i) {
        const auto& o=
            observations[i];

        if (!std::isfinite(o.u) ||
            !std::isfinite(o.v) ||
            !std::isfinite(o.x) ||
            !std::isfinite(o.y) ||
            !std::isfinite(o.z) ||
            o.featureId<0) {
            continue;
        }

        if (o.u<20.0f ||
            o.v<20.0f ||
            o.u>=width-20.0f ||
            o.v>=height-20.0f) {
            continue;
        }

        cv::KeyPoint kp(
            o.u,
            o.v,
            31.0f,
            -1.0f,
            0.0f,
            0,
            i);

        keypoints.push_back(kp);
    }

    if (keypoints.size()<
        static_cast<std::size_t>(
            kMinCaptureObservations)) {
        return false;
    }

    cv::Mat descriptors;
    auto orb=
        makeOrb(
            static_cast<int>(
                keypoints.size()));

    orb->compute(
        image,
        keypoints,
        descriptors);

    if (descriptors.empty() ||
        descriptors.cols!=32 ||
        descriptors.rows<
            kMinCaptureObservations) {
        return false;
    }

    int added=0;

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        if (!captureEnabled_) {
            return false;
        }

        for (int r=0;
             r<descriptors.rows &&
             mapPoints_.size()<
                kMaxMapPoints;
             ++r) {
            const int sourceIndex=
                keypoints[
                    static_cast<std::size_t>(
                        r)].class_id;

            if (sourceIndex<0 ||
                sourceIndex>=
                    observationCount) {
                continue;
            }

            const auto& o=
                observations[
                    sourceIndex];

            auto existing =
                capturedFeatureIndex_.find(
                    o.featureId);

            if (existing !=
                capturedFeatureIndex_.end()) {
                // VINS keeps refining triangulated points while the feature
                // remains in the sliding window. Preserve the original ORB
                // appearance descriptor, but refresh XYZ with the latest
                // optimized world coordinate before the map is persisted.
                const std::size_t index =
                    existing->second;

                if (index < mapPoints_.size()) {
                    mapPoints_[index].x = o.x;
                    mapPoints_[index].y = o.y;
                    mapPoints_[index].z = o.z;
                }
                continue;
            }

            MapPoint point;
            point.x=o.x;
            point.y=o.y;
            point.z=o.z;

            std::memcpy(
                point.descriptor.data(),
                descriptors.ptr<
                    std::uint8_t>(r),
                point.descriptor.size());

            mapPoints_.push_back(point);
            capturedFeatureIndex_[o.featureId] =
                mapPoints_.size() - 1u;
            ++added;
        }

        if (added>=6) {
            ++capturedKeyframes_;

            std::copy(
                liveCameraRwc,
                liveCameraRwc+9,
                lastCaptureRwc_);

            std::copy(
                liveCameraTwc,
                liveCameraTwc+3,
                lastCaptureTwc_);

            lastCaptureTimestampNs_=
                timestampNs;
            haveLastCapturePose_=true;
        }

        stats_.capturedKeyframes=
            capturedKeyframes_;
        stats_.capturedPoints=
            static_cast<int>(
                mapPoints_.size());
        stats_.mapPoints=
            static_cast<int>(
                mapPoints_.size());
    }

    return added>=6;
}

void PersistentRelocalizer::
rebuildDescriptorMatrixLocked() {
    descriptorBytes_.resize(
        mapPoints_.size()*32u);

    for (std::size_t i=0;
         i<mapPoints_.size();
         ++i) {
        std::memcpy(
            descriptorBytes_.data()+
                i*32u,
            mapPoints_[i]
                .descriptor.data(),
            32u);
    }
}

bool PersistentRelocalizer::saveMap(
        const std::string& path) const {
    std::lock_guard<std::mutex> lk(
        mutex_);

    if (mapPoints_.size()<
            static_cast<std::size_t>(
                kMinSavedMapPoints) ||
        capturedKeyframes_<
            kMinSavedKeyframes) {
        return false;
    }

    std::ofstream out(
        path,
        std::ios::binary |
        std::ios::trunc);

    if (!out) return false;

    out.write(
        kMapMagic.data(),
        kMapMagic.size());

    const std::uint32_t count=
        static_cast<std::uint32_t>(
            mapPoints_.size());

    if (!writePod(out,kMapVersion) ||
        !writePod(out,count)) {
        return false;
    }

    for (const auto& point:
         mapPoints_) {
        if (!writePod(out,point.x) ||
            !writePod(out,point.y) ||
            !writePod(out,point.z)) {
            return false;
        }

        out.write(
            reinterpret_cast<
                const char*>(
                    point.descriptor.data()),
            static_cast<
                std::streamsize>(
                    point.descriptor.size()));

        if (!out) return false;
    }

    return static_cast<bool>(out);
}

bool PersistentRelocalizer::loadMap(
        const std::string& path) {
    std::ifstream in(
        path,
        std::ios::binary);

    if (!in) return false;

    std::array<char,8> magic{};
    in.read(
        magic.data(),
        magic.size());

    if (!in ||
        magic!=kMapMagic) {
        return false;
    }

    std::uint32_t version=0;
    std::uint32_t count=0;

    if (!readPod(in,&version) ||
        !readPod(in,&count) ||
        version!=kMapVersion ||
        count<
            static_cast<
                std::uint32_t>(
                    kMinRelocInliers) ||
        count>kMaxMapPoints) {
        return false;
    }

    std::vector<MapPoint> loaded(
        count);

    for (auto& point:loaded) {
        if (!readPod(in,&point.x) ||
            !readPod(in,&point.y) ||
            !readPod(in,&point.z)) {
            return false;
        }

        in.read(
            reinterpret_cast<char*>(
                point.descriptor.data()),
            static_cast<
                std::streamsize>(
                    point.descriptor.size()));

        if (!in ||
            !std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z)) {
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        mapPoints_.swap(loaded);
        rebuildDescriptorMatrixLocked();

        capturedFeatureIndex_.clear();
        captureEnabled_=false;

        mapLoaded_=true;
        localized_=false;
        havePendingAlignment_=false;
        pendingHits_=0;
        lastAttemptNs_=0;

        stats_=
            PersistentRelocalizationStats{};
        stats_.state=1;
        stats_.mapPoints=
            static_cast<int>(
                mapPoints_.size());
        stats_.mapLoaded=true;
    }

    return true;
}

bool PersistentRelocalizer::
tryRelocalize(
        const std::uint8_t* gray,
        int width,
        int height,
        int stride,
        std::uint64_t timestampNs,
        float fx,
        float fy,
        float cx,
        float cy,
        const float liveCameraRwc[9],
        const float liveCameraTwc[3]) {
    if (!gray ||
        width<=64 ||
        height<=64 ||
        stride<width ||
        fx<=1.0f ||
        fy<=1.0f ||
        !liveCameraRwc ||
        !liveCameraTwc) {
        return false;
    }

    std::vector<MapPoint> mapCopy;
    std::vector<std::uint8_t> descCopy;
    bool alreadyLocalized=false;

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        if (!mapLoaded_ ||
            mapPoints_.size()<
                static_cast<
                    std::size_t>(
                        kMinRelocInliers)) {
            return false;
        }

        const std::uint64_t minPeriod=
            localized_
                ? 1'000'000'000ULL
                : 300'000'000ULL;

        if (lastAttemptNs_>0 &&
            timestampNs<
                lastAttemptNs_+
                minPeriod) {
            return localized_;
        }

        lastAttemptNs_=timestampNs;
        alreadyLocalized=localized_;

        mapCopy=mapPoints_;
        descCopy=descriptorBytes_;

        stats_.state=
            localized_ ? 3 : 2;
        ++stats_.attempts;
    }

    cv::Mat image(
        height,
        width,
        CV_8UC1,
        const_cast<std::uint8_t*>(gray),
        static_cast<std::size_t>(
            stride));

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    auto orb=makeOrb(1600);

    orb->detectAndCompute(
        image,
        cv::noArray(),
        keypoints,
        descriptors);

    const int detected=
        descriptors.empty()
            ? 0
            : descriptors.rows;

    if (descriptors.empty() ||
        descriptors.cols!=32 ||
        detected<kMinRelocMatches) {
        std::lock_guard<std::mutex> lk(
            mutex_);
        stats_.lastDetected=detected;
        stats_.lastMatches=0;
        stats_.lastInliers=0;
        return localized_;
    }

    cv::Mat mapDescriptors(
        static_cast<int>(
            mapCopy.size()),
        32,
        CV_8UC1,
        descCopy.data());

    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false);

    std::vector<
        std::vector<cv::DMatch>> knn;

    matcher.knnMatch(
        descriptors,
        mapDescriptors,
        knn,
        2);

    std::unordered_map<
        int,
        cv::DMatch> bestByMapPoint;

    for (const auto& pair:knn) {
        if (pair.size()<2) {
            continue;
        }

        const auto& a=pair[0];
        const auto& b=pair[1];

        if (!(a.distance<
              0.75f*b.distance) ||
            a.distance>72.0f) {
            continue;
        }

        auto it=
            bestByMapPoint.find(
                a.trainIdx);

        if (it==
                bestByMapPoint.end() ||
            a.distance<
                it->second.distance) {
            bestByMapPoint[
                a.trainIdx]=a;
        }
    }

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    objectPoints.reserve(
        bestByMapPoint.size());
    imagePoints.reserve(
        bestByMapPoint.size());

    for (const auto& kv:
         bestByMapPoint) {
        const auto& match=
            kv.second;

        if (match.trainIdx<0 ||
            match.trainIdx>=
                static_cast<int>(
                    mapCopy.size()) ||
            match.queryIdx<0 ||
            match.queryIdx>=
                static_cast<int>(
                    keypoints.size())) {
            continue;
        }

        const auto& point=
            mapCopy[
                static_cast<
                    std::size_t>(
                        match.trainIdx)];

        objectPoints.emplace_back(
            point.x,
            point.y,
            point.z);

        imagePoints.push_back(
            keypoints[
                static_cast<
                    std::size_t>(
                        match.queryIdx)].pt);
    }

    const int matchCount=
        static_cast<int>(
            objectPoints.size());

    if (matchCount<
        kMinRelocMatches) {
        std::lock_guard<std::mutex> lk(
            mutex_);
        stats_.lastDetected=detected;
        stats_.lastMatches=matchCount;
        stats_.lastInliers=0;
        return localized_;
    }

    const cv::Matx33d K(
        fx,0.0,cx,
        0.0,fy,cy,
        0.0,0.0,1.0);

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;

    bool pnpOk=false;

    try {
        pnpOk=
            cv::solvePnPRansac(
                objectPoints,
                imagePoints,
                K,
                cv::noArray(),
                rvec,
                tvec,
                false,
                200,
                4.0,
                0.995,
                inliers,
                cv::SOLVEPNP_EPNP);
    } catch (
        const cv::Exception&) {
        pnpOk=false;
    }

    const int inlierCount=
        pnpOk
            ? inliers.rows
            : 0;

    const float inlierRatio=
        matchCount>0
            ? static_cast<float>(
                  inlierCount)/
              static_cast<float>(
                  matchCount)
            : 0.0f;

    if (!pnpOk ||
        inlierCount<
            kMinRelocInliers ||
        inlierRatio<0.35f) {
        std::lock_guard<std::mutex> lk(
            mutex_);
        stats_.lastDetected=detected;
        stats_.lastMatches=matchCount;
        stats_.lastInliers=inlierCount;
        stats_.lastInlierRatio=inlierRatio;
        return localized_;
    }

    std::vector<cv::Point3f> inlierObject;
    std::vector<cv::Point2f> inlierImage;

    inlierObject.reserve(
        static_cast<
            std::size_t>(
                inlierCount));
    inlierImage.reserve(
        static_cast<
            std::size_t>(
                inlierCount));

    for (int i=0;
         i<inlierCount;
         ++i) {
        const int idx=
            inliers.at<int>(i,0);

        if (idx<0 ||
            idx>=matchCount) {
            continue;
        }

        inlierObject.push_back(
            objectPoints[
                static_cast<
                    std::size_t>(idx)]);

        inlierImage.push_back(
            imagePoints[
                static_cast<
                    std::size_t>(idx)]);
    }

    if (inlierObject.size()>=
        static_cast<
            std::size_t>(
                kMinRelocInliers)) {
        try {
            cv::solvePnP(
                inlierObject,
                inlierImage,
                K,
                cv::noArray(),
                rvec,
                tvec,
                true,
                cv::SOLVEPNP_ITERATIVE);
        } catch (
            const cv::Exception&) {
            // Keep RANSAC result.
        }
    }

    std::vector<cv::Point2f> reprojected;

    try {
        cv::projectPoints(
            inlierObject,
            rvec,
            tvec,
            K,
            cv::noArray(),
            reprojected);
    } catch (
        const cv::Exception&) {
        return alreadyLocalized;
    }

    std::vector<double> errors;
    errors.reserve(
        reprojected.size());

    for (std::size_t i=0;
         i<reprojected.size() &&
         i<inlierImage.size();
         ++i) {
        const cv::Point2f d=
            reprojected[i]-
            inlierImage[i];

        errors.push_back(
            std::sqrt(
                static_cast<double>(
                    d.x*d.x+
                    d.y*d.y)));
    }

    const double medianReproj=
        median(errors);

    if (!std::isfinite(
            medianReproj) ||
        medianReproj>3.0) {
        std::lock_guard<std::mutex> lk(
            mutex_);

        stats_.lastDetected=detected;
        stats_.lastMatches=matchCount;
        stats_.lastInliers=inlierCount;
        stats_.lastInlierRatio=inlierRatio;
        stats_.lastMedianReprojectionPx=
            static_cast<float>(
                medianReproj);

        return localized_;
    }

    cv::Mat RcwCv;
    cv::Rodrigues(
        rvec,
        RcwCv);

    DMat3 Rcw;
    for (int r=0;r<3;++r) {
        for (int c=0;c<3;++c) {
            Rcw.m[r*3+c]=
                RcwCv.at<double>(r,c);
        }
    }

    const DVec3 tcw{
        tvec.at<double>(0,0),
        tvec.at<double>(1,0),
        tvec.at<double>(2,0)
    };

    const DMat3 Rsc=
        transpose(Rcw);

    const DVec3 tsc=
        (Rsc*tcw)*-1.0;

    const DMat3 Rlc=
        matFromArray(
            liveCameraRwc);
    const DVec3 tlc=
        vecFromArray(
            liveCameraTwc);

    const DMat3 Rsl=
        Rsc*
        transpose(Rlc);

    const DVec3 tsl=
        tsc-
        Rsl*tlc;

    if (!finite(Rsl) ||
        !finite(tsl) ||
        std::abs(
            determinant(Rsl)-1.0)>
            0.03 ||
        norm(tsl)>50.0) {
        return alreadyLocalized;
    }

    bool accepted=false;

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        stats_.lastDetected=detected;
        stats_.lastMatches=matchCount;
        stats_.lastInliers=inlierCount;
        stats_.lastInlierRatio=inlierRatio;
        stats_.lastMedianReprojectionPx=
            static_cast<float>(
                medianReproj);

        if (!localized_) {
            if (!havePendingAlignment_) {
                matToArray(
                    Rsl,
                    pendingR_);
                vecToArray(
                    tsl,
                    pendingT_);

                havePendingAlignment_=true;
                pendingHits_=1;
            } else {
                const DMat3 Rp=
                    matFromArray(
                        pendingR_);
                const DVec3 tp=
                    vecFromArray(
                        pendingT_);

                const double angle=
                    rotationAngleDeg(
                        Rp,
                        Rsl);

                const double translation=
                    norm(tp-tsl);

                if (angle<=10.0 &&
                    translation<=0.20) {
                    ++pendingHits_;

                    const DMat3 Ravg=
                        matFromQuat(
                            slerp(
                                quatFromMat(Rp),
                                quatFromMat(Rsl),
                                0.5));

                    const DVec3 tavg=
                        (tp+tsl)*0.5;

                    matToArray(
                        Ravg,
                        pendingR_);
                    vecToArray(
                        tavg,
                        pendingT_);

                    if (pendingHits_>=2) {
                        std::copy(
                            pendingR_,
                            pendingR_+9,
                            alignR_);

                        std::copy(
                            pendingT_,
                            pendingT_+3,
                            alignT_);

                        localized_=true;
                        accepted=true;
                        ++stats_.successes;
                    }
                } else {
                    matToArray(
                        Rsl,
                        pendingR_);
                    vecToArray(
                        tsl,
                        pendingT_);
                    pendingHits_=1;
                }
            }
        } else {
            const DMat3 currentR=
                matFromArray(
                    alignR_);
            const DVec3 currentT=
                vecFromArray(
                    alignT_);

            const double angle=
                rotationAngleDeg(
                    currentR,
                    Rsl);

            const double translation=
                norm(
                    currentT-tsl);

            if (angle<=15.0 &&
                translation<=0.35 &&
                inlierRatio>=0.40f) {
                const double weight=
                    std::clamp(
                        0.08+
                        0.20*
                            static_cast<double>(
                                inlierRatio),
                        0.10,
                        0.25);

                const DMat3 smoothR=
                    matFromQuat(
                        slerp(
                            quatFromMat(
                                currentR),
                            quatFromMat(
                                Rsl),
                            weight));

                const DVec3 smoothT=
                    (1.0-weight)*
                        currentT+
                    weight*tsl;

                matToArray(
                    smoothR,
                    alignR_);
                vecToArray(
                    smoothT,
                    alignT_);

                accepted=true;
                ++stats_.successes;
            }
        }

        stats_.localized=localized_;
        stats_.mapLoaded=mapLoaded_;
        stats_.state=
            localized_ ? 3 : 2;
    }

    return
        localized_ ||
        accepted;
}

bool PersistentRelocalizer::
transformPose(
        const float inRwc[9],
        const float inTwc[3],
        float outRwc[9],
        float outTwc[3]) const {
    if (!inRwc ||
        !inTwc ||
        !outRwc ||
        !outTwc) {
        return false;
    }

    float ar[9];
    float at[3];
    bool mapLoaded=false;
    bool localized=false;

    {
        std::lock_guard<std::mutex> lk(
            mutex_);

        mapLoaded=mapLoaded_;
        localized=localized_;

        std::copy(
            alignR_,
            alignR_+9,
            ar);

        std::copy(
            alignT_,
            alignT_+3,
            at);
    }

    if (!mapLoaded) {
        std::copy(
            inRwc,
            inRwc+9,
            outRwc);

        std::copy(
            inTwc,
            inTwc+3,
            outTwc);

        return true;
    }

    if (!localized) {
        return false;
    }

    const DMat3 A=
        matFromArray(ar);
    const DVec3 a=
        vecFromArray(at);

    const DMat3 R=
        matFromArray(inRwc);
    const DVec3 t=
        vecFromArray(inTwc);

    const DMat3 outR=
        A*R;
    const DVec3 outT=
        A*t+a;

    matToArray(
        outR,
        outRwc);
    vecToArray(
        outT,
        outTwc);

    return true;
}

bool PersistentRelocalizer::
saveAlignment(
        const std::string& path) const {
    std::lock_guard<std::mutex> lk(
        mutex_);

    if (!localized_) {
        return false;
    }

    std::ofstream out(
        path,
        std::ios::binary |
        std::ios::trunc);

    if (!out) return false;

    out.write(
        kAlignMagic.data(),
        kAlignMagic.size());

    if (!writePod(
            out,
            kMapVersion)) {
        return false;
    }

    out.write(
        reinterpret_cast<
            const char*>(
                alignR_),
        9*sizeof(float));

    out.write(
        reinterpret_cast<
            const char*>(
                alignT_),
        3*sizeof(float));

    return static_cast<bool>(out);
}

bool PersistentRelocalizer::
loadAlignment(
        const std::string& path) {
    std::ifstream in(
        path,
        std::ios::binary);

    if (!in) return false;

    std::array<char,8> magic{};
    in.read(
        magic.data(),
        magic.size());

    std::uint32_t version=0;
    float R[9];
    float t[3];

    if (!in ||
        magic!=kAlignMagic ||
        !readPod(in,&version) ||
        version!=kMapVersion) {
        return false;
    }

    in.read(
        reinterpret_cast<char*>(R),
        sizeof(R));

    in.read(
        reinterpret_cast<char*>(t),
        sizeof(t));

    if (!in) return false;

    const DMat3 Re=
        matFromArray(R);
    const DVec3 te=
        vecFromArray(t);

    if (!finite(Re) ||
        !finite(te) ||
        std::abs(
            determinant(Re)-1.0)>
            0.03) {
        return false;
    }

    std::lock_guard<std::mutex> lk(
        mutex_);

    std::copy(
        R,
        R+9,
        alignR_);
    std::copy(
        t,
        t+3,
        alignT_);

    localized_=true;
    stats_.localized=true;
    stats_.state=3;

    return true;
}

bool PersistentRelocalizer::
hasLoadedMap() const {
    std::lock_guard<std::mutex> lk(
        mutex_);
    return mapLoaded_;
}

bool PersistentRelocalizer::
localized() const {
    std::lock_guard<std::mutex> lk(
        mutex_);
    return localized_;
}

PersistentRelocalizationStats
PersistentRelocalizer::stats() const {
    std::lock_guard<std::mutex> lk(
        mutex_);

    PersistentRelocalizationStats out=
        stats_;

    out.mapPoints=
        static_cast<int>(
            mapPoints_.size());

    out.capturedKeyframes=
        capturedKeyframes_;

    out.capturedPoints=
        static_cast<int>(
            mapPoints_.size());

    out.captureEnabled=
        captureEnabled_;

    out.mapLoaded=
        mapLoaded_;

    out.localized=
        localized_;

    return out;
}
