#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
// Test the real identity scorer without changing its production visibility.
#define private public
#include "object_tracker.h"
#undef private
#include "target_mask_temporal.h"

static bool passes(const TargetMaskMotionComparison& c) {
    return c.comparable && c.iou>=.30f && c.areaJump<=.60f && c.centerJump<=.15f;
}
int main() {
    cv::setNumThreads(1); cv::setRNGSeed(171);
    cv::Mat m=cv::Mat::zeros(200,300,CV_8U), next, scratch;
    cv::rectangle(m,cv::Rect(65,75,45,60),cv::Scalar(255),cv::FILLED);
    cv::Matx23d warp(1,0,80,0,1,20);
    cv::warpAffine(m,next,warp,m.size(),cv::INTER_NEAREST);
    auto c=compareTargetMasks(m,next,{50,50,80,100},{130,70,80,100},scratch);
    assert(passes(c) && c.iou>.99f); // zero unaligned overlap; same moving object
    warp=cv::Matx23d(1.5,0,10,0,1.5,-30);
    cv::warpAffine(m,next,warp,m.size(),cv::INTER_NEAREST);
    c=compareTargetMasks(m,next,{50,50,80,100},{85,45,120,150},scratch);
    assert(passes(c));
    next=cv::Mat::zeros(m.size(),CV_8U);
    cv::rectangle(next,{185,75,45,60},cv::Scalar(255),cv::FILLED);
    assert(!passes(compareTargetMasks(m,next,{50,50,80,100},{130,70,80,100},scratch)));
    assert(!passes(compareTargetMasks(m,next,{50,50,80,100},{50,50,250,100},scratch)));

    cv::Mat frame(480,640,CV_8U); cv::randu(frame,0,256);
    cv::GaussianBlur(frame,frame,{3,3},.7);
    cv::Rect box(180,130,180,210);
    ObjectTracker tracker;
    tracker.captureIdentityAnchor(frame,box);
    assert(tracker.identityReady_);
    const float score=tracker.computeIdentityScore(box,frame);
    std::cout << "Identity same-target NCC=" << score << '\n';
    assert(score>.8f);
    cv::Mat unrelated(frame.size(),CV_8U);cv::randu(unrelated,0,256);
    assert(tracker.computeIdentityScore(box,unrelated)<.45f);
    cv::Mat translated;
    cv::warpAffine(frame,translated,cv::Matx23d(1,0,-175,0,1,0),frame.size());
    const float edgeScore=tracker.computeIdentityScore({5,130,180,210},translated);
    std::cout << "Identity clipped-search NCC=" << edgeScore << '\n';
    assert(edgeScore>.75f);

    TargetReacquirer finder;finder.capture(frame,box);assert(finder.ready());
    cv::Mat moved;
    cv::warpAffine(frame,moved,cv::Matx23d(1,0,55,0,1,-15),frame.size());
    cv::Rect recovered;
    assert(!finder.search(moved,1'000'000'000ULL,recovered));
    assert(!finder.search(moved,1'040'000'000ULL,recovered)); // rate limited
    assert(finder.search(moved,1'080'000'000ULL,recovered));
    assert(std::abs(recovered.x-(box.x+55))<=3 && std::abs(recovered.y-(box.y-15))<=3);
    // Unrelated scene cannot sustain the immutable identity.
    for(int i=0;i<12;++i)assert(!finder.search(unrelated,2'000'000'000ULL+i*220'000'000ULL,recovered));
    assert(!finder.search(moved,10'000'000'000ULL,recovered));
    assert(finder.search(moved,10'080'000'000ULL,recovered));
    // Rotating a slender object legitimately changes its axis-aligned aspect.
    cv::Rect slender(250,110,70,240);
    tracker.captureIdentityAnchor(frame,slender);
    tracker.info_.state=TargetState::LOST;
    auto rotation=cv::getRotationMatrix2D({285,230},90,1.);
    cv::Mat rotated;cv::warpAffine(frame,rotated,rotation,frame.size());
    finder.capture(frame,slender);
    assert(!finder.search(rotated,20'000'000'000ULL,recovered));
    assert(finder.search(rotated,20'080'000'000ULL,recovered));
    assert(tracker.adoptNanoBox(recovered,640,480,rotated,20'080'000'000ULL,true));
    // Exercise real LOST -> global search -> TRACKING, then KLT next exposure.
    ObjectTracker lifecycle;
    lifecycle.enabled_=true;
    lifecycle.captureIdentityAnchor(frame,box);
    lifecycle.info_.state=TargetState::LOST;
    lifecycle.updateFrame(moved.data,640,480,int(moved.step),30'000'000'000ULL);
    lifecycle.track(moved.data,640,480,int(moved.step),30'000'000'000ULL);
    assert(lifecycle.info().state==TargetState::LOST);
    lifecycle.updateFrame(moved.data,640,480,int(moved.step),30'080'000'000ULL);
    lifecycle.track(moved.data,640,480,int(moved.step),30'080'000'000ULL);
    assert(lifecycle.isTracking());
    lifecycle.updateFrame(moved.data,640,480,int(moved.step),30'120'000'000ULL);
    lifecycle.track(moved.data,640,480,int(moved.step),30'120'000'000ULL);
    assert(lifecycle.isTracking());
    std::cout << "PASS vision: compensated masks, mismatch rejection, identity scale, edge clipping, 80ms confirmation, lost/return, rotation, KLT handoff\n";
}
