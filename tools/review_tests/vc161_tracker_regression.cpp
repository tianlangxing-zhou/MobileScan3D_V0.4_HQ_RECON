#include "object_tracker.h"
#include <cassert>
#include <iostream>
int main(){
    cv::setNumThreads(1);
    cv::Mat initial(480,640,CV_8U,cv::Scalar(30));cv::Rect box(180,140,200,200);
    cv::RNG rng(30);rng.fill(initial(box),cv::RNG::UNIFORM,0,255);
    cv::GaussianBlur(initial,initial,cv::Size(3,3),.7);
    ObjectTracker tracker;tracker.setEnabled(true);
    assert(tracker.requestTargetRect(float(box.x)/640,float(box.y)/480,float(box.x+box.width)/640,float(box.y+box.height)/480));
    auto frame=[&](const cv::Mat& gray,uint64_t ts){tracker.updateFrame(gray.data,640,480,640,ts);tracker.track(gray.data,640,480,640,ts);};
    frame(initial,1000000000ULL);assert(tracker.info().state==TargetState::TRACKING);
    const auto before=tracker.info();
    // Rotate/scale about the object center: affine translation is non-zero even
    // though the true center stays fixed. VC160 incorrectly adds that offset.
    cv::Mat a=cv::getRotationMatrix2D(cv::Point2f(280,240),4,1.04), moved;
    cv::warpAffine(initial,moved,a,initial.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(30));
    frame(moved,1033333333ULL);auto after=tracker.info();
    assert(after.state==TargetState::TRACKING);
    const double centerError=std::hypot((after.centerXNorm-before.centerXNorm)*640,(after.centerYNorm-before.centerYNorm)*480);
    assert(centerError<2.5);
    // Exercise production state machine beyond local timeout, with no Nano model.
    tracker.markPresenceLost();cv::Mat blank(480,640,CV_8U,cv::Scalar(30));
    for(uint64_t i=0;i<55;++i)frame(blank,2000000000ULL+i*33333333ULL);
    assert(tracker.info().state==TargetState::LOST);
    cv::Mat returned=blank.clone();initial(box).copyTo(returned(cv::Rect(370,160,200,200)));
    frame(returned,10000000000ULL);assert(tracker.info().state==TargetState::LOST);
    frame(returned,10300000000ULL);assert(tracker.info().state==TargetState::TRACKING);
    assert(tracker.info().timestamp==10300000000ULL && tracker.info().trackedPoints>=15);
    assert(tracker.info().globalRecoveries>0);
    tracker.clearTarget();frame(returned,10600000000ULL);
    assert(tracker.info().state!=TargetState::TRACKING);
    std::cout<<"PASS production tracker: affine center error="<<centerError<<" px, LOST->global relock, source timestamp, feature reseed, clear-target\n";
}
