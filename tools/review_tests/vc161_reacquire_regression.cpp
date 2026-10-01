#include "target_reacquirer.h"
#include <cassert>
#include <iostream>
#include <chrono>
int main(){
    cv::setNumThreads(1);
    cv::Mat initial(480,640,CV_8U,cv::Scalar(30));
    cv::Rect original(70,100,200,200);
    cv::Mat texture(200,200,CV_8U);cv::RNG rng(19);rng.fill(texture,cv::RNG::UNIFORM,0,255);
    cv::GaussianBlur(texture,texture,cv::Size(3,3),.6);texture.copyTo(initial(original));
    for(int i=0;i<20;++i)cv::circle(initial,cv::Point(90+(i*29)%150,120+(i*43)%150),5,cv::Scalar(i%2?250:0),2);
    TargetReacquirer finder;finder.capture(initial,original);assert(finder.ready());
    cv::Mat blank(480,640,CV_8U,cv::Scalar(30));cv::Rect result;
    assert(!finder.search(blank,1000000000ULL,result));
    cv::Mat returned=blank.clone();cv::Rect destination(370,160,200,200);
    initial(original).copyTo(returned(destination));
    assert(!finder.search(returned,1300000000ULL,result)); // candidate, not lock
    assert(!finder.search(returned,1300000000ULL,result)); // duplicate exposure
    assert(!finder.search(returned,1400000000ULL,result)); // throttle
    assert(finder.search(returned,1550000000ULL,result));
    assert((result & destination).area()>.9*destination.area());
    // A long absence must not destroy the immutable anchor.
    for(uint64_t i=0;i<50;++i)assert(!finder.search(blank,2000000000ULL+i*250000000ULL,result));
    assert(!finder.search(returned,30000000000ULL,result));
    assert(finder.search(returned,30300000000ULL,result));
    // Similar statistics / unrelated texture cannot relock.
    cv::Mat distractor=blank.clone();rng.fill(distractor(destination),cv::RNG::UNIFORM,0,255);
    for(uint64_t i=0;i<8;++i)assert(!finder.search(distractor,31000000000ULL+i*250000000ULL,result));
    // Rotation and scale: geometric anchor does not depend on raw NCC orientation.
    cv::Mat rotated;cv::Mat a=cv::getRotationMatrix2D(cv::Point2f(170,200),18,1.15);
    a.at<double>(0,2)+=160;a.at<double>(1,2)+=20;
    cv::warpAffine(initial,rotated,a,initial.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(30));
    assert(!finder.search(rotated,40000000000ULL,result));
    assert(finder.search(rotated,40300000000ULL,result));
    finder.reset();assert(!finder.ready());assert(!finder.search(returned,50000000000ULL,result));
    finder.capture(blank,original);assert(!finder.ready());
    std::cout<<"PASS reacquire: translated/rotated/scaled target, >10s absence, two-exposure confirmation, duplicate/throttle, unrelated distractor, reset, low texture\n";
}
