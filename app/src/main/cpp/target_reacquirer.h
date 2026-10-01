#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Immutable user-selected identity. Bounded full-frame ORB search is independent
// of Nano's local search window; two separate exposures must agree before relock.
class TargetReacquirer {
    cv::Ptr<cv::ORB> orb_ = cv::ORB::create(1000, 1.2f, 6, 15, 0, 2,
                                         cv::ORB::HARRIS_SCORE, 31, 12);
    std::vector<cv::KeyPoint> anchorPoints_;
    cv::Mat anchorDescriptors_;
    cv::Rect anchorBox_, pendingBox_;
    cv::Size anchorSize_;
    uint64_t lastSearchTs_=0, pendingTs_=0;
    int confirmations_=0;
public:
    uint64_t searches=0, recoveries=0;
    int lastInliers=0;
    void reset() {
        anchorPoints_.clear();anchorDescriptors_.release();anchorBox_={};
        anchorSize_={};lastSearchTs_=pendingTs_=0;confirmations_=0;pendingBox_={};
        searches=recoveries=0;lastInliers=0;
    }
    void capture(const cv::Mat& gray,const cv::Rect& box) {
        reset();if(gray.empty())return;
        anchorBox_=box & cv::Rect(0,0,gray.cols,gray.rows);anchorSize_=gray.size();
        if(anchorBox_.width<24 || anchorBox_.height<24)return;
        cv::Mat mask=cv::Mat::zeros(gray.size(),CV_8U);
        // Leave the selection border out: it is often background.
        const int mx=std::max(2,anchorBox_.width/10),my=std::max(2,anchorBox_.height/10);
        cv::Rect core(anchorBox_.x+mx,anchorBox_.y+my,anchorBox_.width-2*mx,anchorBox_.height-2*my);
        mask(core).setTo(255);
        orb_->detectAndCompute(gray,mask,anchorPoints_,anchorDescriptors_);
    }
    bool ready()const{return anchorDescriptors_.rows>=12;}
    bool search(const cv::Mat& gray,uint64_t ts,cv::Rect& result) {
        if(!ready() || gray.empty() || gray.size()!=anchorSize_ || !ts)return false;
        // Once a candidate exists, verify on a new exposure promptly. Keep the
        // expensive full-frame search at 5 Hz when there is no candidate.
        const bool pendingFresh = pendingTs_ && ts > pendingTs_ && ts-pendingTs_ <= 750000000ULL;
        const uint64_t interval = pendingFresh ? 80000000ULL : 200000000ULL;
        if(lastSearchTs_ && (ts<=lastSearchTs_ || ts-lastSearchTs_<interval))return false;
        lastSearchTs_=ts;++searches;lastInliers=0;
        cv::Rect candidate;
        if(!locate(gray,candidate)) { confirmations_=0;pendingTs_=0;return false; }
        const float intersection=float((candidate & pendingBox_).area());
        const float unionArea=float(candidate.area()+pendingBox_.area())-intersection;
        const bool consistent=pendingTs_ && ts>pendingTs_ && ts-pendingTs_<=750000000ULL &&
                              unionArea>0 && intersection/unionArea>=.45f;
        confirmations_=consistent?confirmations_+1:1;pendingBox_=candidate;pendingTs_=ts;
        if(confirmations_<2)return false;
        result=candidate;confirmations_=0;pendingTs_=0;++recoveries;return true;
    }
private:
    bool locate(const cv::Mat& gray,cv::Rect& box) {
        std::vector<cv::KeyPoint> points;cv::Mat descriptors;
        orb_->detectAndCompute(gray,cv::noArray(),points,descriptors);
        if(descriptors.rows<12)return false;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<cv::DMatch> reverse;
        matcher.knnMatch(anchorDescriptors_,descriptors,matches,2);
        matcher.match(descriptors,anchorDescriptors_,reverse);
        std::vector<cv::Point2f> from,to;
        for(const auto& pair:matches) {
            if(pair.size()!=2 || pair[0].distance>=.72f*pair[1].distance || pair[0].distance>64)continue;
            const auto& m=pair[0];
            if(m.trainIdx<0 || m.trainIdx>=int(reverse.size()) || reverse[m.trainIdx].trainIdx!=m.queryIdx)continue;
            from.push_back(anchorPoints_[m.queryIdx].pt);to.push_back(points[m.trainIdx].pt);
        }
        if(from.size()<12)return false;
        cv::Mat inliers;
        cv::Mat a=cv::estimateAffinePartial2D(from,to,inliers,cv::RANSAC,2.5,1200,.995,10);
        if(a.empty() || !cv::checkRange(a))return false;
        lastInliers=cv::countNonZero(inliers);
        if(lastInliers<10 || lastInliers<float(from.size())*.65f)return false;
        // Evidence must span the object, not a small repeated logo or corner.
        std::vector<cv::Point2f> supported;
        for(size_t i=0;i<from.size();++i)if(inliers.at<uchar>(int(i)))supported.push_back(from[i]);
        std::vector<cv::Point2f> hull;cv::convexHull(supported,hull);
        if(cv::contourArea(hull)<anchorBox_.area()*.08)return false;
        const double scale=std::hypot(a.at<double>(0,0),a.at<double>(1,0));
        if(scale<.4 || scale>2.)return false;
        std::vector<cv::Point2f> corners={
            {float(anchorBox_.x),float(anchorBox_.y)},
            {float(anchorBox_.x+anchorBox_.width),float(anchorBox_.y)},
            {float(anchorBox_.x+anchorBox_.width),float(anchorBox_.y+anchorBox_.height)},
            {float(anchorBox_.x),float(anchorBox_.y+anchorBox_.height)}};
        cv::transform(corners,corners,a);
        for(const auto& p:corners)if(!std::isfinite(p.x)||!std::isfinite(p.y)||
            std::fabs(p.x)>gray.cols*4 || std::fabs(p.y)>gray.rows*4)return false;
        const cv::Rect full=cv::boundingRect(corners);
        const cv::Rect clipped=full & cv::Rect(0,0,gray.cols,gray.rows);
        if(full.area()<=0 || clipped.area()<.85*full.area() || clipped.width<24 || clipped.height<24)return false;
        box=clipped;return true;
    }
};
