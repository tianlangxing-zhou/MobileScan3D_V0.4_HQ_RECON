#pragma once
#include <cstdint>
#include <vector>
#include <string>
struct Keyframe { uint64_t ts=0; float score=0, sharp=0, exposure=0, novelty=0; float tx=0,ty=0,tz=0; uint32_t features=0; };
class KeyframeEngine {
public:
 void reset();
 bool consider(uint64_t ts,float sharp,float exposure,float novelty,uint32_t features,float tx,float ty,float tz);
 const Keyframe* last() const; size_t size() const; float overlap() const; std::string guidance() const;
private:
 std::vector<Keyframe> k_; float lastNovelty_=1.f; std::string guidance_="继续缓慢环绕目标物体";
};
