#pragma once
#include <cstdint>
struct AiDepthResult { const float* depthMeters=nullptr; const float* confidence=nullptr; int width=0; int height=0; };
class AiBackend {
public:
 virtual ~AiBackend() = default;
 virtual bool initialize(int width,int height)=0;
 virtual AiDepthResult refineDepth(const uint8_t* y,int width,int height)=0;
 virtual bool completeGeometry()=0;
};
class NullAiBackend final : public AiBackend {
public:
 bool initialize(int,int) override { return true; }
 AiDepthResult refineDepth(const uint8_t*,int,int) override { return {}; }
 bool completeGeometry() override { return false; }
};
