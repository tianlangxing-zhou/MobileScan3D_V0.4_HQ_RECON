#include "keyframe_engine.h"
#include <algorithm>
#include <cmath>
void KeyframeEngine::reset(){k_.clear();lastNovelty_=1.f;guidance_="继续缓慢环绕目标物体";}
bool KeyframeEngine::consider(uint64_t ts,float sharp,float exposure,float novelty,uint32_t features,float tx,float ty,float tz){
 float quality=0.38f*sharp+0.22f*exposure+0.20f*std::min(1.f,features/800.f)+0.20f*novelty;
 bool accept=k_.empty() || (quality>0.55f && novelty>0.18f) || (ts-k_.back().ts>400000000ULL && quality>0.42f);
 if(accept){k_.push_back({ts,quality,sharp,exposure,novelty,tx,ty,tz,features}); if(k_.size()>240) k_.erase(k_.begin());}
 lastNovelty_=novelty;
 if(novelty<0.06f) guidance_="视角变化太小：请向左/右移动约 20–40 cm";
 else if(sharp<0.35f) guidance_="画面偏模糊：放慢移动速度并保持对焦";
 else if(exposure<0.35f) guidance_="曝光不足：转向光线更均匀的位置";
 else if(features<180) guidance_="纹理不足：靠近目标并绕到另一侧";
 else guidance_="覆盖良好：继续环绕，优先观察未扫描区域";
 return accept;
}
const Keyframe* KeyframeEngine::last() const { return k_.empty()?nullptr:&k_.back(); }
size_t KeyframeEngine::size() const{return k_.size();}
float KeyframeEngine::overlap() const{return std::clamp(lastNovelty_,0.f,1.f);}
std::string KeyframeEngine::guidance() const{return guidance_;}
