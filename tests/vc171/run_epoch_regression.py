"""Compile and execute the production epoch block with controlled observations.
This isolates state transitions; it is not an Android/JNI integration test.
Usage: python run_epoch_regression.py /path/to/app/src/main/cpp
"""
import pathlib, subprocess, sys, tempfile
cpp=pathlib.Path(sys.argv[1]).resolve()
s=(cpp/'native_engine.cpp').read_text()
def function(name):
 start=s.index(name); opening=s.index('{',start); level=1; end=opening+1
 while level:
  level += (s[end]=='{')-(s[end]=='}');end+=1
 return s[start:end]
constants=s[s.index('static constexpr int kEpochStableFrames'):s.index('// ============================================================================',s.index('static constexpr int kEpochStableFrames'))]
block=s[s.index('    // ---- V0.12 Fusion Calibration Epoch'):s.index('    gPerf[kPerfCalib].push',s.index('    // ---- V0.12 Fusion Calibration Epoch'))]
source='''#include "scan_policy.h"
#include "fusion_evidence.h"
#include "depth_refinement.h"
#include <cassert>
#include <iostream>
static scan_policy::FusionEvidence frozenEvidenceWindow;
static int frozenEvidenceStreak=0;
static uint64_t frozenEvidenceRecoveries=0;
struct CalibrationSource { DepthCalibration value; DepthCalibration calibration()const{return value;} } depthCalibrator;
'''+constants+'\n'+function('static void resetFusionEpoch()')+'\n'+function('static float rawDepthSampleMedian')+'''
void step(uint64_t t,const std::vector<float>& d, bool calibrationUpdatedThisFrame,
          bool frozenEvidenceGood, bool frozenEvidenceContradicted=false) {
    const int w=16,h=16,representation=1;
    const bool calibEnabledEff=true, calibratedNow=depthCalibrator.value.valid;
'''+block+'''
}
int main(){
    std::vector<float> negative(256,-5.f), large(256,100.f);
    assert(rawDepthSampleMedian(negative.data(),16,16,true)==-5.f);
    assert(rawDepthSampleMedian(large.data(),16,16,true)==100.f);
    assert(std::isnan(rawDepthSampleMedian(negative.data(),16,16,false)));
    std::vector<float> missing(256,std::numeric_limits<float>::quiet_NaN());
    resetFusionEpoch();
    depthCalibrator.value.valid=true; depthCalibrator.value.inverseDepthModel=true;
    depthCalibrator.value.scale=1.f;depthCalibrator.value.shift=6.f;
    for(int i=0;i<8;++i)step(1'000'000'000ULL+i*300'000'000ULL,negative,i==0,false);
    assert(epochActive && epochRefRaw==-5.f && epochCalib.toMetric(-5,0)==1.f);
    // Genuine disagreement cannot silently replace the map's calibration.
    depthCalibrator.value.shift=7.f;
    for(int i=0;i<80;++i)step(4'000'000'000ULL+i*300'000'000ULL,negative,true,false,true);
    assert(epochSuspended && epochCalib.shift==6.f && epochReanchors==0);
    // Cache replay must not keep updating the diagnostic EMA.
    const float frozenEma=epochLiveEmaShift;
    depthCalibrator.value.shift=9.f;
    step(29'000'000'000ULL,negative,false,false);
    assert(epochLiveEmaShift==frozenEma);
    // Independently supported old geometry wins over a noisy newer fit.
    for(int i=0;i<6;++i){
        const uint64_t t=30'000'000'000ULL+i*500'000'000ULL;
        step(t,negative,false,true);
        assert(epochSuspended==(i<5));
        step(t+100'000'000ULL,negative,false,false);
    }
    assert(!epochSuspended && epochCalib.shift==6.f && frozenEvidenceRecoveries==1);
    resetFusionEpoch();
    for(int i=0;i<10;++i)step(40'000'000'000ULL+i*300'000'000ULL,missing,false,false);
    assert(!epochActive); // missing depth cannot choose an invented reference q=0
    std::cout<<"PASS production epoch: signed/large q, cold start, no mixed-scale reanchor, no stale EMA, intermittent recovery, empty input\\n";
}
'''
# Keep temporary generated C++ in the current directory, including restricted hosts.
with tempfile.TemporaryDirectory(prefix='epoch_test_',dir='.') as temp:
 root=pathlib.Path(temp);f=root/'epoch.cpp';f.write_text(source);binary=root/'epoch'
 subprocess.run(['g++','-std=c++20','-O2','-I'+str(cpp),str(f),str(cpp/'depth_calib.cpp'),'-o',str(binary)],check=True)
 subprocess.run([str(binary.resolve())],check=True)
