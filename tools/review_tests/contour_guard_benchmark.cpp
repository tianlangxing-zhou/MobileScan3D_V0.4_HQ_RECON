#include "color_contours.h"
#include "fusion_guard.h"
#include <chrono>
#include <iostream>
int main(){const int n=256;const float R[]={1,0,0,0,1,0,0,0,1},K[]={280,280,128,128},T[]={0,0,0};
std::vector<float>d(n*n,1),weight(n*n,1);std::vector<uint8_t>rgb(n*n*3);
for(int y=0;y<n;++y)for(int x=0;x<n;++x)for(int c=0;c<3;++c)rgb[(y*n+x)*3+c]=((x/32+y/32)%2)?230:20;
ColorContours contour;for(int i=0;i<5;++i)contour.build(rgb.data(),n,n,d.data(),n,n,weight.data());
auto start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)contour.build(rgb.data(),n,n,d.data(),n,n,weight.data());
std::cout<<"color_256_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100<<" priority="<<contour.priorityPixels<<'\n';
FusionGuard guard;for(int i=0;i<4;++i){float t[]={i*.12f,0,0};guard.commit(d.data(),n,n,K,R,t,100+i);}
start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)if(!guard.accept(d.data(),n,n,K,R,T,1000+i))return 1;
std::cout<<"guard_4refs_256_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100<<'\n';}
