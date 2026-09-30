#include "hard_surface.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>

namespace {
struct V {
    double x=0,y=0,z=0;
    V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
    V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};}
    V operator*(double s)const{return {x*s,y*s,z*s};}
};
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double length(V a){return std::sqrt(dot(a,a));}
V unit(V a){double n=length(a);return n>1e-15?a*(1/n):V{};}
V get(const std::vector<float>& v,size_t i){return {v[3*i],v[3*i+1],v[3*i+2]};}
void append(std::vector<float>& v,V p){v.insert(v.end(),{float(p.x),float(p.y),float(p.z)});}
struct Sample{V p,n,c;};
struct Plane{V n;double d;std::vector<int> ids;};
// Jacobi eigensolver for symmetric 3x3 covariance: smallest eigenvector is plane normal.
V smallest(double a[3][3]){
    double v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(int it=0;it<24;++it){
        int p=0,q=1;
        for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)if(std::abs(a[i][j])>std::abs(a[p][q])){p=i;q=j;}
        if(std::abs(a[p][q])<1e-14)break;
        double phi=.5*std::atan2(2*a[p][q],a[q][q]-a[p][p]),c=std::cos(phi),s=std::sin(phi);
        for(int k=0;k<3;++k){double x=a[k][p],y=a[k][q];a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
        for(int k=0;k<3;++k){double x=a[p][k],y=a[q][k];a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
        for(int k=0;k<3;++k){double x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;}
    }
    int m=0;for(int i=1;i<3;++i)if(a[i][i]<a[m][m])m=i;
    return unit({v[0][m],v[1][m],v[2][m]});
}
void refine(Plane& p,const std::vector<Sample>& s){
    V center;for(int i:p.ids)center=center+s[i].p;center=center*(1./p.ids.size());
    double a[3][3]{};
    for(int i:p.ids){V d=s[i].p-center;double v[3]={d.x,d.y,d.z};for(int j=0;j<3;++j)for(int k=0;k<3;++k)a[j][k]+=v[j]*v[k];}
    V n=smallest(a);if(dot(n,p.n)<0)n=n*(-1);p.n=n;p.d=dot(n,center);
}
double quantile(std::vector<double> a,double q){size_t k=size_t(q*(a.size()-1));std::nth_element(a.begin(),a.begin()+k,a.end());return a[k];}
std::vector<Plane> detect(const std::vector<Sample>& s,double tol,std::mt19937& rng){
    std::vector<int> rest(s.size());std::iota(rest.begin(),rest.end(),0);
    std::vector<Plane> planes;
    const size_t minimum=std::max(size_t(100),s.size()/20);
    while(planes.size()<12 && rest.size()>=minimum){
        Plane best;
        for(int it=0;it<180;++it){
            int i=rest[rng()%rest.size()];
            V n=s[i].n;
            if(it%2){V b=s[rest[rng()%rest.size()]].p,c=s[rest[rng()%rest.size()]].p;n=unit(cross(b-s[i].p,c-s[i].p));}
            if(length(n)<.9)continue;
            Plane p{n,dot(n,s[i].p),{}};
            for(int j:rest)if(std::abs(dot(n,s[j].p)-p.d)<=tol && std::abs(dot(n,s[j].n))>.94)p.ids.push_back(j);
            if(p.ids.size()>best.ids.size())best=std::move(p);
        }
        if(best.ids.size()<minimum)break;
        refine(best,s);
        best.ids.clear();std::vector<int> next;
        for(int j:rest){if(std::abs(dot(best.n,s[j].p)-best.d)<=tol && std::abs(dot(best.n,s[j].n))>.94)best.ids.push_back(j);else next.push_back(j);}
        if(best.ids.size()<minimum)break;
        rest.swap(next);planes.push_back(std::move(best));
    }
    return planes;
}
bool sampleMesh(const Mesh& m,std::vector<Sample>& out,double& diag,std::mt19937& rng){
    if(m.positions.size()%3 || m.indices.size()%3 || m.vertexCount()<4 || m.triangleCount()<2)return false;
    V lo{1e100,1e100,1e100},hi{-1e100,-1e100,-1e100};
    for(float x:m.positions)if(!std::isfinite(x))return false;
    for(size_t i=0;i<m.vertexCount();++i){V p=get(m.positions,i);lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}
    diag=length(hi-lo);if(diag<1e-6 || diag>1000)return false;
    std::vector<double> area;area.reserve(m.triangleCount());double total=0;
    for(size_t t=0;t<m.indices.size();t+=3){
        for(int k=0;k<3;++k)if(m.indices[t+k]>=m.vertexCount())return false;
        V a=get(m.positions,m.indices[t]),b=get(m.positions,m.indices[t+1]),c=get(m.positions,m.indices[t+2]);
        total+=length(cross(b-a,c-a));area.push_back(total);
    }
    if(total<1e-12)return false;
    std::uniform_real_distribution<double> uni(0,1);
    // Area-proportional triangle samples avoid bias from unequal mesh tessellation.
    for(int i=0;i<6000;++i){
        size_t t=std::upper_bound(area.begin(),area.end(),uni(rng)*total)-area.begin();t=std::min(t,area.size()-1)*3;
        auto ia=m.indices[t],ib=m.indices[t+1],ic=m.indices[t+2];V a=get(m.positions,ia),b=get(m.positions,ib),c=get(m.positions,ic);
        double u=std::sqrt(uni(rng)),v=uni(rng),w0=1-u,w1=u*(1-v),w2=u*v;
        V color{.7,.7,.7};if(m.colors.size()==m.positions.size())color=get(m.colors,ia)*w0+get(m.colors,ib)*w1+get(m.colors,ic)*w2;
        if(!std::isfinite(color.x)||!std::isfinite(color.y)||!std::isfinite(color.z))color={.7,.7,.7};
        out.push_back({a*w0+b*w1+c*w2,unit(cross(b-a,c-a)),color});
    }
    return true;
}
// Only nearly parallel/orthogonal detected planes are regularized. Oblique faces stay oblique.
void regularize(std::vector<Plane>& planes,const std::vector<Sample>& s){
    if(planes.empty())return;
    V a=planes[0].n,b;
    for(size_t i=1;i<planes.size();++i)if(std::abs(dot(a,planes[i].n))<.12){b=unit(planes[i].n-a*dot(a,planes[i].n));break;}
    V axes[3]={a,b,cross(a,b)};
    for(auto& p:planes){
        for(V axis:axes)if(length(axis)>.9 && std::abs(dot(p.n,axis))>.992){p.n=axis*(dot(p.n,axis)>0?1:-1);break;}
        p.d=0;for(int i:p.ids)p.d+=dot(p.n,s[i].p);p.d/=p.ids.size();
    }
}
bool planar(const Mesh& in,const std::vector<Sample>& samples,std::vector<Plane> planes,double tol,Mesh& out,HardSurfaceStats& st){
    if(planes.empty()){st.message="未检测到足够稳定的平面，保留扫描网格";return false;}
    regularize(planes,samples);
    std::vector<V> moved(in.vertexCount());
    for(size_t i=0;i<in.vertexCount();++i){
        V p=get(in.positions,i),q=p;
        // Alternating projections enforce shared-edge intersections without welding hard normals.
        std::array<const Plane*,12> near{};size_t count=0;
        for(auto& f:planes)if(std::abs(dot(f.n,p)-f.d)<tol*1.5)near[count++]=&f;
        for(int iter=0;iter<8;++iter)for(size_t k=0;k<count;++k){auto f=near[k];q=q-f->n*(dot(f->n,q)-f->d);}
        moved[i]=length(q-p)<=tol*3?q:p;
    }
    // Reject a global change if it would flip/collapse any nondegenerate input triangle.
    for(size_t t=0;t<in.indices.size();t+=3){auto a=in.indices[t],b=in.indices[t+1],c=in.indices[t+2];
        V old=cross(get(in.positions,b)-get(in.positions,a),get(in.positions,c)-get(in.positions,a));
        V now=cross(moved[b]-moved[a],moved[c]-moved[a]);
        if(length(old)>1e-12 && (length(now)<length(old)*.05 || dot(unit(old),unit(now))<.25)){
            st.message="规整会破坏局部网格，保留扫描网格";return false;
        }
    }
    Mesh result;std::map<std::pair<uint32_t,int>,uint32_t> mapping;size_t changed=0;
    for(size_t t=0;t<in.indices.size();t+=3){auto a=in.indices[t],b=in.indices[t+1],c=in.indices[t+2];
        V n=unit(cross(moved[b]-moved[a],moved[c]-moved[a]));int label=-1;
        for(size_t j=0;j<planes.size();++j){auto& f=planes[j];if(std::abs(dot(n,f.n))>.995 &&
            std::abs(dot(f.n,moved[a])-f.d)<tol*.2 && std::abs(dot(f.n,moved[b])-f.d)<tol*.2 && std::abs(dot(f.n,moved[c])-f.d)<tol*.2){label=int(j);break;}}
        if(label>=0)++changed;
        for(int k=0;k<3;++k){auto id=in.indices[t+k];auto key=std::make_pair(id,label);
            auto it=mapping.find(key);if(it==mapping.end()){
                uint32_t index=result.vertexCount();mapping[key]=index;
                append(result.positions,moved[id]);
                V normal=label>=0?planes[label].n*(dot(planes[label].n,n)>0?1:-1):
                    (in.normals.size()==in.positions.size()?get(in.normals,id):n);
                append(result.normals,normal);append(result.colors,in.colors.size()==in.positions.size()?get(in.colors,id):V{.7,.7,.7});
                result.indices.push_back(index);
            }else result.indices.push_back(it->second);
        }
    }
    if(!changed){st.message="平面约束不足，保留扫描网格";return false;}
    st.support=float(changed)/in.triangleCount();st.message="已规整稳定平面并保留硬边；未补造缺失表面";out=std::move(result);return true;
}
bool box(const std::vector<Sample>& s,const std::vector<Plane>& planes,bool cube,double tol,double diag,Mesh& out,HardSurfaceStats& st){
    if(planes.size()<3){st.message="至少需要三个方向的平面证据，保留扫描网格";return false;}
    std::array<V,3> ax{};double best=0;
    for(size_t i=0;i<planes.size();++i)for(size_t j=i+1;j<planes.size();++j){
        if(std::abs(dot(planes[i].n,planes[j].n))>.12)continue;
        V a=planes[i].n,b=unit(planes[j].n-a*dot(a,planes[j].n)),c=cross(a,b);
        double support[3]{};
        for(auto& p:planes){double ds[3]={std::abs(dot(a,p.n)),std::abs(dot(b,p.n)),std::abs(dot(c,p.n))};int k=std::max_element(ds,ds+3)-ds;if(ds[k]>.992)support[k]+=p.ids.size();}
        if(*std::min_element(support,support+3)<s.size()*.05)continue;
        double score=support[0]+support[1]+support[2];if(score>best){best=score;ax={a,b,c};}
    }
    if(best<s.size()*.80){st.message="平行／垂直平面支持不足，保留扫描网格";return false;}
    double lo[3],hi[3];
    for(int k=0;k<3;++k){std::vector<double> values;for(auto& p:s)values.push_back(dot(ax[k],p.p));lo[k]=quantile(values,.005);hi[k]=quantile(values,.995);}
    double sizes[3]={hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]};
    if(*std::min_element(sizes,sizes+3)<std::max(tol*8,diag*.025)){st.message="薄片或尺寸退化，保留扫描网格";return false;}
    if(cube){double avg=(sizes[0]+sizes[1]+sizes[2])/3;
        for(double z:sizes)if(std::abs(z-avg)>avg*.06){st.message="三轴尺寸不满足正方体约束，保留扫描网格";return false;}
        for(int k=0;k<3;++k){double mid=(lo[k]+hi[k])*.5;lo[k]=mid-avg*.5;hi[k]=mid+avg*.5;}
    }
    int count[6]{};bool bins[6][36]{};V color[6];double error=0;size_t supported=0;
    for(auto& p:s){double v[3]={dot(ax[0],p.p),dot(ax[1],p.p),dot(ax[2],p.p)},dist=1e100;int f=-1;
        for(int k=0;k<3;++k)for(int side=0;side<2;++side){double d=std::abs(v[k]-(side?hi[k]:lo[k]));if(d<dist){dist=d;f=k*2+side;}}
        int k=f/2;
        if(dist>tol*2 || std::abs(dot(p.n,ax[k]))<.94)continue;
        bool inside=true;for(int j=0;j<3;++j)if(v[j]<lo[j]-tol*2 || v[j]>hi[j]+tol*2)inside=false;if(!inside)continue;
        ++supported;error+=dist*dist;++count[f];color[f]=color[f]+p.c;
        int u=(k+1)%3,vv=(k+2)%3;
        int bx=std::clamp(int(6*(v[u]-lo[u])/(hi[u]-lo[u])),0,5),by=std::clamp(int(6*(v[vv]-lo[vv])/(hi[vv]-lo[vv])),0,5);bins[f][by*6+bx]=true;
    }
    st.support=float(supported)/s.size();st.rms=supported?std::sqrt(error/supported):0;
    if(st.support<.92 || st.rms>tol){st.message="箱体表面残差过大／含凹凸结构，保留扫描网格";return false;}
    int directions=0;
    for(int k=0;k<3;++k){bool direction=false;for(int side=0;side<2;++side){int f=2*k+side;
        int occupied=std::count(bins[f],bins[f]+36,true);
        if(count[f]>=int(s.size()*.035) && occupied>=20){++st.observedFaces;direction=true;}
        // A supported but fragmented face can indicate separated objects.
        else if(count[f]>=int(s.size()*.035)){st.message="面覆盖不足或物体分离，保留扫描网格";return false;}
    }if(direction)++directions;}
    if(directions<3){st.message="缺少三个方向的充分面覆盖，保留扫描网格";return false;}
    st.inferredFaces=6-st.observedFaces;
    Mesh result;
    for(int k=0;k<3;++k)for(int side=0;side<2;++side){int f=k*2+side,u=(k+1)%3,v=(k+2)%3;
        V n=ax[k]*(side?1:-1),col=count[f]?color[f]*(1./count[f]):V{.65,.65,.65};
        uint32_t base=result.vertexCount();
        const int corners[4][2]={{0,0},{1,0},{1,1},{0,1}};
        for(auto& corner:corners){V p=ax[k]*(side?hi[k]:lo[k])+ax[u]*(corner[0]?hi[u]:lo[u])+ax[v]*(corner[1]?hi[v]:lo[v]);append(result.positions,p);append(result.normals,n);append(result.colors,col);}
        if(side)result.indices.insert(result.indices.end(),{base,base+1,base+2,base,base+2,base+3});
        else result.indices.insert(result.indices.end(),{base,base+2,base+1,base,base+3,base+2});
    }
    st.message=std::string(cube?"正方体":"长方体")+"拟合完成；补出未充分观测面 "+std::to_string(st.inferredFaces)+" 个（尺寸／外观需核对）";
    out=std::move(result);return true;
}
} // namespace
bool fitHardSurface(const Mesh& input,HardSurfaceMode mode,float voxelSize,Mesh& output,HardSurfaceStats& stats){
    stats={};if(mode!=HardSurfaceMode::Planar && mode!=HardSurfaceMode::Cuboid && mode!=HardSurfaceMode::Cube){stats.message="无效规整模式";return false;}
    std::mt19937 rng(0x4d533344);std::vector<Sample> samples;double diag=0;
    if(!sampleMesh(input,samples,diag,rng)){stats.message="网格无效或数据不足，保留扫描网格";return false;}
    const double voxel=std::isfinite(voxelSize)&&voxelSize>0?voxelSize:diag*.002;
    const double tol=std::clamp(voxel*1.5,diag*.0015,diag*.012);
    auto planes=detect(samples,tol,rng);stats.planes=planes.size();
    bool ok=mode==HardSurfaceMode::Planar?planar(input,samples,planes,tol,output,stats):box(samples,planes,mode==HardSurfaceMode::Cube,tol,diag,output,stats);
    stats.applied=ok;return ok;
}
