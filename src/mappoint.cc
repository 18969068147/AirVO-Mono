#include "mappoint.h"
#include <iostream>
#include <opencv2/core/eigen.hpp>

Mappoint::Mappoint(): tracking_frame_id(-1), last_frame_seen(-1), local_map_optimization_frame_id(-1),
     _type(Type::UnTriangulated){
}

Mappoint::Mappoint(int& mappoint_id): tracking_frame_id(-1), last_frame_seen(-1),
    local_map_optimization_frame_id(-1), _id(mappoint_id), _type(Type::UnTriangulated){
  if(mappoint_id < 0) exit(0);
}

Mappoint::Mappoint(int& mappoint_id, Eigen::Vector3d& p): tracking_frame_id(-1), last_frame_seen(-1), 
    local_map_optimization_frame_id(-1), _id(mappoint_id), _type(Type::Good), _position(p){
}

Mappoint::Mappoint(int& mappoint_id, Eigen::Vector3d& p, Eigen::Matrix<double, 256, 1>& d):
    tracking_frame_id(-1), last_frame_seen(-1), local_map_optimization_frame_id(-1), 
    _id(mappoint_id), _type(Type::Good), _position(p), _descriptor(d){

}

void Mappoint::SetId(int id){
  //std::cout << "Mappoint::SetId" << id <<std::endl;
  _id = id;
}

int Mappoint::GetId(){
  return _id;
}

void Mappoint::SetType(Type& type){
  _type = type;
}

Mappoint::Type Mappoint::GetType(){
  return _type;
}

void Mappoint::SetBad(){
  _type = Type::Bad;
  _obversers.clear();
}

bool Mappoint::IsBad(){
  return (_type == Type::Bad);
}

void Mappoint::SetGood(){
  _type = Type::Good;
}

bool Mappoint::IsValid(){
  return (_type == Type::Good);
}

void Mappoint::AddObverser(const int& frame_id, const int& keypoint_index){
  _obversers[frame_id] = keypoint_index;
}

void Mappoint::RemoveObverser(const int& frame_id){
  std::map<int, int>::iterator it = _obversers.find(frame_id);
  if(it != _obversers.end()){
    _obversers.erase(it);
  }
}

int Mappoint::ObverserNum(){
  int obverser_num = 0;
  for(auto& kv : _obversers){
    if(kv.second >= 0){
      obverser_num++;
    }
  }
  return obverser_num;
}

void Mappoint::SetPosition(Eigen::Vector3d& p){
  _position = p;
  if(_type == Type::UnTriangulated){
    _type = Type::Good;
  }
}

Eigen::Vector3d& Mappoint::GetPosition(){
  return _position;
}

void Mappoint::SetDescriptor(const Eigen::Matrix<double, 256, 1>& descriptor){
  _descriptor = descriptor;
}

Eigen::Matrix<double, 256, 1>& Mappoint::GetDescriptor(){
  return _descriptor;
}

std::map<int, int>& Mappoint::GetAllObversers(){
  return _obversers;
}

int Mappoint::GetKeypointIdx(int frame_id){
  if(_obversers.count(frame_id) > 0) return _obversers[frame_id];
  return -1;
}

// void Mappoint::UpdateNormalAndDepth(){
   
//     cv::Mat Pos;
//     cv::eigen2cv(_position, Pos);

//     if(_obversers.empty())
//         return;

//     cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
//     int n=0;
//     for(size_t i=0; i<kfs.size(); i++){
//         cv::Mat Owi = kfs[i]->GetCameraCenter();  //camera center(cx,cy,0)
//         cv::Mat normali = mWorldPos - Owi;
//         normal = normal + normali/cv::norm(normali);
//         n++;
//     }
//     FramePtr pRefKF = kfs[0];

//     // cv::Mat PC = Pos - pRefKF->GetCameraCenter();
//     // const float dist = cv::norm(PC);
//     // const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
//     // const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
//     // const int nLevels = pRefKF->mnScaleLevels;
  
//     // mfMaxDistance = dist*levelScaleFactor;
//     // mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
//     mNormalVector = normal/n; // 获得地图点平均的观测方向
  
// }