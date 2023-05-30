#include "map_builder.h"

#include <assert.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"
#include "debug.h"
#include "initializer.h"

MapBuilder::MapBuilder(Configs &configs) : _shutdown(false), _init(false), _track_id(0), _line_track_id(0),
                                           _to_update_local_map(false), _configs(configs),
                                            mpInitializer(static_cast<Initializer *>(NULL)) {
    
    std::cout << "configs.camera_config_path=="<<configs.camera_config_path << std::endl;
    _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
   
    _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
    if (!_superpoint->build()) {
        std::cout << "Error in SuperPoint building" << std::endl;
        exit(0);
    }
    _point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
    _line_detector = std::shared_ptr<LineDetector>(new LineDetector(configs.line_detector_config));

    /*ROS
    _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
    _map = std::shared_ptr<Map>(new Map(_configs.backend_optimization_config, _camera, _ros_publisher));
    */
    // MAP WITHOUT ROS
    _map = std::shared_ptr<Map>(new Map(_configs.backend_optimization_config, _camera));

    _feature_thread = std::thread(boost::bind(&MapBuilder::ExtractFeatureThread, this));
    _tracking_thread = std::thread(boost::bind(&MapBuilder::TrackingThread, this));
}

void MapBuilder::AddInput(InputDataPtr data) {
    // cv::Mat image_left_rect, image_right_rect;
    // _camera->UndistortImage(data->image_left, data->image_right, image_left_rect, image_right_rect);
    
    // data->image_left = image_left_rect;
    // data->image_right = image_right_rect;

    
    while (_data_buffer.size() >= 3 && !_shutdown) {
        usleep(2000);
    }

    _buffer_mutex.lock();
    _data_buffer.push(data);
    _buffer_mutex.unlock();
}

void MapBuilder::ExtractFeatureThread() {
    while (!_shutdown) {
        if (_data_buffer.empty()) {
            usleep(2000);
            continue;
        }
        InputDataPtr input_data;
        _buffer_mutex.lock();
        input_data = _data_buffer.front();
        _data_buffer.pop();
        _buffer_mutex.unlock();

        int frame_id = input_data->index;
        double timestamp = input_data->time;
        cv::Mat image_left_rect = input_data->image_left.clone();
        // cv::Mat image_right_rect = input_data->image_right.clone();

        // construct frame
        FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera, timestamp));
        int currentID = frame->GetFrameId();

        // init
         if (!_init) {
            // _init = InitStereo(frame, image_left_rect, image_right_rect); //双目
            _init = InitMonocular(frame, image_left_rect);
            _last_frame_track_well = _init;

            if (_init) {
                _last_frame = frame;
                _last_image = image_left_rect;
                // _last_right_image = image_right_rect;
                _last_keyimage = image_left_rect;
            }
            /*ROS
            PublishFrame(frame, image_left_rect);
            */
            continue;
        }
        
        // extract features and track last keyframe
        FramePtr last_keyframe = _last_keyframe;
        const Eigen::Matrix<double, 259, Eigen::Dynamic> features_last_keyframe = last_keyframe->GetAllFeatures();

        std::vector<cv::DMatch> matches;
        Eigen::Matrix<double, 259, Eigen::Dynamic> features_left;
        std::vector<Eigen::Vector4d> lines_left;
        ExtractFeatureAndMatch(image_left_rect, features_last_keyframe, features_left, lines_left, matches);
        frame->AddLeftFeatures(features_left, lines_left);

        //undistortion  tum-vi
        frame->UndistortKeyPoints();
        

        TrackingDataPtr tracking_data = std::shared_ptr<TrackingData>(new TrackingData());
        tracking_data->frame = frame;
        tracking_data->ref_keyframe = last_keyframe;
        tracking_data->matches = matches;
        tracking_data->input_data = input_data;
        
        while (_tracking_data_buffer.size() >= 2) {
            usleep(2000);
        }

        _tracking_mutex.lock();
        _tracking_data_buffer.push(tracking_data);
        _tracking_mutex.unlock();
    }
}

void MapBuilder::TrackingThread() {
    while (!_shutdown) {
        if (_tracking_data_buffer.empty()) {
            usleep(2000);
            continue;
        }

        TrackingDataPtr tracking_data;
        _tracking_mutex.lock();
        tracking_data = _tracking_data_buffer.front();
        _tracking_data_buffer.pop();
        _tracking_mutex.unlock();

        FramePtr frame = tracking_data->frame;
        FramePtr ref_keyframe = tracking_data->ref_keyframe;
        InputDataPtr input_data = tracking_data->input_data;
        std::vector<cv::DMatch> matches = tracking_data->matches;

        double timestamp = input_data->time;
        cv::Mat image_left_rect = input_data->image_left.clone();
        // cv::Mat image_right_rect = input_data->image_right.clone();

        // track
        frame->SetPose(_last_frame->GetPose());
        std::function<int()> track_last_frame = [&]() {
            if (_num_since_last_keyframe < 1 || !_last_frame_track_well) return -1;
            // InsertKeyframe(_last_frame, _last_right_image);
            InsertKeyframe(_last_frame, _last_image); 
            _last_keyimage = _last_image;
            matches.clear();
            ref_keyframe = _last_frame;
            return TrackFrame(_last_frame, frame, matches);
        };

        std::cout<< "MapBuilder::TrackingThread()............................................................ " << ref_keyframe->GetFrameId() <<", "<< frame->GetFrameId()<<", "<< matches.size()  << std::endl;
        int num_match = matches.size();
        if (num_match < _configs.keyframe_config.min_num_match) {//如果匹配大到的特征太少，则基于上一帧进行追踪
            num_match = track_last_frame();
        } else {
            num_match = TrackFrame(ref_keyframe, frame, matches); //如果匹配大到的特征太少，则基于上一关键帧进行追踪
            std::cout<< "num_match = TrackFrame " << num_match << std::endl;
            if (num_match < _configs.keyframe_config.min_num_match) {
                num_match = track_last_frame();
            }
        }
        /*ROS
        PublishFrame(frame, image_left_rect);
        */
        
        _last_frame_track_well = (num_match >= _configs.keyframe_config.min_num_match);
        std::cout<< "num_match=track_last_frame===== " << num_match << ", _last_frame_track_well:" << _last_frame_track_well<< std::endl;
        if (!_last_frame_track_well) continue;

        frame->SetPreviousFrame(ref_keyframe);
        _last_frame_track_well = true;

        // for debug
        // SaveTrackingResult(_last_keyimage, image_left_rect, _last_keyframe, frame, matches, _configs.saving_dir);
        std::cout<< "AddKeyframe(ref_keyframe, frame, num_match) " <<AddKeyframe(ref_keyframe, frame, num_match) << std::endl;
        std::cout<< "AddKeyframe(ref_keyframe->GetFrameId() " << ref_keyframe->GetFrameId()  << ", _last_keyframe->GetFrameId="<< _last_keyframe->GetFrameId()  << std::endl;
        if (AddKeyframe(ref_keyframe, frame, num_match) && ref_keyframe->GetFrameId() == _last_keyframe->GetFrameId()) {
            InsertKeyframe(frame, image_left_rect);
            _last_keyimage = image_left_rect;
        }

        _last_frame = frame;
        _last_image = image_left_rect;
        // _last_right_image = image_right_rect;
    }
}

void MapBuilder::ExtractFeatrue(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &points,
                                std::vector<Eigen::Vector4d> &lines) {
    std::function<void()> extract_point = [&]() {
        _gpu_mutex.lock();
        bool good_infer = _superpoint->infer(image, points);
        _gpu_mutex.unlock();
        if (good_infer) {
            std::cout << "Failed when extracting point features !" << std::endl;
            return;
        }
    };

    std::function<void()> extract_line = [&]() {
        _line_detector->LineExtractor(image, lines);
    };

    std::thread point_ectraction_thread(extract_point);
    std::thread line_ectraction_thread(extract_line);

    point_ectraction_thread.join();
    line_ectraction_thread.join();
}


void MapBuilder::ExtractFeatureAndMatch(const cv::Mat &image, const Eigen::Matrix<double, 259, Eigen::Dynamic> &points0,
                                        Eigen::Matrix<double, 259, Eigen::Dynamic> &points1,
                                        std::vector<Eigen::Vector4d> &lines, std::vector<cv::DMatch> &matches) {
    std::function<void()> extract_point_and_match = [&]() {
        auto point0 = std::chrono::steady_clock::now();
        _gpu_mutex.lock();
        if (!_superpoint->infer(image, points1)) {
            _gpu_mutex.unlock();
            std::cout << "Failed when extracting point features !" << std::endl;
            return;
        }
        auto point1 = std::chrono::steady_clock::now();

        matches.clear();
        _point_matching->MatchingPoints(points0, points1, matches);
        _gpu_mutex.unlock();
        auto point2 = std::chrono::steady_clock::now();
        auto point_time = std::chrono::duration_cast<std::chrono::milliseconds>(point1 - point0).count();
        auto point_match_time = std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point1).count();
        // std::cout << "One Frame point Time: " << point_time << " ms." << std::endl;
        // std::cout << "One Frame point match Time: " << point_match_time << " ms." << std::endl;
    };

    std::function<void()> extract_line = [&]() {
        auto line1 = std::chrono::steady_clock::now();
        _line_detector->LineExtractor(image, lines);
        auto line2 = std::chrono::steady_clock::now();
        auto line_time = std::chrono::duration_cast<std::chrono::milliseconds>(line2 - line1).count();
        // std::cout << "One Frame line Time: " << line_time << " ms." << std::endl;
    };

    auto feature1 = std::chrono::steady_clock::now();
    std::thread point_ectraction_thread(extract_point_and_match);
    std::thread line_ectraction_thread(extract_line);

    point_ectraction_thread.join();
    line_ectraction_thread.join();

    auto feature2 = std::chrono::steady_clock::now();
    auto feature_time = std::chrono::duration_cast<std::chrono::milliseconds>(feature2 - feature1).count();
    // std::cout << "One Frame featrue Time: " << feature_time << " ms." << std::endl;
}

void MapBuilder::FeatureMatch(const Eigen::Matrix<double, 259, Eigen::Dynamic> &points0,Eigen::Matrix<double, 259, Eigen::Dynamic> &points1,
                                        std::vector<cv::DMatch> &matches) {
    std::function<void()> point_match = [&]() {
        _gpu_mutex.lock();
        auto point1 = std::chrono::steady_clock::now();
        matches.clear();
        _point_matching->MatchingPoints(points0, points1, matches);
        _gpu_mutex.unlock();

        auto point2 = std::chrono::steady_clock::now();
 
        auto point_match_time = std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point1).count();
        std::cout << "One Frame point match Time: " << point_match_time << " ms." << std::endl;
    };

    auto feature1 = std::chrono::steady_clock::now();
    std::thread point_matching_thread(point_match);

    point_matching_thread.join();

    auto feature2 = std::chrono::steady_clock::now();
    auto feature_time = std::chrono::duration_cast<std::chrono::milliseconds>(feature2 - feature1).count();
    std::cout << "One Frame featrue matching Time: " << feature_time << " ms." << std::endl;
}


cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void MapBuilder::Pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                                      std::vector<cv::KeyPoint> keypoints_2,
                                      std::vector<cv::DMatch> matches,
                                      cv::Mat &R, cv::Mat &t) {
    // 相机内参,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    cv::Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    std::cout << "fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    //-- 计算本质矩阵
    cv::Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    cv::Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    //-- 计算单应矩阵
    //-- 但是本例中场景不是平面，单应矩阵意义不大
    cv::Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    // 此函数仅在Opencv3中提供
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t << std::endl;

}

void triangulation(
  const std::vector<cv::KeyPoint> &keypoint_1,
  const std::vector<cv::KeyPoint> &keypoint_2,
  const std::vector<cv::DMatch> &matches,
  const cv::Mat &R, const cv::Mat &t,
  std::vector<cv::Point3d> &points
);

//三角化,根据匹配点和求解到的三维点。存储在points中
void triangulation(
  const std::vector<cv::KeyPoint> &keypoint_1,
  const std::vector<cv::KeyPoint> &keypoint_2,
  const std::vector<cv::DMatch> &matches,
  const cv::Mat &R, const cv::Mat &t,
  std::vector<cv::Point3d> &points) {
  cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    //根据求解到的RT构造T2矩阵
  cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );
  //相机内参[190.978477, 0, 254.931706, 0, 190.973307, 256.897442, 0, 0, 1]
  cv::Mat K = (cv::Mat_<double>(3, 3) << 190.978477, 0, 254.931706, 0, 190.973307, 256.897442, 0, 0, 1);
  std::vector<cv::Point2f> pts_1, pts_2;
  for (cv::DMatch m:matches) {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }
 
  cv::Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
 
  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    cv::Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}


bool MapBuilder::InitMonocular(FramePtr curframe, cv::Mat &image_left) {
    //Feature extraction
    int currentID = curframe->GetFrameId();
    Eigen::Matrix<double, 259, Eigen::Dynamic> features_left;
    std::vector<Eigen::Vector4d> lines_left;
    ExtractFeatrue(image_left, features_left, lines_left);   // Extract 
    // the number of features under 100; return
    if (features_left.cols() <= 100) {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
        fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        return false;
    }
    curframe->AddLeftFeatures(features_left, lines_left); //add keypoint feature to frame

    //keypoints undistortion fisheye model: tum-vi  
    curframe->UndistortKeyPoints(); 

    if (currentID==0) {
        // Set Reference Frame
        std::cout << "InitMonocular Set Reference Frame ...." << std::endl<< std::endl;
        int feature_num = features_left.cols();
        
        std::vector<cv::KeyPoint> keypoints = curframe->GetAllKeypoints();
        std::cout << "keypoints.size()=="<<keypoints.size()<< std::endl;
        mvbPrevMatched.resize(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); i++){
            mvbPrevMatched[i] = keypoints[i].pt;  //reference frame keypoints
        }
        std::cout << "mvbPrevMatched=="<< mvbPrevMatched.size() << std::endl;
        mpInitializer = new Initializer(curframe, 1.0, 200);
        std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        
        if (feature_num > 100) {
            mInitialFrame = curframe; //初始化参考帧（第一帧）
            _last_frame = curframe;
            _last_keyframe = curframe;
            _last_image = image_left;
            return false;
        }
    } else {
        // Try to initialize
        std::cout << "Try to initialize..... ....." << std::endl;
        //计算两帧的特征点匹配个数
        std::vector<cv::DMatch> matches;
        Eigen::Matrix<double, 259, Eigen::Dynamic> initFeatures = mInitialFrame->GetAllFeatures();
        //Matching
        FeatureMatch(initFeatures, features_left, matches);
        std::cout << "matches........."<< matches.size() << std::endl;


        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        keypoints_1 = mInitialFrame->GetAllKeypoints();
        keypoints_2 = curframe->GetAllKeypoints();
      
        //-- 估计两张图像间运动
        cv::Mat Rcw, tcw;  // Current Camera Rotation,Translation
        Pose_estimation_2d2d(keypoints_1, keypoints_2, matches, Rcw, tcw);
        //-- 三角化
        std::vector<cv::Point3d> mvIniP3D;  
        triangulation(keypoints_1, keypoints_2, matches, Rcw, tcw, mvIniP3D); 
    

        // //mvIniMatches :vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为 -1
        // mvIniMatches.resize(mvbPrevMatched.size());
        // for (size_t i=0;i<mvbPrevMatched.size();i++){
        //     for (size_t j=0; j<matches.size(); j++){
        //         if (i== matches[j].queryIdx){
        //             mvIniMatches[i] = matches[j].trainIdx;
        //         }     
        //     }
        // }
        // std::cout << "mvIniMatches........."<< mvIniMatches.size() << std::endl;

        
        // std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        // std::vector<cv::Point3f> mvIniP3D; // 3d点
        // if (mpInitializer->Initialize(curframe, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
        //     for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        //         if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
        //             mvIniMatches[i] = -1;
        //         }
        //     }

        // Set Frame Poses
        //将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame->SetPose(Eigen::Matrix4d::Identity(4,4));
        mInitialFrame->SetPoseFixed(true);
        
        // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0, 3).col(3));
        Eigen::Matrix4d Tcw_eigen;
        cv::cv2eigen(Tcw, Tcw_eigen);
        curframe->SetPose(Tcw_eigen);
        curframe->SetPoseFixed(true);
        std::cout << "第二关键帧相机位姿curframe->GetPose()：" << curframe->GetPose()<< std::endl;

        //创建初始化地图点MapPoints
        // triangulation函数会得到mvIniP3D
        // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
        CreateInitialMapMonocular(mInitialFrame, curframe, mvIniP3D, matches);
        // }
    }
    return true;
}

void MapBuilder::CreateInitialMapMonocular(FramePtr mInitialFrame, FramePtr curFrame, std::vector<cv::Point3d> mvIniP3D, std::vector<cv::DMatch> matches) {
    std::cout << "CreateInitialMapMonocular======== " << std::endl;
    Eigen::Matrix4d init_pose = curFrame->GetPose();
    Eigen::Matrix3d Rwc = init_pose.block<3, 3>(0, 0);
    Eigen::Vector3d twc = init_pose.block<3, 1>(0, 3);
    // construct mappoints
    Eigen::Matrix<double, 259, -1> features= curFrame->GetAllFeatures();
    size_t feature_num = features.size();
    std::cout << "mvIniP3D.size() ======== "<<mvIniP3D.size() << std::endl;

    //keyframes vector
    std::vector<FramePtr> kfs;
    kfs.push_back(mInitialFrame);
    kfs.push_back(curFrame);

    // reference frame mappoint construct
    std::vector<int> track_ids(feature_num, -1);
    int frame_id = curFrame->GetFrameId();
    std::vector<MappointPtr> new_mappoints;
    Eigen::Vector3d tmp_position; //地图点的世界坐标系
    for (size_t i = 0; i < matches.size(); i++) {
        cv::Mat worldPos(mvIniP3D[i]);  //3d点转换为世界坐标系
        cv::cv2eigen(worldPos, tmp_position);

        track_ids[i] = _track_id++;
        Eigen::Matrix<double, 256, 1> descriptor;
        if (!mInitialFrame->GetDescriptor(matches[i].queryIdx, descriptor)) continue;
        if (!curFrame->GetDescriptor(matches[i].trainIdx, descriptor)) continue;
        MappointPtr mappoint = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i], tmp_position, descriptor));  //构建地图点

        mappoint->AddObverser(0, i); //reference frame (第一帧)
        mappoint->AddObverser(frame_id, i); //std::map<int, int> _obversers;  // frame_id - keypoint_index 

        // mappoint->UpdateNormalAndDepth();

        //将地图点加入frame
        mInitialFrame->InsertMappoint(matches[i].queryIdx, mappoint);//queryIdx代表的特征点序列是keypoints1中的，trainIdx代表的特征点序列是keypoints2中的
        curFrame->InsertMappoint(matches[i].trainIdx, mappoint);

        //将地图点添加之地图
        _map->InsertMappoint(mappoint);
        new_mappoints.push_back(mappoint);
    }
    //SetTrackIds
    mInitialFrame->SetTrackIds(track_ids);
    curFrame->SetTrackIds(track_ids);
    
    //更新updateconeections
    _map->UpdateFrameConnection(mInitialFrame);
    _map->UpdateFrameConnection(curFrame);

    // Insert KFs in the map
    _map->InsertInitKeyframe(mInitialFrame, curFrame);
    _last_keyframe = curFrame;

    //BA POSE Optimization
    CameraPtr camera = curFrame->GetCamera();
    std::vector<CameraPtr> camera_list;
    camera_list.emplace_back(camera);
    BundleAdjustment(kfs, new_mappoints, 20, camera_list);
   
    // Set median depth to 1
    float medianDepth = ComputeSceneMedianDepth(mInitialFrame,2);
    float invMedianDepth = 1.0f / medianDepth;
    std::cout << "medianDepth, invMedianDepth =" << medianDepth<<", " << invMedianDepth << std::endl;
    std::vector<MappointPtr> curAllMapPoints = curFrame->GetAllMappoints();
    if (medianDepth < 0 ||  curAllMapPoints.size() < 100) {
        std::cout << "Wrong initialization, reseting..." << std::endl;
        return;
    }

    // Scale initial baseline
    Eigen::Matrix4d curPose = curFrame->GetPose();
    
    cv::Mat Tc2w;
    cv::eigen2cv(curPose, Tc2w);
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    cv::cv2eigen(Tc2w, curPose);
    std::cout<< "Scale initial baseline:curFrame->GetPose() "<< curFrame->GetPose() <<std::endl;
    curFrame->SetPose(curPose);

    // Scale points
    std::cout << "vpAllMapPoints ======== " << new_mappoints.size()<< std::endl;
    for (size_t iMP = 0; iMP < new_mappoints.size(); iMP++) {
        if (new_mappoints[iMP]) {
            MappointPtr pMP = new_mappoints[iMP];
            Eigen::Vector3d pos = pMP->GetPosition();
            Eigen::Vector3d position = invMedianDepth*pos;
            pMP->SetPosition(position);
        }
    }

    for (MappointPtr mappoint: new_mappoints) {
        _map->InsertMappoint(mappoint);
    }
   
    _ref_keyframe = curFrame;
    _last_frame = curFrame;
    _num_since_last_keyframe = 1;
    _to_update_local_map = true;
    return;
}


float MapBuilder::ComputeSceneMedianDepth(FramePtr initialFrame, const int q){
    Eigen::Matrix4d initPose = initialFrame->GetPose();
    cv::Mat Tcw_;
    cv::eigen2cv(initPose, Tcw_);
    std::cout<<"cv::eigen2cv(initPose, Tcw_);" <<Tcw_<<std::endl; 

    std::vector<MappointPtr> mappoints = initialFrame->GetAllMappoints();
    std::cout<< "ComputeSceneMedianDepth mappoints.size(): "<< mappoints.size() << std::endl;
    std::vector<float> depths;
    depths.reserve(mappoints.size());
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<mappoints.size(); i++){
        if(mappoints[i]){
            MappointPtr pMP = mappoints[i];
            cv::Mat x3Dw;
            cv::eigen2cv(pMP->GetPosition(), x3Dw);
            float z = Rcw2.dot(x3Dw)+zcw;
            depths.push_back(z);
        }
    }

    std::sort(depths.begin(),depths.end());
    return depths[(depths.size()-1)/q];
}

bool MapBuilder::InitStereo(FramePtr frame, cv::Mat &image_left, cv::Mat &image_right) {
    // extract features
    Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;
    std::vector<Eigen::Vector4d> lines_left, lines_right;
    std::vector<cv::DMatch> stereo_matches;
    ExtractFeatrue(image_left, features_left, lines_left);
    int feature_num = features_left.cols();
    if (feature_num < 150) return false;
    ExtractFeatureAndMatch(image_right, features_left, features_right, lines_right, stereo_matches);
    frame->AddLeftFeatures(features_left, lines_left);
    int stereo_point_num = frame->AddRightFeatures(features_right, lines_right, stereo_matches);
    if (stereo_point_num < 100) return false;

    // Eigen::Matrix4d init_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d init_pose;
    init_pose << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 1;
    frame->SetPose(init_pose);
    frame->SetPoseFixed(true);

    Eigen::Matrix3d Rwc = init_pose.block<3, 3>(0, 0);
    Eigen::Vector3d twc = init_pose.block<3, 1>(0, 3);
    // construct mappoints
    std::vector<int> track_ids(feature_num, -1);
    int frame_id = frame->GetFrameId();
    Eigen::Vector3d tmp_position;
    
    std::vector<MappointPtr> new_mappoints;
    for (size_t i = 0; i < feature_num; i++) {
        if (frame->BackProjectPoint(i, tmp_position)) {
            tmp_position = Rwc * tmp_position + twc;
            stereo_point_num++;
            track_ids[i] = _track_id++;
            Eigen::Matrix<double, 256, 1> descriptor;
            if (!frame->GetDescriptor(i, descriptor)) continue;
            MappointPtr mappoint = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i], tmp_position, descriptor));
            mappoint->AddObverser(frame_id, i);
            frame->InsertMappoint(i, mappoint);
            new_mappoints.push_back(mappoint);
        }
    }
    frame->SetTrackIds(track_ids);
    if (stereo_point_num < 100) return false;

    // construct maplines
    size_t line_num = frame->LineNum();
    std::vector<MaplinePtr> new_maplines;
    for (size_t i = 0; i < line_num; i++) {
        frame->SetLineTrackId(i, _line_track_id);
        MaplinePtr mapline = std::shared_ptr<Mapline>(new Mapline(_line_track_id));
        Vector6d endpoints;
        if (frame->TrianguateStereoLine(i, endpoints)) {
            mapline->SetEndpoints(endpoints);
            mapline->SetObverserEndpointStatus(frame_id, 1);
        } else {
            mapline->SetObverserEndpointStatus(frame_id, 0);
        }
        mapline->AddObverser(frame_id, i);
        frame->InsertMapline(i, mapline);
        new_maplines.push_back(mapline);
        _line_track_id++;
    }

    // add frame and mappoints to map
    InsertKeyframe(frame);
    for (MappointPtr mappoint: new_mappoints) {
        _map->InsertMappoint(mappoint);
    }
    for (MaplinePtr mapline: new_maplines) {
        _map->InsertMapline(mapline);
    }
    _ref_keyframe = frame;
    _last_frame = frame;
    return true;
}

int MapBuilder::TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch> &matches) {
    // line tracking
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features0 = frame0->GetAllFeatures();
    Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 = frame1->GetAllFeatures();
    // std::vector<std::map<int, double>> points_on_lines0 = frame0->GetPointsOnLines();
    // std::vector<std::map<int, double>> points_on_lines1 = frame1->GetPointsOnLines();
    // std::vector<int> line_matches;
    // MatchLines(points_on_lines0, points_on_lines1, matches, features0.cols(), features1.cols(), line_matches);
    
    std::vector<int> inliers(frame1->FeatureNum(), -1);
    std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);
    std::vector<MappointPtr>& frame0_mappoints = frame0->GetAllMappoints();
    // std::cout << "FramePoseOptimization frame0_mappoints======== " << frame0_mappoints.size() << std::endl;
    for (auto &match: matches) {
        int idx0 = match.queryIdx;
        int idx1 = match.trainIdx;
        matched_mappoints[idx1] = frame0_mappoints[idx0];
        inliers[idx1] = frame0->GetTrackId(idx0);
    }
    std::cout << "FramePoseOptimization frame1,matched_mappoints,inliers======== " << frame1->GetFrameId() << ", "<< matched_mappoints.size()<< ", "<< inliers.size() << std::endl;

    int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);
    std::cout << "FramePoseOptimization num_inliers======== " << num_inliers << std::endl;
    // update track id` 共有的特征点
    int RM = 0;
    if (num_inliers > _configs.keyframe_config.min_num_match) {
        for (std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();) {
            int idx0 = (*it).queryIdx;
            int idx1 = (*it).trainIdx;
            if (inliers[idx1] > 0) {
                frame1->SetTrackId(idx1, frame0->GetTrackId(idx0));
                frame1->InsertMappoint(idx1, frame0_mappoints[idx0]);
            }

            if (inliers[idx1] > 0) {
                it++;
            } else {
                it = matches.erase(it);
                RM++;
            }
        }
    }

    // //update line track id
    // const std::vector<MaplinePtr> &frame0_maplines = frame0->GetConstAllMaplines();
    // for (size_t i = 0; i < frame0_maplines.size(); i++) {
    //     int j = line_matches[i];
    //     if (j < 0) continue;
    //     int line_track_id = frame0->GetLineTrackId(i);

    //     if (line_track_id >= 0) {
    //         frame1->SetLineTrackId(j, line_track_id);
    //         frame1->InsertMapline(j, frame0_maplines[i]);
    //     }
    // }

    return num_inliers;
}

int MapBuilder::FramePoseOptimization(
        FramePtr frame, std::vector<MappointPtr> &mappoints, std::vector<int> &inliers, int pose_init) {
    // solve PnP using opencv to get initial pose
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    std::vector<int> cv_inliers;
    int num_cv_inliers = SolvePnPWithCV(frame, mappoints, Twc, cv_inliers);
    // std::cout << "FramePoseOptimization->num_cv_inliers " << num_cv_inliers << std::endl;

    Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_frame->GetPose().block<3, 1>(0, 3);
    
    // std::cout << "SolvePnPWithCV->check_dp.norm() " << check_dp.norm() << std::endl;
    if (check_dp.norm() > 0.5 || num_cv_inliers < _configs.keyframe_config.min_num_match) {
        Twc = _last_frame->GetPose();
    }

    // Second, optimization
    MapOfPoses poses;
    MapOfPoints3d points;
    std::vector<CameraPtr> camera_list;
    VectorOfMonoPointConstraints mono_point_constraints;
    VectorOfStereoPointConstraints stereo_point_constraints;

    camera_list.emplace_back(_camera);

    // map of poses
    Pose3d pose;
    pose.p = Twc.block<3, 1>(0, 3);
    pose.q = Twc.block<3, 3>(0, 0);
    int frame_id = frame->GetFrameId();
    poses.insert(std::pair<int, Pose3d>(frame_id, pose));

    // visual constraint construction
    std::vector<size_t> mono_indexes;
    std::vector<size_t> stereo_indexes;
    for (size_t i = 0; i < mappoints.size(); i++) {
        // points
        MappointPtr mpt = mappoints[i];
        if (mpt == nullptr || !mpt->IsValid()) continue;
        Eigen::Vector3d keypoint;
        if (!frame->GetKeypointPosition(i, keypoint)) continue;

        int mpt_id = mpt->GetId();
        Position3d point;
        point.p = mpt->GetPosition();
        point.fixed = true;
        points.insert(std::pair<int, Position3d>(mpt_id, point));

        // visual constraint
        if (keypoint(2) > 0) {
            StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(
                    new StereoPointConstraint());
            stereo_constraint->id_pose = frame_id;
            stereo_constraint->id_point = mpt_id;
            stereo_constraint->id_camera = 0;
            stereo_constraint->inlier = true;
            stereo_constraint->keypoint = keypoint;
            stereo_constraint->pixel_sigma = 0.8;
            stereo_point_constraints.push_back(stereo_constraint);
            stereo_indexes.push_back(i);
        } else {
            MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint());
            mono_constraint->id_pose = frame_id;
            mono_constraint->id_point = mpt_id;
            mono_constraint->id_camera = 0;
            mono_constraint->inlier = true;
            mono_constraint->keypoint = keypoint.head(2);
            mono_constraint->pixel_sigma = 0.8;
            mono_point_constraints.push_back(mono_constraint);
            mono_indexes.push_back(i);
        }

    }
    int num_inliers = FrameOptimization(poses, points, camera_list, mono_point_constraints,
                                        stereo_point_constraints, _configs.tracking_optimization_config);

    if (num_inliers > _configs.keyframe_config.min_num_match) {
        // set frame pose
        Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
        frame_pose.block<3, 3>(0, 0) = poses.begin()->second.q.matrix();
        frame_pose.block<3, 1>(0, 3) = poses.begin()->second.p;
        frame->SetPose(frame_pose);

        // update tracked mappoints
        for (size_t i = 0; i < mono_point_constraints.size(); i++) {
            size_t idx = mono_indexes[i];
            if (!mono_point_constraints[i]->inlier) {
                inliers[idx] = -1;
            }
        }

        for (size_t i = 0; i < stereo_point_constraints.size(); i++) {
            size_t idx = stereo_indexes[i];
            if (!stereo_point_constraints[i]->inlier) {
                inliers[idx] = -1;
            }
        }

    }

    return num_inliers;
}

bool MapBuilder::AddKeyframe(FramePtr last_keyframe, FramePtr current_frame, int num_match) {
    Eigen::Matrix4d frame_pose = current_frame->GetPose();

    Eigen::Matrix4d &last_keyframe_pose = _last_keyframe->GetPose();
    Eigen::Matrix3d last_R = last_keyframe_pose.block<3, 3>(0, 0);
    Eigen::Vector3d last_t = last_keyframe_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d current_R = frame_pose.block<3, 3>(0, 0);
    Eigen::Vector3d current_t = frame_pose.block<3, 1>(0, 3);

    Eigen::Matrix3d delta_R = last_R.transpose() * current_R;
    Eigen::AngleAxisd angle_axis(delta_R);
    double delta_angle = angle_axis.angle();
    double delta_distance = (current_t - last_t).norm();
    int passed_frame_num = current_frame->GetFrameId() - _last_keyframe->GetFrameId();

    bool not_enough_match = (num_match < _configs.keyframe_config.max_num_match);
    bool large_delta_angle = (delta_angle > _configs.keyframe_config.max_angle);
    bool large_distance = (delta_distance > _configs.keyframe_config.max_distance);
    bool enough_passed_frame = (passed_frame_num > _configs.keyframe_config.max_num_passed_frame);
    return (not_enough_match || large_delta_angle || large_distance || enough_passed_frame);
}

void MapBuilder::InsertKeyframe(FramePtr frame, const cv::Mat &image_right) {
    _last_keyframe = frame;

    // Eigen::Matrix<double, 259, Eigen::Dynamic> features_right;
    // std::vector<Eigen::Vector4d> lines_right;
    // std::vector<cv::DMatch> stereo_matches;

    // ExtractFeatureAndMatch(image_right, frame->GetAllFeatures(), features_right, lines_right, stereo_matches);
    // frame->AddRightFeatures(features_right, lines_right, stereo_matches);
    InsertKeyframe(frame);
}

void MapBuilder::InsertKeyframe(FramePtr frame) {
    _last_keyframe = frame;

    // create new track id
    std::vector<int> &track_ids = frame->GetAllTrackIds();
    for (size_t i = 0; i < track_ids.size(); i++) {
        if (track_ids[i] < 0) {
            frame->SetTrackId(i, _track_id++);
        }
    }

    // // create new line track id
    // const std::vector<int> &line_track_ids = frame->GetAllLineTrackId();
    // for (size_t i = 0; i < line_track_ids.size(); i++) {
    //     if (line_track_ids[i] < 0) {
    //         frame->SetLineTrackId(i, _line_track_id++);
    //     }
    // }

    // insert keyframe to map
    _map->InsertKeyframe(frame);

    // update last keyframe
    _num_since_last_keyframe = 1;
    _ref_keyframe = frame;
    _to_update_local_map = true;
}

void MapBuilder::UpdateReferenceFrame(FramePtr frame) {
    int current_frame_id = frame->GetFrameId();
    std::vector<MappointPtr> &mappoints = frame->GetAllMappoints();
    std::map<FramePtr, int> keyframes;
    for (MappointPtr mpt: mappoints) {
        if (!mpt || mpt->IsBad()) continue;
        const std::map<int, int> obversers = mpt->GetAllObversers();
        for (auto &kv: obversers) {
            int observer_id = kv.first;
            if (observer_id == current_frame_id) continue;
            FramePtr keyframe = _map->GetFramePtr(observer_id);
            if (!keyframe) continue;
            keyframes[keyframe]++;
        }
    }
    if (keyframes.empty()) return;

    std::pair<FramePtr, int> max_covi = std::pair<FramePtr, int>(nullptr, -1);
    for (auto &kv: keyframes) {
        if (kv.second > max_covi.second) {
            max_covi = kv;
        }
    }

    if (max_covi.first->GetFrameId() != _ref_keyframe->GetFrameId()) {
        _ref_keyframe = max_covi.first;
        _to_update_local_map = true;
    }
}

void MapBuilder::UpdateLocalKeyframes(FramePtr frame) {
    _local_keyframes.clear();
    std::vector<std::pair<int, FramePtr>> neighbor_frames = _ref_keyframe->GetOrderedConnections(-1);
    for (auto &kv: neighbor_frames) {
        _local_keyframes.push_back(kv.second);
    }
}

void MapBuilder::UpdateLocalMappoints(FramePtr frame) {
    _local_mappoints.clear();
    int current_frame_id = frame->GetFrameId();
    for (auto &kf: _local_keyframes) {
        const std::vector<MappointPtr> &mpts = kf->GetAllMappoints();
        for (auto &mpt: mpts) {
            if (mpt && mpt->IsValid() && mpt->tracking_frame_id != current_frame_id) {
                mpt->tracking_frame_id = current_frame_id;
                _local_mappoints.push_back(mpt);
            }
        }
    }
}

void MapBuilder::SearchLocalPoints(FramePtr frame, std::vector<std::pair<int, MappointPtr>> &good_projections) {
    int current_frame_id = frame->GetFrameId();
    std::vector<MappointPtr> &mpts = frame->GetAllMappoints();
    for (auto &mpt: mpts) {
        if (mpt && !mpt->IsBad()) mpt->last_frame_seen = current_frame_id;
    }

    std::vector<MappointPtr> selected_mappoints;
    for (auto &mpt: _local_mappoints) {
        if (mpt && mpt->IsValid() && mpt->last_frame_seen != current_frame_id) {
            selected_mappoints.push_back(mpt);
        }
    }

    _map->SearchByProjection(frame, selected_mappoints, 1, good_projections);
}

int MapBuilder::TrackLocalMap(FramePtr frame, int num_inlier_thr) {
    if (_to_update_local_map) {
        UpdateLocalKeyframes(frame);
        UpdateLocalMappoints(frame);
    }

    std::vector<std::pair<int, MappointPtr>> good_projections;
    SearchLocalPoints(frame, good_projections);
    if (good_projections.size() < 3) return -1;

    std::vector<MappointPtr> mappoints = frame->GetAllMappoints();
    for (auto &good_projection: good_projections) {
        int idx = good_projection.first;
        if (mappoints[idx] && !mappoints[idx]->IsBad()) continue;
        mappoints[idx] = good_projection.second;
    }

    std::vector<int> inliers(mappoints.size(), -1);
    int num_inliers = FramePoseOptimization(frame, mappoints, inliers, 2);

    // update track id
    if (num_inliers > _configs.keyframe_config.min_num_match && num_inliers > num_inlier_thr) {
        for (size_t i = 0; i < mappoints.size(); i++) {
            if (inliers[i] > 0) {
                frame->SetTrackId(i, mappoints[i]->GetId());
                frame->InsertMappoint(i, mappoints[i]);
            }
        }
    } else {
        num_inliers = -1;
    }
    return num_inliers;
}
/*ROS
void MapBuilder::PublishFrame(FramePtr frame, cv::Mat &image) {
    FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
    FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);

    feature_message->image = image;
    feature_message->keypoints = frame->GetAllKeypoints();;
    feature_message->lines = frame->GatAllLines();
    feature_message->points_on_lines = frame->GetPointsOnLines();
    std::vector<bool> inliers_feature_message;
    frame->GetInlierFlag(inliers_feature_message);
    feature_message->inliers = inliers_feature_message;
    frame_pose_message->pose = frame->GetPose();
    feature_message->line_track_ids = frame->GetAllLineTrackId();

    _ros_publisher->PublishFeature(feature_message);
    _ros_publisher->PublishFramePose(frame_pose_message);
}
*/
void MapBuilder::SaveTrajectory() {
    std::string file_path = ConcatenateFolderAndFileName(_configs.saving_dir, "keyframe_trajectory.txt");
    _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveTrajectory(std::string file_path) {
    _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveMap(const std::string &map_root) {
    _map->SaveMap(map_root);
}

void MapBuilder::ShutDown() {
    _shutdown = true;
    _feature_thread.join();
    _tracking_thread.join();
}
