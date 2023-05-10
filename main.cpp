#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
/*ROS   #include <ros/ros.h>           */

#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"

int main(int argc, char **argv) {
    /*ROS
    ros::init(argc, argv, "air_vo");

    std::string config_path, model_dir;
    ros::param::get("~config_path", config_path);
    ros::param::get("~model_dir", model_dir);
    Configs configs(config_path, model_dir);
    ros::param::get("~dataroot", configs.dataroot);
    ros::param::get("~camera_config_path", configs.camera_config_path);
    ros::param::get("~saving_dir", configs.saving_dir);
    std::string traj_path;
    ros::param::get("~traj_path", traj_path);
    */

    std::string config_path, model_dir;
    config_path = "/home/ru/catkin_ws/src/AirVO/configs/configs_oivio.yaml";
    model_dir = "/home/ru/catkin_ws/src/AirVO/output";
    Configs configs(config_path, model_dir);
    configs.dataroot = "/home/ru/oivio/MN_100_GV_01/husky0";
    configs.camera_config_path = "/home/ru/catkin_ws/src/AirVO/configs/oivio.yaml";
    configs.saving_dir = "/home/ru/catkin_ws/src/AirVO/debug";
    std::string traj_path;
    traj_path = "/home/ru/catkin_ws/src/AirVO/debug/traj.txt";


    Dataset dataset(configs.dataroot);
    MapBuilder map_builder(configs);
    size_t dataset_length = dataset.GetDatasetLength();
    /*ROS for (size_t i = 0; i < dataset_length && ros::ok(); ++i) {  */
    for (size_t i = 0; i < dataset_length; ++i) {
        std::cout << "i ===== " << i << std::endl;
        auto before_infer = std::chrono::steady_clock::now();

        InputDataPtr input_data = dataset.GetData(i);
        if (input_data == nullptr) continue;
        map_builder.AddInput(input_data);

        auto after_infer = std::chrono::steady_clock::now();
        auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
        std::cout << "One Frame Processinh Time: " << cost_time << " ms." << std::endl;
    }
    map_builder.ShutDown();
    map_builder.SaveTrajectory(traj_path);
    /*ROS  ros::shutdown(); */

    return 0;
}
