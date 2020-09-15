#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//线的特征值
const int LINE_MIN_OBS = 5;
//回环信息值
const double LOOP_INFO_VALUE = 50.0;
//#define DEPTH_PRIOR
//#define GT
#define UNIT_SPHERE_ERROR

extern double BASE_LINE;

//初始深度值
extern double INIT_DEPTH;
//关键帧选择阈值（像素单位）
extern double MIN_PARALLAX;
//估计外参
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
//从相机到IMU的旋转矩阵
extern std::vector<Eigen::Matrix3d> RIC;
//从相机到IMU的平移向量
extern std::vector<Eigen::Vector3d> TIC;
//[0,0,G]
extern Eigen::Vector3d G;

//加速度的误差阈值
extern double BIAS_ACC_THRESHOLD;
//Bg的阈值
extern double BIAS_GYR_THRESHOLD;
//最大的解算时间  保证实时性
extern double SOLVER_TIME;
//最大解算器迭代次数（保证实时性）
extern int NUM_ITERATIONS;
//相机与IMU外参的输出路径OUTPUT_PATH + "/extrinsic_parameter.csv"
extern std::string EX_CALIB_RESULT_PATH;

extern std::string VINS_RESULT_PATH;
extern std::string VINS_FOLDER_PATH;

//是否回环
extern int LOOP_CLOSURE;
//最大的回环数目
extern int MIN_LOOP_NUM;
//最大的关键帧数目
extern int MAX_KEYFRAME_NUM;
extern std::string PATTERN_FILE;
extern std::string VOC_FILE;
extern std::string CAM_NAMES;
extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern double ROW, COL;


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    //线特征的优化
    SIZE_LINE = 4
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
