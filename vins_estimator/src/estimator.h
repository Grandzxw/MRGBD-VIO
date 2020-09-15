#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/marginalization_factor.h"

#include "factor/line_parameterization.h"
#include "factor/line_projection_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SE3;
using Sophus::SO3;


//再深度解读
struct RetriveData
{
    /* data */
    int old_index;
    int cur_index;
    double header;
    Vector3d P_old;
    Matrix3d R_old;
    vector<cv::Point2f> measurements;
    vector<int> features_ids; 
    bool relocalized;
    bool relative_pose;
    Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    double loop_pose[7];
};


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> &image, const map<int, vector<pair<int, Vector4d>>> &lines, const std_msgs::Header &header);
  


    // internal
    void clearState();
    //初始化
    bool initialStructure();
    //初始对齐
    bool visualInitialAlign();
    //初始深度对齐
    bool visualInitialAlignWithDepth();
    //位姿恢复
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    //滑窗
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void optimizationwithLine();
    //只有线特征的优化
    void onlyLineOpt();
    //线特征的BA
    void LineBA();
    void LineBAincamera();

    void vector2double();
    void double2vector();
    void double2vector2();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,   //0
        NON_LINEAR  //1
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };


    double frame_cnt_ = 0;
    double sum_solver_time_ = 0.0;
    double mean_solver_time_ = 0.0;
    double sum_marg_time_ = 0.0;
    double mean_marg_time_=0.0;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    //加速度
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;
    //extrinsic
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    //VIO state vector  窗口中的[P,V,R,Ba,Bg]
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];


    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    //窗口中的dt,a,v
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    //初始的外参估计
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double baseline_;
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_LineFeature[NUM_OF_F][SIZE_LINE];      //四参数优化
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];


    RetriveData retrive_pose_data, front_pose;
    vector<RetriveData> retrive_data_vector;

    int loop_window_index;
    bool relocalize;
    
    Vector3d relocalize_t;
    Matrix3d relocalize_r;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;
    
    //是否为关键帧
    //kay是时间戳，val是图像帧
    //图像帧中保存了图像帧的特征点、时间戳、位姿Rt，预积分对象pre_integration，是否是关键帧。
    map<double, ImageFrame> all_image_frame;
    //临时预积分
    IntegrationBase *tmp_pre_integration;

};
