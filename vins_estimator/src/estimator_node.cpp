#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

// #include "loop-closure/loop_closure.h"
// #include "loop-closure/keyframe.h"
// #include "loop-closure/keyframe_database.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

Estimator estimator;


std::condition_variable con;  //条件变量

double current_time = -1;

//无重定位帧
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> linefeature_buf;    //线特征的buf

//图优化的函数     ？
std::mutex m_posegraph_buf;
queue<int> optimize_posegraph_buf;
// queue<KeyFrame*> keyframe_buf;
queue<RetriveData> retrive_data_buf;

int sum_of_wait = 0;

//互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;

//？
std::mutex m_loop_drift;
std::mutex m_keyframedatabase_resample;
std::mutex m_update_visualization;
std::mutex m_keyframe_buf;
std::mutex m_retrive_data_buf;

//IMU项[P,Q,B,Ba,Bg,a,g]
//这里是当前帧融合完之后的结果，也是发布到rviz里面显示的结果
double latest_time;   //上一次的时间
Eigen::Vector3d tmp_P;  //位置
Eigen::Quaterniond tmp_Q;  //旋转
Eigen::Vector3d tmp_V;  //速度
Eigen::Vector3d tmp_Ba;  //加速度偏差
Eigen::Vector3d tmp_Bg;  //陀螺以偏差
Eigen::Vector3d acc_0;  //加速度
Eigen::Vector3d gyr_0;  //角速度


queue<pair<cv::Mat, double>> image_buf;

// //回环检测
// LoopClosure *loop_closure;
// KeyFrameDatabase keyframe_database;

//全局帧的数目
int global_frame_cnt = 0;
//camera param
camodocal::CameraPtr m_camera;
vector<int> erase_index;
std_msgs::Header cur_header;
Eigen::Vector3d relocalize_t{Eigen::Vector3d(0, 0, 0)};
Eigen::Matrix3d relocalize_r{Eigen::Matrix3d::Identity()};

//中值法预测
/*
  使用mid-point方法对imu状态量进行预测  从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
*/
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dt = t - latest_time;
    latest_time = t;
    //线加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};
    //角加速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba - tmp_Q.inverse() * estimator.g);  // Qwi * (ai - ba - Qiw * g)

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;                   // (gyro0 + gyro1)/2 - bg
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);                                         // Qwi * [1, 1/2 w dt]

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba - tmp_Q.inverse() * estimator.g);

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    //？
    tmp_P = relocalize_r * estimator.Ps[WINDOW_SIZE] + relocalize_t;
    tmp_Q = relocalize_r * estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}
/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据           
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg, line_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
        std::pair<sensor_msgs::PointCloudConstPtr,sensor_msgs::PointCloudConstPtr> >>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
            std::pair<sensor_msgs::PointCloudConstPtr,sensor_msgs::PointCloudConstPtr> >> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty() || linefeature_buf.empty())
            return measurements;

        std::cout<<"-------------------------------------\n";
//        std::cout << imu_buf.front()->header.stamp.toSec() << " " << imu_buf.back()->header.stamp.toSec()<<" "<<imu_buf.size() << "\n";
//        std::cout << feature_buf.front()->header.stamp.toSec() << " " << feature_buf.back()->header.stamp.toSec() << "\n";
        if (!(imu_buf.back()->header.stamp > feature_buf.front()->header.stamp)) //如果imu最新数据的时间戳不大于最旧图像的时间戳，那得等imu数据
        {
            ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp < feature_buf.front()->header.stamp)) // 如果imu最老的数据时间戳不小于最旧图像的时间，那得把最老的图像丢弃
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            linefeature_buf.pop();
            continue;
        }
        //得到点与线的特征
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        sensor_msgs::PointCloudConstPtr linefeature_msg = linefeature_buf.front();
        linefeature_buf.pop();

        // 遍历两个图像之间所有的imu数据
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp <= img_msg->header.stamp)
        {
            //emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        //这样就把图像和IMU对应起来了  图像的时间戳处于imu buf时间戳之间
//        std::cout << "measurements size: "<<measurements.size() <<"\n";
        measurements.emplace_back(IMUs, std::make_pair(img_msg,linefeature_msg) );
    }
    return measurements;
}

//imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //可以添加两帧相同的情况

    m_buf.lock();
    //将新帧放入队列中
    imu_buf.push(imu_msg);
    m_buf.unlock();
    //唤醒作用于process线程中的获取观测值数据的函数
    con.notify_one();

    {
        std::lock_guard<std::mutex> lg(m_state);
        //predict imu (no residual error)   //递推得到IMU的PQV--------------------直接进行中值积分哦！积分后的值存在上面的全局变量处
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV  都是在世界坐标系下
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}
//原图的图像回调函数  
void raw_image_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // //将转化为MONO8
    // cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    // //image_pool[img_msg->header.stamp.toNSec()] = img_ptr->image;
    // if(LOOP_CLOSURE)
    // {
    //     i_buf.lock();
    //     image_buf.push(make_pair(img_ptr->image, img_msg->header.stamp.toSec()));
    //     i_buf.unlock();
    // }
}

//点特征的回调函数  图像的回调函数  将feature_msg放入feature_buf当中
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

//线特征的回调函数
void linefeature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    m_buf.lock();
    linefeature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

//IMU的send     processIMU过程
void send_imu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (current_time < 0)
        current_time = t;
    double dt = t - current_time;
    current_time = t;

    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};

    double dx = imu_msg->linear_acceleration.x - ba[0];
    double dy = imu_msg->linear_acceleration.y - ba[1];
    double dz = imu_msg->linear_acceleration.z - ba[2];

    double rx = imu_msg->angular_velocity.x - bg[0];
    double ry = imu_msg->angular_velocity.y - bg[1];
    double rz = imu_msg->angular_velocity.z - bg[2];
    //ROS_DEBUG("IMU %f, dt: %f, acc: %f %f %f, gyr: %f %f %f", t, dt, dx, dy, dz, rx, ry, rz);

    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
}


// thread: visual-inertial odometry  视觉惯性历程计历程
// thread: visual-inertial odometry
void process()
{
    while (true)
    {
          //等待上面两个接收数据完成就会被唤醒，一个是图像特征，一个是IMU数据  加入图像的时间戳
        //在提取measurements时互斥锁m_buf会锁住，此时无法接收数据
        //std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr，sensor_msgs::PointCloudConstPtr>> measurements;
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
        std::pair<sensor_msgs::PointCloudConstPtr,sensor_msgs::PointCloudConstPtr> >> measurements;
        
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        for (auto &measurement : measurements)
        {
            for (auto &imu_msg : measurement.first)
                send_imu(imu_msg);                     // 处理imu数据, 预测 pose

            auto point_and_line_msg = measurement.second;
            auto img_msg = point_and_line_msg.first;
            auto line_msg = point_and_line_msg.second;
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            
            //点特征添加
            TicToc t_s;
            //如何加深度？     加深度信息
            map<int, vector<pair<int, Eigen::Matrix<double, 4, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;

                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;        // 被几号相机观测到的，如果是单目，camera_id = 0
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                
                //添加深度信息
                double depth = img_msg->channels[3].values[i]/1000.0;   //深度值

                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 4, 1> xyz_depth;
                xyz_depth << x,y,z,depth;
                image[feature_id].emplace_back(camera_id, xyz_depth);
            }

            //线特征添加
            map<int, vector<pair<int, Vector4d>>> lines;
            for (unsigned int i = 0; i < line_msg->points.size(); i++)
            {
                int v = line_msg->channels[0].values[i] + 0.5;
                //std::cout<< "receive id: " << v / NUM_OF_CAM << "\n";
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;        // 被几号相机观测到的，如果是单目，camera_id = 0
                //线的两个端点位置
                double x_startpoint = line_msg->points[i].x;
                double y_startpoint = line_msg->points[i].y;
                double x_endpoint = line_msg->channels[1].values[i];
                double y_endpoint = line_msg->channels[2].values[i];
//                ROS_ASSERT(z == 1);
                lines[feature_id].emplace_back(camera_id, Vector4d(x_startpoint, y_startpoint, x_endpoint, y_endpoint));
            }
            //处理image数据  
            estimator.processImage(image,lines, img_msg->header);   // 处理image数据，这时候的image已经是特征点数据，不是原始图像了。
  


            //关键点信息
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            cur_header = header;

            m_loop_drift.lock();

            pubOdometry(estimator, header, relocalize_t, relocalize_r);     //里程计 PQV
            pubKeyPoses(estimator, header, relocalize_t, relocalize_r);     //关键点信息三维坐标
            pubCameraPose(estimator, header, relocalize_t, relocalize_r);   //相机位姿
            pubLinesCloud(estimator, header, relocalize_t, relocalize_r);   //线特征的点云信息
            pubPointCloud(estimator, header, relocalize_t, relocalize_r);   //点特征的点云信息
            pubTF(estimator, header, relocalize_t, relocalize_r);           //相机到IMU外参    
            
            
            m_loop_drift.unlock();
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}



//入口主程序
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_linefeature = n.subscribe("/linefeature_tracker/linefeature", 2000, linefeature_callback);
   // ros::Subscriber sub_raw_image = n.subscribe(IMAGE_TOPIC, 2000, raw_image_callback);

    // thread: visual-inertial odometry
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
