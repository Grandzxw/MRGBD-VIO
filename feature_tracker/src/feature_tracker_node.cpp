#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

//并没有添加重启动？？
ros::Publisher pub_img,pub_match;

//每个相机都有一个FeatureTracker实例，即trackerData[i]    实例化
FeatureTracker trackerData[NUM_OF_CAM];

double first_image_time;
int pub_count = 1;
bool first_image_flag = true;

double frame_cnt = 0;
double sum_time = 0.0;
double mean_time = 0.0;
/**
 * @brief   ROS的回调函数，对新来的图像进行特征点追踪，发布
 * @Description readImage()函数对新来的图像使用光流法进行特征点跟踪
 *              同步之后定义成图像与深度图
 *              追踪的特征点封装成feature_points发布到pub_img的话题下，
 *              图像封装成ptr发布在pub_match下
 * @param[in]   color_msg 输入的RGB图像   depth_msg输入的深度图像
 * @return      void
*/
void img_callback(const sensor_msgs::ImageConstPtr &img_msg,const sensor_msgs::ImageConstPtr &depth_msg)
{
    ROS_INFO("start point callback");
    //判断是否是第一帧,主要是记录了一下时间戳
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
    }

    // frequency control, 如果图像频率低于一个值
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control   
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    //此出可更改为16位
    //将RGB图像编码8UCI转换为mono8
     cv_bridge::CvImageConstPtr ptr;
    //将RGB图像编码8UC1转换为mono8
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    //depth has encoding TYPE_16UC1  深度图像编码
    cv_bridge::CvImageConstPtr depth_ptr;
    // debug use     std::cout<<depth_msg->encoding<<std::endl;
    {
        sensor_msgs::Image img;
        img.header = depth_msg->header;
        img.height = depth_msg->height;
        img.width = depth_msg->width;
        img.is_bigendian = depth_msg->is_bigendian;
        img.step = depth_msg->step;
        img.data = depth_msg->data;
        img.encoding = sensor_msgs::image_encodings::MONO16;
        depth_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);
    }


    cv::Mat show_img = ptr->image;

    TicToc t_r;
    frame_cnt++;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)     // 单目 或者 非双目,   #### readImage： 这个函数里 做了很多很多事, 提取特征 ####
           //FeatureTracker::readImage()函数读取图像数据进行处理
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)),depth_ptr->image.rowRange(ROW * i, ROW * (i + 1)));   // rowRange(i,j) 取图像的i～j行
        else
        {
            if (EQUALIZE)      // 对图像进行直方图均衡化处理
            {
                std::cout <<" EQUALIZE " << std::endl;
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
                trackerData[i].cur_depth = depth_ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
//        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        trackerData[i].showUndistortion();
#endif
    }

    // 双目时
    if ( PUB_THIS_FRAME && STEREO_TRACK && trackerData[0].cur_pts.size() > 0)
    {
        pub_count++;
        r_status.clear();
        r_err.clear();
        TicToc t_o;
        cv::calcOpticalFlowPyrLK(trackerData[0].cur_img, trackerData[1].cur_img, trackerData[0].cur_pts, trackerData[1].cur_pts, r_status, r_err, cv::Size(21, 21), 3);
        ROS_DEBUG("spatial optical flow costs: %fms", t_o.toc());
        vector<cv::Point2f> ll, rr;
        vector<int> idx;
        for (unsigned int i = 0; i < r_status.size(); i++)
        {
            if (!inBorder(trackerData[1].cur_pts[i]))
                r_status[i] = 0;

            if (r_status[i])
            {
                idx.push_back(i);

                Eigen::Vector3d tmp_p;
                trackerData[0].m_camera->liftProjective(Eigen::Vector2d(trackerData[0].cur_pts[i].x, trackerData[0].cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                ll.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));

                trackerData[1].m_camera->liftProjective(Eigen::Vector2d(trackerData[1].cur_pts[i].x, trackerData[1].cur_pts[i].y), tmp_p);
                tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                rr.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
            }
        }
        if (ll.size() >= 8)
        {
            vector<uchar> status;
            TicToc t_f;
            cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 1.0, 0.5, status);
            ROS_DEBUG("find f cost: %f", t_f.toc());
            int r_cnt = 0;
            for (unsigned int i = 0; i < status.size(); i++)
            {
                if (status[i] == 0)
                    r_status[idx[i]] = 0;
                r_cnt += r_status[idx[i]];
            }
        }
    }

   //更新全局ID
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)                   // 单目 或者 非双目
                completed |= trackerData[j].updateID(i);   // 更新检测到的特征的id, 如果一个特征是新的，那就赋予新的id，如果一个特征已经有了id，那就不处理
        if (!completed)
            break;
    }

 //-------------------------------------上面处理完成，下面仅仅是进行发布操作----------------------------------------
   if (PUB_THIS_FRAME)
   {
        cv::Mat show_depth = depth_ptr->image;
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;   //  feature id
        sensor_msgs::ChannelFloat32 u_of_point;    //  u
        sensor_msgs::ChannelFloat32 v_of_point;    //  v
        sensor_msgs::ChannelFloat32 depth_of_point;  //depth

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            if (i != 1 || !STEREO_TRACK)  // 单目
            {
                auto un_pts = trackerData[i].undistortedPoints();
                auto &cur_pts = trackerData[i].cur_pts;
                auto &ids = trackerData[i].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    //存储的其实就是空间点的id
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    //nearest neighbor....fastest  may be changed
                    // show_depth: 480*640   y:[0,480]   x:[0,640]
                    depth_of_point.values.push_back((int)show_depth.at<unsigned short>(round(cur_pts[j].y), round(cur_pts[j].x)));
                    //debug use: print depth pixels
                    //test.push_back((int)show_depth.at<unsigned short>(round(cur_pts[j].y),round(cur_pts[j].x)));
                    ROS_ASSERT(inBorder(cur_pts[j]));
                }
            }
            else if (STEREO_TRACK)  //双目track 
            {
                auto r_un_pts = trackerData[1].undistortedPoints();
                auto &ids = trackerData[0].ids;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (r_status[j])
                    {
                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
                        geometry_msgs::Point32 p;
                        p.x = r_un_pts[j].x;
                        p.y = r_un_pts[j].y;
                        p.z = 1;

                        feature_points->points.push_back(p);
                        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    }
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(depth_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        pub_img.publish(feature_points);


        //显示图片
//        if (SHOW_TRACK)  // SHOW_TRACK
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);

            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
//                tmp_img = trackerData[0].cur_img;
                cv::cvtColor(trackerData[0].cur_img, tmp_img, CV_GRAY2RGB);
                if (i != 1 || !STEREO_TRACK)
                {
                    for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    {
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                        //char name[10];
                        //sprintf(name, "%d", trackerData[i].ids[j]);
                        //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    }
                }
                else
                {
                    for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    {
                        if (r_status[j])
                        {
                            cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 255, 0), 2);
                            cv::line(stereo_img, trackerData[i - 1].cur_pts[j], trackerData[i].cur_pts[j] + cv::Point2f(0, ROW), cv::Scalar(0, 255, 0));
                        }
                    }
                }
            }

            cv::imshow("vis", stereo_img);
            cv::waitKey(5);

            pub_match.publish(ptr->toImageMsg());
        }
    }
    sum_time += t_r.toc();
    mean_time = sum_time/frame_cnt;
//    ROS_INFO("whole point feature tracker processing costs: %f", t_r.toc());
    ROS_INFO("whole point feature tracker processing costs: %f", mean_time);
}

int main(int argc, char **argv)
{
    //ros初始化和设置句柄
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    //设置logger的级别。 只有级别大于或等于level的日志记录消息才会得到处理。
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    //读取每个相机实例读取对应的相机内参
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);
    //判断是否加入鱼眼mask来去除边缘噪声
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }


    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, IMAGE_TOPIC, 1);
    message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, DEPTH_TOPIC, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), sub_image, sub_depth);
    sync.registerCallback(boost::bind(&img_callback, _1, _2));
   
    //发布feature，实例feature_points，跟踪的特征点，给后端优化用
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    //发布feature_img，实例ptr，跟踪的特征点图，给RVIZ用和调试用
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    
    ROS_INFO("point feature detect");
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}
