#pragma once

#include <vector>
using namespace std;
#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Geometry>
using namespace Eigen;
#include <ros/console.h>

class MotionEstimator
{
  public:
    //2d到2d
    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
    //3d到2d
    bool solveRelativeRT_ICP (vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
    //3d到3d
   	bool solveRelativeRT_PNP(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation);
  private:
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};


