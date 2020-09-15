#include "initial_sfm.h"


GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

//关键帧的三角化
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	//定义坐标
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		//已经被三角化  特征点坐标已知
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			//遍历能被该关键帧观测到的特征点
			if (sfm_f[j].observation[k].first == i)
			{
				//图像坐标
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	//特征点个数不够  
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}

	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	//旋转向量与旋转矩阵的互换 
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	//SFM
	//pts_3_vector-世界坐标系下的控制点的坐标  pts_2_vector- 图像坐标点对应控制点的坐标  K-相机的内参矩阵   D-相机的畸变系数  
	//rvec-输出的旋转向量，坐标点从世界坐标系旋转到相机坐标系    t-输出的平移向量，坐标点从世界坐标系平移到相机坐标系 
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;
}


void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

//尺度已知的情况下不用恢复深度值    使用深度值进行判断无需三角化
void GlobalSFM::triangulateTwoFramesWithDepth(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	Matrix3d Pose0_R = Pose0.block< 3,3 >(0,0);
	Matrix3d Pose1_R = Pose1.block< 3,3 >(0,0);
	Vector3d Pose0_t = Pose0.block< 3,1 >(0,3);
	Vector3d Pose1_t = Pose1.block< 3,1 >(0,3);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector3d point0;
		Vector2d point1;
		//同时被两帧观测到能够进行三角化
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation_depth[k].second < 0.1 || sfm_f[j].observation_depth[k].second >10) //max and min measurement
				continue;
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = Vector3d(sfm_f[j].observation[k].second.x()*sfm_f[j].observation_depth[k].second,sfm_f[j].observation[k].second.y()*sfm_f[j].observation_depth[k].second,sfm_f[j].observation_depth[k].second);
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		//3D到2D的投影误差
		if (has_0 && has_1)
		{
			Vector2d residual;
			Vector3d point_3d, point1_reprojected;
			//triangulatePoint(Pose0, Pose1, point0, point1, point_3d);   
			point_3d = Pose0_R.transpose()*point0 - Pose0_R.transpose()*Pose0_t;//shan add:this is point in world;  世界坐标系
			point1_reprojected = Pose1_R*point_3d+Pose1_t;       //相机坐标系

			residual = point1 - Vector2d(point1_reprojected.x()/point1_reprojected.z(),point1_reprojected.y()/point1_reprojected.z());
			//当深度值足够小时进行直接转换
			//std::cout << residual.transpose()<<"norm"<<residual.norm()*460<<endl;
			if (residual.norm() < 1.0/460){
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point_3d(0);
				sfm_f[j].position[1] = point_3d(1);
				sfm_f[j].position[2] = point_3d(2);
			}
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}
	}
}

//求解所有的特征点以及坐标
/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧的位姿和特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）
 * @param[in]  	l 	第l帧
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量
 * @param[in]  	sfm_f		所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)    l是最早的关键帧   倒数第二帧的POSE为relativeR relativeT   求解滑窗内所有图像帧的位姿QT，以及特征点坐标
// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
// l是最早的关键帧   倒数第二帧的POSE为relativeR relativeT   求解滑窗内所有图像帧的位姿QT，以及特征点坐标
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	//特征点个数   定义一个变量找到能够进行三角化的两帧图像
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view   设置初始的两帧位姿
	//第一帧的位姿
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	//倒数第二帧的位姿
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];

	Quaterniond c_Quat[frame_num];
	//for ceres  优化相关
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	//位姿参数
	Eigen::Matrix<double, 3, 4> Pose[frame_num];
	//初始帧的位姿
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];
	//当前帧的位姿
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

	//1、先三角化第l帧（参考帧）与第frame_num-1帧（当前帧）的路标点
	//2、pnp求解从第l+1开始的每一帧到第l帧的变换矩阵R_initial, P_initial，保存在Pose中
	//并与当前帧进行三角化
	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp   从第l帧开始，依次恢复第i帧到当前帧的所有特征点坐标
		if (i > l)
		{
			//先定义初始的上一帧的变换矩阵R，P   恢复了所有的pose
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			//得到第i个关键帧的变换矩阵
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			//得到第i个关键帧的位姿
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}
		//i=l 时先PNP l到当前帧的位姿参数  从第l帧开始恢复第l帧与当前帧之间的所有特征点的三维坐标
		// triangulate point based on the solve pnp result   恢复两帧之间的位姿
		triangulateTwoFramesWithDepth(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//至此得到所有帧的位姿    至此的奥所有帧的位姿
	//从l+1帧开始  恢复第l帧与第i帧的所有特征点坐标
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFramesWithDepth(l, Pose[l], i, Pose[i], sfm_f);
	//从l-1帧开始到第0帧之间的位姿恢复和特征点恢复
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate  恢复两帧之间的点
		triangulateTwoFramesWithDepth(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	//5: triangulate all other points  恢复所有点的三角化  能被两个以上的关键帧观测的都用来做三角化
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		//取出未被三角化的函数
		if(sfm_f[j].state == true) continue;
			    //能被两个帧同时观测到
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector3d point0;
			Vector2d point1;
			int frame_0 = sfm_f[j].observation[0].first;
			//观测到深度值
			if (sfm_f[j].observation_depth[0].second < 0.1 || sfm_f[j].observation_depth[0].second > 10) //max and min measurement
				continue;
			//恢复世界坐标系的尺度问题  三维尺度信息
			point0 = Vector3d(sfm_f[j].observation[0].second.x()*sfm_f[j].observation_depth[0].second,sfm_f[j].observation[0].second.y()*sfm_f[j].observation_depth[0].second,sfm_f[j].observation_depth[0].second);
			//该特征点最后被观测到的帧的id  获取图像坐标
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			//triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            //获取两帧之间的位姿  取起始帧与最后一帧
			Matrix3d Pose0_R = Pose[frame_0].block< 3,3 >(0,0);
			Matrix3d Pose1_R = Pose[frame_1].block< 3,3 >(0,0);
			Vector3d Pose0_t = Pose[frame_0].block< 3,1 >(0,3);
			Vector3d Pose1_t = Pose[frame_1].block< 3,1 >(0,3);
            
			Vector2d residual;
			Vector3d point1_reprojected;
			//triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			//获取世界坐标系的特征点坐标
			point_3d = Pose0_R.transpose()*point0 - Pose0_R.transpose()*Pose0_t;//point in world;
			//转换到最后帧的图像坐标  
			point1_reprojected = Pose1_R*point_3d+Pose1_t;

			residual = point1 - Vector2d(point1_reprojected.x()/point1_reprojected.z(),point1_reprojected.y()/point1_reprojected.z());
            //重投影误差足够小的话，恢复特征点的世界坐标
			if (residual.norm() < 1.0/460) {//reprojection error
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point_3d(0);
				sfm_f[j].position[1] = point_3d(1);
				sfm_f[j].position[2] = point_3d(2);
			}
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//全局图优化
	//full BA   
	ceres::Problem problem;    //构建最小二乘问题    
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		//构建四元数的顺序   ？   先是W
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}
    //添加没有被三角化的点
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}
	}
    //配置求解器
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
		//添加追踪到的特征点信息
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

