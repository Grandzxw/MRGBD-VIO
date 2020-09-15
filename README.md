# MRGBD-VIO
A Multi-Feature tightly-coupled RGB-D visual-inertial SLAM system. The proposed system is the first tightly coupled optimization-based RGB-D-inertial system based on multi-features. This system is runs on **Linux** and **ROS**.

## 1. Prerequisites
1.1 **Ubuntu** and **ROS**
Ubuntu 16.04. ROS Kinetic, [ROS Installation](http://wiki.ros.org/indigo/Installation/Ubuntu)
additional ROS pacakge

```
    sudo apt install ros-Kinetic-desktop-full
```

1.2 **Opencv3**

If you install ROS Kinetic, please update **opencv3** with 
```
    sudo apt-get install ros-kinetic-opencv3
```
1.3 **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html), remember to **make install**.

1.4 **Sophus**
```
    git clone http://github.com/strasdat/Sophus.git
```

## 2. Build MRGBD-VIO on ROS
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/Grandzxw/MRGBD-VIO.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## 3.Run on OpenLORIS dataset

```
  roslaunch vins_estimator realsense_color.launch
  roslaunch vins_estimator vins_rviz.launch
  rosbag play bagname.bag
```
## 4.Run on OpenLORIS dataset

+ [OpenLORIS](https://github.com/lifelong-robotic-vision/lifelong-slam)

## 5. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

