# generator pointclouds by depth,events and odom

## environment
ubuntu18.04, catkin build tools, [cv_bridge3](https://github.com/ros-perception/vision_opencv) for melodic, ROS, cupy, pcl etc.

if you wants optical flow mask ,you need to download [this](https://github.com/heudiasyc/rt_of_low_high_res_event_cameras.git)

## func

### based events optical flow mask
aid to fliter the depth which is static

### generator pointClouds by depth and odom
input: depth image, odom msg
output: point clouds

contains two version --> cpu/gpu

## usage

```
cd ~/catkin_ws/src
git clone https://github.com/genhh/.git
catkin build point_cloud_generator
source ../devel/setup.bash
roslaunch point_cloud_generator final.launch
```