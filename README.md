# point_cloud_generator
a ros node to generator point cloud by depth and odom, which can fliter static object optional.

## environment
ubuntu18.04, catkin build tools, [cv_bridge3](https://github.com/ros-perception/vision_opencv) for melodic, ROS, cupy, pcl etc.

if you wants optical flow mask , you need to download [this](https://github.com/heudiasyc/rt_of_low_high_res_event_cameras.git)

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
# before start, source cv_bridge workspace
# roslaunch point_cloud_generator final.launch 
./start.sh

# need to open new window
#rosrun point_cloud_generator image_saver.py # when you want to save the img from topic, press 's'
sudo apt-get install ros-melodic-ros-numpy
pip3 install open3d
./start_save.sh
# show pcd file
pcl_viewer xxx.pcd
```

## to do list
A bug when start_save quit, the terminal can't work. You need reopen a new terminal, then it can work again.

## reference
depth completion part from [this](https://github.com/kujason/ip_basic.git)

optical flow part from [this](https://github.com/heudiasyc/rt_of_low_high_res_event_cameras.git)

python-pcl error [solution](https://askubuntu.com/questions/1160219/how-to-make-pcl-library-and-python-pcl-run-on-ubuntu-18-04-2)


