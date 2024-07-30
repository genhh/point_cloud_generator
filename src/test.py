import numpy as np
import matplotlib.pyplot as plt
from ip_basic import depth_map_utils
import pcl

def test_img():
    #np.set_printoptions(threshold=np.inf)
    name = ["rt_of_low_high_res_event_cameras_optical_flow_1722222473948471069.npy",
            "rt_of_low_high_res_event_cameras_optical_flow_1722222468997538805.npy",
            "rt_of_low_high_res_event_cameras_optical_flow_1722222479044971227.npy",
            "rt_of_low_high_res_event_cameras_optical_flow_1722222484389434576.npy",
            ]
    depthmap = np.load("/home/ubuntu/imgdata/"+name[3])
    print(np.max(depthmap))
    #print(depthmap[:,:,1].shape)
    #import numpy as np
    x = np.arange(0,260, step=1)
    y = np.arange(0,346, step=1)

    #depthmap = np.load('0000.npy')    #使用numpy载入npy文件
    plt.subplot(121)
    plt.imshow(depthmap[:,:,0])              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
    plt.subplot(122)
    plt.imshow(depthmap[:,:,1])
    # plt.colorbar()                   #添加colorbar
    #plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
    plt.show()                        #在线显示图像
    """
    xx, yy=np.meshgrid(x, y)#网格化坐标
    X, Y=xx.ravel(), yy.ravel()#矩阵扁平化
    bottom=np.zeros_like(X)#设置柱状图的底端位值
    Z=depthmap[:,:,1].ravel()#扁平化矩阵
    width=height=1#每一个柱子的长和宽
    
    #绘图设置
    fig=plt.figure()
    ax=fig.gca(projection='3d')#三维坐标轴
    ax.bar3d(X, Y, bottom, width, height, Z, shade=True)#
    #坐标轴设置
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z(value)')
    plt.show()
    """

def readpcdfile_demo():
	# 加载刚刚保存的点云文件：1.pcd
    cloud = pcl.load_XYZRGB('./1.pcd')
    visual = pcl.pcl_visualization.CloudViewing()# 创建visual类，创建显示窗口，名字为：PCD viewer
    visual.ShowColorCloud(cloud, b'cloud')#cloud前面一定不要忘了'b'，至于为啥这里也不是太懂
    flag = True
    while flag:
        flag != visual.WasStopped()


def depth_completion(image):
    projected_depths = np.float32(image / 256.0)
    final_depths = depth_map_utils.fill_in_fast(
    projected_depths, extrapolate=True, blur_type='bilateral')
    image = (final_depths * 256).astype(np.uint8)

def visual_demo():
    cloud = pcl.PointCloud_PointXYZRGB()#创建pointcloud的类型，着色点云
    points = np.zeros((3000000, 4), dtype=np.float32)
    cloud.from_array(points)
    print(cloud)
    pcl.save(cloud, '/home/ubuntu/1.pcd')
    #函数参数说明：save(cloud: Any,path: {endswith},format: Any = None,binary: bool = False) -> Any
    # cloud：创建的点云对象
    # path：点云文件的保存路径，其余参数不用管


if __name__ == '__main__':
    visual_demo()