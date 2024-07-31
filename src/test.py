import numpy as np
import matplotlib.pyplot as plt
from ip_basic import depth_map_utils
import pcl
import cv2

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

def cal_trans_matrix(): 
  
    # 读取图片  
    imgA = cv2.imread('/home/ubuntu/imgdata/_dvs_render_1722261201126779079.png')
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)   
    imgB = cv2.imread('/home/ubuntu/imgdata/_D435i_depth_image_rect_raw_1722261201126779079.png')  
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    # 使用ORB检测特征点  
    orb = cv2.ORB_create()  
    kpA, desA = orb.detectAndCompute(imgA, None)  
    kpB, desB = orb.detectAndCompute(imgB, None)  
    
    # 匹配特征点  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    matches = bf.match(desA, desB)  
    
    # 排序匹配项  
    matches = sorted(matches, key=lambda x: x.distance)  
    
    # 至少需要一些匹配项来估计变换  
    if len(matches) > 4:  
        src_pts = np.float32([kpA[m.queryIdx].pt for m in matches[:4]]).reshape(-1, 1, 2)  
        dst_pts = np.float32([kpB[m.trainIdx].pt for m in matches[:4]]).reshape(-1, 1, 2)  
    
        # 估计透视变换矩阵  
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  
    
        # 应用透视变换  
        height, width = imgA.shape  
        imgA_transformed = cv2.warpPerspective(imgA, M, (imgB.shape[1], imgB.shape[0]))  
    
        # 这里你可以进一步计算边界框等  
        # 注意：边界框计算依赖于你想要的精度和复杂性  
    
    else:  
        print("Not enough matches are found - %d/%d" % (len(matches), 4))  
    
    # 显示结果（可选）  
    cv2.imshow('Transformed Image A', imgA_transformed)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def match_and_calibrate(image_a, image_b):
    # Convert images to grayscale
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_a, descriptors_b)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points_a = np.zeros((len(matches), 2), dtype=np.float32)
    points_b = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_a[i, :] = keypoints_a[match.queryIdx].pt
        points_b[i, :] = keypoints_b[match.trainIdx].pt

    # Find homography using RANSAC
    H, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC)

    # Get the dimensions of image_a
    h_a, w_a = image_a.shape[:2]

    # Define points in image_a
    corners_a = np.float32([[0, 0], [0, h_a - 1], [w_a - 1, h_a - 1], [w_a - 1, 0]]).reshape(-1, 1, 2)

    # Transform corners to image_b's coordinate system
    corners_b = cv2.perspectiveTransform(corners_a, H)

    # Draw the transformed corners on image_b
    image_b_with_box = image_b.copy()
    cv2.polylines(image_b_with_box, [np.int32(corners_b)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Get the bounding box of the transformed corners
    x, y, w, h = cv2.boundingRect(np.int32(corners_b))

    return image_b_with_box, (x, y, w, h)

  
def detect_circles_in_image(image_path):  
    # 读取图片  
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if img is None:  
        print("Error: 图片未找到或路径错误")  
        return  
  
    # 应用双边滤波  
    bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)  
  
    # 霍夫圆检测  
    # 参数1: 灰度图  
    # 参数2: 检测方法，cv2.HOUGH_GRADIENT 是梯度法  
    # 参数3: dp = 1, 累加器分辨率与图像分辨率的反比  
    # 参数4: minDist = 100, 检测到的圆的中心之间的最小距离  
    # 参数5: param1 = 50, Canny边缘检测器的高阈值  
    # 参数6: param2 = 30, 圆心检测器累加器的阈值  
    # 参数7: minRadius = 0, 圆半径的最小值  
    # 参数8: maxRadius = 0, 圆半径的最大值  
    circles = cv2.HoughCircles(bilateral_filtered, cv2.HOUGH_GRADIENT, 1, 100,  
                               param1=50, param2=30, minRadius=0, maxRadius=0)  
  
    # 确保至少检测到一个圆  
    if circles is not None:  
        circles = np.uint16(np.around(circles))  
        for i in circles[0, :]:
            if img[i[1]][i[0]]>50:
                continue
            # 绘制圆心  
            cv2.circle(img, (i[0], i[1]), 1, (255, 0, 0), 3)  
            # 绘制圆轮廓  
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 255), 3)  
  
    # 显示结果  
    cv2.imshow('Detected Circles', img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  






def compute_dense_optical_flow(image1, image2):
    """
    计算两个图像之间的稠密光流。
    :param image1: 第一幅图像
    :param image2: 第二幅图像
    :return: 稠密光流
    """
    # 将图像转换为灰度图像
    #gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

def create_mask_from_flow(flow):
    """
    根据稠密光流创建掩码 光流为0的位置设置为True。
    :param flow: 稠密光流
    :return: 掩码
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (mag == 0)
    return mask

def apply_mask_to_depth_image(depth_image, mask):
    """
    对深度图像应用掩码，滤除掩码位置的深度值。
    :param depth_image: 16位无符号整型的深度图像
    :param mask: 掩码
    :return: 应用掩码后的深度图像
    """
    depth_image_masked = depth_image.copy()
    depth_image_masked[mask] = 0
    return depth_image_masked

def mask_depth_image_using_optical_flow(depth_image1, depth_image2):
    """
    使用稠密光流计算掩码，并对深度图像应用掩码。
    :param depth_image1: 第一幅16位无符号整型的深度图像
    :param depth_image2: 第二幅16位无符号整型的深度图像
    :return: 应用掩码后的深度图像
    """
    # 将16位深度图像转换为8位图像
    depth_image1_8bit = cv2.convertScaleAbs(depth_image1, alpha=(255.0/65535.0))
    depth_image2_8bit = cv2.convertScaleAbs(depth_image2, alpha=(255.0/65535.0))

    # 计算稠密光流
    flow = compute_dense_optical_flow(depth_image1_8bit, depth_image2_8bit)
    

    # 创建掩码
    mask = create_mask_from_flow(flow)
    
    #cv2.imwrite("flow.png", mask)
    # 应用掩码到深度图像
    depth_image_masked = apply_mask_to_depth_image(depth_image1, mask)
    print(depth_image_masked)
    return depth_image_masked
"""
# 示例使用
if __name__ == "__main__":
    # 加载两幅16位无符号整型的深度图像
    depth_image1 = cv2.imread("/home/ubuntu/imgdata/_D435i_depth_image_rect_raw_1722396043667983055.png", cv2.IMREAD_UNCHANGED)
    depth_image2 = cv2.imread("/home/ubuntu/imgdata/_D435i_depth_image_rect_raw_1722396042672539949.png", cv2.IMREAD_UNCHANGED)

    # 使用稠密光流计算掩码并应用到深度图像
    depth_image_masked = mask_depth_image_using_optical_flow(depth_image1, depth_image2)

    # 保存结果
    cv2.imwrite("masked_depth_image.png", depth_image_masked)
"""



# Example usage
if __name__ == "__main__":
    # Load the images
    image_a = cv2.imread('/home/ubuntu/imgdata/_dvs_render_1722346390904489040.png')
    image_b = cv2.imread('/home/ubuntu/imgdata/_fliter_depth_1722397101039365053.png')
    cv2.imshow("test",image_b.astype(np.uint16))
    ## Match and calibrate
    #result_image, bbox = match_and_calibrate(image_a, image_b)

    ## Display the results
    #print(f"Bounding Box in Image B: {bbox}")
    #cv2.imshow("Image B with Box", result_image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #image_a = np.zeros((4, 2), dtype=np.uint8)
    print(image_b.shape)

    
"""
if __name__ == '__main__':
    cal_trans_matrix()
"""