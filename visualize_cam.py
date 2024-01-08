import matplotlib.pyplot as plt
import numpy as np




def visualize_camera(c2w_matrixs):
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors =['g','b','r']
    for c2w_matrix in c2w_matrixs:

        world_camera_origin = c2w_matrix[:3,3]
        world_camera_axes = c2w_matrix[:3,:3]


        # 绘制相机坐标轴
        for i in range(3):  # world_camera_axes[i, 0]
            ax.plot([world_camera_origin[0], world_camera_origin[0]+world_camera_axes[0][i]],
                    [world_camera_origin[1], world_camera_origin[1]+world_camera_axes[1][i]],
                    [world_camera_origin[2], world_camera_origin[2]+world_camera_axes[2][i]],
                    color=colors[i]
                    )

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置固定单位长度的纵横比
    ax.set_box_aspect([6, 6, 6])  # 在范围 [-3, 3] 之间，每个轴的范围是 6

    # 设置坐标轴范围
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    # 设置图形标题
    ax.set_title('Camera Visualization')

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show(block=True)



# 在z轴平移
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]],dtype=float)

# x轴旋转，x不变
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]],dtype=float)

# y轴旋转，y不变，这样就使得绕物体旋转
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]],dtype=float)


def pose_spherical(theta, phi, radius):
    """
    theta为仰角，(-180,180)
    phi为方位角，30 固定值，多少角度拍一张
    radius为距离球心距离，3 距离多远拍一张

    返回 c2w
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w  # 矩阵乘法的意思
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return  c2w 



c2w_matrixs = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)  # [40,4,4]





# 调用可视化函数
visualize_camera(c2w_matrixs)