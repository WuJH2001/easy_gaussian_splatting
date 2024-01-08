import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import math

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()  # [4,4]
    cam_center = torch.inverse(w2c)[:3, 3]  # 
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],  # 2 * fx / w = near / r, 这个就是高斯代码里的
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)   # 注意这个是转置了的，因为cuda代码就是这样的

    # 因为sh需要动态改变，所以设置为字典形式好一点
    cam = {  "image_height":h,
        "image_width":w,
        "tanfovx":w / (2 * fx),
        "tanfovy":h / (2 * fy),
        "bg":torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        "scale_modifier":1.0,
        "viewmatrix":w2c,
        "projmatrix":full_proj,
        "sh_degree":0,
        "campos":cam_center,
        "prefiltered":False,
        "debug": False}
    return cam


def params2rendervar(params):

    # 进行激活
    rendervar = {
        'means3D': params['means3D'],
         # 'colors_precomp': params['rgb_colors'],
        'shs':   torch.cat((params['feature_dc'], params['feature_rest']), dim=1)  ,      #  [N_points,16,3]
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):  # 使用open3d库
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()  # 初始化点云类
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)   # 构建KD树，以实现高效KNN搜索
    for p in pcd.points:                      # 为每个点找到knn
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)   # 排除本身
        indices.append(i[1:])
        sq_dists.append(d[1:])    # 距离和索引
    return np.array(sq_dists), np.array(indices)


def params2cpu(params):
    res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}  # 第一时刻把所有数据都保存下来
    return res


def save_params(output_params,exp, seq):
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params_3", **output_params)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))