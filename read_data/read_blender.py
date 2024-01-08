import numpy as np
from helpers import setup_camera,o3d_knn,  fov2focal,inverse_sigmoid
import torch
import os
import json
from PIL import Image
import copy
import matplotlib.pyplot as plt

def read_json(path, file_name ):
    cam_centers = []
    dataset = []
    h,w = 800,800
    cx,cy = 400,400
    # f = open('w2c.txt','w')
    with open(os.path.join(path,file_name)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in enumerate(frames):

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # f.write(str(w2c))

            cam_centers.append(c2w[:3,3])  # 其实就是用 [0,0,0,1] 乘出来的
            fx = fov2focal(fovx,w)
            k = np.zeros((3,3),dtype=float)
            k[0][0], k[1][1], k[0][2], k[1][2] = fx, fx, cx, cy
            cam = setup_camera(h, w, k, w2c, near=0.01, far=100)

            image_path = os.path.join(path, frame["file_path"] + ".png")
            # im = np.array(copy.deepcopy(Image.open(image_path)))   # [800,800,4]  错误的
            # im = im[:,:,:3] * im[:,:,3:4]                            # bg = [0,0,0]
            # im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255   # [3,800,800]

            bg = [0,0,0]  # 如果需要白色背景可以改成 [1,1,1]
            norm_data = np.array(Image.open(image_path).convert("RGBA")) / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            im = torch.from_numpy(arr).float().cuda().permute(2, 0, 1).clamp(0.0,1.0)   # 进行 clamp，防止颜色突变


            dataset.append({'cam': cam, 'im': im, 'id': idx})

    # f.close()
    return dataset,cam_centers

def read_blender(path):
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3  # 这个是怎么确定的？
    feature_dc = np.random.random((num_pts,1, 3)) / 255.0
    feature_rest = np.zeros((num_pts,15,3))

    sq_dist, _ = o3d_knn(xyz,3) 
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    # 下面都是没有进行激活的参数
    params = {
        'means3D': xyz,                                    # [N_points, 3]                      
         #'rgb_colors': shs,                               # [N_points, 3]  这个是用于预计算接口的
        'feature_dc':  feature_dc,
        'feature_rest': feature_rest,
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_pts, 1)),  # [N_points, 4]
        'logit_opacities': inverse_sigmoid(0.1 * np.ones((num_pts, 1))),  # 改成 inverse_sigmoid(0.1)
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
    }


    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}
    dataset_train , cam_centers_train = read_json(path,"transforms_train.json")  # cam_centers shape = [n_cameras , 3]
    dataset_test, cam_centers_test = read_json(path,"transforms_test.json")
    cam_centers = cam_centers_train+cam_centers_test
    dataset = dataset_train+dataset_test


    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(num_pts).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(num_pts).cuda().float(),
                 'denom': torch.zeros(num_pts).cuda().float()}
    

    return params, variables,dataset
    


