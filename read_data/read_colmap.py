import os
import struct
from PIL import Image
import numpy as np
from external import build_rotation_np
import torch
from helpers import setup_camera,o3d_knn,inverse_sigmoid,save_intri
from PIL import Image
from helpers import RGB2SH
CAMERA_PARAM={'SIMPLE_PINHOLE':3,'PINHOLE':4}
CAMERA_ID = ["SIMPLE_PINHOLE","PINHOLE"]

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c:char, e, f:float, d:double, h, H, i:int, I:undesigned int, l:long, L:un long, q:long long, Q:un long long}. 
    :param endian_character: Any of {@:native, =:native, <:小端对齐, >:大端对齐, !:大端对齐}  
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)  # fid 是文件句柄
    return struct.unpack(endian_character + format_char_sequence, data)  # 解压



def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    intris = []
    with open(path_to_model_file, "rb") as fid: # cameras.bin 文件
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):  # 静态就一个相机
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            # camera_id = camera_properties[0]
            # model_id = camera_properties[1]
            model_name = CAMERA_ID[camera_properties[1]]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_PARAM[model_name]
            params = read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params)  # 针孔相机4个参数 xy焦距，和宽高（没用到）
            
            if model_name=="PINHOLE":
                focal_x = params[0]
                focal_y = params[1]
            if model_name == "SIMPLE_PINHOLE":
                focal_x = params[0]
                focal_y = params[0]
            k = np.zeros((3,3),dtype=float)
            k[0,0] = focal_x
            k[1,1] = focal_y
            k[0,2] = float(width)/2
            k[1,2] = float(height)/2
            k[2,2] = 1.0
            intris.append(k)
    return intris



def read_extrinsics_binary(path_to_model_file):  # 读取image.bin，将世界坐标系映射到相机坐标系的参数
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:  # image.bin文件
        extris = []
        image_names = []
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]  # 图像数量
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(  # int 4 字节，double 8字节，总共64字节
                fid, num_bytes=64, format_char_sequence="idddddddi")
            # image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])  # 四元组  4，存储的是 R
            tvec = np.array(binary_image_properties[5:8])  # scale  3 ，就是 T
            # camera_id = binary_image_properties[8]   # 静态的就一个摄像机，没必要分这么细

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry  这一段就是为了找到image name
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, num_bytes=8,  # 8638个点
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,  # double double unlong 总共24字节  25914
                                       format_char_sequence="ddq"*num_points2D)
            # xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),  # [8638,2] 从第0个字符起，每隔三个字符取一个，这个好像没用？
            #                        tuple(map(float, x_y_id_s[1::3]))])
            # point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))  

            R = build_rotation_np(qvec)  # w2c 的R，gaussian里面的代码直接用
            w2c = np.zeros((4, 4))
            w2c[:3, :3] = R 
            w2c[:3, 3] = tvec
            w2c[3, 3] = 1.0
            
            extris.append(w2c)
            image_names.append(image_name)


    return extris,image_names


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_colmap(path):
    extrinsic_file = os.path.join(path,"sparse/0/images.bin")
    intrinsic_file = os.path.join(path,"sparse/0/cameras.bin")
    extris,image_names = read_extrinsics_binary(extrinsic_file)  # 读取外参文件 251
    k = read_intrinsics_binary(intrinsic_file)[0]  # 读取相机的内参 k 
    save_intri(k,'colmap',os.path.basename(path))  # 保存内参，用于渲染
    dataset = []
    cam_centers = []
    for idx, w2c in enumerate(extris):
        cam = setup_camera(int(k[0,2]*2),int(k[1,2]*2),k,w2c,0.01,100)  # h,w 一定要是 int
        im = np.array(Image.open(os.path.join(path,"images",image_names[idx])))/255.0 # 不需要后缀也可以读
        im = torch.from_numpy(im).float().cuda().permute(2, 0, 1).clamp(0.0,1.0)
        dataset.append({'cam': cam, 'im': im, 'id': idx})
        cam_centers.append(np.linalg.inv(w2c)[:3,3])
    


    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin") #
    xyz, rgb, _ = read_points3D_binary(bin_path)  # [num_pts,3]

    num_pts = xyz.shape[0]
    feature_dc = RGB2SH(rgb/255.0).reshape((num_pts,1,3))  # [num_pts,1,3]
    feature_rest = np.zeros((num_pts,15,3))

    sq_dist, _ = o3d_knn(xyz,3) 
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    # 下面都是没有进行激活的参数
    params = {
        'means3D': xyz,                                    # [N_points, 3]                      
        'feature_dc':  feature_dc,
        'feature_rest': feature_rest,
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_pts, 1)),  # [N_points, 4]
        'logit_opacities': inverse_sigmoid(0.1 * np.ones((num_pts, 1))),  # 改成 inverse_sigmoid(0.1)
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}

    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(num_pts).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(num_pts).cuda().float(),
                 'denom': torch.zeros(num_pts).cuda().float()}  # denomination
    
    return params,variables,dataset

    
