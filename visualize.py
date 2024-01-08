import os
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as raster_settings
from helpers import setup_camera
from visualize_cam import pose_spherical
from helpers import fov2focal,params2rendervar
import imageio
from PIL import Image

w, h = 800, 800
near, far = 0.01, 100.0
view_scale = 3.9

def render(w2c,k,rendervar):
    cam = setup_camera(w, h, k, w2c, near, far)
    settings  = raster_settings(
            image_height=cam['image_height'],
            image_width=cam['image_width'],
            tanfovx=cam['tanfovx'],
            tanfovy=cam['tanfovy'],
            bg=cam['bg'],
            scale_modifier=cam['scale_modifier'],
            viewmatrix=cam['viewmatrix'],
            projmatrix=cam['projmatrix'],
            sh_degree=cam['sh_degree'],
            campos=cam['campos'],
            prefiltered=cam['prefiltered'],
            debug= cam['debug']
    )
    image,_,depth = Renderer(raster_settings=settings)(**rendervar)
    return image, depth

def load_scene_data(exp,seq):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    rendervar = params2rendervar(params)
    return rendervar


def init_cameras():
    c2w_matrixs = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    fx = fov2focal(0.6911112070083618,800.0)
    k = np.array([[fx,0,400.0],[0,fx,400.0],[0,0,1]],dtype=float)
    return c2w_matrixs, k



def visualize(exp_name, path):
    rendervar = load_scene_data(exp_name,os.path.basename(path))
    c2w_matrixs,k = init_cameras()
    images =[]
    to8b = lambda x : (255*np.clip(x.detach().cpu().numpy(),0,1)).astype(np.uint8)
    for c2w in c2w_matrixs:
        c2w[:3, 1:3] *= -1  # 改为colmap
        w2c = np.linalg.inv(c2w)
        image, depth = render(w2c, k, rendervar)
        images.append(to8b(image).transpose(1,2,0))
    
    imageio.mimwrite(os.path.join(path,'video_rgb.mp4'), images, fps=30)


if __name__=="__main__":
    exp_name= 'blender'
    visualize(exp_name,'/data1/wujiahao/3dgs/output/blender/lego')
    print('done')


