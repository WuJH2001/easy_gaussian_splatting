import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from read_data.read_blender import read_blender
from read_data.read_colmap import read_colmap
import torch
from tqdm import tqdm
from random import randint
from helpers import l1_loss_v1, params2rendervar, params2cpu, save_params
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from external import calc_ssim,calc_psnr,densify
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as raster_settings
import copy
from PIL import Image
from external import get_expon_lr_func


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],  # 需要衰减到 0.0000016*variables['scene_radius']
        'feature_dc': 0.0025,
        'feature_rest':0.000125,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_settings(curr_data):
    cam = raster_settings( 
            image_height=curr_data['cam']['image_height'],
            image_width=curr_data['cam']['image_width'],
            tanfovx=curr_data['cam']['tanfovx'],
            tanfovy=curr_data['cam']['tanfovy'],
            bg=curr_data['cam']['bg'],
            scale_modifier=curr_data['cam']['scale_modifier'],
            viewmatrix=curr_data['cam']['viewmatrix'],
            projmatrix=curr_data['cam']['projmatrix'],
            sh_degree=curr_data['cam']['sh_degree'],
            campos=curr_data['cam']['campos'],
            prefiltered=curr_data['cam']['prefiltered'],
            debug= curr_data['cam']['debug']
    )
    return cam

def show_image(image):
    # 创建PIL图像对象
    show_image = Image.fromarray(np.transpose(np.array(image.detach().clamp(0,1).cpu()*255).astype(np.uint8), (1, 2, 0)))
    # 显示图像
    show_image.show()
    show_image.save('output_image5.png')


def get_loss(params, curr_data,variables):

    rendervar = params2rendervar(params)  # 初始化一些 cuda 输入数据
    rendervar['means2D'].retain_grad()
    raster_settings = get_settings(curr_data)
    render_image, radius, _ = Renderer(raster_settings=raster_settings)(**rendervar)


    gt_image = curr_data["im"]  # 已经在cuda里面了
    loss_l1 = l1_loss_v1(render_image,gt_image)
    # show_image(render_image)
    Loss = 0.8 * loss_l1 + 0.2 * (1.0-calc_ssim(render_image,gt_image))

    psnr = calc_psnr(gt_image,render_image).mean()

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return Loss,variables,psnr

def report_progress(psnr, i , progress_bar, loss, points, every_i=10):  # 修改，添加points loss 当前img
    if i % every_i==0:
        progress_bar.set_postfix({"PSNR": f"{psnr:.{7}f}","Loss":f"{loss:.{7}f}", "points":f"{points}"})
        progress_bar.update(every_i)


def level_up_sh(dataset,todo_dataset):
    for data in dataset:
        data['cam']['sh_degree']+=1
    if todo_dataset is not None:
        for data in todo_dataset:
            data['cam']['sh_degree']+=1


def train(exp_name , datapath):
    if exp_name == "blender":
        params, variables, dataset = read_blender(datapath)
    elif exp_name == "colmap":
        params, variables, dataset = read_colmap(datapath)
    
    optimizer = initialize_optimizer(params, variables)

    progress_bar = tqdm(range(30000), desc="Training progress")
    todo_dataset = []
    xyz_lr_scheduler = get_expon_lr_func(lr_init= 0.00016 * variables['scene_radius'],lr_final= 0.0000016 * variables['scene_radius'],
                                                    lr_delay_mult=0.01,max_steps=30000)
    
    for i in range(1,30001): 
        # 1. 调整学习率
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = xyz_lr_scheduler(i)
                param_group['lr'] = lr
                break
        # 2. 调整 sh维度
        if i % 1000 == 0 and dataset[0]["cam"]['sh_degree'] < 3:
            level_up_sh(dataset,todo_dataset)
        # 3. 提取需要训练到数据
        if not todo_dataset:
            todo_dataset = copy.deepcopy(dataset)
        curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1)) 
        # curr_data = todo_dataset[0]
        # 4. 训练
        loss, variables,psnr = get_loss(params, curr_data, variables)
        loss.backward()
        with torch.no_grad():
            report_progress(psnr, i, progress_bar,loss, params['means3D'].shape[0])
            params, variables = densify(params, variables, optimizer, i)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # 5. 保存模型
        if  i == 7000 or i == 24000 or i == 30000:
            print("save model in {} iteration".format(i))
            output_params = params2cpu(params)
            seq = os.path.basename(datapath)
            save_params(output_params,exp_name,seq)



    progress_bar.close()



if __name__=='__main__':
    # exp_name = "blender"  # 采用的数据集
    exp_name = "colmap"
    # /data1/wujiahao/dataset/gs/4K_Studios_Show_Pair_f16f17 /data1/wujiahao/dataset/nerf/lego  /data1/wujiahao/dataset/gs/1080_Kungfu_Basic_Pair_c24c25
    train(exp_name,"/data1/wujiahao/dataset/gs/1080_Kungfu_Basic_Pair_c24c25") 







