# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import math
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import json

import smpl
from lib.config import load_config
from lib.geo_util import compute_normal_v
from lib.mesh_util import save_obj_mesh, save_obj_mesh_with_color, scalar_to_color, replace_hands_feet, \
    replace_hands_feet_mesh, reconstruction, cal_sdf, sdf2mesh, gen_texture
from lib.model.TNet import TNet
from lib.net_util import batch_rod2quat, homogenize, load_network, get_posemap, compute_knn_feat
from lib.model.IGRSDFNet import IGRSDFNet
from lib.model.LBSNet import LBSNet
from lib.data.CapeDataset import CapeDataset_scan

import logging

from smpl.smpl import create

logging.basicConfig(level=logging.DEBUG)




def Pose2Cano(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
              igr_net, lat_vecs_igr, texture_net,
              model, gt_lbs_smpl,
              multi_pose_loader,
              cuda):
    
    fwd_skin_net.train()
    inv_skin_net.train()
    igr_net.train()

    optimizer_lat_inv = torch.optim.Adam(lat_vecs_inv_skin.parameters(),
                                     lr=opt['training']['lr_inv'])
    optimizer_lat_igr = torch.optim.Adam(lat_vecs_igr.parameters(),
                                         lr=opt['training']['lr_igr'])
    optimizer_texture = torch.optim.Adam(texture_net.parameters(),
                                    lr=opt['training']['lr_t'])

    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)
    
    n_iter = 0
    sdf_accumulate=np.zeros([257,257,257])
    num_accumulate=0
    for train_idx, pose_data in enumerate(multi_pose_loader):
        betas = pose_data['betas'].to(device=cuda)
        body_pose = pose_data['body_pose'].to(device=cuda)
        # scan_cano = train_data['scan_cano'].to(device=cuda).permute(0,2,1)
        scan_posed = pose_data['scan_cano_uni'].to(device=cuda)
        scan_n_posed = pose_data['normals_uni'].to(device=cuda)
        scan_color = pose_data['colors'].to(device=cuda)
        scan_color = scan_color.permute(0, 2, 1)
        scan_vis = pose_data['scan_vis'].to(device=cuda)

        scan_tri = pose_data['scan_tri_posed'].to(device=cuda)
        w_tri = pose_data['w_tri'].to(device=cuda)
        transl = pose_data['transl'].to(device=cuda)
        f_ids = pose_data['f_id'].to(device=cuda)
        smpl_data = pose_data['smpl_data']
        global_orient = body_pose[:, :3]
        body_pose = body_pose[:, 3:]
    
        smpl_neutral = smpl_data['smpl_neutral'].cuda()
        smpl_cano = smpl_data['smpl_cano'].cuda()
        smpl_posed = smpl_data['smpl_posed'].cuda()
        smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
        bmax = smpl_data['bmax'].cuda()
        bmin = smpl_data['bmin'].cuda()
        jT = smpl_data['jT'].cuda()
        inv_rootT = smpl_data['inv_rootT'].cuda()
    
        scan_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_posed))[:, :3,
                     :]  # remove root transform
        scan_n_posed = torch.einsum('bst,bvt->bsv', inv_rootT[:, :3, :3], scan_n_posed)

        scan_tri = torch.einsum('bst,btv->bsv', inv_rootT, homogenize(scan_tri, 1))[:, :3, :]
    
        reference_lbs_scan = compute_knn_feat(scan_posed.permute(0, 2, 1), smpl_posed,
                                              gt_lbs_smpl.expand(scan_posed.shape[0], -1, -1).permute(0, 2, 1))[:,
                             :, 0].permute(0, 2, 1)
    
        if opt['model']['inv_skin_net']['g_dim'] > 0:
            lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
            inv_skin_net.set_global_feat(lat)
    
        for epoch in range(opt['training']['n_epoch_inv']):
            fwd_skin_net.train()
            inv_skin_net.train()
            
            if epoch % opt['training']['resample_every_n_epoch'] == 0:
                multi_pose_loader.dataset.resample_flag = True
            else:
                multi_pose_loader.dataset.resample_flag = False
                
            
            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                          scan_posed, reference_lbs_scan=reference_lbs_scan, jT=jT,
                                                          v_tri=scan_tri, w_tri=w_tri, bmin=bmin[:, :, None],
                                                          bmax=bmax[:, :, None])
            
            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl,
                                                                res_lbs_p['pred_scan_cano'], jT=jT, res_posed=res_lbs_p)
            
            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err = err_lbs_p + err_lbs_c
            err_dict['All'] = err.item()
            
            optimizer_lat_inv.zero_grad()
            err.backward()
            optimizer_lat_inv.step()
            
            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k, v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (
                    epoch, opt['training']['num_epoch_pt2'], train_idx, len(multi_pose_loader), err_txt))
            n_iter += 1



        # START train lat_vecs_igr ###############
        
        igr_net.set_lbsnet(fwd_skin_net)

        n_iter = 0
        max_train_idx = 0
        start_time = time.time()
        current_number_processed_samples = 0
        start_epoch=0
        multi_pose_loader.dataset.resample_flag = True
        
        feat3d_posed=None
        res_lbs_p, err_lbs_p, err_dict_lbs_p = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                            scan_posed, jT=jT, nml_scan=scan_n_posed,
                                                            bmin=bmin[:, :, None], bmax=bmax[:, :, None])
        scan_cano = res_lbs_p['pred_scan_cano']
        normal_cano = res_lbs_p['normal_scan_cano']
        
        if opt['model']['igr_net']['g_dim'] > 0:
            lat = lat_vecs_igr(f_ids)  # (B, Z)
            igr_net.set_global_feat(lat)

        feat3d_cano = None
        smpl_neutral = smpl_neutral.permute(0, 2, 1)

        body_pose_feat = batch_rod2quat(body_pose.reshape(-1, 3)).view(betas.shape[0], -1,
                                                                       opt['model']['igr_net']['pose_dim'])
        igr_net.set_pose_feat(body_pose_feat)

        scan_verts = scan_cano[:, :, torch.randperm(scan_cano.shape[2]).to(cuda)[:opt['data']['num_sample_scan_igr']]]
        smpl_verts = smpl_neutral[:, :,
                     torch.randperm(smpl_neutral.shape[2]).to(cuda)[:opt['data']['num_sample_smpl_igr']]]
        body_verts = torch.cat([scan_verts, smpl_verts], -1)
        body_rand = body_verts + torch.normal(torch.zeros_like(body_verts), opt['data']['sigma_body'])
        bbox_rand = torch.rand_like(body_rand[:, :, :opt['data']['num_sample_bbox_igr']]) * (
                    igr_net.bbox_max - igr_net.bbox_min) + igr_net.bbox_min

        smpl_neutral_n = compute_normal_v(smpl_neutral.permute(0, 2, 1),
                                          smpl_face.expand(smpl_posed.shape[0], -1, -1)).permute(0, 2, 1)
        scan_cano, normal_cano = replace_hands_feet(scan_cano, normal_cano, smpl_neutral, smpl_neutral_n,
                                                    opt['data']['num_sample_surf'],
                                                    vitruvian_angle=model.vitruvian_angle)
        for epoch in range(opt['training']['n_epoch_igr']):

            res_sdf, err_sdf, err_dict = igr_net.forward(feat3d_cano, scan_cano, body_rand, bbox_rand, normal_cano)
            optimizer_lat_igr.zero_grad()
            err_sdf.backward()
            optimizer_lat_igr.step()
            if (n_iter+1) % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k,v in err_dict.items()])
                time_now = time.time()
                duration = time_now-start_time
                current_number_processed_samples += f_ids.shape[0]
                persample_process_time = duration/current_number_processed_samples
                current_number_processed_samples -f_ids.shape[0]
                # print('[%03d/%03d]:[%04d/%04d] %02f FPS, %s' % (epoch, opt['training']['num_epoch_sdf'], train_idx, len(train_data_loader), 1.0/persample_process_time, err_txt))
                epoch_percent = (epoch - start_epoch)/(opt['training']['num_epoch_sdf']-start_epoch)
                if epoch_percent == 0:
                    remaining_time = '..h..m'
                else:
                    remaining_time_in_minute = int(duration/epoch_percent*(1-epoch_percent)/60)
                    remaining_time ='%dh%dm'%(remaining_time_in_minute//60, remaining_time_in_minute%60)

                print('[%03d/%03d]: %02f FPS, %s, %s' % (epoch, opt['training']['num_epoch_sdf'], 1.0/persample_process_time, remaining_time, err_txt))
            n_iter += 1
            current_number_processed_samples += f_ids.shape[0]
        # END train lat_vecs_igr ############
        
        
        # START train texture_net ###########
        
        fwd_skin_net.eval()
        igr_net.eval()
        inv_skin_net.eval()

        for epoch in range(opt['training']['n_t']):
            sdf, last_layer_feature, point_local_feat = igr_net.query(scan_cano, return_last_layer_feature=True)
            # mask points with vis
            scan_color=scan_color[scan_vis]
            point_local_feat=point_local_feat[scan_vis]
            err, err_dict = texture_net(point_local_feat, last_layer_feature, scan_color)
            # 需要借鉴一下 nerf 的 multi-view reconsruction 方法
            # 可以借鉴一下 meta-learning，用来加速

            optimizer_texture.zero_grad()
            err.backward()
            optimizer_texture.step()



        sdf = cal_sdf(igr_net, cuda, torch.eye(4)[None].to(cuda),
                      opt['experiment']['vol_res'], thresh=0.0)
        sdf_accumulate+=sdf
        num_accumulate+=1
    
    sdf_final=sdf_accumulate/num_accumulate
    b_min=igr_net.bbox_min.squeeze().cpu().numpy()
    b_max=igr_net.bbox_max.squeeze().cpu().numpy()

    verts, faces, _, _ =sdf2mesh(sdf_final,b_min,b_max,0.)

    color_verts= gen_texture(igr_net, verts, cuda, torch.eye(4)[None].to(cuda), texture_net)

    # save_obj_mesh('%s/%s_posed%s.obj' % (result_dir, frame_names[0], str(test_idx).zfill(4)), pred_scan_posed, faces)


def run_pose2cano(opt):
    cuda = torch.device('cuda:0')
    
    exp_name = opt['experiment']['name']
    ckpt_dir = '%s/%s' % (opt['experiment']['ckpt_dir'], exp_name)
    result_dir = '%s/%s' % (opt['experiment']['result_dir'], exp_name)
    log_dir = '%s/%s' % (opt['experiment']['log_dir'], exp_name)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Backup config into log_dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as config_file:
        config_file.write(json.dumps(opt))
    
    # Initialize vitruvian SMPL model
    if 'vitruvian_angle' not in opt['data']:
        opt['data']['vitruvian_angle'] = 25
    
    # 1.读取 smpl 模型的固定参数，包括 lbs_weights, J_regressor等
    # 2.初始化姿态(23*4)形状(10)状态值为默认值
    # 3.初始化的内容在之后的 forward 函数(负责 skinning)中被调用
    model = create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                   gender=opt['data']['smpl_gender'], use_face_contour=False,
                   ext='npz').to(cuda)
    
    # Initialize dataset,  smpl model 用來作爲參數輸入 igr， pose_dependent
    train_dataset = CapeDataset_scan(opt['data'], phase='train', smpl=model,
                                     device=cuda)

    
    
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt['training']['batch_size'], shuffle=False,
                                   num_workers=opt['training']['num_threads'], pin_memory=opt['training']['pin_memory'])
    
    # All the hand, face joints are glued to body joints for SMPL
    gt_lbs_smpl = model.lbs_weights[:, :24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:, root] += model.lbs_weights[:, i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0, 2, 1)
    # 用vitruvian pose(姿态)对 Tpose_minimal_v smpl 进行 skinning，输出 vitruvian smpl 的顶点 [1,6890,3]
    # body_neutral_v 使得 smpl 不考虑体型的影响，而采用标准的模板
    # smpl_vitruvian 用来对 igr 和 fwd 进行坐标系范围限定，bbox_min,bbox_max
    smpl_vitruvian = model.initiate_vitruvian(device=cuda,
                                              body_neutral_v=train_dataset.Tpose_minimal_v,
                                              # tpose smpl vertice [N,3]，架空 beta 的影响
                                              vitruvian_angle=opt['data']['vitruvian_angle'])
    
    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32),
                 smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0] - 0.55 * bbox_size[0], bbox_center[1] - 0.6 * bbox_size[1],
                         bbox_center[2] - 1.5 * bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0] + 0.55 * bbox_size[0], bbox_center[1] + 0.6 * bbox_size[1],
                         bbox_center[2] + 1.5 * bbox_size[2]], 0).astype(np.float32)
    
    # Initialize networks
    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'],
                           opt['model']['normalize_posemap'])
    
    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)
    texture_net = TNet(opt['model']['igr_net']).to(cuda)

    
    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    lat_vecs_inv_skin = nn.Embedding(len(train_dataset), opt['model']['inv_skin_net']['g_dim']).to(cuda)
    
    torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)
    torch.nn.init.normal_(lat_vecs_inv_skin.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['inv_skin_net']['g_dim']))
    
    print("igr_net:\n", igr_net)
    print("fwd_skin_net:\n", fwd_skin_net)
    print("inv_skin_net:\n", inv_skin_net)
    
    # Find checkpoints
    ckpt_dict = None
    model_path = '%s/ckpt_latest.pt' % ckpt_dir  # 加全部模型（inv  fwd igr  inv_lat  igr_lat）
    ckpt_dict = torch.load(model_path)

    
    # Load checkpoints
    train_igr_start_epoch = 0
    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
            if 'epoch' in ckpt_dict:
                train_igr_start_epoch = ckpt_dict['epoch']
        else:
            logging.warning("Couldn't find igr_net in checkpoints!")
        
        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            logging.warning("Couldn't find fwd_skin_net in checkpoints!")
        
        if 'inv_skin_net' in ckpt_dict:
            load_network(inv_skin_net, ckpt_dict['inv_skin_net'])
        else:
            logging.warning("Couldn't find inv_skin_net in checkpoints!")
        
        if 'lat_vecs_igr' in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            logging.warning("Couldn't find lat_vecs_igr in checkpoints!")
        
        if 'lat_vecs_inv_skin' in ckpt_dict:
            load_network(lat_vecs_inv_skin, ckpt_dict['lat_vecs_inv_skin'])
        else:
            logging.warning("Couldn't find lat_vecs_inv_skin in checkpoints!")
        if 'texture_net' in ckpt_dict:
            load_network(texture_net, ckpt_dict['texture_net'])
        else:
            print("Couldn't find texture_net in checkpoints!")

    logging.info('train data size: %s' % str(len(train_dataset)))
    
    
    
    # get only valid triangles
    train_data_loader.dataset.compute_valid_tri(inv_skin_net, model, lat_vecs_inv_skin, smpl_vitruvian)


    Pose2Cano(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
                  igr_net, lat_vecs_igr, texture_net,
                  model,  gt_lbs_smpl,
                  train_data_loader,
                  cuda)
    
    with open(os.path.join(result_dir, '../', exp_name + '.txt'), 'w') as finish_file:
        finish_file.write('Done!')
    logging.info('Finished learning geometry!')

def mergeWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Pose2Scan .'
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config file.')
    parser.add_argument('--test_dir', '-t', type=str, \
                    required=True,\
                    help='Path to test directory')
    args = parser.parse_args()

    opt = load_config(args.config, 'configs/default.yaml')

    run_pose2cano(opt)




if __name__ == '__main__':
    print(f'train_scanimate pid : {os.getpid()}')
    input()
    
    mergeWrapper()