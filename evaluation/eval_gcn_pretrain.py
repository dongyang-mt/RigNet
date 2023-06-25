import glob
import os
import cv2
import shutil
import numpy as np
from utils.rig_parser import Info
from utils.io_utils import readPly
from utils.eval_utils import getJointArr, chamfer_dist
from utils.vis_utils import draw_shifted_pts, draw_joints
from utils import binvox_rw
from utils.mst_utils import flip, inside_check
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans


def kmeans_cluster(pts_in, k, sample_weight=None):
    #print('k=',k)
    kms = KMeans(n_clusters=k, n_jobs=-1).fit(pts_in, sample_weight=sample_weight)
    return kms


def eval_jointnet():
    info_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/rig_info/"
    res_folders = ["/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift/best_13/",
                   "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift_2/best_3/",
                   "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift_3/best_2/"]
    for res_folder in res_folders:
        chf_dist_all = 0.0
        ply_list = glob.glob(os.path.join(res_folder, '*.ply'))
        for ply_filename in ply_list:
            model_id = ply_filename.split('/')[-1].split('.')[0]
            rig_info_filename = os.path.join(info_folder, '{:s}.txt'.format(model_id))
            shifted_pts = readPly(ply_filename)
            rig_info = Info(rig_info_filename)
            joints = getJointArr(rig_info)
            chf_dist = chamfer_dist(shifted_pts, joints)
            chf_dist_all += chf_dist
        print('{:s} {:f}'.format(res_folder, chf_dist_all / len(ply_list)))


def eval_masknet():
    mask_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/pretrain_attention/"
    res_folders = ["/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift/best_13/"]
    for res_folder in res_folders:
        pred_list = glob.glob(os.path.join(res_folder, '*_attn.npy'))
        # l1_dist_total = 0.0
        bce_loss_total = 0.0
        for pred_mask_filename in pred_list:
            model_id = pred_mask_filename.split('/')[-1].split('_')[0]
            gt_mask_filename = os.path.join(mask_folder, '{:s}.txt'.format(model_id))
            pred_mask = np.load(pred_mask_filename).squeeze()
            gt_mask = np.loadtxt(gt_mask_filename)
            gt_mask = torch.from_numpy(gt_mask).float().to(device)
            pred_mask = torch.from_numpy(pred_mask).float().to(device)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask.float())
            bce_loss_total += bce_loss
            # l1_dist = np.mean(np.abs(gt_mask-pred_mask))
            # l1_dist_total += l1_dist
        print('{:s} {:f}'.format(res_folder, bce_loss_total / len(pred_list)))


def show_results():
    mesh_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/obj/"
    info_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/rig_info/"
    vox_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/vox/"
    pred_points_folder = "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/pretrain_jointnet/best_100/"
    pred_mask_folder = "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/pretrain_masknet2/best_87/"
    ply_list = glob.glob(os.path.join(pred_points_folder, '*.ply'))
    for ply_filename in ply_list:
        model_id = ply_filename.split('/')[-1].split('.')[0]
        rig_info_filename = os.path.join(info_folder, '{:s}.txt'.format(model_id))
        pred_mask_filename = os.path.join(pred_mask_folder, '{:s}_attn.npy'.format(model_id))
        mesh_filename = os.path.join(mesh_folder, '{:s}.obj'.format(model_id))

        rig_info = Info(rig_info_filename)
        joints = getJointArr(rig_info)
        shifted_pts = readPly(ply_filename)
        pred_mask = np.load(pred_mask_filename)

        # inside check
        vox_file = os.path.join(vox_folder, '{:s}.binvox'.format(model_id))
        with open(vox_file, 'rb') as f2:
            vox = binvox_rw.read_as_3d_array(f2)
        shifted_pts, index_inside = inside_check(shifted_pts, vox)
        pred_mask = pred_mask[index_inside, :]
        img = draw_shifted_pts(mesh_filename, shifted_pts, pred_mask)
        kms = kmeans_cluster(shifted_pts, len(joints), pred_mask.squeeze(1))
        pred_joints = kms.cluster_centers_
        pred_joints, _ = flip(pred_joints)
        img = draw_shifted_pts(mesh_filename, pred_joints)


def eval_finetune_res():
    from utils.cluster_utils import meanshift_cluster, nms_meanshift
    mesh_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/obj/"
    info_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/rig_info/"
    vox_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/vox/"
    threshold = 5e-6
    res_folders = ["/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift/best_34/",
                   "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift_2/best_39/",
                   "/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift_3/best_33/"]
    for res_folder in res_folders:
        chf_dist_all = 0.0
        ply_list = glob.glob(os.path.join(res_folder, '*.ply'))
        for ply_filename in ply_list:
            model_id = ply_filename.split('/')[-1].split('.')[0]
            attn_filename = ply_filename.replace('.ply', '_attn.npy')
            mesh_filename = os.path.join(mesh_folder, '{:s}.obj'.format(model_id))
            vox_filename = os.path.join(vox_folder, '{:s}.obj'.format(model_id))
            attn = np.load(attn_filename)
            vox_file = os.path.join(vox_folder, '{:s}.binvox'.format(model_id))
            with open(vox_file, 'rb') as f2:
                vox = binvox_rw.read_as_3d_array(f2)
            bandwidth = np.load(ply_filename.replace('.ply', '_bandwidth.npy'))[0]
            rig_info_filename = os.path.join(info_folder, '{:s}.txt'.format(model_id))
            shifted_pts = readPly(ply_filename)
            img = draw_shifted_pts(mesh_filename, shifted_pts, weights=attn)
            cv2.imwrite(os.path.join(res_folder, "{:s}_pts.png".format(model_id)), img[:,:,::-1])

            shifted_pts, index_inside = inside_check(shifted_pts, vox)
            attn = attn[index_inside, :]
            shifted_pts = shifted_pts[attn.squeeze() > 1e-3]
            attn = attn[attn.squeeze() > 1e-3]
            # symmetrize points by reflecting
            shifted_pts_reflect = shifted_pts * np.array([[-1, 1, 1]])
            shifted_pts = np.concatenate((shifted_pts, shifted_pts_reflect), axis=0)
            attn = np.tile(attn, (2, 1))
            shifted_pts = meanshift_cluster(shifted_pts, bandwidth, attn)

            Y_dist = np.sum(((shifted_pts[np.newaxis, ...] - shifted_pts[:, np.newaxis, :]) ** 2), axis=2)
            density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
            density = np.sum(density, axis=0)
            density_sum = np.sum(density)
            shifted_pts = shifted_pts[density / density_sum > threshold]
            attn = attn[density / density_sum > threshold][:, 0]
            density = density[density / density_sum > threshold]

            pred_joints = nms_meanshift(shifted_pts, density, bandwidth)
            pred_joints, _ = flip(pred_joints)
            img = draw_joints(mesh_filename, pred_joints)
            cv2.imwrite(os.path.join(res_folder, "{:s}_joint.png".format(model_id)), img[:,:,::-1])
            np.save(os.path.join(res_folder, "{:s}_joint.npy".format(model_id)), pred_joints)

            rig_info = Info(rig_info_filename)
            joints_gt = getJointArr(rig_info)
            chf_dist = chamfer_dist(pred_joints, joints_gt)
            chf_dist_all += chf_dist
        print('{:s} {:f}'.format(res_folder, chf_dist_all / len(ply_list)))


if __name__ == '__main__':
    #eval_jointnet()
    #eval_masknet()
    #show_results()
    eval_finetune_res()

