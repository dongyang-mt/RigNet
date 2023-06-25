import glob
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.rig_parser import Skel, Info
from utils.eval_utils import getJointArr, chamfer_dist, joint2bone_chamfer_dist, bone2bone_chamfer_dist, edit_dist
from utils.vis_utils import draw_shifted_pts, draw_joints, show_obj_skel
from utils.io_utils import readPly
from utils.tree_utils import TreeNode
from utils import binvox_rw
from utils.mst_utils import loadSkel_recur, increase_cost_for_outside_bone, primMST


def get_joint_with_name(skel):
    joints = []
    names = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            joint_ = np.array(p_node.pos)
            joint_ = joint_[np.newaxis, :]
            joints.append(joint_)
            names.append(p_node.name)
            next_level += p_node.children
        this_level = next_level
    joints = np.concatenate(joints, axis=0)
    return joints, names


def load_featuresize(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    fs_dict = {}
    for li in lines:
        words = li.strip().split()
        fs_dict[words[0]] = float(words[1])
    return fs_dict


def eval_skeleton():
    info_folder = '/media/zhanxu/4T1/ModelResource_Dataset/info/'
    obj_folder = '/media/zhanxu/4T1/ModelResource_Dataset/obj_fixed/'
    featuresize_folder = '/media/zhanxu/4T1/ModelResource_Dataset/joint_featuresize/'
    test_list = np.loadtxt('/media/zhanxu/4T1/ModelResource_Dataset/test_final.txt', dtype=np.int)
    # res_folders = ['/media/zhanxu/WD/comparasion_results/v2v/raw_results/']
    # res_folders = ['/media/zhanxu/4T1/defiant_data/comparasion_results/pinocchio/converted_skel/']
    # res_folders = ['/media/zhanxu/WD/comparasion_results/l1_median']
    res_folders = ['/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift3/5e6/',
                   '/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift3/1e5/',
                   '/mnt/gypsum/home/zhanxu/Proj/rigNet/results/gcn_meanshift3/2e5/',]

    for res_folder in res_folders:
        # pred_skel_list = glob.glob(os.path.join(res_folder, '*_joint.npy'))
        # pred_skel_list = glob.glob(os.path.join(res_folder, '*.txt'))
        # chamfer_total = chamfer_joint2bone_total = chamfer_projected_total = 0.0
        chamfer_total = 0.0
        j2b_chamfer_total = 0.0
        joint_IoU_total = 0.0
        joint_precision_total = 0.0
        joint_recall_total = 0.0
        edit_total = 0.0
        b2b_chamfer_total = 0.0
        # bone_acc = 0.0
        num_invalid = 0
        for model_id in test_list:
            #print(model_id)
            # pred_joint_file = os.path.join(res_folder, '{:d}_joint.npy'.format(model_id))
            pred_skel_file = os.path.join(res_folder, '{:d}_skel.txt'.format(model_id))
            fs_file = os.path.join(featuresize_folder, '{:d}.txt'.format(model_id))
            fs_dict = load_featuresize(fs_file)
            if not os.path.exists(pred_skel_file):
                num_invalid += 1
                continue
            pred_skel = Skel(pred_skel_file)
            gt_skel = Info(os.path.join(info_folder, '{:d}.txt'.format(model_id)))
            # mesh_file = os.path.join(obj_folder, '{:d}.obj'.format(model_id))
            # pred_joint = np.load(pred_joint_file)
            # pred_joint = readPly(pred_joint_file)
            # img = draw_joints(mesh_file, pred_joint)
            # cv2.imwrite(os.path.join(res_folder, '{:s}.jpg'.format(model_id)), img[:, :, ::-1])
            pred_joint = getJointArr(pred_skel)
            gt_joint, gt_joint_name = get_joint_with_name(gt_skel)
            fs = [fs_dict[i] for i in gt_joint_name]
            fs = np.array(fs)
            # print(len(gt_joint), len(pred_joint))
            if len(pred_joint) == 0:
                num_invalid += 1
                continue

            chamfer_score = chamfer_dist(pred_joint, gt_joint)
            chamfer_total += chamfer_score
            j2b_chamfer = joint2bone_chamfer_dist(pred_skel, gt_skel)
            j2b_chamfer_total += j2b_chamfer

            dist_matrix = np.sqrt(np.sum((pred_joint[np.newaxis, ...] - gt_joint[:, np.newaxis, :]) ** 2, axis=2))
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            fs_threshod = fs[row_ind]
            joint_IoU = 2 * np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / (len(pred_joint) + len(gt_joint))
            joint_IoU_total += joint_IoU
            joint_precision = np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / len(pred_joint)
            joint_precision_total += joint_precision
            joint_recall = np.sum(dist_matrix[row_ind, col_ind] < fs_threshod) / len(gt_joint)
            joint_recall_total += joint_recall

            bone2bone_dist_ = bone2bone_chamfer_dist(pred_skel, gt_skel)
            b2b_chamfer_total += bone2bone_dist_
            # adj_gt = gt_skel.adjacent_matrix()
            # adj_pred = pred_skel.adjacent_matrix()
            # acc_ = np.sum(np.logical_and(adj_gt == 1.0, adj_pred == 1.0)) / np.sum(adj_gt == 1.0)  # the same for adj_pred
            # bone_acc += acc_
            edit_distance_ = edit_dist(pred_skel, gt_skel)
            edit_total += edit_distance_

        print(num_invalid)
        chamfer_total /= (len(test_list) - num_invalid)
        j2b_chamfer_total /= (len(test_list) - num_invalid)
        b2b_chamfer_total /= (len(test_list) - num_invalid)
        joint_precision_total /= (len(test_list) - num_invalid)
        joint_recall_total /= (len(test_list) - num_invalid)
        joint_IoU_total /= (len(test_list) - num_invalid)
        edit_total /= (len(test_list) - num_invalid)
        '''print('{:s}\n'.format(res_folder),
              '\tchamfer_distance {:.03f}%\n'.format(chamfer_total * 100),
              '\tj2b_chamfer_distance {:.03f}%\n'.format(j2b_chamfer_total * 100),
              '\tb2b_chamfer_distance {:.03f}%\n'.format(b2b_chamfer_total * 100),
              '\tjoint_IoU {:.03f}%\n'.format(joint_IoU_total * 100),
              '\tjoint_precision {:.03f}%\n'.format(joint_precision_total * 100),
              '\tjoint_recall {:.03f}%\n'.format(joint_recall_total * 100),
              '\tedit distance {:.03f}\n'.format(edit_total))'''
        print('{:.03f}% {:.03f}% {:.03f}% {:.03f}% {:.03f}% {:.03f}% {:.03f}'.format(chamfer_total * 100, j2b_chamfer_total * 100,
                                                                                     b2b_chamfer_total * 100, joint_IoU_total * 100,
                                                                                     joint_precision_total * 100, joint_recall_total * 100,
                                                                                     edit_total))


if __name__ == '__main__':
    eval_skeleton()

