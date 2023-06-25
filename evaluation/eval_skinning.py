import sys
sys.path.append("./")
import os
import glob
import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from utils.io_utils import output_rigging
from utils.rig_parser import Info


def eval_our_model():
    info_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/rig_info_remesh/"
    obj_remesh_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/obj_remesh/"
    obj_fixed_folder = "/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/obj/"
    output_folders = ["../results/skinnet_dglf1ring/"]
    test_model_list = np.loadtxt('/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/test_final.txt', dtype=np.int)

    for output_folder in output_folders:
        prec_total = []
        rec_total = []
        l1_dist_total = []
        l1_max_total = []
        for model_id in test_model_list:
            #model_id = 783
            #print(model_id)
            info = Info(os.path.join(info_folder, "{:d}.txt".format(model_id)))
            mesh_remesh = o3d.io.read_triangle_mesh(os.path.join(obj_remesh_folder, "{:d}.obj".format(model_id)))
            mesh_remesh_vert = np.asarray(mesh_remesh.vertices)
            mesh_fixed = o3d.io.read_triangle_mesh(os.path.join(obj_fixed_folder, "{:d}.obj".format(model_id)))
            mesh_fixed_vert = np.asarray(mesh_fixed.vertices)
            vertice_remesh_distance = np.sqrt(np.sum((mesh_remesh_vert[np.newaxis, ...] - mesh_fixed_vert[:, np.newaxis, :]) ** 2, axis=2))
            vertice_raw_id = np.argmin(vertice_remesh_distance, axis=1)
            with open(os.path.join(output_folder, '{:d}_bone_names.txt'.format(model_id)), 'r') as fin:
                lines = fin.readlines()
            bone_names = []
            for li in lines:
                bone_names.append(li.strip().split())
            skin_pred_full = np.load(os.path.join(output_folder, "{:d}_full_pred.npy".format(model_id)))

            joint_names = list(set([bone_name[0] for bone_name in bone_names]))
            skin_pred_full_new = np.zeros((len(skin_pred_full), len(joint_names)))
            for i in range(len(joint_names)):
                joint_name = joint_names[i]
                for j in range(len(bone_names)):
                    if bone_names[j][0] == joint_name:
                        skin_pred_full_new[:, i] += skin_pred_full[:, j]
            skin_pred_full = skin_pred_full_new

            skin_gt_full = np.zeros_like(skin_pred_full)
            for v in range(len(info.joint_skin)):
                skin = info.joint_skin[v]
                for w in np.arange(1, len(skin), 2):
                    col_id = joint_names.index(skin[w])
                    skin_gt_full[v, col_id] = skin[w+1]

            skin_pred_full = skin_pred_full[vertice_raw_id]
            skin_gt_full = skin_gt_full[vertice_raw_id]
            skin_pred_full = skin_pred_full[np.abs(np.sum(skin_gt_full, axis=1)-1) < 1e-2]
            skin_gt_full = skin_gt_full[np.abs(np.sum(skin_gt_full, axis=1) - 1) < 1e-2]

            precision = np.sum(np.logical_and(skin_pred_full>0, skin_gt_full>0)) / (np.sum(skin_pred_full>0) + 1e-10)
            recall = np.sum(np.logical_and(skin_pred_full>0, skin_gt_full>0)) / (np.sum(skin_gt_full>0) + 1e-10)
            mean_l1_dist = np.sum(np.abs(skin_pred_full - skin_gt_full)) / len(skin_pred_full)
            max_l1_dist = np.max(np.sum(np.abs(skin_pred_full - skin_gt_full), axis=1))
            prec_total.append(precision)
            rec_total.append(recall)
            l1_dist_total.append(mean_l1_dist)
            l1_max_total.append(max_l1_dist)
        prec_total = np.array(prec_total)
        rec_total = np.array(rec_total)
        l1_dist_total = np.array(l1_dist_total)
        l1_max_total = np.array(l1_max_total)
        print('result name: ', output_folder,
              'precision: ', np.mean(prec_total),
              'recall: ', np.mean(rec_total),
              'avg_l1_dist: ', np.mean(l1_dist_total),
              'max_l1_dist: ', np.mean(l1_max_total))


def eval_neural_skinning_model():
    info_folder = "/media/zhanxu/4T1/ModelResource_Dataset/info_remesh/"
    obj_remesh_folder = "/media/zhanxu/4T1/ModelResource_Dataset/obj_remesh/"
    obj_fixed_folder = "/media/zhanxu/4T1/ModelResource_Dataset/obj_fixed/"
    output_folder = "../results/neural_skinning/"
    test_model_list = np.loadtxt('/media/zhanxu/4T1/ModelResource_Dataset/test_final.txt', dtype=int)

    prec_total = []
    rec_total = []
    l1_dist_total = []
    l1_max_total = []
    for model_id in test_model_list:
        print(model_id)
        info = Info(os.path.join(info_folder, "{:d}.txt".format(model_id)))
        mesh_remesh = Mesh_obj(os.path.join(obj_remesh_folder, "{:d}.obj".format(model_id)))
        mesh_fixed = Mesh_obj(os.path.join(obj_fixed_folder, "{:d}.obj".format(model_id)))
        vertice_remesh_distance = np.sqrt(
            np.sum((mesh_remesh.v[np.newaxis, ...] - mesh_fixed.v[:, np.newaxis, :]) ** 2, axis=2))
        vertice_raw_id = np.argmin(vertice_remesh_distance, axis=1)
        with open(os.path.join(output_folder, '{:d}_bone_names.txt'.format(model_id)), 'r') as fin:
            lines = fin.readlines()
        bone_names = []
        for li in lines:
            bone_names.append(li.strip().split())
        skin_pred_full = np.load(os.path.join(output_folder, "{:d}_full_pred.npy".format(model_id)))

        joint_names = list(set([bone_name[0] for bone_name in bone_names]))
        skin_pred_full_new = np.zeros((len(skin_pred_full), len(joint_names)))
        for i in range(len(joint_names)):
            joint_name = joint_names[i]
            for j in range(len(bone_names)):
                if bone_names[j][0] == joint_name:
                    skin_pred_full_new[:, i] += skin_pred_full[:, j]
        skin_pred_full = skin_pred_full_new

        skin_gt_full = np.zeros_like(skin_pred_full)
        for v in range(len(info.joint_skin)):
            skin = info.joint_skin[v]
            for w in np.arange(1, len(skin), 2):
                col_id = joint_names.index(skin[w])
                skin_gt_full[v, col_id] = skin[w + 1]

        skin_pred_full = skin_pred_full[vertice_raw_id]
        skin_gt_full = skin_gt_full[vertice_raw_id]
        skin_pred_full = skin_pred_full[np.abs(np.sum(skin_gt_full, axis=1) - 1) < 1e-2]
        skin_gt_full = skin_gt_full[np.abs(np.sum(skin_gt_full, axis=1) - 1) < 1e-2]

        precision = np.sum(np.logical_and(skin_pred_full > 0, skin_gt_full > 0)) / (np.sum(skin_pred_full > 0) + 1e-10)
        recall = np.sum(np.logical_and(skin_pred_full > 0, skin_gt_full > 0)) / (np.sum(skin_gt_full > 0) + 1e-10)
        mean_l1_dist = np.sum(np.abs(skin_pred_full - skin_gt_full)) / len(skin_pred_full)
        max_l1_dist = np.max(np.sum(np.abs(skin_pred_full - skin_gt_full), axis=1))
        prec_total.append(precision)
        rec_total.append(recall)
        l1_dist_total.append(mean_l1_dist)
        l1_max_total.append(max_l1_dist)
    prec_total = np.array(prec_total)
    rec_total = np.array(rec_total)
    l1_dist_total = np.array(l1_dist_total)
    l1_max_total = np.array(l1_max_total)
    print('precision: ', np.mean(prec_total),
          'recall: ', np.mean(rec_total),
          'avg_l1_dist: ', np.mean(l1_dist_total),
          'max_l1_dist: ', np.mean(l1_max_total))


def eval_geodesic_voxel_results():
    maya_results_folder = "/media/zhanxu/4T1/ModelResource_Dataset/geodesic_voxel_results/"
    test_model_list = np.loadtxt('/media/zhanxu/4T1/ModelResource_Dataset/test_final.txt', dtype=np.int)
    info_folder = "/media/zhanxu/4T1/ModelResource_Dataset/info/"
    prec_total = []
    rec_total = []
    l1_dist_total = []
    l1_max_total = []
    for model_id in test_model_list:
        print(model_id)
        maya_filename = os.path.join(maya_results_folder, '{:d}.txt'.format(model_id))
        info_gt = Info(os.path.join(info_folder, "{:d}.txt".format(model_id)))
        info_pred = Info(maya_filename)

        joint_names = list(info_gt.get_joint_dict().keys())
        skin_gt_full = np.zeros((len(info_gt.joint_skin), len(joint_names)))
        skin_pred_full = np.zeros((len(info_pred.joint_skin), len(joint_names)))

        for v in range(len(info_gt.joint_skin)):
            skin = info_gt.joint_skin[v]
            for w in np.arange(1, len(skin), 2):
                col_id = joint_names.index(skin[w])
                skin_gt_full[v, col_id] = skin[w + 1]

        for v in range(len(info_pred.joint_skin)):
            skin = info_pred.joint_skin[v]
            for w in np.arange(1, len(skin), 2):
                col_id = joint_names.index(skin[w])
                skin_pred_full[v, col_id] = skin[w + 1]

        skin_pred_full[skin_pred_full < 0.35] = 0
        skin_pred_full = skin_pred_full / (np.sum(skin_pred_full, axis=1, keepdims=True) + 1e-10)

        skin_pred_full = skin_pred_full[np.abs(np.sum(skin_gt_full, axis=1) - 1) < 1e-2]
        skin_gt_full = skin_gt_full[np.abs(np.sum(skin_gt_full, axis=1) - 1) < 1e-2]

        precision = np.sum(np.logical_and(skin_pred_full > 0, skin_gt_full > 0)) / (np.sum(skin_pred_full > 0) + 1e-10)
        recall = np.sum(np.logical_and(skin_pred_full > 0, skin_gt_full > 0)) / (np.sum(skin_gt_full > 0) + 1e-10)
        mean_l1_dist = np.sum(np.abs(skin_pred_full - skin_gt_full)) / len(skin_pred_full)
        max_l1_dist = np.max(np.sum(np.abs(skin_pred_full - skin_gt_full), axis=1))
        prec_total.append(precision)
        rec_total.append(recall)
        l1_dist_total.append(mean_l1_dist)
        l1_max_total.append(max_l1_dist)
    prec_total = np.array(prec_total)
    rec_total = np.array(rec_total)
    l1_dist_total = np.array(l1_dist_total)
    l1_max_total = np.array(l1_max_total)
    print('precision: ', np.mean(prec_total),
          'recall: ', np.mean(rec_total),
          'mean_l1_dist: ', np.mean(l1_dist_total),
          'max_l1_dist: ', np.mean(l1_max_total))


def eval_our_model_on_predicted_skeleton(best_threshold):
    global device
    from utils.rigging_parser.skel_parser import Skel
    num_nearest_bone = 5
    working_folder = "/mnt/defiant/home/zhan/Proj/model_resource_v2/p2p_data/skinning_data/test_my/"
    checkpoint = torch.load('/mnt/gypsum/home/zhanxu/Proj/point_skeleton/checkpoints/skin17/model_best.pth.tar')
    model = SKINNET2(two_branches=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()  # switch to test mode

    model_list = glob.glob(os.path.join(working_folder, '*_v.txt'))
    for v_filename in model_list:
        #v_filename = os.path.join(working_folder, '9479_v.txt')
        model_id = v_filename.split('/')[-1].split('_')[0]
        print(model_id)
        v_np = np.loadtxt(v_filename)
        v = torch.from_numpy(v_np).float()
        tpl_e_np = np.loadtxt(os.path.join(working_folder, '{:s}_tpl_e.txt'.format(model_id))).T
        euc_e = np.loadtxt(os.path.join(working_folder, '{:s}_euc_e.txt'.format(model_id))).T
        tpl_e = torch.from_numpy(tpl_e_np).long()
        euc_e = torch.from_numpy(euc_e).long()
        tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
        euc_e, _ = add_self_loops(euc_e, num_nodes=v.size(0))
        batch = np.zeros(len(v))
        batch = torch.from_numpy(batch).long()
        skin_input, skin_nn, _, loss_mask = load_skin(os.path.join(working_folder, '{:s}_skin.txt'.format(model_id)))
        skin_input = torch.from_numpy(skin_input).float()
        skin_nn_tensor = torch.from_numpy(skin_nn).long()
        skin_label = torch.from_numpy(np.zeros((len(skin_input), num_nearest_bone))).float()
        loss_mask_tensor = torch.from_numpy(loss_mask).long()
        num_skin = skin_input.shape[0]
        data = Data(x=v[:, 3:6], pos=v[:, 0:3], batch=batch, skin_input=skin_input, skin_label=skin_label,
                    skin_nn=skin_nn_tensor, loss_mask=loss_mask_tensor, num_skin=[num_skin],
                    tpl_edge_index=tpl_e, euc_edge_index=euc_e).to(device)

        with torch.no_grad():
            skel_filename = os.path.join(working_folder, '{:s}_skel.txt'.format(model_id))
            skin_reg_pred, skin_cls_pred = model(data)
            loss_mask = data.loss_mask.float()
            skin_cls_pred = torch.softmax(skin_cls_pred, dim=1)
            skin_cls_pred = skin_cls_pred * loss_mask
            #skinning_pred = skin_cls_pred.data.cpu().numpy()

            skel = Skel(skel_filename)
            full_bones, _, _ = get_joint_bone_with_leaf(skel.root)
            skin_pred_full = np.zeros((len(skin_cls_pred), len(full_bones)))
            for v in range(len(skin_cls_pred)):
                for nn_id in range(len(skin_nn[v, :])):
                    skin_pred_full[v, skin_nn[v, nn_id]] = skin_cls_pred[v, nn_id]
            skin_pred_full = post_filter(skin_pred_full, tpl_e_np, v_np, num_ring=2)
            #for v in range(len(skin_pred_full)):
            #    skinning_pred[v, :] = skin_pred_full[v, skin_nn[v, :]]

            #skin_cls_pred = torch.nn.functional.threshold(skin_cls_pred, best_threshold, 0.0)

            skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.25] = 0.0
            skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
            output_rigging(skel_filename, skin_pred_full, None, working_folder + 'rig/', model_id)


if __name__ == '__main__':
    eval_our_model()
    #eval_our_model_on_predicted_skeleton()
    #eval_neural_skinning_model()
    #eval_geodesic_voxel_results()