import glob
import os
import numpy as np
from utils import binvox_rw
from utils.tree_utils import TreeNode
from utils.vis_utils import show_obj_skel
from utils.rig_parser import Info
from utils.eval_utils import bone2bone_chamfer_dist, edit_dist
from utils.mst_utils import loadSkel_recur, primMST


def eval_pair_cls():
    res_folders = ['/mnt/gypsum/home/zhanxu/Proj/rigNet/results/pair_full_3/best_296/']
    #mesh_folder = '/media/zhanxu/4T1/ModelResource_Dataset/obj_remesh/'
    skel_folder = '/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/rig_info/'
    for res_folder in res_folders:
        print(res_folder)
        res_list = glob.glob(os.path.join(res_folder, '*.npy'))
        acc = []
        edit_distance = []
        bone2bone_distance = []
        for embedding_file in res_list:
            model_id = embedding_file.split('/')[-1].split('_')[0]
            cost_matrix = np.load(embedding_file)
            #mesh_file = os.path.join(mesh_folder, '{:s}.obj'.format(model_id))
            #mesh = Mesh_obj(mesh_file)
            rig_info = Info(os.path.join(skel_folder, '{:s}.txt'.format(model_id)))
            joint_pos = rig_info.get_joint_dict()
            joint_name_list = list(joint_pos.keys())
            joint_pos_list = np.array(list(joint_pos.values()))
            cost_matrix = cost_matrix + cost_matrix.transpose()
            parent, key = primMST(cost_matrix, 0)
            rig_info_res = Info()
            for i in range(len(parent)):
                if parent[i] == -1:
                    rig_info_res.root = TreeNode(joint_name_list[i], joint_pos_list[i])
                    break
            loadSkel_recur(rig_info_res.root, i, joint_name_list, joint_pos_list, parent)
            #img = show_obj_skel(mesh, rig_info_res.root)
            #cv2.imwrite(embedding_file.replace('_cost.npy', '.jpg'), img)

            adj_gt = rig_info.adjacent_matrix()
            adj_pred = rig_info_res.adjacent_matrix()
            acc_ = np.sum(np.logical_and(adj_gt == 1.0, adj_pred == 1.0)) / np.sum(adj_pred == 1.0)  # the same for adj_pred
            acc.append(acc_)
            ## get edit distance
            edit_distance_ = edit_dist(rig_info, rig_info_res)
            edit_distance.append(edit_distance_)
            ## get bone2bone chamfer distance
            bone2bone_dist_ = bone2bone_chamfer_dist(rig_info, rig_info_res)
            bone2bone_distance.append(bone2bone_dist_)
        print('accuracy: {:f}'.format(np.mean(acc)))
        print('edit distance: {:f}'.format(np.mean(edit_distance)))
        print('bone2bone_chamfer: {:f}'.format(np.mean(bone2bone_distance)))


if __name__ == '__main__':
    eval_pair_cls()
