"""
单帧3D姿态生成脚本
根据文本描述生成静态的单帧3D姿态（而不是运动序列）

用法示例:
python sample_single_pose.py --text_prompt "A person raising both hands" --dataset_name t2m
"""

import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
from models.AE import AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, kit_kinematic_chain, t2m_kinematic_chain
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


def plot_single_pose_3d(joints, kinematic_tree, title="", save_path=None, figsize=(8, 8)):
    """
    绘制单帧3D姿态
    
    Args:
        joints: (22, 3) 单帧关节点坐标
        kinematic_tree: 骨架连接关系
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴
    RADIUS = 4
    ax.set_xlim3d([-RADIUS / 2, RADIUS / 2])
    ax.set_ylim3d([0, RADIUS])
    ax.set_zlim3d([-RADIUS / 3., RADIUS * 2 / 3.])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, pad=20)
    
    # 调整视角
    ax.view_init(elev=110, azim=-90)
    ax.dist = 7.5
    
    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
               c='red', marker='o', s=50, alpha=0.8)
    
    # 定义不同部位的颜色
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    # 绘制骨架连接
    for i, chain in enumerate(kinematic_tree):
        if i < len(colors):
            linewidth = 4.0
        else:
            linewidth = 2.0
        color = colors[i % len(colors)]
        
        for j in range(len(chain) - 1):
            parent_idx = chain[j]
            child_idx = chain[j + 1]
            ax.plot([joints[parent_idx, 0], joints[child_idx, 0]],
                   [joints[parent_idx, 1], joints[child_idx, 1]],
                   [joints[parent_idx, 2], joints[child_idx, 2]],
                   color=color, linewidth=linewidth, alpha=0.8)
    
    # 添加关节点标签（可选）
    # for idx in range(len(joints)):
    #     ax.text(joints[idx, 0], joints[idx, 1], joints[idx, 2], 
    #             str(idx), fontsize=8, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存图片: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    #################################################################################
    #                                       Data                                    #
    #################################################################################
    dim_pose = 64 if args.dataset_name == 'kit' or args.dataset_name =='eval_kit'else 67
    nb_joints = 21 if args.dataset_name == 'kit' or args.dataset_name =='eval_kit' else 22
    data_root = f'{args.dataset_dir}/KIT-ML/' if args.dataset_name == 'kit' or args.dataset_name =='eval_kit' else f'{args.dataset_dir}/HumanML3D/'
    
    if args.dataset_name =="t2m":
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
    elif args.dataset_name =="kit":
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
    elif args.dataset_name =="eval_t2m":
        mean =np.load(pjoin('./utils/eval_mean_std/t2m','eval_mean.npy'))
        std =np.load(pjoin('./utils/eval_mean_std/t2m','eval_std.npy'))
    elif args.dataset_name =="eval_kit":
        mean =np.load(pjoin('./utils/eval_mean_std/kit','eval_mean.npy'))
        std =np.load(pjoin('./utils/eval_mean_std/kit','eval_std.npy'))
    
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    result_dir = pjoin('./generation', args.name + '_' + args.dataset_name + '_single_pose')
    os.makedirs(result_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])
    if torch.cuda.is_available():
        ae=ae.cuda()

    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text')
    model_dir = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    assert len(unexpected_keys2) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys2])

    length_estimator = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location='cpu')
    length_estimator.load_state_dict(ckpt['estimator'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ae.to(device)
    ema_mardm = ema_mardm.to(device)
    length_estimator = length_estimator.to(device)

    ae.eval()
    ema_mardm.eval()
    length_estimator.eval()
    
    #################################################################################
    #                                     Sampling                                  #
    #################################################################################
    prompt_list = []
    
    if args.text_prompt != "":
        prompt_list.append(args.text_prompt)
    elif args.text_path != "":
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    prompt_list.append(line)
    else:
        raise ValueError("需要提供 --text_prompt 或 --text_path 参数！")

    # 使用固定的短序列长度（最少4帧）
    # 生成后我们只取指定帧
    token_lens = torch.LongTensor([args.sequence_length // 4] * len(prompt_list))
    token_lens = token_lens.to(device).long()
    m_length = token_lens * 4
    
    captions = prompt_list
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain

    print(f"📝 文本提示词: {captions}")
    print(f"🎯 提取帧索引: {args.frame_index} (从生成的{args.sequence_length}帧序列中)")
    print(f"🔄 重复次数: {args.repeat_times}")
    print(f"=" * 60)

    all_results = []

    for r in range(args.repeat_times):
        print(f"\n-->重复 {r+1}/{args.repeat_times}")
        with torch.no_grad():
            pred_latents = ema_mardm.generate(captions, token_lens, args.time_steps, args.cfg,
                                              temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder)
            pred_motions = ae.decode(pred_latents)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = pred_motions * std + mean

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            s_path = pjoin(result_dir, str(k))
            os.makedirs(s_path, exist_ok=True)
            
            # 截取到指定长度
            joint_data = joint_data[:m_length[k]]
            
            # 转换为XYZ坐标
            joint_sequence = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            
            # 提取指定帧
            if args.frame_index >= len(joint_sequence):
                print(f"⚠️  警告: 帧索引 {args.frame_index} 超出范围 (序列长度: {len(joint_sequence)})，使用最后一帧")
                frame_idx = len(joint_sequence) - 1
            else:
                frame_idx = args.frame_index
            
            single_pose = joint_sequence[frame_idx]  # (22, 3)
            
            print(f"  样本 {k}: \"{caption}\"")
            print(f"  - 姿态形状: {single_pose.shape}")
            print(f"  - 提取帧: {frame_idx}/{len(joint_sequence)}")
            
            # 保存文件名
            base_name = f"caption:{caption[:30]}_sample{k}_repeat{r}_frame{frame_idx}"
            
            # 保存为NPY格式
            npy_path = pjoin(s_path, base_name + ".npy")
            np.save(npy_path, single_pose)
            print(f"  ✅ NPY文件: {npy_path}")
            
            # 保存为JSON格式（方便阅读）
            if args.save_json:
                json_data = {
                    "caption": caption,
                    "frame_index": frame_idx,
                    "num_joints": nb_joints,
                    "joints": single_pose.tolist(),
                    "joint_names": [
                        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
                        "right_foot", "neck", "left_collar", "right_collar", "head",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist"
                    ] if nb_joints == 22 else None
                }
                json_path = pjoin(s_path, base_name + ".json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"  ✅ JSON文件: {json_path}")
            
            # 可视化并保存图片
            if args.save_image:
                img_path = pjoin(s_path, base_name + ".png")
                plot_single_pose_3d(single_pose, kinematic_chain, 
                                   title=f"{caption}\n(Frame {frame_idx})",
                                   save_path=img_path)
            
            # 收集结果
            all_results.append({
                'caption': caption,
                'frame_index': frame_idx,
                'pose': single_pose,
                'npy_path': npy_path
            })
            
            print()
    
    print("=" * 60)
    print(f"✅ 完成！共生成 {len(all_results)} 个单帧姿态")
    print(f"📁 保存目录: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从文本生成单帧3D姿态')
    
    # 模型参数
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m', 
                       choices=['t2m', 'kit', 'eval_t2m', 'eval_kit'])
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    # 生成参数
    parser.add_argument("--seed", type=int, default=3407, 
                       help="随机种子")
    parser.add_argument("--time_steps", default=18, type=int,
                       help="扩散步数")
    parser.add_argument("--cfg", default=4.5, type=float,
                       help="Classifier-free guidance强度")
    parser.add_argument("--temperature", default=1, type=float,
                       help="采样温度")
    parser.add_argument('--hard_pseudo_reorder', action="store_true",
                       help="使用硬性伪重排序")
    
    # 输入参数
    parser.add_argument('--text_prompt', default='', type=str,
                       help='单个文本提示词，例如: "A person raising both hands"')
    parser.add_argument('--text_path', type=str, default="",
                       help='包含多个文本提示词的文件路径（每行一个）')
    
    # 姿态提取参数
    parser.add_argument("--sequence_length", default=16, type=int,
                       help="生成的序列长度（帧数），必须是4的倍数。生成短序列更快。")
    parser.add_argument("--frame_index", default=-1, type=int,
                       help="从生成序列中提取的帧索引。-1表示中间帧，0表示第一帧，-2表示最后一帧")
    parser.add_argument("--repeat_times", default=3, type=int,
                       help="为每个提示词生成多少个不同的姿态")
    
    # 输出参数
    parser.add_argument('--save_json', action='store_true',
                       help='是否保存JSON格式（方便查看坐标）')
    parser.add_argument('--save_image', action='store_true',
                       help='是否保存PNG图片（可视化）')
    
    args = parser.parse_args()
    
    # 参数验证和处理
    if args.sequence_length % 4 != 0:
        raise ValueError(f"sequence_length 必须是4的倍数，当前值: {args.sequence_length}")
    
    if args.sequence_length < 4:
        raise ValueError(f"sequence_length 必须至少为4，当前值: {args.sequence_length}")
    
    # 处理frame_index
    if args.frame_index == -1:
        # 默认使用中间帧
        args.frame_index = args.sequence_length // 2
    elif args.frame_index == -2:
        # 最后一帧
        args.frame_index = args.sequence_length - 1
    elif args.frame_index < 0:
        # 负索引
        args.frame_index = args.sequence_length + args.frame_index
    
    if args.frame_index < 0 or args.frame_index >= args.sequence_length:
        raise ValueError(f"frame_index {args.frame_index} 超出范围 [0, {args.sequence_length})")
    
    main(args)

