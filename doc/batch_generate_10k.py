"""
批量生成10K静态姿态
支持多GPU并行、自定义输出路径、以文本命名文件
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
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import re
import shutil
from tqdm import tqdm


def sanitize_filename(text, max_length=200):
    """
    将文本转换为合法的文件名
    """
    # 移除或替换非法字符
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # 替换空格为下划线
    text = text.replace(' ', '_')
    # 移除开头的"A_person_"
    if text.startswith('A_person_'):
        text = text[9:]
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length]
    return text


def plot_single_pose_3d(joints, kinematic_tree, title="", save_path=None, figsize=(8, 8)):
    """
    绘制单帧3D姿态
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
    ax.set_title(title, fontsize=10, pad=10)
    
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
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
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
    
    # 临时结果目录
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # 最终结果目录
    final_dir = args.final_dir
    os.makedirs(final_dir, exist_ok=True)

    print(f"加载模型...")
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

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

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ae = ae.to(device)
    ema_mardm = ema_mardm.to(device)
    length_estimator = length_estimator.to(device)

    ae.eval()
    ema_mardm.eval()
    length_estimator.eval()
    
    print(f"✅ 模型加载完成，使用设备: {device}")
    
    #################################################################################
    #                                     Sampling                                  #
    #################################################################################
    # 读取文本提示词
    prompt_list = []
    with open(args.text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompt_list.append(line)
    
    print(f"📝 共读取 {len(prompt_list)} 条文本描述")
    print(f"📁 临时目录: {temp_dir}")
    print(f"📁 最终目录: {final_dir}")
    print(f"🎯 序列长度: {args.sequence_length} 帧")
    print(f"🎯 提取帧索引: {args.frame_index}")
    print(f"=" * 80)
    
    # 使用固定的序列长度
    token_lens = torch.LongTensor([args.sequence_length // 4] * len(prompt_list))
    token_lens = token_lens.to(device).long()
    m_length = token_lens * 4
    
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain
    
    # 批量生成
    batch_size = args.batch_size
    num_batches = (len(prompt_list) + batch_size - 1) // batch_size
    
    success_count = 0
    error_count = 0
    
    for batch_idx in tqdm(range(num_batches), desc=f"GPU {args.gpu_id} 生成进度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompt_list))
        batch_prompts = prompt_list[start_idx:end_idx]
        batch_token_lens = token_lens[start_idx:end_idx]
        batch_m_length = m_length[start_idx:end_idx]
        
        try:
            with torch.no_grad():
                pred_latents = ema_mardm.generate(batch_prompts, batch_token_lens, args.time_steps, args.cfg,
                                                  temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder)
                pred_motions = ae.decode(pred_latents)
                pred_motions = pred_motions.detach().cpu().numpy()
                data = pred_motions * std + mean
            
            # 处理每个样本
            for i, (caption, joint_data) in enumerate(zip(batch_prompts, data)):
                try:
                    # 截取到指定长度
                    joint_data = joint_data[:batch_m_length[i]]
                    
                    # 转换为XYZ坐标
                    joint_sequence = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
                    
                    # 提取指定帧
                    if args.frame_index >= len(joint_sequence):
                        frame_idx = len(joint_sequence) - 1
                    else:
                        frame_idx = args.frame_index
                    
                    single_pose = joint_sequence[frame_idx]  # (22, 3)
                    
                    # 生成文件名
                    safe_name = sanitize_filename(caption)
                    
                    # 保存到临时目录（用于缓存）
                    temp_npy = pjoin(temp_dir, f"{safe_name}_temp.npy")
                    np.save(temp_npy, single_pose)
                    
                    # 保存到最终目录
                    final_npy = pjoin(final_dir, f"{safe_name}.npy")
                    final_json = pjoin(final_dir, f"{safe_name}.json")
                    final_png = pjoin(final_dir, f"{safe_name}.png")
                    
                    # 保存NPY
                    np.save(final_npy, single_pose)
                    
                    # 保存JSON
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
                    with open(final_json, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    # 保存PNG
                    plot_single_pose_3d(single_pose, kinematic_chain, 
                                       title=caption if len(caption) < 50 else caption[:47] + "...",
                                       save_path=final_png)
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\n❌ 处理失败: {caption[:50]}... 错误: {e}")
                    continue
        
        except Exception as e:
            error_count += len(batch_prompts)
            print(f"\n❌ 批次 {batch_idx} 生成失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"✅ GPU {args.gpu_id} 完成！")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {error_count} 个")
    print(f"   总计: {len(prompt_list)} 个")
    print(f"📁 最终结果保存在: {final_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量生成10K静态姿态')
    
    # 模型参数
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    # 生成参数
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument('--hard_pseudo_reorder', action="store_true")
    
    # 输入输出
    parser.add_argument('--text_path', type=str, required=True, help='文本描述文件路径')
    parser.add_argument('--temp_dir', type=str, default='./SinglePose_temp', help='临时目录')
    parser.add_argument('--final_dir', type=str, default='./SinglePose', help='最终结果目录')
    
    # 姿态参数
    parser.add_argument("--sequence_length", default=16, type=int)
    parser.add_argument("--frame_index", default=-1, type=int)
    
    # GPU和批处理
    parser.add_argument("--gpu_id", type=int, default=0, help='使用的GPU ID')
    parser.add_argument("--batch_size", type=int, default=8, help='批处理大小')
    
    args = parser.parse_args()
    
    # 处理frame_index
    if args.frame_index == -1:
        args.frame_index = args.sequence_length // 2
    elif args.frame_index == -2:
        args.frame_index = args.sequence_length - 1
    elif args.frame_index < 0:
        args.frame_index = args.sequence_length + args.frame_index
    
    main(args)


