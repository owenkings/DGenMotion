"""
快速评估脚本 - 用于快速对比不同优化方案
只运行3次迭代（而不是20次），评估部分测试集
适合快速验证改动效果，不适合正式评估
"""
import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.AE import AE_models
from models.MARDM import MARDM_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
from utils.eval_utils import evaluation_mardm
import argparse
import json
from datetime import datetime


def quick_eval_mardm(args, model_name, model_type):
    """快速评估MARDM模型"""
    print(f"\n{'='*80}")
    print(f"快速评估: {model_name}")
    print(f"{'='*80}")
    
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    #################################################################################
    #                                    Eval Data                                  #
    #################################################################################
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
    
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = 196
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    split_file = pjoin(data_root, 'test.txt')
    
    # 使用较小的batch_size以加快速度
    eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, args.dataset_name, 
                                      motion_dir, text_dir, 4, max_motion_length, 20, evaluation=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=2, drop_last=True,
                             collate_fn=collate_fn, shuffle=True)
    
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, model_name, 'model')
    
    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), 
                     map_location='cpu')
    ae.load_state_dict(ckpt['ae'])
    
    ema_mardm = MARDM_models[model_type](ae_dim=ae.output_emb_width, cond_mode='text')
    model_file = os.path.join(model_dir, 'latest.tar')
    checkpoint = torch.load(model_file, map_location='cpu')
    missing_keys, unexpected_keys = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_wrapper = Evaluators(args.dataset_name, device=device)
    
    ae.eval()
    ae.to(device)
    ema_mardm.eval()
    ema_mardm.to(device)
    
    #################################################################################
    #                              Quick Evaluation Loop                            #
    #################################################################################
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []
    
    # 只运行3次迭代（而不是20次）
    repeat_time = args.repeat_times
    print(f"运行 {repeat_time} 次迭代（快速模式）")
    
    for i in range(repeat_time):
        print(f"\n迭代 {i+1}/{repeat_time}")
        with torch.no_grad():
            best_fid, best_div, best_top1, best_top2, best_top3 = 1000, 0, 0, 0, 0
            best_matching, best_mm, clip_score = 100, 0, -1
            
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score, writer, save_now = evaluation_mardm(
                model_file, eval_loader, ema_mardm, ae, None, i, best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, 
                train_mean=mean, train_std=std, time_steps=args.time_steps, cond_scale=args.cfg, 
                temperature=args.temperature, cal_mm=args.cal_mm, draw=False, 
                hard_pseudo_reorder=args.hard_pseudo_reorder)
        
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mm.append(best_mm)
        clip_scores.append(clip_score)
        
        # 实时显示进度
        print(f"  FID: {best_fid:.4f}, R-Precision TOP1: {best_top1:.4f}, CLIP: {clip_score:.4f}")
    
    # 计算统计
    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    clip_scores = np.array(clip_scores)
    
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'dataset': args.dataset_name,
        'repeat_times': repeat_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'FID': {'mean': float(np.mean(fid)), 'std': float(np.std(fid))},
            'Diversity': {'mean': float(np.mean(div)), 'std': float(np.std(div))},
            'R-Precision_TOP1': {'mean': float(np.mean(top1)), 'std': float(np.std(top1))},
            'R-Precision_TOP2': {'mean': float(np.mean(top2)), 'std': float(np.std(top2))},
            'R-Precision_TOP3': {'mean': float(np.mean(top3)), 'std': float(np.std(top3))},
            'Matching_Score': {'mean': float(np.mean(matching)), 'std': float(np.std(matching))},
            'Multimodality': {'mean': float(np.mean(mm)), 'std': float(np.std(mm))},
            'CLIP_Score': {'mean': float(np.mean(clip_scores)), 'std': float(np.std(clip_scores))}
        }
    }
    
    print(f"\n{'='*80}")
    print(f"快速评估结果 - {model_name}")
    print(f"{'='*80}")
    print(f"FID:              {np.mean(fid):.4f} ± {np.std(fid):.4f}")
    print(f"Diversity:        {np.mean(div):.4f} ± {np.std(div):.4f}")
    print(f"R-Precision TOP1: {np.mean(top1):.4f} ± {np.std(top1):.4f}")
    print(f"R-Precision TOP2: {np.mean(top2):.4f} ± {np.std(top2):.4f}")
    print(f"R-Precision TOP3: {np.mean(top3):.4f} ± {np.std(top3):.4f}")
    print(f"Matching Score:   {np.mean(matching):.4f} ± {np.std(matching):.4f}")
    print(f"Multimodality:    {np.mean(mm):.4f} ± {np.std(mm):.4f}")
    print(f"CLIP-Score:       {np.mean(clip_scores):.4f} ± {np.std(clip_scores):.4f}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MARDM快速评估脚本')
    parser.add_argument('--name', type=str, default='MARDM_SiT_XL')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument('--cal_mm', action="store_false")
    parser.add_argument('--hard_pseudo_reorder', action="store_true")
    parser.add_argument('--repeat_times', type=int, default=3, 
                       help='评估重复次数（默认3次，正式评估用20次）')
    
    args = parser.parse_args()
    
    # 运行评估
    results = quick_eval_mardm(args, args.name, args.model)
    
    # 保存结果
    os.makedirs('quick_eval_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'quick_eval_results/quick_eval_{args.name}_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {output_file}")
    
    # 创建latest链接
    latest_file = f'quick_eval_results/quick_eval_{args.name}_latest.json'
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"最新结果: {latest_file}")


if __name__ == "__main__":
    main()

