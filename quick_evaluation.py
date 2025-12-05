"""
快速评估脚本 - 用于验证评估流程
只运行少量迭代以快速检查系统状态
"""
import os
import sys
import json
import time
import torch
import numpy as np
from os.path import join as pjoin
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from models.AE import AE_models
from models.MARDM import MARDM_models
from torch.utils.data import DataLoader
from utils.datasets import Text2MotionDataset, collate_fn
from utils.evaluators import Evaluators


def quick_check_environment():
    """快速检查环境"""
    print("="*80)
    print("环境检查")
    print("="*80)
    
    info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print()
    return info


def check_models_exist(dataset_name='t2m'):
    """检查模型文件是否存在"""
    print("="*80)
    print("模型文件检查")
    print("="*80)
    
    models_to_check = {
        'AE': 'AE/model/latest.tar' if dataset_name == 't2m' else 'AE/model/net_best_fid.tar',
        'MARDM_SiT_XL': 'MARDM_SiT_XL/model/latest.tar',
        'MARDM_DDPM_XL': 'MARDM_DDPM_XL/model/latest.tar',
        'text_mot_match': 'text_mot_match/model/finest.tar',
        'text_mot_match_clip': 'text_mot_match_clip/model/finest.tar'
    }
    
    status = {}
    for model_name, model_path in models_to_check.items():
        full_path = pjoin('./checkpoints', dataset_name, model_path)
        exists = os.path.exists(full_path)
        status[model_name] = exists
        
        if exists:
            size = os.path.getsize(full_path) / (1024**2)  # MB
            print(f"✓ {model_name}: 存在 ({size:.1f} MB)")
        else:
            print(f"✗ {model_name}: 不存在")
    
    print()
    return status


def check_dataset_exist(dataset_name='t2m'):
    """检查数据集是否存在"""
    print("="*80)
    print("数据集检查")
    print("="*80)
    
    if dataset_name == 't2m':
        data_root = './datasets/HumanML3D/'
    else:
        data_root = './datasets/KIT-ML/'
    
    required_files = [
        'Mean.npy',
        'Std.npy',
        'test.txt',
        'train.txt',
        'new_joint_vecs',
        'texts'
    ]
    
    status = {}
    for file_name in required_files:
        file_path = pjoin(data_root, file_name)
        exists = os.path.exists(file_path)
        status[file_name] = exists
        
        if exists:
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"✓ {file_name}: 存在 ({size:.1f} KB)")
            else:
                # 目录
                count = len(os.listdir(file_path))
                print(f"✓ {file_name}: 存在 ({count} 个文件)")
        else:
            print(f"✗ {file_name}: 不存在")
    
    print()
    return status


def quick_test_ae(dataset_name='t2m', num_batches=3):
    """快速测试AE模型"""
    print("="*80)
    print("快速测试 AE 模型")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据准备
        if dataset_name == "t2m":
            data_root = './datasets/HumanML3D/'
            dim_pose = 67
        else:
            data_root = './datasets/KIT-ML/'
            dim_pose = 64
        
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        mean = np.load(pjoin(data_root, 'Mean.npy'))
        std = np.load(pjoin(data_root, 'Std.npy'))
        eval_mean = np.load(f'utils/eval_mean_std/{dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'test.txt')
        
        eval_dataset = Text2MotionDataset(eval_mean, eval_std, split_file, dataset_name, 
                                         motion_dir, text_dir, 4, 196, 20, evaluation=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, num_workers=2, 
                                drop_last=True, collate_fn=collate_fn, shuffle=True)
        
        # 加载模型
        ae = AE_models['AE_Model'](input_width=dim_pose)
        ckpt_path = pjoin('./checkpoints', dataset_name, 'AE', 'model',
                         'latest.tar' if dataset_name == 't2m' else 'net_best_fid.tar')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ae.load_state_dict(ckpt['ae'])
        ae.eval()
        ae.to(device)
        
        print(f"模型参数量: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f}M")
        
        # 测试几个批次
        total_time = 0
        num_samples = 0
        
        for i, batch in enumerate(eval_loader):
            if i >= num_batches:
                break
            
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
            motion = motion.to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = ae(motion)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            batch_time = time.time() - start_time
            total_time += batch_time
            num_samples += motion.shape[0]
            
            print(f"批次 {i+1}: {motion.shape[0]} 样本, {batch_time*1000:.2f}ms")
        
        avg_time = (total_time / num_samples) * 1000
        print(f"\n平均每样本时间: {avg_time:.2f}ms")
        print(f"吞吐量: {num_samples / total_time:.2f} samples/sec")
        print("✓ AE模型测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ AE模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def quick_test_mardm(dataset_name='t2m', model_name='MARDM_SiT_XL', 
                     model_type='MARDM-SiT-XL', num_samples=3):
    """快速测试MARDM模型"""
    print("="*80)
    print(f"快速测试 {model_name} 模型")
    print("="*80)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据准备
        if dataset_name == "t2m":
            dim_pose = 67
        else:
            dim_pose = 64
        
        # 加载AE
        ae = AE_models['AE_Model'](input_width=dim_pose)
        ae_ckpt_path = pjoin('./checkpoints', dataset_name, 'AE', 'model',
                            'latest.tar' if dataset_name == 't2m' else 'net_best_fid.tar')
        ae_ckpt = torch.load(ae_ckpt_path, map_location='cpu')
        ae.load_state_dict(ae_ckpt['ae'])
        ae.eval()
        ae.to(device)
        
        # 加载MARDM
        mardm = MARDM_models[model_type](ae_dim=ae.output_emb_width, cond_mode='text')
        mardm_ckpt_path = pjoin('./checkpoints', dataset_name, model_name, 'model', 'latest.tar')
        mardm_ckpt = torch.load(mardm_ckpt_path, map_location='cpu')
        mardm.load_state_dict(mardm_ckpt['ema_mardm'], strict=False)
        mardm.eval()
        mardm.to(device)
        
        print(f"MARDM参数量: {sum(p.numel() for p in mardm.parameters()) / 1e6:.2f}M")
        print(f"AE参数量: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f}M")
        
        # 测试生成
        test_texts = [
            "A person is walking forward.",
            "A person is running.",
            "A person is jumping."
        ]
        
        cfg_scale = 4.5 if dataset_name == 't2m' else 2.5
        time_steps = 18
        motion_length = 32  # latent space length
        
        total_time = 0
        
        for i in range(num_samples):
            text = test_texts[i % len(test_texts)]
            m_lengths = torch.tensor([motion_length]).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                latents = mardm.generate([text], m_lengths, time_steps, cfg_scale)
                motions = ae.decode(latents)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            gen_time = time.time() - start_time
            total_time += gen_time
            
            print(f"样本 {i+1}: \"{text[:30]}...\", {gen_time*1000:.2f}ms, "
                  f"输出形状: {motions.shape}")
        
        avg_time = (total_time / num_samples) * 1000
        print(f"\n平均生成时间: {avg_time:.2f}ms")
        print(f"✓ {model_name}模型测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ {model_name}模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='MARDM快速评估')
    parser.add_argument('--dataset_name', type=str, default='t2m',
                       choices=['t2m', 'kit'],
                       help='数据集名称')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MARDM项目快速评估")
    print(f"数据集: {args.dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    results = {}
    
    # 1. 环境检查
    results['environment'] = quick_check_environment()
    
    # 2. 模型文件检查
    results['models'] = check_models_exist(args.dataset_name)
    
    # 3. 数据集检查
    results['dataset'] = check_dataset_exist(args.dataset_name)
    
    # 4. 快速测试AE
    results['ae_test'] = quick_test_ae(args.dataset_name, num_batches=3)
    print()
    
    # 5. 快速测试MARDM-SiT-XL
    results['mardm_sit_test'] = quick_test_mardm(args.dataset_name, 'MARDM_SiT_XL', 
                                                  'MARDM-SiT-XL', num_samples=3)
    print()
    
    # 6. 快速测试MARDM-DDPM-XL
    results['mardm_ddpm_test'] = quick_test_mardm(args.dataset_name, 'MARDM_DDPM_XL',
                                                   'MARDM-DDPM-XL', num_samples=3)
    print()
    
    # 总结
    print("="*80)
    print("快速评估总结")
    print("="*80)
    
    all_models_exist = all(results['models'].values())
    all_dataset_exist = all(results['dataset'].values())
    all_tests_pass = (results.get('ae_test', False) and 
                     results.get('mardm_sit_test', False) and 
                     results.get('mardm_ddpm_test', False))
    
    print(f"模型文件: {'✓ 完整' if all_models_exist else '✗ 缺失'}")
    print(f"数据集: {'✓ 完整' if all_dataset_exist else '✗ 缺失'}")
    print(f"功能测试: {'✓ 通过' if all_tests_pass else '✗ 失败'}")
    
    if all_models_exist and all_dataset_exist and all_tests_pass:
        print("\n✓ 系统状态良好，可以运行完整评估")
        print("\n运行完整评估:")
        print("  bash run_full_evaluation.sh")
    else:
        print("\n✗ 系统存在问题，请检查上述输出")
    
    # 保存结果
    os.makedirs('./evaluation_results', exist_ok=True)
    output_file = f'./evaluation_results/quick_check_{args.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()

