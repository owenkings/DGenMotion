#!/usr/bin/env python3
"""
Checkpoint 信息查看工具

用法:
    python utils/checkpoint_info.py --name FSQ_MARDM_DiT_XL
    python utils/checkpoint_info.py --name FSQ_MARDM_DiT_XL --checkpoint net_best_fid.tar
"""

import torch
import os
import argparse
from pathlib import Path


def get_checkpoint_info(checkpoint_path):
    """获取 checkpoint 的详细信息"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        info = {
            'path': checkpoint_path,
            'epoch': ckpt.get('ep', 'N/A'),
            'iterations': ckpt.get('total_it', 'N/A'),
            'keys': list(ckpt.keys()),
            'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        }
        
        # 检查是否有模型权重
        if 'ema_mardm' in ckpt:
            info['has_ema'] = True
            info['ema_params'] = sum(p.numel() for p in ckpt['ema_mardm'].values() if isinstance(p, torch.Tensor))
        else:
            info['has_ema'] = False
        
        if 'mardm' in ckpt:
            info['has_model'] = True
        else:
            info['has_model'] = False
            
        return info
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='查看 Checkpoint 信息')
    parser.add_argument('--name', type=str, required=True, help='模型名称')
    parser.add_argument('--dataset', type=str, default='t2m', help='数据集名称')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='checkpoints 根目录')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='指定 checkpoint 文件 (latest.tar 或 net_best_fid.tar)')
    
    args = parser.parse_args()
    
    model_dir = Path(args.checkpoints_dir) / args.dataset / args.name / 'model'
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    print("=" * 70)
    print(f"Checkpoint 信息: {args.name}")
    print("=" * 70)
    print(f"目录: {model_dir}\n")
    
    # 列出所有 checkpoint 文件
    checkpoint_files = {
        'latest.tar': model_dir / 'latest.tar',
        'net_best_fid.tar': model_dir / 'net_best_fid.tar',
    }
    
    if args.checkpoint:
        # 查看指定的 checkpoint
        if args.checkpoint in checkpoint_files:
            checkpoint_path = checkpoint_files[args.checkpoint]
        else:
            checkpoint_path = model_dir / args.checkpoint
        
        print(f"📁 查看: {args.checkpoint}")
        print("-" * 70)
        info = get_checkpoint_info(checkpoint_path)
        
        if info is None:
            print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        elif 'error' in info:
            print(f"❌ 加载错误: {info['error']}")
        else:
            print(f"Epoch: {info['epoch']}")
            print(f"Iterations: {info['iterations']:,}")
            print(f"文件大小: {info['size_mb']:.2f} MB")
            print(f"包含的键: {', '.join(info['keys'])}")
            if info.get('has_ema'):
                print(f"✅ 包含 EMA 模型权重 ({info['ema_params']:,} 参数)")
            if info.get('has_model'):
                print(f"✅ 包含模型权重")
            print(f"\n路径: {info['path']}")
    else:
        # 查看所有 checkpoint
        print("📊 所有 Checkpoint:\n")
        
        for name, path in checkpoint_files.items():
            print(f"【{name}】")
            print("-" * 70)
            info = get_checkpoint_info(path)
            
            if info is None:
                print(f"  ❌ 文件不存在")
            elif 'error' in info:
                print(f"  ❌ 加载错误: {info['error']}")
            else:
                print(f"  Epoch: {info['epoch']}")
                print(f"  Iterations: {info['iterations']:,}")
                print(f"  文件大小: {info['size_mb']:.2f} MB")
                if info.get('has_ema'):
                    print(f"  ✅ 包含 EMA 模型权重")
                if info.get('has_model'):
                    print(f"  ✅ 包含模型权重")
            print()
        
        # 推荐
        print("=" * 70)
        print("💡 推荐使用:")
        print("  - 评估/推理: net_best_fid.tar (FID 最低)")
        print("  - 继续训练: latest.tar (最新状态)")
        print("=" * 70)


if __name__ == '__main__':
    main()

