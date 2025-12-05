"""
简单的环境和文件检查脚本
不导入模型代码，只检查文件和基础环境
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path


def check_python_packages():
    """检查Python包"""
    print("="*80)
    print("Python包检查")
    print("="*80)
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'psutil': 'psutil',
    }
    
    results = {}
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = {'installed': True, 'version': version}
            print(f"✓ {name}: {version}")
        except ImportError:
            results[package] = {'installed': False}
            print(f"✗ {name}: 未安装")
    
    # 特殊检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA: 不可用 (将使用CPU)")
    except:
        pass
    
    print()
    return results


def check_directory_structure():
    """检查目录结构"""
    print("="*80)
    print("目录结构检查")
    print("="*80)
    
    required_dirs = [
        'checkpoints',
        'datasets',
        'models',
        'utils',
        'diffusions',
        'glove'
    ]
    
    results = {}
    for dir_name in required_dirs:
        exists = os.path.isdir(dir_name)
        results[dir_name] = exists
        
        if exists:
            count = len(os.listdir(dir_name))
            print(f"✓ {dir_name}/: 存在 ({count} 个项目)")
        else:
            print(f"✗ {dir_name}/: 不存在")
    
    print()
    return results


def check_models(dataset_name='t2m'):
    """检查模型文件"""
    print("="*80)
    print(f"模型文件检查 (数据集: {dataset_name})")
    print("="*80)
    
    models = {
        'AE': f'checkpoints/{dataset_name}/AE/model/latest.tar' if dataset_name == 't2m' 
              else f'checkpoints/{dataset_name}/AE/model/net_best_fid.tar',
        'MARDM_SiT_XL': f'checkpoints/{dataset_name}/MARDM_SiT_XL/model/latest.tar',
        'MARDM_DDPM_XL': f'checkpoints/{dataset_name}/MARDM_DDPM_XL/model/latest.tar',
        'text_mot_match': f'checkpoints/{dataset_name}/text_mot_match/model/finest.tar',
        'text_mot_match_clip': f'checkpoints/{dataset_name}/text_mot_match_clip/model/finest.tar',
        'length_estimator': f'checkpoints/{dataset_name}/length_estimator/model/finest.tar'
    }
    
    results = {}
    for model_name, model_path in models.items():
        exists = os.path.isfile(model_path)
        results[model_name] = exists
        
        if exists:
            size_mb = os.path.getsize(model_path) / (1024**2)
            if size_mb > 1024:
                size_str = f"{size_mb/1024:.2f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"
            print(f"✓ {model_name}: {size_str}")
        else:
            print(f"✗ {model_name}: 不存在")
            print(f"  路径: {model_path}")
    
    print()
    return results


def check_dataset(dataset_name='t2m'):
    """检查数据集"""
    print("="*80)
    print(f"数据集检查 (数据集: {dataset_name})")
    print("="*80)
    
    if dataset_name == 't2m':
        data_root = 'datasets/HumanML3D'
        dataset_full_name = 'HumanML3D'
    else:
        data_root = 'datasets/KIT-ML'
        dataset_full_name = 'KIT-ML'
    
    required_items = {
        'Mean.npy': 'file',
        'Std.npy': 'file',
        'test.txt': 'file',
        'train.txt': 'file',
        'new_joint_vecs': 'dir',
        'texts': 'dir'
    }
    
    results = {}
    for item_name, item_type in required_items.items():
        item_path = os.path.join(data_root, item_name)
        
        if item_type == 'file':
            exists = os.path.isfile(item_path)
            if exists:
                size_kb = os.path.getsize(item_path) / 1024
                print(f"✓ {item_name}: {size_kb:.1f} KB")
            else:
                print(f"✗ {item_name}: 不存在")
        else:  # directory
            exists = os.path.isdir(item_path)
            if exists:
                count = len(os.listdir(item_path))
                print(f"✓ {item_name}/: {count} 个文件")
            else:
                print(f"✗ {item_name}/: 不存在")
        
        results[item_name] = exists
    
    # 检查eval_mean_std
    eval_mean_path = f'utils/eval_mean_std/{dataset_name}/eval_mean.npy'
    eval_std_path = f'utils/eval_mean_std/{dataset_name}/eval_std.npy'
    
    eval_mean_exists = os.path.isfile(eval_mean_path)
    eval_std_exists = os.path.isfile(eval_std_path)
    
    print(f"\n评估用统计文件:")
    print(f"{'✓' if eval_mean_exists else '✗'} eval_mean.npy")
    print(f"{'✓' if eval_std_exists else '✗'} eval_std.npy")
    
    results['eval_mean'] = eval_mean_exists
    results['eval_std'] = eval_std_exists
    
    print()
    return results


def check_evaluation_scripts():
    """检查评估脚本"""
    print("="*80)
    print("评估脚本检查")
    print("="*80)
    
    scripts = [
        'evaluation_AE.py',
        'evaluation_MARDM.py',
        'train_AE.py',
        'train_MARDM.py',
        'sample.py',
        'edit.py'
    ]
    
    results = {}
    for script in scripts:
        exists = os.path.isfile(script)
        results[script] = exists
        
        if exists:
            size_kb = os.path.getsize(script) / 1024
            print(f"✓ {script}: {size_kb:.1f} KB")
        else:
            print(f"✗ {script}: 不存在")
    
    print()
    return results


def generate_summary_report(all_results, dataset_name):
    """生成总结报告"""
    print("="*80)
    print("检查总结")
    print("="*80)
    
    # 统计
    packages_ok = all(r.get('installed', False) for r in all_results['packages'].values())
    dirs_ok = all(all_results['directories'].values())
    models_ok = all(all_results['models'].values())
    dataset_ok = all(all_results['dataset'].values())
    scripts_ok = all(all_results['evaluation_scripts'].values())
    
    print(f"Python包: {'✓ 完整' if packages_ok else '✗ 缺失'}")
    print(f"目录结构: {'✓ 完整' if dirs_ok else '✗ 缺失'}")
    print(f"模型文件: {'✓ 完整' if models_ok else '✗ 缺失'}")
    print(f"数据集: {'✓ 完整' if dataset_ok else '✗ 缺失'}")
    print(f"评估脚本: {'✓ 完整' if scripts_ok else '✗ 缺失'}")
    
    print()
    
    if all([packages_ok, dirs_ok, models_ok, dataset_ok, scripts_ok]):
        print("✓ 系统检查通过，可以运行评估")
        print("\n建议的下一步:")
        print("1. 运行快速测试 (可选):")
        print("   python quick_evaluation.py --dataset_name t2m")
        print("\n2. 运行完整评估:")
        print("   bash run_full_evaluation.sh t2m")
        print("\n3. 或者分步运行:")
        print("   python evaluation_AE.py --name AE --dataset_name t2m")
        print("   python evaluation_MARDM.py --name MARDM_SiT_XL --model MARDM-SiT-XL --dataset_name t2m --cfg 4.5")
        print("   python evaluation_MARDM.py --name MARDM_DDPM_XL --model MARDM-DDPM-XL --dataset_name t2m --cfg 4.5")
    else:
        print("✗ 系统检查未通过，存在以下问题:")
        
        if not packages_ok:
            print("\n缺失的Python包:")
            for pkg, info in all_results['packages'].items():
                if not info.get('installed', False):
                    print(f"  - {pkg}")
        
        if not dirs_ok:
            print("\n缺失的目录:")
            for dir_name, exists in all_results['directories'].items():
                if not exists:
                    print(f"  - {dir_name}/")
        
        if not models_ok:
            print("\n缺失的模型文件:")
            for model_name, exists in all_results['models'].items():
                if not exists:
                    print(f"  - {model_name}")
        
        if not dataset_ok:
            print("\n缺失的数据集文件:")
            for item_name, exists in all_results['dataset'].items():
                if not exists:
                    print(f"  - {item_name}")
        
        if not scripts_ok:
            print("\n缺失的脚本文件:")
            for script, exists in all_results['evaluation_scripts'].items():
                if not exists:
                    print(f"  - {script}")
    
    return all([packages_ok, dirs_ok, models_ok, dataset_ok, scripts_ok])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MARDM环境检查')
    parser.add_argument('--dataset_name', type=str, default='t2m',
                       choices=['t2m', 'kit'],
                       help='数据集名称 (t2m=HumanML3D, kit=KIT-ML)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MARDM项目环境检查")
    print(f"数据集: {args.dataset_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': args.dataset_name
    }
    
    # 运行所有检查
    all_results['packages'] = check_python_packages()
    all_results['directories'] = check_directory_structure()
    all_results['models'] = check_models(args.dataset_name)
    all_results['dataset'] = check_dataset(args.dataset_name)
    all_results['evaluation_scripts'] = check_evaluation_scripts()
    
    # 生成总结
    all_ok = generate_summary_report(all_results, args.dataset_name)
    
    # 保存结果
    os.makedirs('evaluation_results', exist_ok=True)
    output_file = f'evaluation_results/environment_check_{args.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n检查结果已保存到: {output_file}")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

