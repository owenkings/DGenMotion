#!/usr/bin/env python3
"""
生成模型评估报告的 Markdown 文件

从 eval.log 和 TensorBoard 日志中提取数据，为每个模型生成详细的评估报告。
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import json

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available, will only parse eval.log files")


def parse_eval_log(eval_log_path):
    """解析 eval.log 文件，提取最终评估结果"""
    if not os.path.exists(eval_log_path):
        return None
    
    with open(eval_log_path, 'r') as f:
        content = f.read()
    
    if not content.strip():
        return None
    
    result = {}
    
    # 解析 FID
    fid_match = re.search(r'FID:\s*([\d.]+)', content)
    if fid_match:
        result['FID'] = float(fid_match.group(1))
    
    # 解析 Diversity
    div_match = re.search(r'Diversity:\s*([\d.]+)', content)
    if div_match:
        result['Diversity'] = float(div_match.group(1))
    
    # 解析 TOP1, TOP2, TOP3
    top_match = re.search(r'TOP1:\s*([\d.]+).*?TOP2\.\s*([\d.]+).*?TOP3\.\s*([\d.]+)', content)
    if top_match:
        result['Top1'] = float(top_match.group(1))
        result['Top2'] = float(top_match.group(2))
        result['Top3'] = float(top_match.group(3))
    
    # 解析 Matching
    match_match = re.search(r'Matching:\s*([\d.]+)', content)
    if match_match:
        result['Matching'] = float(match_match.group(1))
    
    # 解析 Multimodality
    mm_match = re.search(r'Multimodality:([\d.]+)', content)
    if mm_match:
        result['Multimodality'] = float(mm_match.group(1))
    
    # 解析 CLIP Score
    clip_match = re.search(r'CLIP-Score:([\d.]+)', content)
    if clip_match:
        result['CLIP_Score'] = float(clip_match.group(1))
    
    # 解析 MAE (for AE models)
    mae_match = re.search(r'MAE:([\d.]+)', content)
    if mae_match:
        result['MAE'] = float(mae_match.group(1))
    
    return result


def extract_tensorboard_data(model_dir):
    """从 TensorBoard 日志中提取评估数据"""
    if not HAS_TENSORBOARD:
        return []
    
    model_dir = Path(model_dir)
    events_files = list(model_dir.glob('events.out.tfevents.*'))
    if not events_files:
        return []
    
    try:
        ea = EventAccumulator(str(model_dir))
        ea.Reload()
        
        # 检查是否有评估数据
        if 'scalars' not in ea.Tags():
            return []
        
        tags = ea.Tags()['scalars']
        
        # 提取所有评估指标
        metrics_dict = {}
        
        # 提取 Test/FID (使用 ./Test/FID 或 Test/FID)
        fid_tag = './Test/FID' if './Test/FID' in tags else ('Test/FID' if 'Test/FID' in tags else None)
        if fid_tag:
            fid_scalars = ea.Scalars(fid_tag)
            for scalar in fid_scalars:
                epoch = int(scalar.step)
                if epoch not in metrics_dict:
                    metrics_dict[epoch] = {}
                metrics_dict[epoch]['epoch'] = epoch
                metrics_dict[epoch]['FID'] = scalar.value
        
        # 提取其他指标
        tag_mapping = {
            './Test/Diversity': 'Diversity',
            'Test/Diversity': 'Diversity',
            './Test/top1': 'Top1',
            'Test/top1': 'Top1',
            './Test/top2': 'Top2',
            'Test/top2': 'Top2',
            './Test/top3': 'Top3',
            'Test/top3': 'Top3',
            './Test/matching_score': 'matching_score',
            'Test/matching_score': 'matching_score',
            './Test/clip_score': 'clip_score',
            'Test/clip_score': 'clip_score',
            'Val/loss': 'Val_Loss'
        }
        
        for tag_pattern, key in tag_mapping.items():
            if tag_pattern in tags:
                scalars = ea.Scalars(tag_pattern)
                for scalar in scalars:
                    epoch = int(scalar.step)
                    if epoch not in metrics_dict:
                        metrics_dict[epoch] = {'epoch': epoch}
                    metrics_dict[epoch][key] = scalar.value
        
        # 转换为列表并排序
        metrics = sorted(metrics_dict.values(), key=lambda x: x.get('epoch', 0))
        return metrics
    except Exception as e:
        print(f"Warning: Could not parse TensorBoard data from {model_dir}: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_model_info(model_dir):
    """获取模型信息"""
    model_name = Path(model_dir).name
    
    # 尝试从 checkpoint 获取信息
    checkpoint_path = Path(model_dir) / 'model' / 'latest.tar'
    epoch = None
    if checkpoint_path.exists():
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            epoch = ckpt.get('ep', None)
        except:
            pass
    
    return {
        'name': model_name,
        'epoch': epoch
    }


def generate_markdown_report(model_dir, output_dir):
    """为单个模型生成 Markdown 报告"""
    model_dir = Path(model_dir)
    model_name = model_dir.name
    
    # 获取模型信息
    info = get_model_info(model_dir)
    
    # 解析 eval.log
    eval_log_path = model_dir / 'eval' / 'eval.log'
    final_results = parse_eval_log(eval_log_path)
    
    # 提取 TensorBoard 数据
    tb_data = extract_tensorboard_data(model_dir / 'model')
    
    # 生成 Markdown
    md_lines = []
    
    # 标题
    md_lines.append(f"# {model_name}")
    md_lines.append("")
    
    # 配置信息
    md_lines.append("## Configuration")
    md_lines.append("")
    md_lines.append(f"- **Model**: {model_name}")
    if info['epoch']:
        md_lines.append(f"- **Total Epochs**: {info['epoch']}")
    if tb_data:
        md_lines.append(f"- **Evaluation Rounds**: {len(tb_data)}")
    md_lines.append("")
    
    # 最终结果
    if final_results:
        md_lines.append("## Final Results")
        md_lines.append("")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        
        metric_names = {
            'FID': 'FID',
            'Diversity': 'Diversity',
            'Top1': 'Top1',
            'Top2': 'Top2',
            'Top3': 'Top3',
            'Matching': 'Matching',
            'Multimodality': 'Multimodality',
            'CLIP_Score': 'CLIP Score',
            'MAE': 'MAE'
        }
        
        for key, display_name in metric_names.items():
            if key in final_results:
                md_lines.append(f"| {display_name} | {final_results[key]:.4f} |")
        md_lines.append("")
    
    # 每个 epoch 的数据（如果有）
    if tb_data:
        md_lines.append("## All Metrics by Epoch")
        md_lines.append("")
        
        # 表头
        headers = ['Epoch', 'FID', 'Diversity', 'Top1', 'Top2', 'Top3', 'Matching']
        if any('Val_Loss' in d for d in tb_data):
            headers.append('Val Loss')
        if any('clip_score' in d for d in tb_data):
            headers.append('CLIP Score')
        
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["-------"] * len(headers)) + "|")
        
        # 数据行
        for data in tb_data:
            row = []
            row.append(f"{data.get('epoch', '')}")
            
            # FID
            fid = data.get('FID')
            row.append(f"{fid:.4f}" if fid is not None else "")
            
            # Diversity
            div = data.get('Diversity')
            row.append(f"{div:.4f}" if div is not None else "")
            
            # Top1, Top2, Top3
            row.append(f"{data.get('Top1', 0):.4f}" if data.get('Top1') is not None else "")
            row.append(f"{data.get('Top2', 0):.4f}" if data.get('Top2') is not None else "")
            row.append(f"{data.get('Top3', 0):.4f}" if data.get('Top3') is not None else "")
            
            # Matching
            match = data.get('matching_score')
            row.append(f"{match:.4f}" if match is not None else "")
            
            # Val Loss
            if 'Val Loss' in headers:
                val_loss = data.get('Val_Loss')
                row.append(f"{val_loss:.6f}" if val_loss is not None else "")
            
            # CLIP Score
            if 'CLIP Score' in headers:
                clip = data.get('clip_score')
                row.append(f"{clip:.4f}" if clip is not None else "")
            
            md_lines.append("| " + " | ".join(row) + " |")
        
        md_lines.append("")
        
        # Best Values
        if tb_data:
            md_lines.append("## Best Values")
            md_lines.append("")
            md_lines.append("| Metric | Best Value | Epoch |")
            md_lines.append("|--------|------------|-------|")
            
            metrics_to_track = {
                'FID': (min, True),  # 越小越好
                'Diversity': (max, True),  # 越大越好
                'Top1': (max, True),
                'Top2': (max, True),
                'Top3': (max, True),
                'matching_score': (min, True),  # 越小越好
                'Val_Loss': (min, True)
            }
            
            for metric, (func, track) in metrics_to_track.items():
                values = [(d.get('epoch', 0), d.get(metric)) for d in tb_data if d.get(metric) is not None]
                if values:
                    best_epoch, best_value = func(values, key=lambda x: x[1])
                    metric_name = metric.replace('_', ' ').title()
                    if metric == 'matching_score':
                        metric_name = 'Matching'
                    elif metric == 'FID':
                        metric_name = 'FID'  # 保持大写
                    elif metric == 'Val_Loss':
                        metric_name = 'Val Loss'  # 显示为 "Val Loss"
                    md_lines.append(f"| {metric_name} | {best_value:.4f} | {best_epoch} |")
            
            md_lines.append("")
    
    # 保存文件
    output_path = Path(output_dir) / f"{model_name}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ Generated: {output_path}")
    return output_path


def main():
    checkpoints_dir = Path('checkpoints/t2m')
    output_dir = Path('checkpoints/t2m/reports')
    output_dir.mkdir(exist_ok=True)
    
    # 查找所有有评估结果的模型
    model_dirs = []
    for model_dir in checkpoints_dir.iterdir():
        if model_dir.is_dir():
            eval_dir = model_dir / 'eval'
            if eval_dir.exists() and (eval_dir / 'eval.log').exists():
                model_dirs.append(model_dir)
    
    print(f"Found {len(model_dirs)} models with evaluation results")
    print("=" * 70)
    
    for model_dir in sorted(model_dirs):
        try:
            generate_markdown_report(model_dir, output_dir)
        except Exception as e:
            print(f"❌ Error processing {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"✅ Generated {len(model_dirs)} reports in {output_dir}")


if __name__ == '__main__':
    main()

