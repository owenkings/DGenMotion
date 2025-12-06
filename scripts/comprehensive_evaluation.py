"""
MARDM项目综合评估脚本
评估所有模型并收集完整的性能指标
"""
import os
import sys
import json
import time
import torch
import psutil
import subprocess
from datetime import datetime
from pathlib import Path

# 切换到项目根目录（脚本在scripts/目录下）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
print(f"工作目录: {os.getcwd()}")

class ComprehensiveEvaluator:
    def __init__(self, dataset_name='t2m'):
        self.dataset_name = dataset_name
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_name,
            'system_info': self.get_system_info(),
            'models': {}
        }
        
    def get_system_info(self):
        """获取系统信息"""
        info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info['gpu_memory'] = [f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB" 
                                 for i in range(torch.cuda.device_count())]
        
        info['cpu_count'] = psutil.cpu_count()
        info['ram_total'] = f"{psutil.virtual_memory().total / 1024**3:.2f} GB"
        
        return info
    
    def evaluate_ae(self):
        """评估AutoEncoder模型"""
        print("\n" + "="*80)
        print("评估 AutoEncoder 模型")
        print("="*80)
        
        model_name = 'AE'
        start_time = time.time()
        
        # 检查模型是否存在
        model_path = f'./checkpoints/{self.dataset_name}/{model_name}/model'
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            return None
        
        # 运行评估
        cmd = [
            'python', 'evaluation_AE.py',
            '--name', model_name,
            '--dataset_name', self.dataset_name,
            '--num_workers', '4'
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            eval_time = time.time() - start_time
            
            # 解析评估日志
            eval_log_path = f'./checkpoints/{self.dataset_name}/{model_name}/eval/eval.log'
            metrics = self.parse_eval_log(eval_log_path)
            
            self.results['models'][model_name] = {
                'evaluation_time': f"{eval_time:.2f}s",
                'metrics': metrics,
                'status': 'success' if result.returncode == 0 else 'failed',
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            }
            
            print(f"✓ AE评估完成，耗时: {eval_time:.2f}秒")
            return metrics
            
        except subprocess.TimeoutExpired:
            print("✗ AE评估超时")
            self.results['models'][model_name] = {'status': 'timeout'}
            return None
        except Exception as e:
            print(f"✗ AE评估出错: {str(e)}")
            self.results['models'][model_name] = {'status': 'error', 'error': str(e)}
            return None
    
    def evaluate_mardm(self, model_name, model_type, cfg_scale):
        """评估MARDM模型"""
        print("\n" + "="*80)
        print(f"评估 {model_name} 模型")
        print("="*80)
        
        start_time = time.time()
        
        # 检查模型是否存在
        model_path = f'./checkpoints/{self.dataset_name}/{model_name}/model'
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            return None
        
        # 运行评估
        cmd = [
            'python', 'evaluation_MARDM.py',
            '--name', model_name,
            '--model', model_type,
            '--dataset_name', self.dataset_name,
            '--cfg', str(cfg_scale),
            '--num_workers', '4',
            '--time_steps', '18'
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4小时超时
            eval_time = time.time() - start_time
            
            # 解析评估日志
            eval_log_path = f'./checkpoints/{self.dataset_name}/{model_name}/eval/eval.log'
            metrics = self.parse_eval_log(eval_log_path)
            
            self.results['models'][model_name] = {
                'model_type': model_type,
                'cfg_scale': cfg_scale,
                'evaluation_time': f"{eval_time:.2f}s",
                'metrics': metrics,
                'status': 'success' if result.returncode == 0 else 'failed',
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            }
            
            print(f"✓ {model_name}评估完成，耗时: {eval_time:.2f}秒")
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"✗ {model_name}评估超时")
            self.results['models'][model_name] = {'status': 'timeout'}
            return None
        except Exception as e:
            print(f"✗ {model_name}评估出错: {str(e)}")
            self.results['models'][model_name] = {'status': 'error', 'error': str(e)}
            return None
    
    def parse_eval_log(self, log_path):
        """解析评估日志文件"""
        if not os.path.exists(log_path):
            return None
        
        metrics = {}
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
            # 解析指标
            import re
            
            # FID
            fid_match = re.search(r'FID:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if fid_match:
                metrics['FID'] = {'mean': float(fid_match.group(1)), 'conf': float(fid_match.group(2))}
            
            # Diversity
            div_match = re.search(r'Diversity:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if div_match:
                metrics['Diversity'] = {'mean': float(div_match.group(1)), 'conf': float(div_match.group(2))}
            
            # R-Precision (TOP1, TOP2, TOP3)
            top1_match = re.search(r'TOP1:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top1_match:
                metrics['R-Precision_TOP1'] = {'mean': float(top1_match.group(1)), 'conf': float(top1_match.group(2))}
            
            top2_match = re.search(r'TOP2\.\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top2_match:
                metrics['R-Precision_TOP2'] = {'mean': float(top2_match.group(1)), 'conf': float(top2_match.group(2))}
            
            top3_match = re.search(r'TOP3\.\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top3_match:
                metrics['R-Precision_TOP3'] = {'mean': float(top3_match.group(1)), 'conf': float(top3_match.group(2))}
            
            # Matching Score
            matching_match = re.search(r'Matching:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if matching_match:
                metrics['Matching_Score'] = {'mean': float(matching_match.group(1)), 'conf': float(matching_match.group(2))}
            
            # Multimodality
            mm_match = re.search(r'Multimodality:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if mm_match:
                metrics['Multimodality'] = {'mean': float(mm_match.group(1)), 'conf': float(mm_match.group(2))}
            
            # CLIP Score
            clip_match = re.search(r'CLIP-Score:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if clip_match:
                metrics['CLIP_Score'] = {'mean': float(clip_match.group(1)), 'conf': float(clip_match.group(2))}
            
            # MAE (for AE)
            mae_match = re.search(r'MAE:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if mae_match:
                metrics['MAE'] = {'mean': float(mae_match.group(1)), 'conf': float(mae_match.group(2))}
                
        except Exception as e:
            print(f"解析日志文件出错: {str(e)}")
            return None
        
        return metrics
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("\n" + "="*80)
        print("MARDM项目综合评估")
        print(f"数据集: {self.dataset_name}")
        print(f"开始时间: {self.results['timestamp']}")
        print("="*80)
        
        # 1. 评估AE
        self.evaluate_ae()
        
        # 2. 评估MARDM-SiT-XL
        cfg_scale = 4.5 if self.dataset_name == 't2m' else 2.5
        self.evaluate_mardm('MARDM_SiT_XL', 'MARDM-SiT-XL', cfg_scale)
        
        # 3. 评估MARDM-DDPM-XL
        self.evaluate_mardm('MARDM_DDPM_XL', 'MARDM-DDPM-XL', cfg_scale)
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存评估结果"""
        output_dir = './evaluation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'{output_dir}/comprehensive_eval_{self.dataset_name}_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {output_file}")
        
        # 同时保存一个易读的文本版本
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(self.format_results_text())
        
        print(f"文本报告已保存到: {txt_file}")
    
    def format_results_text(self):
        """格式化结果为文本"""
        lines = []
        lines.append("="*80)
        lines.append("MARDM项目综合评估报告")
        lines.append("="*80)
        lines.append(f"\n评估时间: {self.results['timestamp']}")
        lines.append(f"数据集: {self.results['dataset']}")
        
        lines.append("\n" + "-"*80)
        lines.append("系统信息")
        lines.append("-"*80)
        for key, value in self.results['system_info'].items():
            lines.append(f"{key}: {value}")
        
        lines.append("\n" + "="*80)
        lines.append("模型评估结果")
        lines.append("="*80)
        
        for model_name, model_results in self.results['models'].items():
            lines.append(f"\n{'='*80}")
            lines.append(f"模型: {model_name}")
            lines.append(f"{'='*80}")
            
            if 'status' in model_results and model_results['status'] != 'success':
                lines.append(f"状态: {model_results['status']}")
                if 'error' in model_results:
                    lines.append(f"错误: {model_results['error']}")
                continue
            
            if 'model_type' in model_results:
                lines.append(f"模型类型: {model_results['model_type']}")
                lines.append(f"CFG Scale: {model_results['cfg_scale']}")
            
            lines.append(f"评估时间: {model_results.get('evaluation_time', 'N/A')}")
            
            if 'metrics' in model_results and model_results['metrics']:
                lines.append("\n指标结果:")
                lines.append("-"*40)
                for metric_name, metric_value in model_results['metrics'].items():
                    if isinstance(metric_value, dict):
                        lines.append(f"{metric_name:20s}: {metric_value['mean']:.4f} ± {metric_value['conf']:.4f}")
                    else:
                        lines.append(f"{metric_name:20s}: {metric_value}")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """打印评估总结"""
        print("\n" + "="*80)
        print("评估总结")
        print("="*80)
        
        for model_name, model_results in self.results['models'].items():
            print(f"\n{model_name}:")
            if 'status' in model_results and model_results['status'] != 'success':
                print(f"  状态: {model_results['status']}")
                continue
            
            if 'metrics' in model_results and model_results['metrics']:
                for metric_name, metric_value in model_results['metrics'].items():
                    if isinstance(metric_value, dict):
                        print(f"  {metric_name:20s}: {metric_value['mean']:.4f} ± {metric_value['conf']:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MARDM项目综合评估')
    parser.add_argument('--dataset_name', type=str, default='t2m', 
                       choices=['t2m', 'kit'],
                       help='数据集名称 (t2m=HumanML3D, kit=KIT-ML)')
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(dataset_name=args.dataset_name)
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

