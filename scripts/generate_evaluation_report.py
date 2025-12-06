"""
生成MARDM项目综合评估报告
整合所有评估结果并生成详细报告
"""
import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

# 切换到项目根目录（脚本在scripts/目录下）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
print(f"工作目录: {os.getcwd()}")


class ReportGenerator:
    def __init__(self, dataset_name='t2m'):
        self.dataset_name = dataset_name
        self.checkpoints_dir = './checkpoints'
        self.results_dir = './evaluation_results'
        
        self.report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_name,
            'models': {}
        }
    
    def load_eval_logs(self):
        """加载所有评估日志"""
        models = ['AE', 'MARDM_SiT_XL', 'MARDM_DDPM_XL']
        
        for model_name in models:
            eval_log_path = os.path.join(self.checkpoints_dir, self.dataset_name, 
                                        model_name, 'eval', 'eval.log')
            
            if os.path.exists(eval_log_path):
                print(f"加载 {model_name} 评估结果...")
                metrics = self.parse_eval_log(eval_log_path)
                self.report['models'][model_name] = {
                    'metrics': metrics,
                    'log_path': eval_log_path
                }
            else:
                print(f"警告: 未找到 {model_name} 的评估日志")
    
    def parse_eval_log(self, log_path):
        """解析评估日志"""
        metrics = {}
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            import re
            
            # FID
            fid_match = re.search(r'FID:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if fid_match:
                metrics['FID'] = {
                    'mean': float(fid_match.group(1)),
                    'conf': float(fid_match.group(2))
                }
            
            # Diversity
            div_match = re.search(r'Diversity:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if div_match:
                metrics['Diversity'] = {
                    'mean': float(div_match.group(1)),
                    'conf': float(div_match.group(2))
                }
            
            # R-Precision
            top1_match = re.search(r'TOP1:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top1_match:
                metrics['R-Precision_TOP1'] = {
                    'mean': float(top1_match.group(1)),
                    'conf': float(top1_match.group(2))
                }
            
            top2_match = re.search(r'TOP2\.\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top2_match:
                metrics['R-Precision_TOP2'] = {
                    'mean': float(top2_match.group(1)),
                    'conf': float(top2_match.group(2))
                }
            
            top3_match = re.search(r'TOP3\.\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if top3_match:
                metrics['R-Precision_TOP3'] = {
                    'mean': float(top3_match.group(1)),
                    'conf': float(top3_match.group(2))
                }
            
            # Matching Score
            matching_match = re.search(r'Matching:\s*([\d.]+),\s*conf\.\s*([\d.]+)', content)
            if matching_match:
                metrics['Matching_Score'] = {
                    'mean': float(matching_match.group(1)),
                    'conf': float(matching_match.group(2))
                }
            
            # Multimodality
            mm_match = re.search(r'Multimodality:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if mm_match:
                metrics['Multimodality'] = {
                    'mean': float(mm_match.group(1)),
                    'conf': float(mm_match.group(2))
                }
            
            # CLIP Score
            clip_match = re.search(r'CLIP-Score:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if clip_match:
                metrics['CLIP_Score'] = {
                    'mean': float(clip_match.group(1)),
                    'conf': float(clip_match.group(2))
                }
            
            # MAE (for AE)
            mae_match = re.search(r'MAE:\s*([\d.]+),\s*conf\.?\s*([\d.]+)', content)
            if mae_match:
                metrics['MAE'] = {
                    'mean': float(mae_match.group(1)),
                    'conf': float(mae_match.group(2))
                }
        
        except Exception as e:
            print(f"解析日志时出错: {str(e)}")
        
        return metrics
    
    def load_performance_results(self):
        """加载性能分析结果"""
        # 查找最新的性能分析结果
        pattern = f'{self.results_dir}/performance_profile_{self.dataset_name}_*.json'
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f"加载性能分析结果: {latest_file}")
            
            try:
                with open(latest_file, 'r') as f:
                    perf_data = json.load(f)
                
                self.report['performance'] = perf_data
            except Exception as e:
                print(f"加载性能数据时出错: {str(e)}")
        else:
            print("警告: 未找到性能分析结果")
    
    def generate_markdown_report(self):
        """生成Markdown格式的报告"""
        lines = []
        
        # 标题
        lines.append("# MARDM项目评估报告")
        lines.append("")
        lines.append(f"**生成时间**: {self.report['timestamp']}")
        lines.append(f"**数据集**: {self.report['dataset']}")
        lines.append("")
        
        # 目录
        lines.append("## 目录")
        lines.append("")
        lines.append("1. [评估概述](#评估概述)")
        lines.append("2. [模型性能指标](#模型性能指标)")
        lines.append("3. [性能分析](#性能分析)")
        lines.append("4. [指标说明](#指标说明)")
        lines.append("5. [结论与建议](#结论与建议)")
        lines.append("")
        
        # 评估概述
        lines.append("## 评估概述")
        lines.append("")
        lines.append("本报告对MARDM项目的所有模型进行了全面评估，包括：")
        lines.append("")
        lines.append("- **AutoEncoder (AE)**: 运动数据压缩与重建模型")
        lines.append("- **MARDM-SiT-XL**: 基于Scalable Interpolant Transformers的扩散模型")
        lines.append("- **MARDM-DDPM-XL**: 基于DDPM的扩散模型")
        lines.append("")
        
        # 模型性能指标
        lines.append("## 模型性能指标")
        lines.append("")
        
        # 创建对比表格
        if self.report['models']:
            lines.append("### 主要指标对比")
            lines.append("")
            
            # 表头
            lines.append("| 指标 | AE | MARDM-SiT-XL | MARDM-DDPM-XL |")
            lines.append("|------|-----|--------------|---------------|")
            
            # 收集所有指标
            all_metrics = set()
            for model_name, model_data in self.report['models'].items():
                if 'metrics' in model_data and model_data['metrics']:
                    all_metrics.update(model_data['metrics'].keys())
            
            # 填充表格
            for metric_name in sorted(all_metrics):
                row = [metric_name]
                
                for model_name in ['AE', 'MARDM_SiT_XL', 'MARDM_DDPM_XL']:
                    if model_name in self.report['models']:
                        metrics = self.report['models'][model_name].get('metrics', {})
                        if metric_name in metrics:
                            value = metrics[metric_name]
                            if isinstance(value, dict):
                                row.append(f"{value['mean']:.4f} ± {value['conf']:.4f}")
                            else:
                                row.append(f"{value:.4f}")
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")
                
                lines.append("| " + " | ".join(row) + " |")
            
            lines.append("")
        
        # 详细指标
        for model_name in ['AE', 'MARDM_SiT_XL', 'MARDM_DDPM_XL']:
            if model_name in self.report['models']:
                lines.append(f"### {model_name} 详细指标")
                lines.append("")
                
                metrics = self.report['models'][model_name].get('metrics', {})
                if metrics:
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, dict):
                            lines.append(f"- **{metric_name}**: {metric_value['mean']:.4f} ± {metric_value['conf']:.4f}")
                        else:
                            lines.append(f"- **{metric_name}**: {metric_value:.4f}")
                    lines.append("")
                else:
                    lines.append("*暂无评估数据*")
                    lines.append("")
        
        # 性能分析
        lines.append("## 性能分析")
        lines.append("")
        
        if 'performance' in self.report:
            perf = self.report['performance']
            
            # 系统信息
            if 'system_info' in perf:
                lines.append("### 测试环境")
                lines.append("")
                sys_info = perf['system_info']
                lines.append(f"- **PyTorch版本**: {sys_info.get('pytorch_version', 'N/A')}")
                lines.append(f"- **CUDA可用**: {sys_info.get('cuda_available', 'N/A')}")
                if sys_info.get('cuda_available'):
                    lines.append(f"- **CUDA版本**: {sys_info.get('cuda_version', 'N/A')}")
                    lines.append(f"- **GPU**: {', '.join(sys_info.get('gpu_names', []))}")
                lines.append(f"- **CPU核心数**: {sys_info.get('cpu_count', 'N/A')}")
                lines.append(f"- **内存**: {sys_info.get('ram_total', 'N/A')}")
                lines.append("")
            
            # 模型性能
            if 'models' in perf:
                for model_name, model_perf in perf['models'].items():
                    if model_perf is None:
                        continue
                    
                    lines.append(f"### {model_name} 性能")
                    lines.append("")
                    lines.append(f"- **参数量**: {model_perf.get('parameters_M', 0):.2f}M")
                    lines.append("")
                    
                    # AE模型的序列长度测试
                    if 'sequence_length_tests' in model_perf:
                        lines.append("#### 推理时间 (不同序列长度)")
                        lines.append("")
                        lines.append("| 序列长度 | 平均时间 (ms) | 吞吐量 (samples/s) |")
                        lines.append("|----------|---------------|-------------------|")
                        
                        for key, value in model_perf['sequence_length_tests'].items():
                            seq_len = key.replace('seq_len_', '')
                            lines.append(f"| {seq_len} | {value['mean_time_ms']:.2f} ± {value['std_time_ms']:.2f} | {value['throughput_samples_per_sec']:.2f} |")
                        
                        lines.append("")
                    
                    # MARDM模型的动作长度测试
                    if 'motion_length_tests' in model_perf:
                        lines.append("#### 生成时间 (不同动作长度)")
                        lines.append("")
                        lines.append("| 动作长度 | 帧数 | 平均时间 (ms) | 峰值内存 (MB) |")
                        lines.append("|----------|------|---------------|---------------|")
                        
                        for key, value in model_perf['motion_length_tests'].items():
                            motion_len = key.replace('motion_len_', '')
                            mem = value.get('peak_memory_mb', 'N/A')
                            mem_str = f"{mem:.2f}" if isinstance(mem, (int, float)) else mem
                            lines.append(f"| {motion_len} | {value['frames']} | {value['mean_time_ms']:.2f} ± {value['std_time_ms']:.2f} | {mem_str} |")
                        
                        lines.append("")
                    
                    # 批大小测试
                    if 'batch_size_tests' in model_perf:
                        lines.append("#### 批处理性能")
                        lines.append("")
                        lines.append("| 批大小 | 批处理时间 (ms) | 单样本时间 (ms) | 吞吐量 (samples/s) |")
                        lines.append("|--------|-----------------|-----------------|-------------------|")
                        
                        for key, value in model_perf['batch_size_tests'].items():
                            batch_size = key.replace('batch_', '')
                            if 'status' in value:
                                lines.append(f"| {batch_size} | {value['status']} | - | - |")
                            else:
                                lines.append(f"| {batch_size} | {value['mean_time_ms']:.2f} | {value['time_per_sample_ms']:.2f} | {value['throughput_samples_per_sec']:.2f} |")
                        
                        lines.append("")
                    
                    # 时间步测试
                    if 'timestep_tests' in model_perf:
                        lines.append("#### 采样步数影响")
                        lines.append("")
                        lines.append("| 采样步数 | 总时间 (ms) | 每步时间 (ms) |")
                        lines.append("|----------|-------------|---------------|")
                        
                        for key, value in model_perf['timestep_tests'].items():
                            steps = key.replace('steps_', '')
                            lines.append(f"| {steps} | {value['mean_time_ms']:.2f} | {value['time_per_step_ms']:.2f} |")
                        
                        lines.append("")
        else:
            lines.append("*暂无性能分析数据*")
            lines.append("")
        
        # 指标说明
        lines.append("## 指标说明")
        lines.append("")
        lines.append("### 质量指标")
        lines.append("")
        lines.append("- **FID (Fréchet Inception Distance)**: 衡量生成动作与真实动作分布的距离，越低越好")
        lines.append("- **Diversity**: 生成动作的多样性，反映模型生成不同动作的能力")
        lines.append("- **R-Precision (TOP1/TOP2/TOP3)**: 文本-动作检索准确率，越高越好")
        lines.append("- **Matching Score**: 文本与动作的匹配程度，越低越好")
        lines.append("- **Multimodality**: 同一文本生成多个不同动作的能力，越高越好")
        lines.append("- **CLIP-Score**: 基于CLIP的文本-动作对齐分数，越高越好")
        lines.append("- **MAE (Mean Absolute Error)**: 重建误差，仅用于AE模型，越低越好")
        lines.append("")
        
        lines.append("### 性能指标")
        lines.append("")
        lines.append("- **推理时间**: 单次前向传播所需时间")
        lines.append("- **生成时间**: 完整生成一个动作序列所需时间")
        lines.append("- **吞吐量**: 每秒可处理的样本数")
        lines.append("- **峰值内存**: 推理过程中的最大GPU内存占用")
        lines.append("")
        
        # 结论与建议
        lines.append("## 结论与建议")
        lines.append("")
        lines.append("### 模型对比")
        lines.append("")
        
        # 自动生成结论
        if 'MARDM_SiT_XL' in self.report['models'] and 'MARDM_DDPM_XL' in self.report['models']:
            sit_metrics = self.report['models']['MARDM_SiT_XL'].get('metrics', {})
            ddpm_metrics = self.report['models']['MARDM_DDPM_XL'].get('metrics', {})
            
            if sit_metrics and ddpm_metrics:
                lines.append("#### MARDM-SiT-XL vs MARDM-DDPM-XL")
                lines.append("")
                
                # 比较FID
                if 'FID' in sit_metrics and 'FID' in ddpm_metrics:
                    sit_fid = sit_metrics['FID']['mean']
                    ddpm_fid = ddpm_metrics['FID']['mean']
                    
                    if sit_fid < ddpm_fid:
                        lines.append(f"- **FID**: SiT-XL ({sit_fid:.4f}) 优于 DDPM-XL ({ddpm_fid:.4f})，改进 {((ddpm_fid - sit_fid) / ddpm_fid * 100):.2f}%")
                    else:
                        lines.append(f"- **FID**: DDPM-XL ({ddpm_fid:.4f}) 优于 SiT-XL ({sit_fid:.4f})，改进 {((sit_fid - ddpm_fid) / sit_fid * 100):.2f}%")
                
                # 比较R-Precision
                if 'R-Precision_TOP1' in sit_metrics and 'R-Precision_TOP1' in ddpm_metrics:
                    sit_top1 = sit_metrics['R-Precision_TOP1']['mean']
                    ddpm_top1 = ddpm_metrics['R-Precision_TOP1']['mean']
                    
                    if sit_top1 > ddpm_top1:
                        lines.append(f"- **R-Precision TOP1**: SiT-XL ({sit_top1:.4f}) 优于 DDPM-XL ({ddpm_top1:.4f})")
                    else:
                        lines.append(f"- **R-Precision TOP1**: DDPM-XL ({ddpm_top1:.4f}) 优于 SiT-XL ({sit_top1:.4f})")
                
                lines.append("")
        
        lines.append("### 改进建议")
        lines.append("")
        lines.append("1. **模型优化**")
        lines.append("   - 考虑模型剪枝或量化以提高推理速度")
        lines.append("   - 探索更高效的采样策略以减少生成时间")
        lines.append("")
        lines.append("2. **数据增强**")
        lines.append("   - 增加训练数据多样性")
        lines.append("   - 探索数据增强技术")
        lines.append("")
        lines.append("3. **评估扩展**")
        lines.append("   - 添加用户研究评估")
        lines.append("   - 增加更多下游任务评估")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append(f"*报告生成时间: {self.report['timestamp']}*")
        
        return "\n".join(lines)
    
    def save_report(self):
        """保存报告"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON
        json_file = f'{self.results_dir}/evaluation_report_{self.dataset_name}_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON报告已保存: {json_file}")
        
        # 保存Markdown
        md_file = f'{self.results_dir}/evaluation_report_{self.dataset_name}_{timestamp}.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        print(f"Markdown报告已保存: {md_file}")
        
        # 创建一个最新报告的链接
        latest_md = f'{self.results_dir}/evaluation_report_{self.dataset_name}_latest.md'
        with open(latest_md, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown_report())
        
        print(f"最新报告: {latest_md}")
    
    def generate(self):
        """生成完整报告"""
        print("="*80)
        print("生成评估报告")
        print("="*80)
        
        print("\n加载评估日志...")
        self.load_eval_logs()
        
        print("\n加载性能分析结果...")
        self.load_performance_results()
        
        print("\n生成报告...")
        self.save_report()
        
        print("\n✓ 报告生成完成!")


def main():
    parser = argparse.ArgumentParser(description='生成MARDM评估报告')
    parser.add_argument('--dataset_name', type=str, default='t2m',
                       choices=['t2m', 'kit'],
                       help='数据集名称')
    args = parser.parse_args()
    
    generator = ReportGenerator(dataset_name=args.dataset_name)
    generator.generate()


if __name__ == '__main__':
    main()

