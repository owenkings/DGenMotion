"""
MARDM性能分析脚本
测量推理时间、内存使用、吞吐量等性能指标
"""
import os
import sys
import json
import time
import torch
import psutil
import numpy as np
from os.path import join as pjoin
from datetime import datetime
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from models.AE import AE_models
from models.MARDM import MARDM_models


class PerformanceProfiler:
    def __init__(self, dataset_name='t2m'):
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if dataset_name == "t2m":
            self.dim_pose = 67
        else:
            self.dim_pose = 64
        
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_name,
            'device': str(self.device),
            'models': {}
        }
    
    def get_gpu_memory(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,  # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
            }
        return None
    
    def get_cpu_memory(self):
        """获取CPU内存使用情况"""
        process = psutil.Process()
        return {
            'rss': process.memory_info().rss / 1024**2,  # MB
            'vms': process.memory_info().vms / 1024**2,  # MB
            'percent': process.memory_percent()
        }
    
    def profile_ae_model(self, model_name='AE', num_samples=100):
        """性能分析AE模型"""
        print(f"\n{'='*80}")
        print(f"性能分析: {model_name}")
        print(f"{'='*80}")
        
        try:
            # 加载模型
            ae = AE_models['AE_Model'](input_width=self.dim_pose)
            ckpt_path = pjoin('./checkpoints', self.dataset_name, model_name, 'model',
                             'latest.tar' if self.dataset_name == 't2m' else 'net_best_fid.tar')
            
            if not os.path.exists(ckpt_path):
                print(f"模型文件不存在: {ckpt_path}")
                return None
            
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ae.load_state_dict(ckpt['ae'])
            ae.eval()
            ae.to(self.device)
            
            # 重置GPU内存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # 模型大小
            model_size = sum(p.numel() for p in ae.parameters()) / 1e6  # Million parameters
            trainable_size = sum(p.numel() for p in ae.parameters() if p.requires_grad) / 1e6
            
            print(f"模型参数量: {model_size:.2f}M (可训练: {trainable_size:.2f}M)")
            
            # 测试不同序列长度
            sequence_lengths = [64, 128, 196]
            length_results = {}
            
            for seq_len in sequence_lengths:
                print(f"\n测试序列长度: {seq_len}")
                
                # 准备测试数据
                test_input = torch.randn(1, seq_len, self.dim_pose).to(self.device)
                
                # 预热
                with torch.no_grad():
                    for _ in range(10):
                        _ = ae(test_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 测量推理时间
                inference_times = []
                for _ in range(num_samples):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output = ae(test_input)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    inference_times.append(time.time() - start_time)
                
                # 统计
                inference_times = np.array(inference_times) * 1000  # 转换为ms
                
                length_results[f'seq_len_{seq_len}'] = {
                    'mean_time_ms': float(np.mean(inference_times)),
                    'std_time_ms': float(np.std(inference_times)),
                    'min_time_ms': float(np.min(inference_times)),
                    'max_time_ms': float(np.max(inference_times)),
                    'median_time_ms': float(np.median(inference_times)),
                    'throughput_samples_per_sec': 1000.0 / np.mean(inference_times)
                }
                
                print(f"  平均推理时间: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")
                print(f"  吞吐量: {1000.0 / np.mean(inference_times):.2f} samples/sec")
            
            # 批处理测试
            batch_sizes = [1, 4, 8, 16, 32]
            batch_results = {}
            seq_len = 128  # 使用中等长度
            
            print(f"\n测试不同批大小 (序列长度={seq_len}):")
            
            for batch_size in batch_sizes:
                try:
                    test_input = torch.randn(batch_size, seq_len, self.dim_pose).to(self.device)
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(5):
                            _ = ae(test_input)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    
                    # 测量
                    inference_times = []
                    for _ in range(50):
                        start_time = time.time()
                        
                        with torch.no_grad():
                            output = ae(test_input)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        inference_times.append(time.time() - start_time)
                    
                    inference_times = np.array(inference_times) * 1000
                    
                    batch_results[f'batch_{batch_size}'] = {
                        'mean_time_ms': float(np.mean(inference_times)),
                        'std_time_ms': float(np.std(inference_times)),
                        'throughput_samples_per_sec': batch_size * 1000.0 / np.mean(inference_times),
                        'time_per_sample_ms': float(np.mean(inference_times) / batch_size)
                    }
                    
                    if torch.cuda.is_available():
                        batch_results[f'batch_{batch_size}']['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
                    
                    print(f"  Batch {batch_size}: {np.mean(inference_times):.2f}ms, "
                          f"{batch_size * 1000.0 / np.mean(inference_times):.2f} samples/sec")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Batch {batch_size}: OOM")
                        batch_results[f'batch_{batch_size}'] = {'status': 'OOM'}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise
            
            # 内存使用
            memory_info = {
                'gpu': self.get_gpu_memory(),
                'cpu': self.get_cpu_memory()
            }
            
            self.results['models'][model_name] = {
                'model_type': 'AutoEncoder',
                'parameters_M': model_size,
                'trainable_parameters_M': trainable_size,
                'sequence_length_tests': length_results,
                'batch_size_tests': batch_results,
                'memory_usage': memory_info
            }
            
            print(f"\n✓ {model_name} 性能分析完成")
            return self.results['models'][model_name]
            
        except Exception as e:
            print(f"✗ {model_name} 性能分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def profile_mardm_model(self, model_name='MARDM_SiT_XL', model_type='MARDM-SiT-XL', 
                           ae_name='AE', num_samples=20):
        """性能分析MARDM模型"""
        print(f"\n{'='*80}")
        print(f"性能分析: {model_name}")
        print(f"{'='*80}")
        
        try:
            # 加载AE
            ae = AE_models['AE_Model'](input_width=self.dim_pose)
            ae_ckpt_path = pjoin('./checkpoints', self.dataset_name, ae_name, 'model',
                                'latest.tar' if self.dataset_name == 't2m' else 'net_best_fid.tar')
            
            if not os.path.exists(ae_ckpt_path):
                print(f"AE模型文件不存在: {ae_ckpt_path}")
                return None
            
            ae_ckpt = torch.load(ae_ckpt_path, map_location='cpu')
            ae.load_state_dict(ae_ckpt['ae'])
            ae.eval()
            ae.to(self.device)
            
            # 加载MARDM
            mardm = MARDM_models[model_type](ae_dim=ae.output_emb_width, cond_mode='text')
            mardm_ckpt_path = pjoin('./checkpoints', self.dataset_name, model_name, 'model', 'latest.tar')
            
            if not os.path.exists(mardm_ckpt_path):
                print(f"MARDM模型文件不存在: {mardm_ckpt_path}")
                return None
            
            mardm_ckpt = torch.load(mardm_ckpt_path, map_location='cpu')
            mardm.load_state_dict(mardm_ckpt['ema_mardm'], strict=False)
            mardm.eval()
            mardm.to(self.device)
            
            # 重置GPU内存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # 模型大小
            model_size = sum(p.numel() for p in mardm.parameters()) / 1e6
            trainable_size = sum(p.numel() for p in mardm.parameters() if p.requires_grad) / 1e6
            
            print(f"模型参数量: {model_size:.2f}M (可训练: {trainable_size:.2f}M)")
            
            # 测试不同序列长度的生成
            motion_lengths = [16, 32, 49]  # latent space lengths (对应64, 128, 196帧)
            length_results = {}
            
            cfg_scale = 4.5 if self.dataset_name == 't2m' else 2.5
            time_steps = 18
            
            # 准备测试文本
            test_texts = ["A person is walking forward."]
            
            for motion_len in motion_lengths:
                print(f"\n测试动作长度: {motion_len} (约{motion_len*4}帧)")
                
                m_lengths = torch.tensor([motion_len]).to(self.device)
                
                # 预热
                with torch.no_grad():
                    for _ in range(3):
                        _ = mardm.generate(test_texts, m_lengths, time_steps, cfg_scale)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # 测量生成时间
                generation_times = []
                for _ in range(num_samples):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        latents = mardm.generate(test_texts, m_lengths, time_steps, cfg_scale)
                        motions = ae.decode(latents)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    generation_times.append(time.time() - start_time)
                
                generation_times = np.array(generation_times) * 1000  # ms
                
                length_results[f'motion_len_{motion_len}'] = {
                    'mean_time_ms': float(np.mean(generation_times)),
                    'std_time_ms': float(np.std(generation_times)),
                    'min_time_ms': float(np.min(generation_times)),
                    'max_time_ms': float(np.max(generation_times)),
                    'median_time_ms': float(np.median(generation_times)),
                    'frames': motion_len * 4
                }
                
                if torch.cuda.is_available():
                    length_results[f'motion_len_{motion_len}']['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
                
                print(f"  平均生成时间: {np.mean(generation_times):.2f} ± {np.std(generation_times):.2f} ms")
                print(f"  每帧时间: {np.mean(generation_times) / (motion_len * 4):.2f} ms/frame")
            
            # 测试不同时间步数
            print(f"\n测试不同采样步数 (动作长度={motion_lengths[1]}):")
            timestep_results = {}
            m_lengths = torch.tensor([motion_lengths[1]]).to(self.device)
            
            for steps in [10, 18, 25, 50]:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                generation_times = []
                for _ in range(num_samples):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        latents = mardm.generate(test_texts, m_lengths, steps, cfg_scale)
                        motions = ae.decode(latents)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    generation_times.append(time.time() - start_time)
                
                generation_times = np.array(generation_times) * 1000
                
                timestep_results[f'steps_{steps}'] = {
                    'mean_time_ms': float(np.mean(generation_times)),
                    'std_time_ms': float(np.std(generation_times)),
                    'time_per_step_ms': float(np.mean(generation_times) / steps)
                }
                
                print(f"  {steps}步: {np.mean(generation_times):.2f}ms, "
                      f"{np.mean(generation_times) / steps:.2f}ms/step")
            
            # 内存使用
            memory_info = {
                'gpu': self.get_gpu_memory(),
                'cpu': self.get_cpu_memory()
            }
            
            self.results['models'][model_name] = {
                'model_type': model_type,
                'parameters_M': model_size,
                'trainable_parameters_M': trainable_size,
                'motion_length_tests': length_results,
                'timestep_tests': timestep_results,
                'memory_usage': memory_info,
                'default_cfg_scale': cfg_scale,
                'default_time_steps': time_steps
            }
            
            print(f"\n✓ {model_name} 性能分析完成")
            return self.results['models'][model_name]
            
        except Exception as e:
            print(f"✗ {model_name} 性能分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_profiling(self):
        """运行完整性能分析"""
        print(f"\n{'='*80}")
        print("MARDM性能分析")
        print(f"数据集: {self.dataset_name}")
        print(f"设备: {self.device}")
        print(f"开始时间: {self.results['timestamp']}")
        print(f"{'='*80}")
        
        # 1. 分析AE
        self.profile_ae_model('AE', num_samples=100)
        
        # 2. 分析MARDM-SiT-XL
        self.profile_mardm_model('MARDM_SiT_XL', 'MARDM-SiT-XL', num_samples=20)
        
        # 3. 分析MARDM-DDPM-XL
        self.profile_mardm_model('MARDM_DDPM_XL', 'MARDM-DDPM-XL', num_samples=20)
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存性能分析结果"""
        output_dir = './evaluation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'{output_dir}/performance_profile_{self.dataset_name}_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n性能分析结果已保存到: {output_file}")
        
        # 保存文本版本
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(self.format_results_text())
        
        print(f"文本报告已保存到: {txt_file}")
    
    def format_results_text(self):
        """格式化结果为文本"""
        lines = []
        lines.append("="*80)
        lines.append("MARDM性能分析报告")
        lines.append("="*80)
        lines.append(f"\n分析时间: {self.results['timestamp']}")
        lines.append(f"数据集: {self.results['dataset']}")
        lines.append(f"设备: {self.results['device']}")
        
        for model_name, model_results in self.results['models'].items():
            lines.append(f"\n{'='*80}")
            lines.append(f"模型: {model_name}")
            lines.append(f"{'='*80}")
            
            if model_results is None:
                lines.append("分析失败")
                continue
            
            lines.append(f"模型类型: {model_results.get('model_type', 'N/A')}")
            lines.append(f"参数量: {model_results.get('parameters_M', 0):.2f}M")
            
            # 序列长度测试
            if 'sequence_length_tests' in model_results:
                lines.append("\n序列长度测试:")
                for key, value in model_results['sequence_length_tests'].items():
                    lines.append(f"  {key}:")
                    lines.append(f"    推理时间: {value['mean_time_ms']:.2f} ± {value['std_time_ms']:.2f} ms")
                    lines.append(f"    吞吐量: {value['throughput_samples_per_sec']:.2f} samples/sec")
            
            # 动作长度测试
            if 'motion_length_tests' in model_results:
                lines.append("\n动作长度测试:")
                for key, value in model_results['motion_length_tests'].items():
                    lines.append(f"  {key} ({value['frames']}帧):")
                    lines.append(f"    生成时间: {value['mean_time_ms']:.2f} ± {value['std_time_ms']:.2f} ms")
                    if 'peak_memory_mb' in value:
                        lines.append(f"    峰值内存: {value['peak_memory_mb']:.2f} MB")
            
            # 批大小测试
            if 'batch_size_tests' in model_results:
                lines.append("\n批大小测试:")
                for key, value in model_results['batch_size_tests'].items():
                    if 'status' in value:
                        lines.append(f"  {key}: {value['status']}")
                    else:
                        lines.append(f"  {key}:")
                        lines.append(f"    批处理时间: {value['mean_time_ms']:.2f} ms")
                        lines.append(f"    单样本时间: {value['time_per_sample_ms']:.2f} ms")
                        lines.append(f"    吞吐量: {value['throughput_samples_per_sec']:.2f} samples/sec")
            
            # 时间步测试
            if 'timestep_tests' in model_results:
                lines.append("\n采样步数测试:")
                for key, value in model_results['timestep_tests'].items():
                    lines.append(f"  {key}:")
                    lines.append(f"    总时间: {value['mean_time_ms']:.2f} ms")
                    lines.append(f"    每步时间: {value['time_per_step_ms']:.2f} ms")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """打印性能总结"""
        print(f"\n{'='*80}")
        print("性能分析总结")
        print(f"{'='*80}")
        
        for model_name, model_results in self.results['models'].items():
            if model_results is None:
                continue
            
            print(f"\n{model_name}:")
            print(f"  参数量: {model_results.get('parameters_M', 0):.2f}M")
            
            if 'sequence_length_tests' in model_results:
                # 取中等长度的结果
                mid_key = list(model_results['sequence_length_tests'].keys())[1]
                mid_result = model_results['sequence_length_tests'][mid_key]
                print(f"  推理时间 (典型): {mid_result['mean_time_ms']:.2f}ms")
            
            if 'motion_length_tests' in model_results:
                mid_key = list(model_results['motion_length_tests'].keys())[1]
                mid_result = model_results['motion_length_tests'][mid_key]
                print(f"  生成时间 (典型): {mid_result['mean_time_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description='MARDM性能分析')
    parser.add_argument('--dataset_name', type=str, default='t2m',
                       choices=['t2m', 'kit'],
                       help='数据集名称')
    args = parser.parse_args()
    
    profiler = PerformanceProfiler(dataset_name=args.dataset_name)
    profiler.run_full_profiling()


if __name__ == '__main__':
    main()

