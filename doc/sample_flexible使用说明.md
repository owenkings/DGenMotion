# sample_flexible.py 使用说明

## 功能介绍

`sample_flexible.py` 是 `sample.py` 的增强版本，新增了**灵活控制输出格式**的功能。

### 新增参数

```bash
--output_format [video|npy|both]
```

- **`video`**: 仅生成 MP4 视频文件
- **`npy`**: 仅生成 NPY 数据文件（跳过耗时的视频渲染）
- **`both`**: 同时生成 MP4 视频 + NPY 文件（默认）

---

## 使用示例

### 1️⃣ 同时生成视频和NPY文件（默认）

```bash
python sample_flexible.py \
  --name MARDM_SiT_XL \
  --text_prompt "A person is running on a treadmill." \
  --output_format both
```

或者省略参数（默认就是 both）：

```bash
python sample_flexible.py \
  --name MARDM_SiT_XL \
  --text_prompt "A person is running on a treadmill."
```

**输出：**
- ✅ `caption:A person is running on a treadmill._sample0_repeat0_len192.mp4`
- ✅ `caption:A person is running on a treadmill._sample0_repeat0_len192.npy`

---

### 2️⃣ 仅生成NPY文件（快速模式）

适合批量生成数据，跳过视频渲染以节省时间：

```bash
python sample_flexible.py \
  --name MARDM_SiT_XL \
  --text_prompt "A person is running on a treadmill." \
  --output_format npy
```

**输出：**
- ✅ `caption:A person is running on a treadmill._sample0_repeat0_len192.npy`
- ⏭️  跳过视频生成

**优势：** 大幅加快生成速度（视频渲染通常很慢）

---

### 3️⃣ 仅生成MP4视频

适合只需要可视化结果的场景：

```bash
python sample_flexible.py \
  --name MARDM_SiT_XL \
  --text_prompt "A person is running on a treadmill." \
  --output_format video
```

**输出：**
- ✅ `caption:A person is running on a treadmill._sample0_repeat0_len192.mp4`
- ⏭️  跳过NPY文件保存

---

### 4️⃣ 批量生成（从文件读取提示词）

创建一个文本文件 `prompts.txt`：

```
A person is walking forward.#120
A person is jumping.#80
A person is dancing.#160
```

运行：

```bash
python sample_flexible.py \
  --name MARDM_SiT_XL \
  --text_path prompts.txt \
  --output_format npy \
  --repeat_times 3
```

这将为每个提示词生成 3 次，总共 9 个 NPY 文件，无视频。

---

## 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--name` | str | MARDM | 模型checkpoint名称 |
| `--ae_name` | str | AE | AutoEncoder名称 |
| `--ae_model` | str | AE_Model | AE模型类型 |
| `--model` | str | MARDM-SiT-XL | 使用的MARDM模型 |
| `--dataset_name` | str | t2m | 数据集名称 (t2m/kit/eval_t2m/eval_kit) |
| `--dataset_dir` | str | ./datasets | 数据集目录 |
| `--checkpoints_dir` | str | ./checkpoints | checkpoint目录 |
| `--seed` | int | 3407 | 随机种子 |
| `--time_steps` | int | 18 | 扩散步数 |
| `--cfg` | float | 4.5 | Classifier-free guidance权重 |
| `--temperature` | float | 1.0 | 采样温度 |
| `--text_prompt` | str | "" | 单个文本提示词 |
| `--text_path` | str | "" | 文本提示词文件路径 |
| `--motion_length` | int | 0 | 动作长度（0表示自动估计） |
| `--repeat_times` | int | 1 | 每个提示词重复生成次数 |
| `--hard_pseudo_reorder` | flag | False | 是否使用硬伪重排序 |
| **`--output_format`** | **str** | **both** | **输出格式: video/npy/both** ⭐ |

---

## 性能对比

| 输出格式 | 速度 | 磁盘占用 | 适用场景 |
|---------|------|---------|---------|
| `npy` | 🚀 最快 | 💾 小 | 批量数据生成、训练数据准备 |
| `video` | 🐌 慢 | 📼 中 | 仅需可视化验证 |
| `both` | 🐢 最慢 | 💾📼 大 | 完整保存所有结果 |

---

## 典型使用流程

### 场景1: 快速生成大量训练数据

```bash
# 第一步：快速生成NPY数据
python sample_flexible.py \
  --text_path train_prompts.txt \
  --output_format npy \
  --repeat_times 10

# 第二步：从NPY文件中挑选几个可视化
python convert_npy_to_video.py --input ./generation/MARDM_SiT_XL_t2m/0/*.npy
```

### 场景2: 论文/展示用高质量可视化

```bash
python sample_flexible.py \
  --text_prompt "A person performs a graceful dance." \
  --output_format both \
  --repeat_times 5 \
  --motion_length 200
```

### 场景3: 调试模型输出

```bash
python sample_flexible.py \
  --text_prompt "Test motion." \
  --output_format video \
  --cfg 2.0
```

---

## 输出目录结构

```
generation/
└── MARDM_SiT_XL_t2m/
    ├── 0/
    │   ├── caption:A person is running on a treadmill._sample0_repeat0_len192.mp4
    │   └── caption:A person is running on a treadmill._sample0_repeat0_len192.npy
    ├── 1/
    │   └── ...
    └── 2/
        └── ...
```

---

## 注意事项

1. **视频生成很慢**：如果只需要数据，使用 `--output_format npy` 可以节省大量时间
2. **NPY文件格式**：形状为 `(seq_len, 22, 3)` (t2m) 或 `(seq_len, 21, 3)` (kit)
3. **需要Mean和Std文件**：确保 `./datasets/HumanML3D/Mean.npy` 和 `Std.npy` 存在
4. **GPU内存**：如果内存不足，减少 `--repeat_times` 或批量大小

---

## 与原版sample.py的区别

| 特性 | sample.py | sample_flexible.py |
|------|-----------|-------------------|
| 输出格式控制 | ❌ 固定both | ✅ 可选 video/npy/both |
| 视频生成 | ✅ 总是生成 | ✅ 可选 |
| NPY保存 | ✅ 总是保存 | ✅ 可选 |
| 速度优化 | ❌ 无 | ✅ 可跳过视频渲染 |
| 输出提示 | 基础 | 详细（带emoji和文件路径） |

---

## 快速上手

最简单的命令：

```bash
# 只生成NPY（最快）
python sample_flexible.py --text_prompt "A person walks." --output_format npy

# 只生成视频（可视化）
python sample_flexible.py --text_prompt "A person walks." --output_format video

# 全都要（完整）
python sample_flexible.py --text_prompt "A person walks." --output_format both
```

---

## 问题排查

### Q: 提示 "FileNotFoundError: Mean.npy"
**A:** 确保数据集目录下有 Mean.npy 和 Std.npy 文件

```bash
ls ./datasets/HumanML3D/Mean.npy
ls ./datasets/HumanML3D/Std.npy
```

如果没有，从 `./datasets/HumanML3D/HumanML3D/` 复制：

```bash
cp ./datasets/HumanML3D/HumanML3D/Mean.npy ./datasets/HumanML3D/
cp ./datasets/HumanML3D/HumanML3D/Std.npy ./datasets/HumanML3D/
```

### Q: 生成速度太慢
**A:** 使用 `--output_format npy` 跳过视频渲染

### Q: 想从NPY文件后续生成视频
**A:** 可以单独写一个转换脚本，或者重新运行时使用 `video` 格式

