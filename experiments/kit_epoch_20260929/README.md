# KIT：FSQ、FSQ＋JiT 与直接 APG

本目录保留当前运行版本的核心训练、评估、APG 和恢复实现。发布版与已运行版本的差异集中在两处：`prepare_epoch_run.py` 改为必填 `--root`，移除私有机器路径并检查源码复制目标不能位于源目录内部；`benchmark_control.py` 修正 KIT 并发负载检测和 SiT 动态 NFE 检查。核心 trainer/evaluator/APG 的数值实现不因发布而改变。

## 实验定义

| 训练任务 | 生成头 | 每个完整 epoch 的 validation |
| --- | --- | --- |
| `K_FSQ` | `FSQ-MARDM-SiT-XL`，原 SiT MLP/velocity 目标 | 原生 CFG |
| `K_FSQ_JiT` | `FSQ-MARDM-DiT-XL`，项目 JiT-style clean-coordinate MSE | 同一 EMA 权重分别使用 CFG、直接 APG |

两个生成器均从随机初始化训练 500 epoch，batch=16、seed=3407，无基于 FID 的早停。它们共享并冻结同一个 KIT FSQ AE：该 AE 已完成 50 epoch，当前冻结的是其中按 validation 选出的第 21 epoch 权重，不是第 50 epoch 的末尾权重。直接 APG 只改变推理，不增加第三条训练链，也不进行 APG 蒸馏。每轮两个 JiT sampler 在独立进程内使用同一 seed=3407，不消耗训练随机流；选模只看 validation。

KIT 原文件可为 251D；本实现取前 64D、21 joints，必须配套 64D 训练统计量和 KIT evaluator。FSQ levels 为 `[8,8,8,5,5,5]`。外层 18 轮、CFG=2.5；APG 为 beta=-0.5、eta=0、norm_threshold=0。JiT 采用固定 50 次迭代，原 SiT 保留自适应 `dopri5`；SiT 的 50 个输出网格点不是固定 50 次网络前向，不应宣称两者 NFE 相等。

## 本地资产与配置

运行需要 Linux、CUDA、已安装的项目依赖，以及足够容纳两条训练链和验证进程的 GPU/磁盘资源。已运行环境观测到的主要版本为：PyTorch `2.2.0`、CUDA `12.1`、torchdiffeq `0.2.5`、timm `1.0.9`、NumPy `1.21.5`、SciPy `1.10.1`、einops `0.6.1`。这些版本是现有环境记录，不是对新机器安装兼容性的保证。仓库 `environment.yml` 是历史完整环境导出，仅作依赖核对参考，尚未据此验证干净环境重建。

代码需要官方 [OpenAI CLIP](https://github.com/openai/CLIP) 提供的 `import clip`；仅安装 `open-clip-torch` 不能替代该模块。当前运行采用本地 `CLIP/` 源码目录，核验到的 Git 提交为 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。也可在训练前预装同一提交，例如在已配置的 Python 环境中执行以下安装命令；此处只给出示例，本次文档发布未执行安装：

```bash
python -m pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
```

若采用本地源码方式，应将该固定提交放在独立源码根目录的 `CLIP/` 下，使其随 prepare 一起复制；不要混用未知提交的 `clip` 实现。

训练和评估脚本不下载数据或模型。KIT 数据、文本、冻结 split、训练/评估统计量、两套 KIT evaluator、CLIP 权重都必须在开跑前已备齐。CLIP 的 ViT-B/32、ViT-B/16 权重缓存分别为 `~/.cache/clip/ViT-B-32.pt`、`~/.cache/clip/ViT-B-16.pt`；两者均按官方 URL 中的 SHA-256 校验，缺失或不匹配立即报错，绝不在训练或评估过程中自动下载。

`--root` 指向用户自己的资产工作区，例如 `/repro/asset-workspace`，不是隐含指向此 checkout。准备脚本要求该目录已包含：

```text
runs/paper_controls_20260920/
  KIT_base/config.json
  KIT_tokenizer/COMPLETE
  KIT_tokenizer/best.pt
  assets/...
```

`KIT_base/config.json` 提供真实本地路径、冻结哈希、模型与训练配置。AE 必须有原训练产生的 checkpoint/config/`COMPLETE` 来源记录：完成 50 epoch，所选 AE 哈希与 `COMPLETE` 一致，训练统计量也必须匹配；不能手写成功标记绕过检查。这里重用的是 AE 和资产配置，不载入旧基础生成器，准备脚本会移除 `base_checkpoint`。

`configs/K_FSQ.example.json` 和 `configs/K_FSQ_JiT.example.json` 是**准备后训练任务配置**的脱敏示例，保留本次运行的哈希、预算和采样规则；它们不含资产，也不能直接运行。所有 `/repro/...` 都需对应真实本地文件。路径可以调整，但不能删除哈希检查；如果资产不同，应说明来源并核验/锁定新的真实哈希，不应声称与本次资产完全相同。

示例中的 `code_root` 指向准备脚本将生成的 `experiments/kit_epoch_20260929/code`。用于准备的旧 `KIT_base/config.json` 则必须将 `code_root` 指向一个**独立源码目录**。例如，在此发布 checkout 根目录执行：

```bash
mkdir -p /repro/asset-workspace/source
git archive HEAD | tar -x -C /repro/asset-workspace/source
```

随后在真实旧资产配置中设置 `paths.code_root=/repro/asset-workspace/source`，并核对该独立源码的本地依赖。不把运行产物或数据写入源码目录。准备脚本会将其复制到本实验的 `code/`，并逐个核对 Python 源文件哈希；新 run 或目标 `code/` 已存在时拒绝覆盖。

## 准备、真实检查、启动与恢复

在发布 checkout 根目录、已激活项目环境后执行。`--help` 不读取资产。

```bash
python experiments/kit_epoch_20260929/prepare_epoch_run.py --help
python experiments/kit_epoch_20260929/prepare_epoch_run.py --root /repro/asset-workspace
python experiments/kit_epoch_20260929/run_real_checks.py --run /repro/asset-workspace/runs/kit_epoch_20260929 --tag real_checks_v1
python experiments/kit_epoch_20260929/supervise_epoch.py --run /repro/asset-workspace/runs/kit_epoch_20260929
```

真实检查对两种模型分别进行连续 8 更新、4＋恢复至 8 更新的一致性比较，并检查梯度与三条生成采样路径；全部成功才写入 `checks/READY.json`，supervisor 会再核对配置和源码身份。检查失败时先读取日志，修复后使用新的检查 tag。短检查通过不等于完成 500 epoch 或获得正式生成质量结果。本发布说明不宣称已在读者环境通过这些检查。

supervisor 当前会启动最多两条训练链，验证期间训练模型仍占用显存；先根据真实检查峰值确认容量，50＋500 epoch 的预算和每轮 validation 均有实际成本。本次数据快照的计划值是每轮 274 更新、每条 137,000 更新，上限 1,500 次生成 validation；真实每轮更新数由训练数据与有效 batch 计算。换数据后应核查该计划值，不能把它视为通用常数。建议在 `tmux` 或集群作业环境内运行 supervisor，使终端断开后继续工作。

意外中断后先确认旧 supervisor/训练进程已经退出，再显式恢复，不能重复运行 prepare 或并行启动同一 run：

```bash
python experiments/kit_epoch_20260929/supervise_epoch.py --run /repro/asset-workspace/runs/kit_epoch_20260929 --resume
```

## 保存与结果边界

- `last.pt` 是 latest 完整恢复点，含 student/EMA、optimizer/scheduler、随机状态、数据游标和选模状态；`last.prev.pt` 保留上一恢复点，`latest.json` 说明映射。
- `best.pt`/`best.json` 记录 CFG 最小 validation FID，JiT 另有 `best_apg.pt`/`best_apg.json` 和 `selection_apg.json`；平局保留较早 epoch。best 文件用于推理，完整续训使用 `last.pt`。
- 每个 sampler 成功后独立提交状态。恢复时补齐未完成的同轮验证，不重复已提交的成功验证；两个 best 可能来自不同 epoch，应分别报告轮次。
- 临时 EMA 快照有界保留，当前 best 引用的快照不被清理；全部历史验证指标保留。达到预算并完成验证才写 `COMPLETE` 和 `final.pt`。
- 中途每轮评估不是 test 选模。示例里的 test 字段不代表本流程会自动执行最终 test；最终 test 还需通过评估器的候选锁定流程。

`REMOTE_SOURCE_MANIFEST.json` 用于核对发布源码来源。共享负载下的运行耗时不构成独占速度基准，直接 APG 的效果也需要实际同协议结果支持。

## Benchmark publication notes

The published benchmark records the per-draw NFE distribution. Its legacy scalar
is null when draws differ. Adaptive SiT checkpoints must be timed separately;
alternate-checkpoint timing reuse requires an explicitly fixed-step sampler.
The Linux /proc worker check is best-effort observation, not proof of exclusive GPU access.

REMOTE_SOURCE_MANIFEST.json records original remote bytes, before publication-only
prepare/benchmark changes. RELEASE_VALIDATION.json identifies these differences.
Do not replace source files underneath an active run; changed source requires a new
run/check identity.
