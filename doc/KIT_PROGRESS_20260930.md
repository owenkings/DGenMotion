# KIT 逐轮实验进度快照

采集时间：2026-09-30 22:20:06（Asia/Shanghai）。这是一份当时的现场快照，不代表实时状态或训练已完成。

| 方法 | 已完成训练轮次 / 预算 | 已完成对应验证轮次 | 最低 validation generation FID | 最优轮次 |
|---|---:|---:|---:|---:|
| FSQ（SiT/MLP velocity + CFG） | 156 / 500 | 155 | 0.95466653 | 98 |
| FSQ＋JiT（clean prediction + CFG） | 92 / 500 | 92 | 1.28496905 | 90 |
| FSQ＋JiT＋直接 APG | 共享 JiT 权重 | 91 | 2.27207064 | 91 |

两条训练进程存活、supervisor 心跳正常，分别正在验证第156轮CFG和第92轮APG。
`last.pt`、`last.prev.pt`、`best.pt` 均已保存；JiT 另有 `best_apg.pt`。
APG 是采样引导，不是第三条独立训练或蒸馏训练。

协议：KIT val，seed3407，batch32、shuffle=True、drop_last=True，9批共288条生成序列；
每个完整训练epoch后评估。JiT CFG和APG使用同轮EMA权重及相同输入随机种子。
此表分别报告各分支迄今最低验证FID，不是多个随机种子的均值，也不是KIT test成绩。
复用的AE训练预算为50轮，实际为第21轮选出的best权重；两个生成器从头训练，预算各500轮。

当前观察：KIT上还没有APG改善的证据。第91轮相同JiT权重下，CFG FID为
1.30357127，APG为2.27207064；当前各自最优结果也以CFG较好。
FSQ与JiT已训练的轮次不同，暂不能将本表作为最终架构优劣结论。
HumanML与KIT以及validation与test指标应分开报告。

最近完整轮次的中位耗时约为FSQ 13.3分钟、JiT双评16.4分钟。
若此速度保持不变，从本快照时刻到500轮约还需77小时和112小时；
并发负载和自适应求解调用数会改变耗时，该估计不是完成时间保证。

验证依据：远程正式运行的progress、selection、checkpoint元数据和各轮summary；
本地保留完整采集记录。公开仓库只包含不带机器连接信息的摘要。
发布前已通过4项CPU状态逻辑检查；此前真实KIT短测验证了两套模型的有限非零梯度、
连续8步与4＋4恢复逐项一致，以及FSQ CFG、JiT CFG、JiT APG三分支真实生成。
这些检查验证实现和恢复能力，不证明收敛质量。
