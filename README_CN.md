# SC-HGNN-Rank复现材料

本包对应当前手稿采用的原SC-HGNN-Rank模型，包含真实处理后输入、固定划分、
正式运行记录、65份主模型检查点，以及匹配比较、敏感性、消融和纯图对照代码。
未用探索性的改进模型替代原模型。

## 安装与快速运行

使用Python 3.12，先完整解压文件。创建并激活虚拟环境后，安装依赖：

```text
python -m pip install -r environment/requirements-cpu.txt
python run.py smoke --device cpu --output outputs/quick_check
```

快速检查包括输入与13折核验、图和张量重建、检查点推理、外部证据及优先级重算，
并对8种比较模型各执行3个epoch的短训练。成功后生成
`outputs/quick_check/SMOKE_SUCCESS.json`。短训练不会替换正式结果。

GPU环境使用`environment/requirements-cuda126.txt`；macOS使用根目录的
`requirements.txt`。具体版本见`environment/`。

## 常用命令

```text
python run.py validate --output outputs/input_check
python run.py prepare --output outputs/rebuilt_inputs
python run.py infer --output outputs/checkpoint_predictions
python run.py scores --output outputs/evidence_scores
python run.py weights --output outputs/score_weight_sensitivity
python run.py summary --output outputs/reference_summary
python run.py ablation --mode check --output outputs/ablation_check
python run.py graph-only --mode check --output outputs/graph_only_check
```

`summary`从包内正式运行记录重新计算比较与统计，不训练新模型。
`infer`默认使用检查点清单中的第一份模型；可用`--checkpoint`指定其他模型。
`scores`按可评价来源均值重算外部证据及24条保留候选的优先级。

完整重训命令：

```text
python run.py train --phase all --device auto --output outputs/full_training
python run.py summary --results outputs/full_training --output outputs/new_summary
```

主流程包括520次匹配比较、210次新增训练敏感性分析及78次证据屏蔽分析；
敏感性分析另复用30次基线。模块消融和纯图对照分别另有260和65次正式运行。
所有运行均使用固定13折和相应已记录种子。完整训练建议使用GPU。

模型输入为600条候选、960个节点、9,576条边和873条关系扰动比较规则。
生成的文件统一写入`--output`指定目录，不覆盖包内输入和正式参考结果。
软件许可沿用`TERMS.md`，未另行授予新的许可证。

## 配套文件

`SC_HGNN_Rank_Explorer.html`是可直接用浏览器打开的独立文件，图片和数据均已内嵌。
增补数据包包含空间展示编号与来源对应、完整49张切片映射权重及补充源数据工作簿。
英文完整运行说明见`README.md`，图表对应关系见`docs/FIGURE_SOURCE_MAP.md`。
