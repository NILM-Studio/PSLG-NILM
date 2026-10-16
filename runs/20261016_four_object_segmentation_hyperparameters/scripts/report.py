"""Build the stage-one frozen configuration, tables, and ten-panel diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from choose_finalists import OBJECTS

EXP = Path(__file__).resolve().parents[1]
METHODS = ("prim-glr","pelt-rbf","fluss-threshold","clasp","clap","e2usd")
FROZEN = json.loads((EXP/"data/frozen_inputs_and_split.json").read_text())
SELECTION = json.loads((EXP/"data/screening_selection.json").read_text())


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def validation_records(obj,method,entry):
    path=Path(entry["validation_execution"])/"analysis/records.json"
    return json.loads(path.read_text())


def main():
    if not (EXP/"review/human_review_decisions.json").exists():
        raise RuntimeError("Ten-panel human review decisions are required before final report")
    chosen={}
    csv_rows=[]
    candidate_rows=[]
    panels=[]
    for obj in OBJECTS:
        chosen[obj]={}
        for method in METHODS:
            safe=method.replace("-","_")
            entry=json.loads((EXP/"data"/f"selected_{safe}.json").read_text())[obj]
            if "human_review" not in entry:
                raise RuntimeError(f"Unapproved selection: {obj} {method}")
            finalist=json.loads((EXP/"data"/f"finalists_{safe}.json").read_text())[obj]
            selected=entry["selected"]
            chosen[obj][method]={"params":entry["run_params"],
                                 "validation_execution":entry["validation_execution"],
                                 "score":selected["score"],
                                 "usable_on_validation":selected["success"]>0,
                                 "validation_success":selected["success"],
                                 "validation_unsupported":selected["unsupported"],
                                 "validation_score_components":selected["score_components"]}
            for rank,row in enumerate(finalist["ranking"],1):
                candidate_rows.append({"object":obj,"method":method,"development_rank":rank,
                     "config_id":row["config_id"],"params":json.dumps(row["params"],sort_keys=True),
                     "success":row["success"],"unsupported":row["unsupported"],
                     "score":row["score"],"is_baseline":row["is_baseline"],
                     "chosen_for_validation":any(f["parent_config_id"]==row["config_id"]
                                                  for f in finalist["finalists"])})
            rows=validation_records(obj,method,entry)
            components={name:selected["score_components"].get(name) for name in
              ("nrmse","cut_density","short30_fraction","proxy_miss_fraction","zero_cut_fraction")}
            csv_rows.append({"object":obj,"method":method,
              "params":json.dumps(entry["run_params"],sort_keys=True),
              "success":selected["success"],"unsupported":selected["unsupported"],
              "validation_attempts":len(SELECTION[obj]["validation"])*(3 if method=="e2usd" else 1),
              "score":selected["score"],**components,
              "validation_execution":entry["validation_execution"]})
    json_write(EXP/"configs/frozen.json",{
      "experiment":"20261016_four_object_segmentation_hyperparameters",
      "status":"frozen_after_ten_panel_human_review",
      "methods":list(METHODS),"objects":chosen,
      "stage2_evaluation_split_status":"sealed",
      "selection_rule":"protocol.md: success count, fixed scalar score, runtime, parameter dictionary"})
    for file,rows in ((EXP/"analysis/candidate_rankings.csv",candidate_rows),
                      (EXP/"analysis/selected_validation_summary.csv",csv_rows)):
        file.parent.mkdir(parents=True,exist_ok=True)
        with file.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    lines=["# 四对象切分超参数冻结报告","",
      "状态：第一阶段已完成筛参和验证，第二阶段正式实验一尚未启动。","",
      "## 设计和限制","",
      "四对象共 1,807 条活动；按核心时间冻结开发/验证/评价划分。四对象的验证活动分别为 12、12、11、12 条。", "各对象、各方法在相同验证 ID 上比较参数；E2USD 每配置使用 3 个种子，其余固定种子 0。短序列不适用独立计数。",
      "开发侧的活动与先前全量调参复核有重叠，因此此次时间划分是回顾性划分；不能声称完全独立的外部验证。", 
      "无人工真值边界；代理跃迁是弱参考。得分是事先冻结的工程折中，不是真实切分准确率。CSR_SE 尚未恢复，未用于筛参。评价侧保持封存。", 
      "FLUSS 初次验证出现近乎全零切点，随后扩大 CAC 阈值并重跑；此次修订发生在查看验证结果之后，因此 FLUSS 的第二轮验证属于探索性确认，详见 `amendment1_fluss.md`。", 
      "Clasp 使用 claspy 官方实现；CLaP 使用同对象冻结的 Clasp 初始化。K=2/3/3/4 属于后续聚类，不参与切分选参。",
      "", "## 冻结参数与验证结果", "",
      "| 对象 | 方法 | 参数 | 成功/尝试 | 分数 | NRMSE | 切点/10分钟 | 短段率 | 代理漏检率 | 零切点率 |", 
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def show(value,digits=2):
        return "—" if value is None else f"{value:.{digits}f}"
    for row in csv_rows:
        parts=[row["object"],row["method"],"`"+row["params"]+"`",
               f"{row['success']}/{row['validation_attempts']}",
               show(row['score'],4),show(row['nrmse'],4),
               show(row['cut_density']),show(row['short30_fraction']),
               show(row['proxy_miss_fraction']),show(row['zero_cut_fraction'])]
        lines.append("| "+" | ".join(parts)+" |")
    lines += ["", "完整候选排名：`analysis/candidate_rankings.csv`。验证分量：`analysis/selected_validation_summary.csv`。机器可读冻结配置：`configs/frozen.json`。", 
              "成功/尝试为 0 的行只记录候选配置，不代表方法已可用；正式实验一需单独处理不适用，不能将其作为有效零切点。", 
              "", "## 十子图检查页", "", 
              "审核图库位于 `review/figures/`。每张 5×2 十子图固定一条活动，逐格比较候选参数；详见 `review/index.md` 与 `review/figure_manifest.json`。", ""]
    lines += ["", "## 后续", "", 
              "正式实验一使用冻结参数运行封存评价侧，再按论文协议完成活动状态提取与切分对照。若恢复 CSR_SE，需先冻结定义和共享参考；不能根据评价侧结果重新调参。", ""]
    (EXP/"hyperparameters.md").write_text("\n".join(lines),encoding="utf-8")
    out=Path(os.environ.get("PSLG_EXECUTION_DIR",EXP/"analysis"))
    json_write(out/"analysis/report_manifest.json",{
        "frozen_config_sha256":hashlib.sha256((EXP/"configs/frozen.json").read_bytes()).hexdigest(),
        "hyperparameters_sha256":hashlib.sha256((EXP/"hyperparameters.md").read_bytes()).hexdigest(),
        "ten_panel_count":len(json.loads((EXP/"review/figure_manifest.json").read_text())["figures"]),
        "candidate_count":len(candidate_rows),
        "selected_count":len(csv_rows)})
    print("REPORT",len(csv_rows),"human-approved configurations")


if __name__=="__main__":main()
