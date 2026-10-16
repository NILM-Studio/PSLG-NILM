"""Check ten-panel gallery artifacts and build a human-review index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

EXP=Path(__file__).resolve().parents[1]
METHODS=("prim-glr","pelt-rbf","fluss-threshold","clasp","clap","e2usd")
OBJECTS=("refit_h7_dishwasher","refit_h20_tumble_dryer",
         "ukdale_h2_washing_machine","ukdale_h1_washing_machine_hotpoint")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    manifests=[];figures=[];missing=[];review_rows=[]
    for method in METHODS:
        safe=method.replace("-","_")
        for split in ("development","validation"):
            path=EXP/"review/manifests"/f"{split}_{safe}.json"
            if not path.exists():
                missing.append(f"{method} {split}")
                continue
            data=json.loads(path.read_text())
            if len(data["figures"])!=40:
                raise RuntimeError(f"Expected 40 figures in {path}, got {len(data['figures'])}")
            for figure in data["figures"]:
                if len(figure["panels"])!=10:
                    raise RuntimeError(f"Not ten panels: {figure['png']}")
                for key in ("png","svg"):
                    target=EXP/figure[key]
                    if not target.is_file() or target.stat().st_size==0:
                        raise FileNotFoundError(target)
                with Image.open(EXP/figure["png"]) as image:
                    image.verify()
                ET.parse(EXP/figure["svg"])
                for panel in figure["panels"]:
                    if panel["status"] not in ("unused","missing"):
                        if panel["drawn_cut_lines"]!=len(panel["cuts"]):
                            raise RuntimeError(f"Cut-line mismatch: {figure['png']} panel {panel['panel']}")
                        if len(panel["cut_timestamps_utc"])!=len(panel["cuts"]):
                            raise RuntimeError(f"Cut-timestamp mismatch: {figure['png']} panel {panel['panel']}")
                figures.append({"method":method,"split":split,"object":figure["object"],
                                "activity_id":figure["activity_id"],"png":figure["png"],
                                "svg":figure["svg"],"input_sha256":figure["input_sha256"],
                                "panels":figure["panels"]})
            manifests.append({"method":method,"split":split,"path":str(path.relative_to(EXP)),
                              "sha256":sha(path),"figure_count":len(data["figures"])})
        screen_path=EXP/"data"/f"finalists_{safe}.json"
        if screen_path.exists():
            data=json.loads(screen_path.read_text())
            for obj,block in data.items():
                shortlist={row["parent_config_id"] for row in block["finalists"]}
                for rank,item in enumerate(block["ranking"],1):
                    if rank<=10 or item["is_baseline"] or item["config_id"] in shortlist:
                        review_rows.append({"object":obj,"method":method,
                          "development_rank":rank,"config_id":item["config_id"],
                          "params":json.dumps(item["params"],ensure_ascii=False,sort_keys=True),
                          "development_success":item["success"],
                          "development_score":item["score"],
                          "validation_finalist":item["config_id"] in shortlist,
                          "is_baseline":item["is_baseline"],
                          "human_decision":"","human_note":""})
    if args.require_complete and missing:
        raise RuntimeError("Missing galleries: "+", ".join(missing))
    payload={"experiment":"20261016_four_object_segmentation_hyperparameters",
             "status":"awaiting_human_review", "manifest_count":len(manifests),
             "figure_count":len(figures),"missing":missing,
             "manifests":manifests,"figures":figures}
    root=EXP/"review"
    (root/"figure_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    if review_rows:
        with (root/"human_review_template.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(review_rows[0]))
            writer.writeheader();writer.writerows(review_rows)
    lines=["# 四对象切分超参数十子图人工审核", "",
      "**状态：待用户审核。** 所有机器排序只作建议；未写最终冻结配置，正式实验一未启动。", "",
      "每张图固定一条活动原始功率波形，5×2 十格比较不同参数。黑线为功率，红虚线为原始切点，灰区为上下文；各格使用相同坐标。格子不足十组时标为无更多候选。",
      "开发集图覆盖候选排名靠前配置与默认值；验证集图覆盖入围配置。切点是否对应有效运行阶段、是否漏掉长平台转折或过度切碎，须由人判断。", "",
      "CLaP 图若使用 `data/provisional_clasp_initializer.json`，只属于暂定 Clasp 初始化下的探索性对照；Clasp 人审选择不同参数时必须重跑 CLaP。", "",
      f"当前 {len(figures)} 张 PNG + {len(figures)} 张 SVG；缺失：{', '.join(missing) if missing else '无'}。",
      "逐子图哈希、切点索引、UTC 时间和绘图线数见 `figure_manifest.json`；审核表见 `human_review_template.csv`。",
      "", "## 图库", ""]
    for obj in OBJECTS:
        lines += [f"### {obj}",""]
        for method in METHODS:
            for split in ("development","validation"):
                subset=[f for f in figures if f["object"]==obj and f["method"]==method and f["split"]==split]
                if not subset:
                    lines.append(f"- {method} / {split}: 待生成")
                    continue
                links=" · ".join(f"[{f['activity_id']}]({f['png'].replace('review/','')})" for f in subset)
                lines.append(f"- {method} / {split}: {links}")
        lines.append("")
    lines += ["## 审核规则", "",
      "请根据十子图对每个对象×方法给出可接受配置、需排除的明显错误切点与理由。Clasp 参数还决定 CLaP 的初始边界；若修改 Clasp，CLaP 图必须重跑。",
      "同一图中的参数和切点可由 manifest 精确追溯；机器评分、代理跃迁与少量图均不能单独证明真实边界。",
      "人工审核记录到位后才运行带 `--review-decisions` 的冻结脚本。", ""]
    (root/"index.md").write_text("\n".join(lines),encoding="utf-8")
    if "PSLG_EXECUTION_DIR" in os.environ:
        out=Path(os.environ["PSLG_EXECUTION_DIR"])/"analysis/gallery_qa.json"
        out.write_text(json.dumps({"status":"passed","figures":len(figures),
                    "missing":missing,"manifest_sha256":sha(root/"figure_manifest.json")},indent=2)+"\n")
    print("GALLERY_INDEX",len(figures),"figures",len(missing),"missing")


if __name__=="__main__":main()
