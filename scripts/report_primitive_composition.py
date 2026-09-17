"""Read audited composition outputs and create a separate report, CSV and PNGs.

This command never reconstructs, modifies or regenerates a waveform. It only
needs a copied study directory (JSON and NPZ), not the original source CSVs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from scripts.audit_primitive_composition import audit_composition


def _support(row):
    supported = row["transition_supported_edges"]
    fallback = len(row["transition_fallback_edges"])
    return "none" if not supported else "partial" if fallback else "full"


def _case_key(case, ratios):
    return (case["seed"], ratios[(case["seed"], case["budget_tag"])],
            tuple(case["class_mode"]), case["anchor_activity_id"], case["case_id"])


def _representatives(manifest, maximum):
    """Round-robin budgets, choosing each class/mode first; never rank scores."""
    ratios = {(row["seed"], row["budget_tag"]): row["real_ratio"] for row in manifest["budgets"]}
    ordered = sorted(manifest["cases"], key=lambda case: _case_key(case, ratios))
    preferred, extra, seen = defaultdict(list), defaultdict(list), set()
    for case in ordered:
        budget = case["seed"], case["budget_tag"]
        group = (*budget, tuple(case["class_mode"]))
        (extra if group in seen else preferred)[budget].append(case)
        seen.add(group)
    buckets = [preferred[key] + extra[key] for key in preferred]
    selected = []
    for index in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket):
                selected.append(bucket[index])
    return selected[:maximum]


def _number(value):
    return "NA" if value is None else f"{float(value):.5g}"


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows], ""]


def _rows(manifest, images):
    rows = []
    for case in manifest["cases"]:
        for method in manifest["methods"]:
            result = case["results"][method]
            rows.append({
                "case_id": case["case_id"], "seed": case["seed"],
                "budget_tag": case["budget_tag"], "class_id": case["class_mode"][0],
                "mode_id": case["class_mode"][1], "anchor_activity_id": case["anchor_activity_id"],
                "method": method, "support": _support(result),
                "state_blocks": len(case["template"]),
                "supported_edges": result["transition_supported_edges"],
                "fallback_edges": len(result["transition_fallback_edges"]),
                "single_donor_cycle": result["single_donor_cycle"],
                "nearest_donor_resampled_nrmse": result["nearest_donor_resampled_nrmse"],
                "nearest_donor_activity_id": result.get("nearest_donor_activity_id", ""),
                "largest_donor_sample_fraction": result.get("largest_donor_sample_fraction", ""),
                "duration_target_cost": result.get("duration_target_cost", ""),
                "unit_selection_objective": result.get("unit_selection_objective", ""),
                "boundary_objective": result["boundary_objective"],
                "transition_objective": result["transition_objective"],
                "energy_wh": result["metrics"]["energy_wh"],
                "mean_watts": result["metrics"]["mean_watts"],
                "peak_watts": result["metrics"]["peak_watts"],
                "block_state_labels": ";".join(str(block["state_label"]) for block in case["template"]),
                "block_donor_activity_ids": ";".join(str(source["activity_id"]) for source in result["sources"]),
                "waveform_file": result["file"], "image": images.get(case["case_id"], ""),
            })
    return rows


CSV_FIELDS = [
    "case_id", "seed", "budget_tag", "class_id", "mode_id", "anchor_activity_id", "method", "support",
    "state_blocks", "supported_edges", "fallback_edges", "single_donor_cycle",
    "nearest_donor_resampled_nrmse", "nearest_donor_activity_id", "largest_donor_sample_fraction",
    "duration_target_cost", "unit_selection_objective", "boundary_objective", "transition_objective", "energy_wh",
    "mean_watts", "peak_watts", "block_state_labels", "block_donor_activity_ids", "waveform_file", "image",
]


def _plot_case(root, manifest, case, target):
    # Import only when there is something to plot. A canvas avoids changing the
    # application's global matplotlib backend or pyplot state.
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    methods = list(manifest["methods"])
    entries = dict(case["results"])
    if "anchor_reference" in case:
        methods.insert(0, "real_anchor_reference")
        entries["real_anchor_reference"] = case["anchor_reference"]
    waves = {}
    for method in methods:
        with np.load(root / entries[method]["file"], allow_pickle=False) as data:
            waves[method] = (data["timestamp"].copy(), data["appliance"].copy())
    period = manifest["configuration"]["sample_period"]
    edges = np.r_[0, np.cumsum([block["length_samples"] for block in case["template"]])] * period
    duration = float(edges[-1])
    maximum = max(float(np.max(power)) for _, power in waves.values())
    # Label at most 12 wide-enough blocks; the CSV keeps every ordered source.
    label_indices = [index for index, width in enumerate(np.diff(edges)) if width / duration >= 0.045]
    if len(label_indices) > 12:
        label_indices = [label_indices[index] for index in np.linspace(0, len(label_indices) - 1, 12, dtype=int)]
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10}):
        figure = Figure(figsize=(14, 2.45 * len(methods) + 1.1), facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(methods), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        for axis, method in zip(axes, methods):
            result = entries[method]
            is_reference = method == "real_anchor_reference"
            timestamp, power = waves[method]
            axis.plot(timestamp, power, color="#24648a", linewidth=1.1)
            for index, edge in enumerate(edges[1:-1]):
                axis.axvline(edge, color="#67747b", linestyle="--", linewidth=0.65, alpha=0.7)
            for index in label_indices:
                midpoint = (edges[index] + edges[index + 1]) / 2
                axis.text(midpoint, 0.97,
                          f"S{case['template'][index]['state_label']}\n"
                          f"{'A' if is_reference else 'D'}{case['anchor_activity_id'] if is_reference else result['sources'][index]['activity_id']}",
                          transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            title = ("Real training anchor | visual template reference, not a reconstruction target"
                     if is_reference else
                     f"{method} | support={_support(result)} | single donor={result['single_donor_cycle']} | "
                     f"nearest donor NRMSE={_number(result['nearest_donor_resampled_nrmse'])}")
            axis.set_title(title, loc="left")
            axis.set_ylabel("Power (W)")
            axis.set_ylim(0, max(maximum * 1.18, 1.0))
            axis.set_xlim(0, duration)
            axis.grid(axis="y", alpha=0.18)
        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(f"{case['case_id']} | Class/Mode={case['class_mode']} | inherited state blocks", y=0.989)
        figure.text(0.5, 0.012,
                    "S: inherited state ID; D: donor cycle ID; A: anchor cycle ID. No seam smoothing. "
                    "Crowded labels omitted; ordered states/donors in cases.csv.\n"
                    "Shared axes; integrity audit does not establish physical quality or source reconstruction.",
                    ha="center", va="bottom", fontsize=8)
        figure.tight_layout(rect=(0, 0.06, 1, 0.965))
        figure.savefig(target, dpi=130)
        figure.clear()


def _markdown(root, output, manifest, summary, audit, rows, images):
    link = lambda filename: quote(os.path.relpath(root / filename, output).replace(os.sep, "/"))
    lines = ["# 基元状态拼接：已有输出的描述性报告", "",
             f"状态：`{summary.get('status', 'not_recorded')}`。本报告读取现有 JSON/NPZ，未重新生成或改变信号。",
             f"本次独立审计：passed={audit['passed']}，检查 {audit['paired_cases_checked']} 个配对案例、"
             f"{audit['waveform_files_checked']} 个波形文件。", "",
             "**审计只证明配对、声明的来源预算、哈希和数值一致性，不证明物理合理、NILM 提升，"
             "也未用原始源 CSV 重构波形核验。状态编号与 Class/Mode 为继承的经验标签，不是人工标注程序。**", "",
             "## 覆盖与兼容性支持", "",
             "full：全部状态转移有足够供体支持；partial：部分边回退；none：全部边回退。"
             "这些是参考模型的支持情况，不是质量等级。", ""]
    coverage = []
    for budget in manifest["budgets"]:
        cases = [case for case in manifest["cases"]
                 if (case["seed"], case["budget_tag"]) == (budget["seed"], budget["budget_tag"])]
        support = Counter(_support(case["results"][manifest["methods"][0]]) for case in cases)
        requested = budget["requested_anchors"]
        coverage.append([budget["seed"], budget["budget_tag"], requested, len(cases), requested - len(cases),
                         _number(len(cases) / requested if requested else None),
                         support["full"], support["partial"], support["none"]])
    lines += _table(["seed", "预算", "请求", "配对成功", "跳过", "成功比例", "full", "partial", "none"], coverage)
    lines += [f"旧版兼容字段：paired_cases={summary.get('paired_cases', len(manifest['cases']))}；"
              f"cases_with_supported_transitions={summary.get('cases_with_supported_transitions', 'NA')}；"
              f"skipped_cases={summary.get('skipped_cases', len(manifest['skipped']))}。", ""]
    reasons = Counter(row["reason"] for row in manifest["skipped"])
    if reasons:
        lines += _table(["共同跳过原因", "数量"], sorted(reasons.items()))
    lines += ["## 按预算、方法、支持程度分层的重放诊断", "",
              "单供体比例越高，越需要检查是否只是在复用一个已有周期。最近供体 NRMSE 很低也可能表示重放，"
              "不能解释为生成更好。NRMSE 为对所有同组预算供体做完整周期长度适配后的最小 RMSE / max(供体峰值, 1 W)。", ""]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["budget_tag"], (row["class_id"], row["mode_id"]),
                row["method"], row["support"])].append(row)
    replay = []
    ratios = {(row["seed"], row["budget_tag"]): row["real_ratio"] for row in manifest["budgets"]}
    for key in sorted(groups, key=lambda key: (key[0], ratios[key[:2]], key[2], key[3], key[4])):
        values = groups[key]
        errors = [row["nearest_donor_resampled_nrmse"] for row in values]
        replay.append([*key, len(values), _number(np.mean([row["single_donor_cycle"] for row in values])),
                       _number(np.median(errors)), _number(np.percentile(errors, 10)), _number(np.percentile(errors, 90))])
    lines += _table(["seed", "预算", "Class/Mode", "方法", "支持", "案例数", "单供体比例", "NRMSE 中位数", "P10", "P90"], replay)
    lines += ["## 独立验证分布", "",
              "只描述同 Class/Mode 的验证集分布，未使用测试集选方法。能量与功率的 Wasserstein 距离"
              "保留原量纲；不同量纲不能直接相加排名。验证样本少时不据此宣布某方法获胜。", ""]
    evaluation = summary.get("evaluation")
    if evaluation:
        validation = evaluation.get("cycle_validation", [])
    elif manifest.get("schema_version", 1) == 1:
        validation = summary.get("validation_diagnostics", [])
        lines += ["旧版 validation_diagnostics 混合不同支持程度，仅作描述，不能据此比较全支持方法效果。", ""]
    else:
        validation = []
    if validation:
        lines += _table(["seed", "预算", "Class/Mode", "方法", "支持", "状态", "生成数", "验证数", "能量 W1 (Wh)",
                         "均值 W1 (W)", "峰值 W1 (W)"], [
            [row["seed"], row["budget_tag"], row["class_mode"], row["method"],
             row.get("support_status", "mixed/legacy"), row.get("status", "not_recorded"), row["generated_cycles"],
             row["validation_cycles"], *[_number(row[name].get("wasserstein"))
                                         for name in ("energy_wh", "mean_watts", "peak_watts")]]
            for row in validation])
    else:
        lines += ["现有 summary 未记录同组验证分布。", ""]
    if evaluation:
        transitions = evaluation.get("transition_validation", [])
        if transitions:
            lines += ["按状态转移类型分层的验证比较：跳变保留符号，局部斜率单位 W/s；"
                      "NA 表示无可比观测，不能当作零误差。low_support 表示样本不足，只作描述。", ""]
            lines += _table(["seed", "预算", "Class/Mode", "方法", "支持", "转移", "状态", "跳变 W1 (W)",
                             "左斜率 W1 (W/s)", "右斜率 W1 (W/s)"], [
                [row["seed"], row["budget_tag"], row["class_mode"], row["method"],
                 row.get("support_status", "unknown"), f"{row['left_state']} → {row['right_state']}", row["status"],
                 *[_number(row.get(name, {}).get("wasserstein")) for name in
                   ("signed_jump_watts", "left_slope_watts_per_second", "right_slope_watts_per_second")]]
                for row in transitions])
        lines += ["新版评价已记录于 " + f"[composition_summary.json]({link('composition_summary.json')})："
                  "`evaluation.coverage`（覆盖）、`evaluation.method_summary`（分层方法统计）、"
                  "`evaluation.cycle_validation`（同组周期分布）、`evaluation.transition_validation`（按转移类型比较验证分布）、"
                  "`evaluation.metric_definitions`（指标定义与限制）。请连同样本数、支持层级阅读。", ""]
    else:
        lines += ["这是旧版评价输出：未包含 `evaluation` 的状态转移分层评价，缺失项不能视为已通过。", ""]
    lines += ["## 波形图", "",
              "图先按 seed、预算比例、Class/Mode、anchor ID 排序，再轮流从各预算取图，优先覆盖不同 Class/Mode；"
              "不按评分或视觉好坏挑选。各方法独立子图、共享时间/功率范围，虚线标状态边界；"
              "新版首行为真实训练锚点，只作模板参考，不能视作待重建的唯一真值。S 为状态编号，D 为逐块供体周期。拥挤处略去标注，完整有序来源在 [cases.csv](cases.csv)。", ""]
    for case_id, filename in images.items():
        lines += [f"![{case_id}]({quote(filename)})", ""]
    if not images:
        lines += ["没有可展示的案例，或 max_cases=0。", ""]
    lines += ["## 方法与解释边界", "",
              "`random` 为随机选择；`boundary_greedy` 为局部端点选择；`boundary_dp` 为全局端点选择；"
              "`transition_dp` 为经验转移兼容性选择。若存在 Unit Selection 方法，它是目标匹配与连接成本的"
              "领域适配对照，不是语音论文的原样复现。所有方法保留已有状态模板与共同候选池。", "",
              "优化目标下降是搜索的预期结果，不是独立效果证据；合法电器切换允许功率阶跃，接缝差越小不一定更合理。"
              "全回退与部分回退应独立于全支持案例解释；上游表示拟合范围仍未认证。", "",
              f"完整数据：[manifest]({link('composition_manifest.json')})（`methods`、`cases[].results`、逐块来源）、"
              f"[summary]({link('composition_summary.json')})（`validation_diagnostics`、`paired_comparisons`、"
              "`limitations`）、[逐案例指标](cases.csv)。本报告不宣称服务器真实效果已得到验证。", ""]
    return "\n".join(lines)


def build_report(study_dir, output_dir, max_cases=6):
    """Audit a saved study and write a new report directory without overwrites."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 0:
        raise ValueError("max_cases must be a nonnegative integer")
    root, output = Path(study_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"report output already exists: {output}")
    audit = audit_composition(root)
    if not audit["passed"]:
        raise ValueError("composition audit failed: " + "; ".join(audit["errors"]))
    manifest = json.loads((root / "composition_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "composition_summary.json").read_text(encoding="utf-8"))
    if summary.get("input_fingerprint") != manifest["input_fingerprint"]:
        raise ValueError("summary fingerprint differs from manifest")
    if summary.get("paired_cases") != len(manifest["cases"]):
        raise ValueError("summary case count differs from manifest")
    for case in manifest["cases"]:
        for row in case["results"].values():
            source = (root / row["file"]).resolve()
            if output.is_relative_to(source.parent):
                raise ValueError("report output must not be inside a source waveform directory")
    chosen = _representatives(manifest, max_cases)
    images = {case["case_id"]: f"case_{index:03d}.png" for index, case in enumerate(chosen, 1)}
    rows = _rows(manifest, images)
    markdown = _markdown(root, output, manifest, summary, audit, rows, images)
    # All input checks above are read-only. Never reuse a prior or partial report.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for case in chosen:
        _plot_case(root, manifest, case, output / images[case["case_id"]])
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return {"output_dir": str(output), "report": str(output / "report.md"), "audit": audit,
            "paired_cases": len(manifest["cases"]), "methods": manifest["methods"],
            "case_method_rows": len(rows), "plotted_cases": list(images),
            "images": [str(output / filename) for filename in images.values()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cases", default=6, type=int)
    args = parser.parse_args()
    try:
        result = build_report(args.study_dir, args.output_dir, args.max_cases)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        parser.exit(1, f"[composition-report] {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
