"""Record machine-suggested ClaSP initializers for exploratory CLaP gallery only."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from choose_finalists import OBJECTS, score

EXP=Path(__file__).resolve().parents[1]


def main():
    result={}
    for obj in OBJECTS:
        paths=sorted((EXP/"executions"/f"validate_{obj}_clasp").glob("*/analysis/records.json"))
        records=None;source=None
        for path in reversed(paths):
            run=json.loads((path.parents[1]/"run.json").read_text())
            if run.get("exit_code")==0:
                records=json.loads(path.read_text());source=path.parents[1];break
        if records is None:raise RuntimeError(f"Clasp validation incomplete for {obj}")
        grouped=defaultdict(list)
        for row in records:grouped[row["config_id"]].append(row)
        ranked=[]
        for config_id,rows in grouped.items():
            value,_=score(rows)
            ranked.append({"config_id":config_id,"params":rows[0]["params"],
                           "success":sum(r["status"]=="success" for r in rows),
                           "score":value})
        ranked.sort(key=lambda x:(-x["success"],x["score"],
                    json.dumps(x["params"],sort_keys=True)))
        winner=ranked[0]
        result[obj]={"params":winner["params"],"config_id":winner["config_id"],
                     "validation_execution":str(source),
                     "status":"machine_suggestion_for_exploratory_clap_only",
                     "human_approved":False}
    path=EXP/"data/provisional_clasp_initializer.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(path)
    for obj,item in result.items():print(obj,item["params"],item["config_id"])


if __name__=="__main__":main()
