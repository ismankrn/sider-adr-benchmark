"""One-off repair for lc_*.csv written by scripts 08 and 09 before append_rows aligned columns."""
import csv, sys, shutil, pandas as pd
metrics = ["auroc","auprc","mcc","balanced_acc","accuracy","f1","precision","recall","specificity","tp","fp","tn","fn","threshold"]
tuned = [f"{k}_tuned" for k in ["auroc","auprc","mcc","balanced_acc","accuracy","f1","precision","recall","specificity"]] + ["threshold_tuned"]
schema09 = ["dataset","seed","fold","n","n_train","task","model","pos_rate","train_time"] + metrics + tuned
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path))); header = rows[0]
    if all(len(r) == len(header) for r in rows[1:]):
        print(path, "sudah konsisten"); continue
    shutil.copy(path, path + ".bak"); out = []
    for r in rows[1:]:
        if len(r) == len(header): out.append(dict(zip(header, r)))
        elif len(r) == len(schema09): out.append(dict(zip(schema09, r)))
        else: raise SystemExit(f"{path}: baris tak dikenal {len(r)} kolom")
    pd.DataFrame(out).to_csv(path, index=False); print(path, "diperbaiki:", len(out), "baris")
