"""08 — CORAL, CORN and MiniLM-embedding baselines on the same split as the R experiments.

Needs: data/processed/triage_full.csv, data/raw/nhamcs_data_2018_22.csv (chief complaint text),
       revision/results/final/{split_test_rows,split_train_rows}.csv and {e3,e4}_columns.txt
       (written by R/final_baselines_tree.R, which also fixes the 80/20 split with seed 42).
Writes: revision/results/final/pred_{CORAL,CORN,MiniLM_*}.csv and neural_baselines.csv.
        TRIAGE_SPLIT=stratified switches input and output to revision/results/final_stratified.
Run:    python pipeline/08_baselines_neural.py   (a few minutes on a GPU)
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")

import os
OUT = cfg.ROOT / "revision" / "results" / ("final" if os.environ.get("TRIAGE_SPLIT", "random") == "random" else "final_stratified")
SEED = 42
K = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def qwk(y, p, k=K):
    cm = np.zeros((k, k))
    np.add.at(cm, (y - 1, p - 1), 1)
    n = cm.sum()
    exp = np.outer(cm.sum(1) / n, cm.sum(0) / n)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    return 1.0 - (w * cm / n).sum() / (w * exp).sum()


def decode(raw, t):
    return 1 + (raw[:, None] >= np.asarray(t)[None, :]).sum(1)


def find_thresholds(raw, y):
    """Same grid as R/lib_triage.R::find_thresholds."""
    best, bq = (0.32, 0.45, 0.58, 0.72), -np.inf
    for t1 in np.arange(0.20, 0.401, 0.02):
        for t2 in np.arange(0.40, 0.551, 0.02):
            for t3 in np.arange(0.55, 0.701, 0.02):
                for t4 in np.arange(0.70, 0.851, 0.02):
                    if not (t1 < t2 < t3 < t4):
                        continue
                    q = qwk(y, decode(raw, (t1, t2, t3, t4)))
                    if q > bq:
                        best, bq = (t1, t2, t3, t4), q
    return np.round(best, 2)


def row(label, n_feat, y, p, **extra):
    return dict(model=label, n_feat=n_feat, qwk=round(qwk(y, p), 4), acc=round(float((p == y).mean()), 4),
                within1=round(float((np.abs(p - y) <= 1).mean()), 4),
                mae=round(float(np.abs(p - y).mean()), 4), **extra)


def save(name, rows, y, p, raw=None):
    d = pd.DataFrame({"row": rows + 1, "y": y, "pred": p})
    if raw is not None:
        d["raw"] = raw
    d.to_csv(OUT / f"pred_{name}.csv", index=False)


class OrdinalMLP(nn.Module):
    def __init__(self, d_in, kind):
        super().__init__()
        self.kind = kind
        self.body = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        if kind == "coral":  # one shared weight vector, K-1 biases
            self.w = nn.Linear(128, 1, bias=False)
            self.b = nn.Parameter(torch.zeros(K - 1))
        else:                # corn: K-1 independent conditional classifiers
            self.out = nn.Linear(128, K - 1)

    def forward(self, x):
        h = self.body(x)
        return self.w(h) + self.b if self.kind == "coral" else self.out(h)


def coral_loss(z, y):
    levels = (y[:, None] > torch.arange(K - 1, device=z.device)[None, :]).float()
    return -(F.logsigmoid(z) * levels + (F.logsigmoid(z) - z) * (1 - levels)).sum(1).mean()


def corn_loss(z, y):
    total, n = 0.0, 0
    for k in range(K - 1):
        m = y >= k
        if m.sum() == 0:
            continue
        total = total + F.binary_cross_entropy_with_logits(z[m, k], (y[m] > k).float(), reduction="sum")
        n += int(m.sum())
    return total / n


def predict_labels(model, x):
    z = model(x)
    if model.kind == "coral":
        pr = torch.sigmoid(z)
    else:
        pr = torch.cumprod(torch.sigmoid(z), dim=1)
    return (1 + (pr > 0.5).sum(1)).cpu().numpy()


def train_mlp(kind, xtr, ytr, xva, yva, xte, max_epochs=80, patience=8):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = OrdinalMLP(xtr.shape[1], kind).to(DEV)
    loss_fn = coral_loss if kind == "coral" else corn_loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xtr_t, ytr_t = torch.tensor(xtr, dtype=torch.float32, device=DEV), torch.tensor(ytr - 1, device=DEV)
    xva_t, yva_t = torch.tensor(xva, dtype=torch.float32, device=DEV), torch.tensor(yva - 1, device=DEV)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xtr_t), device=DEV)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(xva_t), yva_t))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return predict_labels(model, torch.tensor(xte, dtype=torch.float32, device=DEV)), ep + 1


XGB_KW = dict(objective="reg:logistic", max_depth=8, learning_rate=0.04, subsample=0.85,
              colsample_bytree=0.85, min_child_weight=5, reg_lambda=2.0, n_estimators=600,
              tree_method="hist", device=DEV, random_state=0)


def xgb_ordinal(xtr, ytr, xte):
    m = xgb.XGBRegressor(**XGB_KW).fit(xtr, (ytr - 1) / 4)
    return m.predict(xte)


def main():
    rows = []
    d = pd.read_csv(cfg.DATA_PROCESSED)
    y_all = d["triage_acuity"].to_numpy()
    tr = pd.read_csv(OUT / "split_train_rows.csv")["row"].to_numpy() - 1
    te = pd.read_csv(OUT / "split_test_rows.csv")["row"].to_numpy() - 1
    ytr, yte = y_all[tr], y_all[te]
    g4 = (OUT / "e4_columns.txt").read_text().split("\n")
    g4 = [c for c in g4 if c]
    g3 = [c for c in (OUT / "e3_columns.txt").read_text().split("\n") if c]
    assert len(g4) == 640 and len(g3) == 114

    # CORAL / CORN on the 640 E4 features; 10% of the training rows are held out for early stopping
    X = d[g4].to_numpy(dtype=np.float32)
    sc = StandardScaler().fit(X[tr])
    Xs = np.nan_to_num(sc.transform(X)).astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr))
    n_va = int(0.1 * len(tr))
    va_i, fit_i = tr[perm[:n_va]], tr[perm[n_va:]]
    for kind in ("coral", "corn"):
        p, ep = train_mlp(kind, Xs[fit_i], y_all[fit_i], Xs[va_i], y_all[va_i], Xs[te])
        name = kind.upper()
        r = row(name, len(g4), yte, p, epochs=ep)
        rows.append(r)
        save(name, te, yte, p)
        print(r, flush=True)

    # MiniLM sentence embeddings replace the TF-IDF block: E3 features (114) + 384 dims
    emb_path = OUT / "minilm_embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        from sentence_transformers import SentenceTransformer
        raw = pd.read_csv(cfg.DATA_RAW, low_memory=False)
        raw = pd.concat([raw[raw.year.isin([2018, 2019, 2020])], raw[raw.year.isin([2021, 2022])]])
        assert (raw["target_triage_acuity"].to_numpy() == y_all).all(), "raw/processed row order differs"
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
        emb = st.encode(raw["chief_complaint_text"].fillna("").tolist(), batch_size=256,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, emb)
    XM = np.hstack([d[g3].to_numpy(dtype=np.float32), emb])
    raw_te = xgb_ordinal(XM[tr], ytr, XM[te])
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(tr)) % 5
    oof = np.zeros(len(tr))
    for f in range(5):
        oof[fold == f] = xgb_ordinal(XM[tr][fold != f], ytr[fold != f], XM[tr][fold == f])
    th_train = find_thresholds(oof, ytr)   # out-of-fold training scores only
    th_test = find_thresholds(raw_te, yte)  # test predictions, as in R/reproduce_E4.R
    for proto, th in (("test_tuned", th_test), ("train_tuned", th_train)):
        p = decode(raw_te, th)
        r = row("MiniLM", XM.shape[1], yte, p, protocol=proto, t1=th[0], t2=th[1], t3=th[2], t4=th[3])
        rows.append(r)
        save(f"MiniLM_{proto}", te, yte, p, raw_te)
        print(r, flush=True)

    pd.DataFrame(rows).to_csv(OUT / "neural_baselines.csv", index=False)


if __name__ == "__main__":
    main()
