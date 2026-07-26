import random
import numpy as np
import pandas as pd
import os
import argparse
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import SurvivalModel



def concordance_index_simple(times, risks, events):
    times = np.asarray(times, dtype=float)
    risks = np.asarray(risks, dtype=float)
    events = np.asarray(events, dtype=int)

    concordant = permissible = tied = 0.0
    n = len(times)

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            # i가 더 빨리 사건 발생(time이 더 작음)한 경우만 비교
            if times[i] >= times[j]:
                continue
            permissible += 1.0
            # 더 빨리 죽은 i가 risk가 더 커야 concordant
            if risks[i] > risks[j]:
                concordant += 1.0
            elif risks[i] == risks[j]:
                tied += 1.0

    if permissible == 0:
        return 0.5
    return (concordant + 0.5 * tied) / permissible

# =========================
# Config
# =========================
CSV_PATH = '/home/yuz/wsi-survival/clinical_survival_processed_final.csv'

FEATURE_COLS = [
    'demographic.age_at_index',
    'stage_numeric',
    'gender_numeric',
    'race_asian',
    'race_black or african american',
    'race_not reported',
    'race_white',
    'dx_Lobular',
    'dx_Other'
]

EPOCHS = 50
LR = 1e-4
SEED = 42
BAG_BATCH_SIZE = 1
ACCUM_BAGS = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== Regularization / Early stopping =====
WEIGHT_DECAY = 1e-4          
PATIENCE = 30                
MIN_DELTA = 1e-4             
# CKPT_PATH = "best_survival_attn.pt"
# history = []


class EarlyStopping:
    """
    mode='max' : metric이 클수록 좋음 (C-index)
    """
    def __init__(self, patience=10, min_delta=1e-4, mode="max", ckpt_path="best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_path = ckpt_path

        self.best = None
        self.num_bad_epochs = 0

    def _is_improved(self, metric):
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > (self.best + self.min_delta)
        else:  # mode == "min"
            return metric < (self.best - self.min_delta)


    def step(self, metric, model, optimizer, epoch):
        if self._is_improved(metric):
            self.best = metric
            self.num_bad_epochs = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_c": self.best,
            }, self.ckpt_path)

            return True, False
        else:
            self.num_bad_epochs += 1
            should_stop = self.num_bad_epochs >= self.patience
            return False, should_stop     


# =========================
# Dataset: bag 1개씩 반환
# =========================
class SurvivalDataset(Dataset):
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.patient_ids = self.data["cases.submitter_id"].unique()

        if "feature_path" not in self.data.columns:
            raise ValueError("CSV에 'feature_path' 컬럼이 없습니다.")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        rows = self.data[self.data["cases.submitter_id"] == patient_id]

        slide_feats = []

        
        MAX_PATCHES = 4096

        PT_DIR = "/home/yuz/wsi-survival/features_uni/pt_files"

        for _, row in rows.iterrows():
            pt_name = os.path.basename(row["feature_path"])
            pt_path = os.path.join(PT_DIR, pt_name)

            h = torch.load(pt_path).float()

            if h.shape[0] > MAX_PATCHES:
                patch_idx = torch.randperm(h.shape[0])[:MAX_PATCHES]
                h = h[patch_idx]

            slide_feats.append(h)  # each h: [num_patches, 1024]

        row = rows.iloc[0]

        clinical_np = pd.to_numeric(row[FEATURE_COLS], errors="coerce").to_numpy(dtype=np.float32)
        clinical_np = np.nan_to_num(clinical_np, nan=0.0, posinf=0.0, neginf=0.0)
        clinical = torch.from_numpy(clinical_np)

        time = torch.tensor(float(row["survival_time"]), dtype=torch.float32)
        event = torch.tensor(float(row["event"]), dtype=torch.float32)

        return slide_feats, clinical, time, event

def collate_fn(batch):
    # batch_size=1이므로 batch 안에 환자 1명만 들어있음
    slides, clinical, time, event = batch[0]
    return slides, clinical, time, event


# =========================
# Cox loss (mini-batch)
# =========================
def cox_loss(risk, time, event):
    if torch.sum(event) == 0:
        return risk.sum() * 0.0

    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.sum((risk - log_cumsum) * event) / torch.sum(event)
    return loss

@torch.no_grad()
def eval_c_index(model, loader):
    model.eval()
    risks, times, events = [], [], []

    for (h, clinical, time, event) in loader:
        h = [slide.to(DEVICE) for slide in h]
        clinical = clinical.to(DEVICE)

        r = model(h, clinical)
        risks.append(float(r.item()))
        times.append(float(time.item()))
        events.append(int(event.item()))

    return concordance_index_simple(times, np.array(risks), events)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    updates = 0

    # 누적용 버퍼
    buf_risk, buf_time, buf_event = [], [], []

    optimizer.zero_grad(set_to_none=True)

    for step, (h, clinical, time, event) in enumerate(
        tqdm(loader, desc="Training", leave=False),
        start=1
    ):
        h = [slide.to(DEVICE) for slide in h]
        clinical = clinical.to(DEVICE)
        time = time.to(DEVICE)
        event = event.to(DEVICE)

        risk = model(h, clinical)  # scalar
        buf_risk.append(risk)
        buf_time.append(time.squeeze())
        buf_event.append(event.squeeze())

        # ACCUM_BAGS개 모이면 Cox loss 한 번 계산해서 업데이트
        if len(buf_risk) == ACCUM_BAGS:
            risks = torch.stack(buf_risk)      # [B]
            times = torch.stack(buf_time)      # [B]
            events = torch.stack(buf_event)    # [B]

            loss = cox_loss(risks, times, events)

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item())
                updates += 1

            buf_risk, buf_time, buf_event = [], [], []

    # epoch 끝에 버퍼에 남은 게 있으면 마지막 업데이트(선택)
    if len(buf_risk) > 1:
        risks = torch.stack(buf_risk)
        times = torch.stack(buf_time)
        events = torch.stack(buf_event)
        loss = cox_loss(risks, times, events)

        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            updates += 1

    return total_loss / max(updates, 1)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="attention",
    choices=["attention", "transmil", "mamba", "mean"]
)


def build_aggregator(name):
    if name == "attention":
        from aggregators.attention import AttentionAggregator
        return AttentionAggregator(embed_dim=1024, out_dim=512)

    elif name == "transmil":
        from aggregators.transmil import TransMILAggregator
        return TransMILAggregator()

    elif name == "mamba":
        from aggregators.mambamil import MambaMIL
        return MambaMIL(
            in_dim=1024,
            n_classes=512,
            dropout=0.25,
            act="relu"
        )

    elif name == "mean":
        from aggregators.mean import MeanPoolingAggregator
        return MeanPoolingAggregator()

    else:
        raise ValueError(f"Unknown aggregator: {name}")


def main():
    args = parser.parse_args()
    set_seed(SEED)

    df = pd.read_csv(CSV_PATH)

    AGGREGATOR_NAME = args.model
    RESULT_PATH = f"cv_results_{AGGREGATOR_NAME}.csv"

    patient_df = df.drop_duplicates("cases.submitter_id").reset_index(drop=True)
    patients = patient_df["cases.submitter_id"].values
    events = patient_df["event"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if os.path.exists(RESULT_PATH):
        prev_results = pd.read_csv(RESULT_PATH)
        completed_folds = set(prev_results["fold"].astype(int).tolist())
        fold_results = prev_results.to_dict("records")
    else:
        completed_folds = set()
        fold_results = []


    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, events), start=1):

        if fold in completed_folds:
            print(f"\n========== Fold {fold}/5 already completed. Skip. ==========")
            continue

        print(f"\n========== Fold {fold}/5 ==========")

        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        train_df = df[df["cases.submitter_id"].isin(train_patients)]
        val_df = df[df["cases.submitter_id"].isin(val_patients)]

        train_set = SurvivalDataset(train_df)
        val_set = SurvivalDataset(val_df)

        train_loader = DataLoader(
            train_set,
            batch_size=BAG_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=BAG_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            collate_fn=collate_fn,
        )

        aggregator = build_aggregator(AGGREGATOR_NAME)

        CKPT_PATH = f"best_survival_{AGGREGATOR_NAME}_fold{fold}.pt"
        LAST_CKPT_PATH = f"last_survival_{AGGREGATOR_NAME}_fold{fold}.pt"
        
        print("=" * 50)
        print(f"Aggregator : {AGGREGATOR_NAME}")
        print(f"Fold       : {fold}")
        print(f"Device     : {DEVICE}")
        print(f"Train size : {len(train_set)}")
        print(f"Val size   : {len(val_set)}")
        print(f"Val events : {int(val_df.drop_duplicates('cases.submitter_id')['event'].sum())}")
        print(f"LR         : {LR}")
        print(f"Epochs     : {EPOCHS}")
        print(f"Accum bags : {ACCUM_BAGS}")
        print(f"Checkpoint : {CKPT_PATH}")
        print(f"Last ckpt  : {LAST_CKPT_PATH}")
        print("=" * 50) 

        model = SurvivalModel(aggregator).to(DEVICE)

        # 파라미터 수 계산
        n_params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        
        print(f"\nTrainable Parameters: {n_params:,}")
        print(f"({n_params/1e6:.2f} M)")


        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # AdamW + weight decay

        early_stopper = EarlyStopping(
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            mode="max",
            ckpt_path=CKPT_PATH,
        )

        start_epoch = 1
        history = []

        if os.path.exists(LAST_CKPT_PATH):
            ckpt = torch.load(LAST_CKPT_PATH, map_location=DEVICE)

            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])

            start_epoch = ckpt["epoch"] + 1
            early_stopper.best = ckpt.get("best_c", None)
            early_stopper.num_bad_epochs = ckpt.get("num_bad_epochs", 0)
            history = ckpt.get("history", [])

            print(
                f"Resumed Fold {fold} from epoch {ckpt['epoch']} | "
                f"best C-index {early_stopper.best:.4f}"
            )


        for epoch in range(start_epoch, EPOCHS + 1):
            print(f"\n===== Fold {fold} | Epoch {epoch}/{EPOCHS} =====")
            
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_c = eval_c_index(model, val_loader)


            improved, should_stop = early_stopper.step(val_c, model, optimizer, epoch)
            status = "↑" if improved else "-"

            history.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_c_index": val_c,
                "best_c_index": early_stopper.best,
            })   
            pd.DataFrame(history).to_csv(
                f"history_{AGGREGATOR_NAME}_fold{fold}.csv",
                index=False
            )            

            torch.save({
                "fold": fold,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_c": early_stopper.best,
                "num_bad_epochs": early_stopper.num_bad_epochs,
                "history": history,
            }, LAST_CKPT_PATH)


            print(
                f"Fold {fold} | "
                f"Epoch {epoch:02d} | "
                f"loss {train_loss:.4f} | "
                f"c-index {val_c:.4f} | "
                f"best {early_stopper.best:.4f} {status}"
            )

            if should_stop:
                print(f"Early stopping triggered (patience={PATIENCE}).")
                break

        fold_results.append({
            "fold": fold,
            "best_c_index": early_stopper.best,
            "train_size": len(train_set),
            "val_size": len(val_set),
            "val_events": int(val_df.drop_duplicates("cases.submitter_id")["event"].sum()),
            
        })

        pd.DataFrame(fold_results).to_csv(
            RESULT_PATH,
            index=False
        )

    result_df = pd.DataFrame(fold_results)

    print("\n========== 5-Fold CV Result ==========")
    print(result_df)

    if len(result_df) > 0:
        print(f"Mean C-index: {result_df['best_c_index'].mean():.4f}")
        print(f"Std C-index : {result_df['best_c_index'].std():.4f}")


if __name__ == "__main__":
    main()
