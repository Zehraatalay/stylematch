import copy 
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

from sklearn.metrics import(
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

def set_seed(seed = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model,loader,device):

    model.eval()
    labels = []
    probs = []

    loss_sum = 0.0
    n = 0

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    use_amp = (
        device.type == "cuda"
    )

    for img_a, img_b, y in loader:
        img_a = img_a.to(
            device,
            non_blocking = True
        )

        img_b = img_b.to(
            device,
            non_blocking = True
        )

        y = y.to(
            device,
            non_blocking = True
        )

        with torch.autocast(
            device_type = device.type,
            dtype = torch.float16,
            enabled = use_amp
        ):
            logits = model(
                img_a,
                img_b
            )

            loss = criterion(
                logits,
                y
            )
        probabilities = (
            torch.sigmoid(logits)
        )

        labels.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        loss_sum += (
            loss.item()
            *y.size(0)
        )

        n += y.size(0)
    if n == 0:
        return{
            "loss": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan")
        }
    labels = np.asarray(
        labels,
        dtype = np.int64
    )

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

    preds = (
        probs >=0.5

    ).astype(np.int64)

    result = {
        "loss":
            float(loss_sum/ n),
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    preds
                )
            ),
        "precision":
            float(
                precision_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),
        "recall":
            float(
                recall_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),
        "f1":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division=0
                )
            )

    }

    try:
        result["roc_auc"] = float(
            roc_auc_score(
                labels,
                probs
            )
        )
    except ValueError:

        result["roc_auc"] = (
            float("nan")
        )
    return result


def train_epochs(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
    epochs,
    patience = 2,
    stage_name= "train",
    checkpoint_path = None
        
):
    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_auc = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(
        model.state_dict()
    )

    wait = 0
    history = []

    if checkpoint_path is not None:
        checkpoint_path = Path(
            checkpoint_path
        )
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
    for epoch in range(
        1,
        epochs + 1
    ):
        start_time = time.time()
        model.train()

        loss_sum = 0.0
        n = 0
        for img_a,img_b,y in train_loader:
            img_a = img_a.to(
                device,
                non_blocking= True
            )

            img_b = img_b.to(
                device,
                non_blocking = True
            )
            y = y.to(
                device,
                non_blocking = True
            )

            optimizer.zero_grad(
                set_to_none = True
            )

            with torch.autocast(
                device_type = device.type,
                dtype = torch.float16,
                enabled = use_amp
            ):
                logits = model(
                    img_a,
                    img_b
                )

                loss = criterion(
                    logits,
                    y
                )
            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()
            loss_sum += (
                loss.item()
                *y.size(0)
            )

            n += y.szie(0)
        train_loss = (
            loss_sum
            / max(n,1)
        )

        val_metrics = evaluate(
            model,
            valid_loader,
            device
        )

        record = {
            "epoch":
                epoch,
            "train_loss":
                float(train_loss),
            **val_metrics
        }
        history.append(record)
        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[{stage_name} "
                        f"{epoch:02d}/{epochs}] "
                        f"train_loss={train_loss:.4f} "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"val_acc={val_metrics['accuracy']:.4f} "
                        f"val_f1={val_metrics['f1']:.4f} "
                        f"val_auc={val_metrics['roc_auc']:.4f} "
                        f"time={elapsed:.1f}s"
        )

        auc = val_metrics["roc_auc"]

        if (
                   np.isfinite(auc)
                   and auc > best_auc
               ):
       
                   best_auc = float(auc)
       
                   best_epoch = epoch
       
                   best_state = copy.deepcopy(
                       model.state_dict()
                   )
       
                   wait = 0
       
                   if checkpoint_path is not None:
       
                       torch.save(
                           {
                               "model_state":
                                   best_state,
       
                               "optimizer_state":
                                   optimizer.state_dict(),
       
                               "epoch":
                                   epoch,
       
                               "best_val_auc":
                                   best_auc,
       
                               "history":
                                   history,
       
                               "stage":
                                   stage_name
                           },
                           checkpoint_path
                       )
       
                       print(
                           "[checkpoint] saved ->",
                           checkpoint_path
                       )
       
        else:
       
                   wait += 1
       
                   if wait >= patience:
       
                       print(
                           "[early-stop] "
                           f"no ROC-AUC improvement "
                           f"for {patience} epochs"
                       )
       
                       break
       
    model.load_state_dict(
        best_state
           )
       
    print(
               f"[best] epoch={best_epoch} "
               f"val_auc={best_auc:.4f}"
           )
       
    return (
               history,
               best_auc
           )
       
       
    def save_json(obj, path):
       
           path = Path(path)
       
           path.parent.mkdir(
               parents=True,
               exist_ok=True
           )
       
           with open(
               path,
               "w",
               encoding="utf-8"
           ) as f:
       
               json.dump(
                   obj,
                   f,
                   indent=2,
                   ensure_ascii=False
               )

            
                        
