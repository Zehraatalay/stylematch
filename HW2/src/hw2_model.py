from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



def load_feature_split(prepared_dir:Path,split_name: str) -> tuple[np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    feature_path = prepared_dir / "features" / f"{split_name}_features.npz"
    data = np.load(feature_path)
    meta = {
        "image_indices": data["image_indices"],
        "window_indices": data["window_indices"],
        "window_boxes": data["window_boxes"],
    }

    return data["features"].astype(np.float32),data["labels"].astype(np.int64),meta



class MLPDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
        batch_norm: bool,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    


@dataclass
class TrainingConfig:
    name: str
    batch_size: int
    learning_rate: float
    patience : int
    hidden_dims: list[int]
    dropout: float
    batch_norm: bool
    max_epochs: int = 25


DEFAULT_CONFIGS = [
    TrainingConfig("cfg_01",64,1e-3,5,[512],0.0,False),
    TrainingConfig("cfg_02", 64, 1e-3, 5, [1024, 256], 0.2, False),
    TrainingConfig("cfg_03", 128, 1e-3, 6, [1024, 512], 0.2, True),
    TrainingConfig("cfg_04", 128, 3e-4, 8, [1024, 512], 0.3, True),
    TrainingConfig("cfg_05", 64, 3e-4, 8, [512, 256], 0.3, True),
    TrainingConfig("cfg_06", 128, 1e-3, 6, [512, 256], 0.2, False),

]


def compute_metrics(true_labels:np.ndarray,pred_labels:np.ndarray,label_order:list[str])-> dict:
    cm = confusion_matrix(true_labels,pred_labels,labels=list(range(len(label_order))))
    report = classification_report(
        true_labels,
        pred_labels,
        labels=list(range(len(label_order))),
        target_names=label_order,
        output_dict=True,
        zero_division=0,
    )
    return{
        "accuracy": float(accuracy_score(true_labels,pred_labels)),
        "macro_f1": float(f1_score(true_labels,pred_labels,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(true_labels,pred_labels,average="weighted",zero_division=0)),
        "classification_report":report,
        "confusion_matrix": cm.tolist(),
    }


def standardize_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_scaled = ((train_x - mean) / std).astype(np.float32)
    val_scaled = ((val_x - mean) / std).astype(np.float32)
    test_scaled = None if test_x is None else ((test_x - mean) / std).astype(np.float32)
    return train_scaled, val_scaled, test_scaled, mean, std


def build_class_weights(labels:np.ndarray,num_classes: int) ->torch.Tensor:
    counts = np.bincount(labels,minlength=num_classes).astype(np.float32)
    counts[counts== 0.0] = 1.0
    weights = counts.sum() / (num_classes*counts)
    return torch.tensor(weights,dtype = torch.float32)



def make_loader(features: np.ndarray,labels: np.ndarray,batch_size: int, shuffle:bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features),torch.from_numpy(labels))
    return DataLoader(dataset,batch_size = batch_size,shuffle=shuffle)


def predict_model(model: nn.Module,loader:DataLoader,device:torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            predictions.append(torch.argmax(logits,dim=1).cpu().numpy())
        return np.concatenate(predictions,axis=0)
    

def train_single_config(
        train_x:np.ndarray,
        train_y:np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        config: TrainingConfig,
        label_order:list[str],
        device: torch.device,
        output_dir:Path,


) -> dict:
    model = MLPDetector(
        input_dim = train_x.shape[1],
        output_dim = len(label_order),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batch_norm= config.batch_norm



    ).to(device)
    class_weights = build_class_weights(train_y,len(label_order)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)



    train_loader = make_loader(train_x,train_y,config.batch_size,shuffle=False)
    val_loader = make_loader(val_x,val_y,config.batch_size,shuffle=False)


    best_state = None
    best_metrics = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x,batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits,batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_x.shape[0]
        val_pred = predict_model(model,val_loader,device)
        metrics = compute_metrics(val_y,val_pred,label_order)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = epoch_loss/float(len(train_y))
        history.append(metrics)


        if best_metrics is None or(
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["weighted_f1"],
        ) > (
            best_metrics["macro_f1"],
            best_metrics["accuracy"],
            best_metrics["weighted_f1"],


        ):
            best_metrics = metrics
            best_state = {key: value.cpu() for key,value in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait +=1
            if wait>= config.patience:
                break
    model_path = output_dir / f"{config.name}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": config.__dict__,
            "best_epoch": best_epoch,
        },
        model_path,
    )

    return {
        "config" : config.__dict__,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history" : history,
        "model_path": str(model_path.resolve()),

    }


def save_json(path:Path,payload: dict | list) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding= "utf-8")
def run_validation_search(
        prepared_dir:Path,
        outputs_dir:Path,
        label_order:list[str],
        seed: int,
        config_names: list[str] | None = None,
        max_epochs_override: int | None = None,


) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_x,train_y,_ = load_feature_split(prepared_dir, "train")
    val_x, val_y, _ = load_feature_split(prepared_dir, "validation")
    train_mask = train_y>=0
    val_mask = val_y >= 0
    train_x = train_x[train_mask]
    train_y = train_y[train_mask]
    val_x = val_x[val_mask]
    val_y = val_y[val_mask]
    train_x,val_x,_,mean,std = standardize_features(train_x,val_x,None)


    validation_dir = outputs_dir / "validation"
    models_dir = validation_dir / "models"
    config_results_dir = validation_dir / "config_results"
    models_dir.mkdir(parents = True,exist_ok = True)
    config_results_dir.mkdir(parents = True,exist_ok = True)
   
    np.save(validation_dir / "scaler_mean.npy", mean)
    np.save(validation_dir / "scaler_std.npy",std)



    selected_configs = DEFAULT_CONFIGS
    if config_names:
        allowed = set(config_names)
        selected_configs = [config for config in DEFAULT_CONFIGS if config.name in allowed]
        missing = sorted(allowed.difference(config.name for config in selected_configs))
        if missing:
            raise ValueError(f"bilinmeyen config isimleri : {',' . join(missing)}")
        
    if max_epochs_override is not None:
        selected_configs = [
            TrainingConfig(
                name = config.name,
                batch_size = config.batch_size,
                learning_rate = config.learning_rate,
                patience = min(config.patience,max_epochs_override),
                hidden_dims = config.hidden_dims,
                dropout = config.dropout,
                batch_norm = config.batch_norm,
                max_epochs = max_epochs_override
                
            )
            for config in selected_configs
        ]


    results = []
    for config in selected_configs:
            result_path = config_results_dir / f"{config.name}.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding = "utf-8"))
            else:

                result = train_single_config(
                    train_x = train_x,
                    train_y = train_y,
                    val_x = val_x,
                    val_y = val_y,
                    config = config,
                    label_order = label_order,
                    device= device,
                    output_dir= models_dir,
                )
                save_json(result_path,result)
            results.append(result)
        
    results.sort(
            key= lambda item:(
                item["best_metrics"]["macro_f1"],
                item["best_metrics"]["accuracy"],
                item["best_metrics"]["weighted_f1"],


            ),
            reverse=True,
        )
    best = results[0]
    save_json(validation_dir / "grid_search.json",results)
    save_json(validation_dir / "best_config.json",best)
    save_json(validation_dir / "metrics.json",best["best_metrics"])
    print(json.dumps(best,indent=2,ensure_ascii=False))
    return best
    



def load_trained_model(
        model_path: Path,
        label_order: list[str],
        input_dim: int,
        device: torch.device,

) -> tuple[nn.Module,dict]:
    checkpoint = torch.load(model_path,map_location=device)
    config = checkpoint["config"]
    model = MLPDetector(
        input_dim=input_dim,
        output_dim=len(label_order),
        hidden_dims=config["hidden_dims"],
        dropout= config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model,checkpoint
                           
    




                    
        
    