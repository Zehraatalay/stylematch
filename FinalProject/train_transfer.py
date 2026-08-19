import argparse 
import copy 
import gc 
import torch 
from torch.utils.data import DataLoader
from config import (OUTPUT_DIR, PAIR_DIR, SEED, SPLITS)
from dataset import PolyvorePairDataset 
from model_transfer import (TransferSiameseNet)
from train_utils import (evaluate, save_json, set_seed, train_epochs)

def make_loader(split, batch_size, augment, max_pairs) : 
    ds = PolyvorePairDataset(SPLITS[split]["parquet"], PAIR_DIR / f"{split}_pairs.json", mode = "transfer", augment=augment, max_pairs = max_pairs)
    workers = 2 
    loader = DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    return ds, loader

def main() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="transfer")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--finetune_head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze", choices=["layer4", "layer3_layer4"], default="layer4")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--head_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_train_pairs", type=int, default=20000)
    parser.add_argument("--max_valid_pairs", type=int, default=5000)
    parser.add_argument("--max_test_pairs", type=int, default=None)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()
    if args.final : 
        args.max_train_pairs = None
        args.max_valid_pairs = None 
        args.head_epochs = 2
        args.finetune_epochs = 6
    set_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)
    print("[configuration]", vars(args))
    run_dir = (OUTPUT_DIR / "transfer" / args.run_name)

    run_dir.mkdir(parents = True, exist_ok = True)
    train_ds, train_loader = make_loader("train", args.batch_size, True, args.max_train_pairs)
    valid_ds, valid_loader = make_loader("valid", args.batch_size, False, args.max_valid_pairs)
    model = TransferSiameseNet(args.embed_dim, args.dropout, pretrained=True).to(device)
    optimizer = (torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW)

    model.freeze_backbone()
    optimizer1 = optimizer(model.head_parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    hist1, auc1 = train_epochs(model, train_loader, valid_loader, optimizer1, device, args.head_epochs, args.patience, args.run_name + "-head", run_dir / "stage1_best.pt")
    state1 = copy.deepcopy(model.state_dict()) 
    if args.unfreeze == "layer4" : 
        model.unfreeze_layer4()
    else : 
        model.unfreeze_layer3_layer4()

    optimizer2 = optimizer([{"params" : model.backbone_trainable_parameters(), "lr" : args.backbone_lr}, {"params" : model.head_parameters(), "lr" : args.finetune_head_lr}], weight_decay=args.weight_decay)
    hist2, auc2 = train_epochs(model, train_loader, valid_loader, optimizer2, device, args.finetune_epochs, args.patience, args.run_name + "-finetune", run_dir / "stage2_best.pt")
    state2 = copy.deepcopy(model.state_dict())
    if auc1 > auc2 : 
        model.load_state_dict(state1)
        best_auc = auc1
        best_stage = "stage1"
    else : 
        model.load_state_dict(state2) 
        best_auc = auc2 
        best_stage = "stage2"
    torch.save({"model_state" : model.state_dict(), "args" : vars(args), "best_val_auc" : best_auc, "best_stage" : best_stage}, run_dir / "best.pt")
    save_json({"history_stage1" : hist1, "history_stage2" : hist2, "best_val_auc" : best_auc, "best_stage" : best_stage, "args" : vars(args)}, run_dir / "history.json")

    del train_loader
    del valid_loader
    del train_ds
    del valid_ds
    gc.collect()
    if not args.skip_test : 
        test_ds, test_loader = make_loader("test", args.batch_size, False, args.max_test_pairs)
        metrics = evaluate(model, test_loader, device)
        print("[TEST]", {k : round(v, 4) for k, v in metrics.items()})
        save_json(metrics, run_dir / "test_metrics.json")
    else : 
        print("[TEST] skipped")

if __name__ == "__main__" : 
    main()   
