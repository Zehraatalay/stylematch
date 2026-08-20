import argparse
import gc

import torch

from torch.utils.data import DataLoader

from config import (
    OUTPUT_DIR,
    PAIR_DIR,
    SEED,
    SPLITS
)

from dataset import PolyvorePairDataset

from model_scratch import (
    ScratchEarlyFusionNet
)

from train_utils import (
    evaluate,
    save_json,
    set_seed,
    train_epochs
)


def make_loader(
    split,
    batch_size,
    augment,
    max_pairs
):

    ds = PolyvorePairDataset(
        SPLITS[split]["parquet"],
        PAIR_DIR / f"{split}_pairs.json",
        mode="scratch",
        augment=augment,
        max_pairs=max_pairs
    )

    workers = 2

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            workers > 0
        )
    )

    return ds, loader


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        default="scratch"
    )

    parser.add_argument(
        "--optimizer",
        choices=[
            "adam",
            "adamw"
        ],
        default="adamw"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=24
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max_train_pairs",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--max_valid_pairs",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--max_test_pairs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--final",
        action="store_true"
    )

    parser.add_argument(
        "--skip_test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.final:

        args.max_train_pairs = None
        args.max_valid_pairs = None

        # Upper bound only.
        # Early stopping can stop earlier.
        args.epochs = 8

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print("[device]", device)
    print("[configuration]", vars(args))

    run_dir = (
        OUTPUT_DIR
        / "scratch"
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_ds, train_loader = make_loader(
        "train",
        args.batch_size,
        True,
        args.max_train_pairs
    )

    valid_ds, valid_loader = make_loader(
        "valid",
        args.batch_size,
        False,
        args.max_valid_pairs
    )

    model = ScratchEarlyFusionNet(
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        base_channels=args.base_channels
    ).to(device)

    if args.optimizer == "adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    else:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    history, best_auc = train_epochs(
        model,
        train_loader,
        valid_loader,
        optimizer,
        device,
        args.epochs,
        args.patience,
        args.run_name,
        checkpoint_path=(
            run_dir / "best.pt"
        )
    )

    save_json(
        {
            "best_val_auc":
                best_auc,

            "history":
                history,

            "args":
                vars(args)
        },

        run_dir
        / "history.json"
    )

    del train_loader
    del valid_loader
    del train_ds
    del valid_ds

    gc.collect()

    if not args.skip_test:

        test_ds, test_loader = make_loader(
            "test",
            args.batch_size,
            False,
            args.max_test_pairs
        )

        metrics = evaluate(
            model,
            test_loader,
            device
        )

        print(
            "[TEST]",
            {
                k: round(v, 4)
                for k, v
                in metrics.items()
            }
        )

        save_json(
            metrics,
            run_dir
            / "test_metrics.json"
        )

    else:

        print(
            "[TEST] skipped"
        )


if __name__ == "__main__":
    main()