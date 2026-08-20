import argparse
import subprocess
import sys

def run(cmd):
    print("\n" + "="*70)
    print(
        " ".join(cmd),
        flush=True
    )

    print("=" * 70)

    subprocess.run(
        cmd,
        check=True
    )
def tune_scratch():
    experiments = [
        {
            "name": "S1_baseline",
                        "optimizer": "adam",
                        "lr": "3e-4",
                        "dropout": "0.3",
                        "weight_decay": "0",
                        "embed": "128",
                        "channels": "24"
        },

        {
            "name": "S2_adamw",
            "optimizer": "adamw",
            "lr": "3e-4",
            "dropout": "0.4",
            "weight_decay": "1e-4",
            "embed": "128",
            "channels": "24"
        },

        {
            "name": "S3_capacity",
                        "optimizer": "adamw",
                        "lr": "3e-4",
                        "dropout": "0.3",
                        "weight_decay": "1e-4",
                        "embed": "256",
                        "channels": "32"
        }
    ]

    for e in experiments:
        run([
            sys.executable,
            "train_scratch.py",

            "--run_name",
            e["name"],

            "--optimizer",
            e["optimizer"],
            "--lr",
            e["lr"],
            "--dropout",
            e["droput"],
            "--weight_decay",
            e["weight_decay"],

            "--embed_dim",
            e["embed"],
            "--base_channels",
            e["channels"],
            "--epochs",
            "4",

            "--patience",
            "2",
            "--skip_test"

        ])
def tune_transfer():
    experiments = [
        {
            "name": "T1_baseline",
                        "optimizer": "adam",
                        "embed": "128",
                        "dropout": "0.3",
                        "unfreeze": "layer4",
                        "backbone_lr": "1e-5"
        },
        {
            "name": "T2_adamw",
                        "optimizer": "adamw",
                        "embed": "256",
                        "dropout": "0.4",
                        "unfreeze": "layer4",
                        "backbone_lr": "1e-5"
        },

        {
                    "name": "T3_deeper",
                    "optimizer": "adamw",
                    "embed": "128",
                    "dropout": "0.3",
                    "unfreeze": "layer3_layer4",
                    "backbone_lr": "3e-6"
                }
    ]

    for e in experiments:
        run([
            sys.executable,
            "train_transfer.py",
            "--run_name",
            e["name"],

            "--optimizer",
            e["optimizer"],
            "--embed_dim",
            e["embed"],
            "--dropout",
            e["dropout"],
            "--unfreeze",
            e["unfreeze"],
            "--backbone_lr",
            e["unfreeze"],
            "--backbone_lr",
            e["backbone_lr"],

            "--head_epochs",
            "1",
            "--finetune_epochs",
            "3",
            "--patience",
            "2",
            "--skip_test"
        ])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=[
            "scratch",
            "transfer"
        ],
        required= True
    )

    args = parser.parse_args()
    if args.method == "scratch":
        tune_scratch()
    else:
        tune_transfer()
if __name__ == "__main__":
    main()