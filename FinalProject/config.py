import os
from pathlib import Path


DATA_ROOT = Path(
    os.environ.get(
        "POLYVORE_ROOT",
                "/kaggle/working/polyvore-outfits"
    )
)

OUTPUT_DIR = Path(
     os.environ.get(
            "STYLEMATCH_OUTPUT_DIR",
            "/kaggle/working/outputs"
        )
)
PAIR_DIR = DATA_ROOT / "pairs_stylematch"

SEED = 42
IMAGE_SIZE = 224

SPLITS = {
     "train": {
            "json":
                DATA_ROOT
                / "strict_disjoint"
                / "train.json",
    
            "parquet":
                DATA_ROOT
                / "data"
                / "disjoint"
                / "train.parquet",
        },
    
        "valid": {
            "json":
                DATA_ROOT
                / "strict_disjoint"
                / "valid.json",
    
            "parquet":
                DATA_ROOT
                / "data"
                / "disjoint"
                / "validation.parquet",
        },
    
        "test": {
            "json":
                DATA_ROOT
                / "strict_disjoint"
                / "test.json",
    
            "parquet":
                DATA_ROOT
                / "data"
                / "disjoint"
                / "test.parquet",
        },
}

METADATA_PATH = (
    DATA_ROOT
    /"Polyvore_item_metadata.json"
)