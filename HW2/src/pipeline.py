from __future__ import annotations

import json
from pathlib import Path

from models import SCALES, ensure_dir, save_json, load_fashionpedia, get_category_names
from indexing import build_indices, choose_categories, select_unique_objects
from sampling import make_positive, build_negative_samples


def prepare_fashionpedia_data(output_dir:Path,num_classes:int,selected_category_ids:list[int] | None,
                            templates_per_class:int,positives_per_class:int,background_per_split: int,
                            image_size: int,seed: int,min_object_area_ratio:float,min_bbox_size:int) -> None:
    dataset_dict = load_fashionpedia()
    category_names = get_category_names(dataset_dict)
    category_index,row_index = build_indices(dataset_dict,min_object_area_ratio,min_bbox_size)
    selected = choose_categories(category_index,category_names,num_classes,templates_per_class,positives_per_class,selected_category_ids)



    templates_dir = ensure_dir(output_dir/"templates")
    validation_dir = ensure_dir(output_dir /"validation")
    test_dir = ensure_dir(output_dir/"test")
    manifests_dir = ensure_dir(output_dir /"manifests")


    template_manifest,validation_manifest,test_manifest = [], [], []
    validation_used_images: set[int] = set()
    test_used_images:set[int] = set()
    crop_scales = [1.25,1.75,2.25]

    for class_index,class_info in enumerate(selected):
        category_id = int(class_info["id"])
        label_name = str(class_info["name"])


        train_candidates = category_index["train"][category_id]
        val_candidates = category_index["val"][category_id]

        template_records = select_unique_objects(train_candidates,templates_per_class,set(),seed + class_index * 31)
        used_train_images = {r.image_id for r in template_records}
        validation_used_images.update(used_train_images)
        validation_records = select_unique_objects(
            train_candidates,
            positives_per_class,
            used_train_images,
            seed + class_index * 37
)
        validation_used_images.update(r.image_id for r in validation_records)

        test_records = select_unique_objects(val_candidates, positives_per_class, set(), seed + class_index * 41)
        test_used_images.update(r.image_id for r in test_records)


        for i, record in enumerate(test_records):
         row = dataset_dict[record.split_name][record.row_index]
        path = test_dir / label_name / f"test_pos_c{category_id}_{i:04d}.png"
        test_manifest.append(make_positive(
            row, record.bbox, image_size, crop_scales[i % 3], path,
            class_index, label_name, category_id, record.image_id, split="test"
    ))

        for i,record in enumerate(validation_records):
            row = dataset_dict[record.split_name][record.row_index]
            path = validation_dir / label_name / f"validation_pos_c{category_id}_{i:04d}.png"
            validation_manifest.append(make_positive(
                row,record.bbox,image_size,crop_scales[i % 3],path,
                class_index,label_name,category_id,record.image_id,split ="split"))
            
    selected_category_set = {int(item["id"]) for item in selected}
    validation_manifest.extend(build_negative_samples(
        dataset_dict["train"], row_index["train"], selected_category_set, validation_used_images,
        background_per_split, seed + 701, "validation", validation_dir,
    ))
    test_manifest.extend(build_negative_samples(
        dataset_dict["val"], row_index["val"], selected_category_set, test_used_images,
        background_per_split, seed + 907, "test", test_dir,
    ))

    metadata = {
        "dataset_name": "detection-datasets/fashionpedia",
        "image_size": image_size,
        "scales" : SCALES,
        "selected_classes": selected,
        "label_order": [c["name"]for c in selected] + ["background"],
        "templates_per_class": templates_per_class,
        "positives_per_class": positives_per_class,
        "min_object_area_ratio": min_object_area_ratio,
        "min_bbox_size": min_bbox_size,
        "seed": seed,
        "counts" : {
            "templates": len(template_manifest),
            "validation": len(validation_manifest),
            "test": len(test_manifest),

        },

    }

    save_json(manifests_dir/ "templates.json",template_manifest)
    save_json(manifests_dir / "validation.json", validation_manifest)
    save_json(manifests_dir / "test.json", test_manifest)
    save_json(manifests_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


    
