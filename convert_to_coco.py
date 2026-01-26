import os
import supervision as sv

dataset_dir = "yolo_dataset"
coco_output_dir = "coco_dataset"

for split in ["train", "val", "test"]:
    images_path = os.path.join(dataset_dir, "images", split)
    labels_path = os.path.join(dataset_dir, "labels", split)
    
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=images_path,
        annotations_directory_path=labels_path,
        data_yaml_path=os.path.join(dataset_dir, "data.yaml")
    )
    
    os.makedirs(os.path.join(coco_output_dir, split), exist_ok=True)
    ds.as_coco(
        images_directory_path=os.path.join(coco_output_dir, split),
        annotations_path=os.path.join(coco_output_dir, split, "_annotations.coco.json")
    )
    
    print(f"Done {split} set")
    
print("Done")