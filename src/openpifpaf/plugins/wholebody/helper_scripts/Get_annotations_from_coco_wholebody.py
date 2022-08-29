import json
import copy
import numpy as np


def main():
    # List of all the annotation types that should be used
    ann_types = ["keypoints", "foot_kpts", "face_kpts", "lefthand_kpts", "righthand_kpts"]

    orig_file = "../../data-mscoco/annotations_wholebody/coco_wholebody_train_v1.0.json"
    new_file = "../../data-mscoco/annotations/"
    "final_person_keypoints_train2017_wholebody_pifpaf_style.json"

    # =============================================================================
    # orig_file = "../../data-mscoco/annotations_wholebody/coco_wholebody_val_v1.0.json"
    # new_file = "../../data-mscoco/annotations/"
    # "final_person_keypoints_val2017_wholebody_pifpaf_style.json"
    # =============================================================================

    handle_validity = True
    drop_attribute_list = ["face_box", "face_kpts", "face_valid", "foot_kpts", "foot_valid",
                           "lefthand_box", "lefthand_kpts", "lefthand_valid",
                           "num_keypoints", "righthand_box", "righthand_kpts",
                           "righthand_valid", "segmentation"]
    with open(orig_file, 'r') as f:
        orig_data = json.load(f)
    new_data = copy.deepcopy(orig_data)
    new_data["annotations"] = []
    discard_count = 0
    crowd_count = 0
    for ann_dict in orig_data["annotations"]:
        if not all(x == 0 for x in ann_dict["keypoints"]):  # If all zero, only bbox
            new_dict = copy.deepcopy(ann_dict)
            for entry in drop_attribute_list:
                new_dict.pop(entry)
            ann = []
            for key in ann_types:
                ann = ann + ann_dict[key]
            if handle_validity:
                for jj, name in enumerate(["face", "foot", "lefthand", "righthand"]):
                    if not ann_dict[name + "_valid"]:
                        if name == "face":
                            if np.any(np.array(ann[23 * 3:91 * 3])) > 0:
                                print("face")
                            ann[23 * 3:91 * 3] = [0.0] * 68
                        elif name == "foot":
                            if np.any(np.array(ann[17 * 3:23 * 3])) > 0:
                                print("foot")
                            ann[17 * 3:23 * 3] = [0.0] * 6
                        elif name == "lefthand":
                            if np.any(np.array(ann[91 * 3:112 * 3])) > 0:
                                print("LH")
                                print(ann_dict["image_id"])
                                print(ann[91 * 3:112 * 3])
                            ann[91 * 3:112 * 3] = [0.0] * 21
                        elif name == "righthand":
                            if np.any(np.array(ann[112 * 3:133 * 3])) > 0:
                                print("RH")
                            ann[112 * 3:133 * 3] = [0.0] * 21
                        else:
                            raise Exception("Unknown")
                        if name != "foot":
                            bb = ann_dict[name + "_box"]
                            area = bb[2] * bb[3]
                            im_id = ann_dict["image_id"]
                            old_id = ann_dict["id"]
                            mask_id = int(str(old_id) + '00' + str(jj))
                            if area > 0.0:
                                new_data["annotations"].append({
                                    'image_id': im_id,
                                    'category_id': 1,
                                    'iscrowd': 1,
                                    'id': mask_id,
                                    'area': area,
                                    'bbox': bb,
                                    'num_keypoints': 0,
                                    'keypoints': [0.0] * 133 * 3,
                                    'segmentation': []})
                                crowd_count += 1
            new_dict["keypoints"] = ann
            new_dict['num_keypoints'] = sum(x > 0 for x in ann[2::3])
            new_data["annotations"].append(new_dict)
        else:
            discard_count += 1
    with open(new_file, 'w') as f:
        json.dump(new_data, f)
    print("\nCreated a new json file with " + str(len(new_data["annotations"]))
          + " annotations of which " + str(crowd_count) + " were crowd annotations and discarded "
          + str(discard_count) + " annotations from the original file.")


if __name__ == "__main__":
    main()
