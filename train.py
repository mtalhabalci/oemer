import sys
import time
import os
import argparse

import tensorflow as tf

from oemer import train
from oemer import classifier


def write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)

parser = argparse.ArgumentParser(description="Train models and manage checkpoints.")
parser.add_argument("model_name", help="Model type to train",
                    choices=[
                        "segnet", "unet",
                        "unet_from_checkpoint", "segnet_from_checkpoint",
                        "rests_above8", "rests", "all_rests", "sfn", "clef"
                    ])
parser.add_argument("--dataset-path", dest="dataset_path", default=None,
                    help="Dataset root path. For segnet, expects 'images/' and 'segmentation/' subfolders.")
parser.add_argument("--source-path", dest="source_path", default=None,
                    help="Source checkpoints directory to copy from (default depends on model)")
parser.add_argument("--target-path", dest="target_path", default=None,
                    help="Target directory in Drive to copy checkpoints into")
args = parser.parse_args()

def get_model_base_name(model_name: str) -> str:
    timestamp = str(round(time.time()))
    return f"{model_name}_{timestamp}"

model_type = args.model_name

def prepare_classifier_data():
    if not os.path.exists("train_data"):
        classifier.collect_data(2000)

if model_type == "segnet":
    ds_path = args.dataset_path or "/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn"
    model = train.train_model(ds_path, data_model=model_type, steps=1500, epochs=15)
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
    import shutil
    import os
    source_path = args.source_path or "/content/oemer/checkpoints/"
    target_path = args.target_path or "/content/drive/MyDrive/omr_dataset/train/ds2_dense_segnet/15epoch1500step/"
    # 🎯 Drive'a taşı
    os.makedirs(target_path, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    print(f"✅ Model başarıyla Drive'a taşındı: {target_path}")
elif model_type == "unet":
    ds_path = args.dataset_path or "CvcMuscima-Distortions"
    model = train.train_model(ds_path, data_model=model_type, steps=1500, epochs=15)
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
    
    import shutil
    import os
    source_path = args.source_path or "/content/oemer/checkpoints/"
    target_path = args.target_path or "/content/drive/MyDrive/oemer_dataset/trainedmodel/15epoch1500step/"
    # 🎯 Drive'a taşı
    os.makedirs(target_path, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    print(f"✅ Model başarıyla Drive'a taşındı: {target_path}")
    
elif model_type == "unet_from_checkpoint" or model_type == "segnet_from_checkpoint":
    model = tf.keras.models.load_model("seg_unet", custom_objects={"WarmUpLearningRate": train.WarmUpLearningRate})
    filename = get_model_base_name(model_type.split("_")[0])
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model_type == "rests_above8":
    prepare_classifier_data()
    classifier.train_rests_above8(get_model_base_name(model_type))
elif model_type == "rests":
    prepare_classifier_data()
    classifier.train_rests(get_model_base_name(model_type))
elif model_type == "all_rests":
    prepare_classifier_data()
    classifier.train_all_rests(get_model_base_name(model_type))
elif model_type == "sfn":
    prepare_classifier_data()
    classifier.train_sfn(get_model_base_name(model_type))
elif model_type == "clef":
    prepare_classifier_data()
    classifier.train_clefs(get_model_base_name(model_type))
else:
    print("Unknown model: " + model_type)
    sys.exit(1)