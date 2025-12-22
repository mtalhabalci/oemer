import sys
import time
import os
import argparse

import tensorflow as tf
import re
import glob

from oemer import train
from oemer import classifier


def write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)

# Environment helpers for smart defaults
def _is_colab() -> bool:
    try:
        return os.path.exists("/content")
    except Exception:
        return False

def _drive_mounted() -> bool:
    try:
        return os.path.exists("/content/drive/MyDrive")
    except Exception:
        return False

def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)

parser = argparse.ArgumentParser(description="Train models and manage checkpoints.")
parser.add_argument("model_name", help="Model type to train",
                    choices=[
                        "segnet", "unet",
                        "unet_from_checkpoint", "segnet_from_checkpoint",
                        "rests_above8", "rests", "all_rests", "sfn", "clef"
                    ])
parser.add_argument("--epochs", dest="epochs", type=int, default=None,
                    help="Number of epochs to train (default: 15)")
parser.add_argument("--steps", dest="steps", type=int, default=None,
                    help="Steps per epoch (default: 1500)")
parser.add_argument("--dataset-path", dest="dataset_path", default=None,
                    help="Dataset root path. For segnet, expects 'images/' and 'segmentation/' subfolders.")
parser.add_argument("--source-path", dest="source_path", default=None,
                    help="Source checkpoints directory to copy from (default depends on model)")
parser.add_argument("--target-path", dest="target_path", default=None,
                    help="Target directory in Drive to copy checkpoints into")
parser.add_argument("--final-path", dest="final_path", default=None,
                    help="Target directory in Drive to copy the final model folder into")
args = parser.parse_args()

def get_model_base_name(model_name: str) -> str:
    timestamp = str(round(time.time()))
    return f"{model_name}_{timestamp}"

model_type = args.model_name

def prepare_classifier_data():
    if not os.path.exists("train_data"):
        classifier.collect_data(2000)

if model_type == "segnet":
    # Resolve dataset path with environment-aware defaults
    ds_path = (
        args.dataset_path
        or _env("OEMER_DATASET_PATH")
        or ("/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn" if _is_colab() else None)
        or ("dataset/ds2/ds2_dense_tmn" if os.path.isdir("dataset/ds2/ds2_dense_tmn") else None)
    )
    if not ds_path or not os.path.isdir(ds_path):
        print(f"Dataset path not found. Provide --dataset-path or set OEMER_DATASET_PATH. Got: {ds_path}")
        sys.exit(2)
    model = train.train_model(
        ds_path,
        data_model=model_type,
        steps=args.steps or 1500,
        epochs=args.epochs or 15,
    )
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    # Best checkpoint'i bul ve nihai klasöre kopyala
    ckpt_dir = os.path.abspath("checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.weights.h5"))
    def _f1_of(p):
        m = re.search(r"valf1_(\d+\.\d+)", os.path.basename(p))
        return float(m.group(1)) if m else -1.0
    best_ckpt = max(ckpts, key=_f1_of) if ckpts else None
    if best_ckpt is None and ckpts:
        # fallback: en yeni dosya
        best_ckpt = max(ckpts, key=lambda p: os.path.getmtime(p))
    if best_ckpt:
        import shutil as _shutil
        _shutil.copyfile(best_ckpt, os.path.join(filename, "weights.weights.h5"))
        _shutil.copyfile(best_ckpt, os.path.join(filename, "weights.h5"))
        # Ensure full model reflects best weights
        try:
            model.load_weights(best_ckpt)
        except Exception:
            pass
    else:
        # Eğitim weights'lerini doğrudan kaydet (son çare)
        weights_new = os.path.join(filename, "weights.weights.h5")
        model.save_weights(weights_new)
        try:
            import shutil as _shutil
            _shutil.copyfile(weights_new, os.path.join(filename, "weights.h5"))
        except Exception:
            pass
    # Save full model in Keras format
    try:
        model.save(os.path.join(filename, "model.keras"))
    except Exception as e:
        print(f"Full model save skipped: {e}")
    import shutil
    import os
    source_path = args.source_path or _env("OEMER_SOURCE_PATH") or ckpt_dir
    # Only set a default target when Drive is mounted; else require --target-path
    target_path = args.target_path or _env("OEMER_TARGET_PATH") or (
        "/content/drive/MyDrive/omr_dataset/train/ds2_dense_segnet/15epoch1500step/" if _drive_mounted() else None
    )
    if target_path:
        # 🎯 Drive'a taşı
        os.makedirs(target_path, exist_ok=True)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        print(f"✅ Model başarıyla hedefe kopyalandı: {target_path}")
    else:
        print("ℹ️ Hedef kopyalama atlandı. --target-path verin veya OEMER_TARGET_PATH ayarlayın.")
    # Optionally copy final model folder
    final_base = args.final_path or _env("OEMER_FINAL_PATH")
    if final_base:
        dst_final = os.path.join(final_base, filename)
        os.makedirs(final_base, exist_ok=True)
        shutil.copytree(filename, dst_final, dirs_exist_ok=True)
        print(f"✅ Nihai model çıktı klasörü kopyalandı: {dst_final}")
elif model_type == "unet":
    ds_path = (
        args.dataset_path
        or _env("OEMER_DATASET_PATH")
        or ("CvcMuscima-Distortions" if os.path.isdir("CvcMuscima-Distortions") else None)
    )
    if not ds_path or not os.path.isdir(ds_path):
        print(f"Dataset path not found. Provide --dataset-path or set OEMER_DATASET_PATH. Got: {ds_path}")
        sys.exit(2)
    model = train.train_model(
        ds_path,
        data_model=model_type,
        steps=args.steps or 1500,
        epochs=args.epochs or 15,
    )
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    # Best checkpoint'i bul ve nihai klasöre kopyala
    ckpt_dir = os.path.abspath("checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.weights.h5"))
    def _f1_of(p):
        m = re.search(r"valf1_(\d+\.\d+)", os.path.basename(p))
        return float(m.group(1)) if m else -1.0
    best_ckpt = max(ckpts, key=_f1_of) if ckpts else None
    if best_ckpt is None and ckpts:
        best_ckpt = max(ckpts, key=lambda p: os.path.getmtime(p))
    if best_ckpt:
        import shutil as _shutil
        _shutil.copyfile(best_ckpt, os.path.join(filename, "weights.weights.h5"))
        _shutil.copyfile(best_ckpt, os.path.join(filename, "weights.h5"))
        try:
            model.load_weights(best_ckpt)
        except Exception:
            pass
    else:
        weights_new = os.path.join(filename, "weights.weights.h5")
        model.save_weights(weights_new)
        try:
            import shutil as _shutil
            _shutil.copyfile(weights_new, os.path.join(filename, "weights.h5"))
        except Exception:
            pass
    try:
        model.save(os.path.join(filename, "model.keras"))
    except Exception as e:
        print(f"Full model save skipped: {e}")
    
    import shutil
    import os
    source_path = args.source_path or _env("OEMER_SOURCE_PATH") or ckpt_dir
    target_path = args.target_path or _env("OEMER_TARGET_PATH") or (
        "/content/drive/MyDrive/oemer_dataset/trainedmodel/15epoch1500step/" if _drive_mounted() else None
    )
    if target_path:
        # 🎯 Hedefe kopyala (Drive varsa)
        os.makedirs(target_path, exist_ok=True)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        print(f"✅ Model başarıyla hedefe kopyalandı: {target_path}")
    else:
        print("ℹ️ Hedef kopyalama atlandı. --target-path verin veya OEMER_TARGET_PATH ayarlayın.")
    final_base = args.final_path or _env("OEMER_FINAL_PATH")
    if final_base:
        dst_final = os.path.join(final_base, filename)
        os.makedirs(final_base, exist_ok=True)
        shutil.copytree(filename, dst_final, dirs_exist_ok=True)
        print(f"✅ Nihai model çıktı klasörü kopyalandı: {dst_final}")
    
elif model_type == "unet_from_checkpoint" or model_type == "segnet_from_checkpoint":
    model = tf.keras.models.load_model("seg_unet", custom_objects={"WarmUpLearningRate": train.WarmUpLearningRate})
    filename = get_model_base_name(model_type.split("_")[0])
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    # Eldeki model weights'lerini kaydet
    weights_new = os.path.join(filename, "weights.weights.h5")
    model.save_weights(weights_new)
    try:
        import shutil as _shutil
        _shutil.copyfile(weights_new, os.path.join(filename, "weights.h5"))
    except Exception:
        pass
    try:
        model.save(os.path.join(filename, "model.keras"))
    except Exception as e:
        print(f"Full model save skipped: {e}")
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