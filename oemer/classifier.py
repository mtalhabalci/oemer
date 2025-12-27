import random
import pickle
import os
from pathlib import Path
import os
from os import remove
from pathlib import Path
from PIL import Image

import augly.image as imaugs
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from oemer.bbox import get_bbox, merge_nearby_bbox, draw_bounding_boxes, rm_merge_overlap_bbox
from oemer.build_label import find_example


SVM_PARAM_GRID = {
    'degree': [2, 3, 4],
    'decision_function_shape': ['ovo', 'ovr'],
    'C':[0.1, 1, 10, 100],
    'gamma':[0.0001, 0.001, 0.1, 1],
    'kernel':['rbf', 'poly']
}
TARGET_WIDTH = 40
TARGET_HEIGHT = 70
DISTANCE = 10

def _is_colab() -> bool:
    try:
        return os.path.exists("/content")
    except Exception:
        return False

def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)

# Ortam/Colab duyarlı varsayılan veri yolu:
# 1) OEMER_DATASET_PATH env değişkeni varsa onu kullan
# 2) Colab ise Drive'daki ds2_dense_tmn/segmentation yolunu kullan
# 3) Yerelde varsayılan ./ds2_dense/segmentation
DATASET_PATH = (
    _env("OEMER_DATASET_PATH")
    or ("/content/drive/MyDrive/omr_dataset/dataset/ds2/ds2_dense_tmn/segmentation" if _is_colab() else "./ds2_dense/segmentation")
)


def _collect(color, out_path, samples=100):
    out_path = Path(out_path)
    # Klasörleri ebeveynleriyle birlikte oluştur
    out_path.mkdir(parents=True, exist_ok=True)

    cur_samples = 0
    add_space = 10
    idx = 0
    while cur_samples < samples:
        arr = find_example(DATASET_PATH, color)
        if arr is None:
            continue
        arr[arr!=200] = 0
        boxes = get_bbox(arr)
        if len(boxes) > 1:
            boxes = merge_nearby_bbox(boxes, DISTANCE)
        boxes = rm_merge_overlap_bbox(boxes)
        for box in boxes:
            if idx >= samples:
                break
            print(f"{idx+1}/{samples}", end='\r')
            patch = arr[box[1]-add_space:box[3]+add_space, box[0]-add_space:box[2]+add_space]
            ratio = random.choice(np.arange(0.6, 1.3, 0.1))
            tar_w = int(ratio * patch.shape[1])
            tar_h = int(ratio * patch.shape[0])
            img = imaugs.resize(Image.fromarray(patch.astype(np.uint8)), width=tar_w, height=tar_h)

            seed = random.randint(0, 1000)
            np.float = float  # Monkey patch to workaround removal of np.float
            img = imaugs.perspective_transform(img, seed=seed, sigma=3)
            img = np.where(np.array(img)>0, 255, 0)
            Image.fromarray(img.astype(np.uint8)).save(out_path / f"{idx}.png")
            idx += 1

        cur_samples += len(boxes)
    print()


def collect_data(samples=400):
    color_map = {
        74: "sharp",
        70: "flat",
        72: "natural",
        97: 'rest_whole',
        98: 'rest_half',
        99: 'rest_quarter',
        100: 'rest_8th',
        101: 'rest_16th',
        102: 'rest_32nd',
        103: 'rest_64th',
        10: 'gclef',
        13: 'fclef',
        
        # tmn sembolleri
        216: "tmn_9_diyez",
        215: "tmn_9_bemol",
        214: "tmn_8_diyez",
        213: "tmn_8_bemol",
        212: "tmn_5_diyez",
        211: "tmn_4_bemol",
        210: "tmn_1_diyez",
        209: "tmn_1_bemol",
    }

    for color, name in color_map.items():
        print('Current', name)
        _collect(color, f"train_data/{name}", samples=samples)
        _collect(color, f"test_data/{name}", samples=samples)


def train(folders):
    class_map = {idx: Path(ff).name for idx, ff in enumerate(folders)}
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.glob('*.png'):
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1

    print("Train model")
    model = svm.SVC()#C=0.1, gamma=0.0001, kernel='poly', degree=2, decision_function_shape='ovo')
    #model = AdaBoostClassifier(n_estimators=50)
    #model = BaggingClassifier(n_estimators=50)  # For sfn classification
    #model = RandomForestClassifier(n_estimators=50)
    #model = GradientBoostingClassifier(n_estimators=50, verbose=1)
    #model = GridSearchCV(svm.SVC(), SVM_PARAM_GRID)
    #model = KNeighborsClassifier(n_neighbors=len(folders))#, weights='distance')
    model.fit(train_x, train_y)
    return model, class_map

def build_class_map(folders):
    return {idx: Path(ff).name for idx, ff in enumerate(folders)}

def train_tf(folders):
    import tensorflow as tf
    class_map = build_class_map(folders)
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.iterdir():
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1
    train_x = np.array(train_x)[..., np.newaxis]
    train_y = tf.one_hot(train_y, len(folders))
    output_types = (tf.uint8, tf.uint8)
    output_shapes = ((TARGET_HEIGHT, TARGET_WIDTH, 1), (len(folders)))
    dataset = tf.data.Dataset.from_generator(lambda: zip(train_x, train_y), output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.shuffle(len(train_y), reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(16)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(folders), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, batch_size=16, epochs=10)
    return model, class_map


def test(model, folders):
    test_x = []
    test_y = []
    samples = 100
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        files = list(folder.glob('*.png'))
        random.shuffle(files)
        for ff in files:
            if idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            test_x.append(arr)
            test_y.append(cidx)
            idx += 1

    pred_y = model.predict(test_x)
    tp_idx = (pred_y == test_y)
    tp = len(pred_y[tp_idx])
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def test_tf(model, folders):
    test_x = []
    test_y = []
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        files = list(folder.iterdir())
        random.shuffle(files)
        for ff in files:
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            test_x.append(arr)
            test_y.append(cidx)

    test_x = np.array(test_x)[..., np.newaxis]
    test_y = np.array(test_y)
    test_result = []
    batch_size = 16
    for idx in range(0, len(test_x), batch_size):
        data = test_x[idx:idx+batch_size]
        pred = model.predict(data)
        pidx = np.argmax(pred, axis=-1)
        test_result.extend(list(pidx))

    test_result = np.array(test_result)
    tp = test_result[test_result==test_y].size
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


def predict(region, model_name):
    if np.max(region) == 1:
        region *= 255
    # Tek bir sabit kök: repo kökü (..../oemer/oemer/oemer -> repo root)
    here = Path(__file__).resolve()
    # Repo kökü: .../repo/oemer (iki seviye yukarı)
    repo_root = here.parent.parent
    # Modellerin yeri: <repo>/oemer/sklearn_models
    base_dir = os.path.join(str(repo_root), "sklearn_models")
    nested_dir = os.path.join(base_dir, model_name)
    # Nested yapıyı tercih et
    pkl_path_nested = os.path.join(nested_dir, f"{model_name}.model")
    keras_path_nested = os.path.join(nested_dir, f"{model_name}.keras")
    meta_path_nested = os.path.join(nested_dir, f"{model_name}_meta.pkl")
    # Düz (flat) yapı da desteklenir
    pkl_path = os.path.join(base_dir, f"{model_name}.model")
    keras_path = os.path.join(base_dir, f"{model_name}.keras")
    meta_path = os.path.join(base_dir, f"{model_name}_meta.pkl")

    # Önce pickle (pointer ya da sklearn)
    if os.path.exists(pkl_path_nested) or os.path.exists(pkl_path):
        load_path = pkl_path_nested if os.path.exists(pkl_path_nested) else pkl_path
        m_info = pickle.load(open(load_path, "rb"))
        if 'keras_path' in m_info:
            import tensorflow as tf
            w = m_info['w']; h = m_info['h']; class_map = m_info['class_map']
            model_file = m_info.get('keras_path')
            if not model_file or not os.path.isabs(model_file):
                model_file = os.path.join(os.path.dirname(load_path), f"{model_name}.keras")
            if not os.path.exists(model_file):
                model_file = keras_path_nested if os.path.exists(keras_path_nested) else keras_path
            model = tf.keras.models.load_model(model_file)
            img = Image.fromarray(region.astype(np.uint8)).resize((w, h))
            arr = np.array(img)[np.newaxis, ..., np.newaxis]
            pred = model.predict(arr)
            idx = int(np.argmax(pred, axis=-1)[0])
            return class_map[idx]
        # sklearn modeli
        model = m_info['model']
        w = m_info['w']; h = m_info['h']
        img = Image.fromarray(region.astype(np.uint8)).resize((w, h))
        pred = model.predict(np.array(img).reshape(1, -1))
        return m_info['class_map'][pred[0]]

    # Keras + meta ikilisi
    if (os.path.exists(keras_path_nested) and os.path.exists(meta_path_nested)) or (os.path.exists(keras_path) and os.path.exists(meta_path)):
        import tensorflow as tf
        meta_file = meta_path_nested if os.path.exists(meta_path_nested) else meta_path
        keras_file = keras_path_nested if os.path.exists(keras_path_nested) else keras_path
        meta = pickle.load(open(meta_file, "rb"))
        w = meta['w']; h = meta['h']; class_map = meta['class_map']
        model = tf.keras.models.load_model(keras_file)
        img = Image.fromarray(region.astype(np.uint8)).resize((w, h))
        arr = np.array(img)[np.newaxis, ..., np.newaxis]
        pred = model.predict(arr)
        idx = int(np.argmax(pred, axis=-1)[0])
        return class_map[idx]

    raise FileNotFoundError(
        f"No model found for '{model_name}'. Expected nested or flat under {base_dir}."
    )

def train_rests_above8(filename = "rests_above8.model"):
    folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_rests(filename = "rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_all_rests(filename = "all_rests.model"):
    folders = ["rest_whole", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_sfn(filename = "sfn.model"):
    base = Path("train_data")
    folders = ["sharp", "flat", "natural"]
    if base.exists():
        # Auto-detect TMN classes collected via categories.json
        detected = []
        for d in base.iterdir():
            try:
                if d.is_dir():
                    n = d.name
                    if n in {"sharp", "flat", "natural"} or n.startswith("tmn_"):
                        detected.append(n)
            except Exception:
                continue
        detected = sorted(set(detected))
        if len(detected) >= 3:
            folders = detected
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    # Küçük format + nested kayıt: repo_kökü/sklearn_models/sfn/
    try:
        import tensorflow as tf
        here = Path(__file__).resolve()
        # Repo kökü: .../repo/oemer (iki seviye yukarı)
        repo_root = here.parent.parent
        # Kayıt yeri: <repo>/oemer/sklearn_models/sfn
        base_dir = os.path.join(str(repo_root), "sklearn_models")
        nested_dir = os.path.join(base_dir, "sfn")
        os.makedirs(nested_dir, exist_ok=True)
        keras_out = os.path.join(nested_dir, "sfn.keras")
        meta_out = os.path.join(nested_dir, "sfn_meta.pkl")
        pointer_out = os.path.join(nested_dir, "sfn.model")
        model.save(keras_out)
        pickle.dump({"class_map": class_map, "w": TARGET_WIDTH, "h": TARGET_HEIGHT}, open(meta_out, "wb"))
        pickle.dump({"keras_path": keras_out, "class_map": class_map, "w": TARGET_WIDTH, "h": TARGET_HEIGHT}, open(pointer_out, "wb"))
        # Eski çağrılar için filename'e de pointer yaz (eğer farklıysa)
        if filename and filename != pointer_out:
            try:
                pickle.dump({"keras_path": keras_out, "class_map": class_map, "w": TARGET_WIDTH, "h": TARGET_HEIGHT}, open(filename, "wb"))
            except Exception:
                pass
        print(f"✅ SFN küçük format üretildi: {keras_out}, {pointer_out}")
    except Exception as e:
        # Geriye uyumluluk: büyük pickle yaz
        output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
        pickle.dump(output, open(filename, "wb"))
        print(f"⚠️ Küçük format başarısız, büyük pickle yazıldı: {filename} -> {e}")


def train_clefs(filename = "clef.model"):
    folders = ["gclef", "fclef"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(filename, "wb"))


def train_noteheads():
    folders = ["notehead_solid", "notehead_hollow"]
    model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    test_tf(model, [f"test_data/{folder}" for folder in folders])
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(f"notehead.model", "wb"))
    

if __name__ == "__main__":
    samples = 400
    # collect_data(samples=samples)

    # folders = ["gclef", "fclef"]; model_name = "clef"
    # folders = ["sharp", "flat", "natural"]; model_name = "sfn"
    folders = ["rest_whole", "rest_quarter", "rest_8th"]; model_name = "rests"
    # folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]; model_name = "rests_above8"

    #folders = ['clefs', 'sfns']; model_name = 'clefs_sfns'
    #folders = ["rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]

    # Sklearn model
    model, class_map = train([f"ds2_dense/train_data/{folder}" for folder in folders])
    test(model, [f"ds2_dense/test_data/{folder}" for folder in folders])

    # TF-based model
    # model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    # test_tf(model, [f"test_data/{folder}" for folder in folders])

    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(f"oemer/sklearn_models/{model_name}.model", "wb"))
