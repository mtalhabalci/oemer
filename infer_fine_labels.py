import argparse
import os
import cv2
import numpy as np
from PIL import Image

from oemer.inference import inference as seg_inference
from oemer.inference import resize_image as _infer_resize
from oemer import classifier


def _crop_region(binary_map: np.ndarray, bbox, pad: int = 2) -> np.ndarray:
    x, y, w, h = bbox
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(binary_map.shape[0], y + h + pad)
    x2 = min(binary_map.shape[1], x + w + pad)
    region = binary_map[y1:y2, x1:x2]
    # Ensure 0/255 for classifier.predict
    region = (region > 0).astype(np.uint8) * 255
    return region


def main():
    parser = argparse.ArgumentParser(description="Segment an image and draw bboxes with fine class names (SFN/TMN, clefs, rests, noteheads).")
    parser.add_argument("--model-dir", required=True, help="Final model directory containing arch.json/weights.h5 or model.keras")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="annotated_fine.png", help="Output annotated image path")
    parser.add_argument("--step-size", type=int, default=128, help="Sliding window step size (default: 128)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--use-tf", action="store_true", help="Use TensorFlow for segmentation inference (default: onnxruntime)")
    parser.add_argument("--rest-model", default=None, help="Rest classifier model name (rests | all_rests | rests_above8). If omitted, auto-selects the most detailed available.")
    parser.add_argument("--clef-area-th", type=int, default=1200, help="Area threshold to treat a group-3 bbox as clef (else SFN/TMN)")
    parser.add_argument("--stem_aspect_th", type=float, default=3.0, help="Aspect ratio h/w threshold to consider a bbox as stem/barline (skip labeling)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Resolve rest model name (auto-prefer all_rests if available)
    from pathlib import Path
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # .../repo/oemer
    rest_base = os.path.join(str(repo_root), "sklearn_models")
    def _exists(model_name: str) -> bool:
        nested = os.path.join(rest_base, model_name, f"{model_name}.model")
        flat = os.path.join(rest_base, f"{model_name}.model")
        return os.path.exists(nested) or os.path.exists(flat)
    chosen_rest = args.rest_model
    if not chosen_rest:
        if _exists("all_rests"):
            chosen_rest = "all_rests"
        elif _exists("rests"):
            chosen_rest = "rests"
        elif _exists("rests_above8"):
            chosen_rest = "rests_above8"
        else:
            chosen_rest = "rests"  # fallback

    # Run segmentation inference
    class_map, _ = seg_inference(
        model_path=args.model_dir,
        img_path=args.image,
        step_size=args.step_size,
        batch_size=args.batch_size,
        manual_th=None,
        use_tf=args.use_tf,
    )

    # Align drawing with inference resizing
    _pil = Image.open(args.image).convert("RGB")
    _pil_resized = _infer_resize(_pil)
    base_gray = np.array(_pil_resized.convert("L"))
    annotated = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    # Prepare binary maps per group
    group1 = (class_map == 1).astype(np.uint8)  # Stems/Rests/Barlines
    group2 = (class_map == 2).astype(np.uint8)  # Noteheads
    group3 = (class_map == 3).astype(np.uint8)  # Clefs/Keys/Accidentals (TMN)

    colors = {
        "sfn": (0, 128, 255),
        "tmn": (0, 128, 255),
        "clef": (255, 0, 128),
        "rest": (0, 200, 0),
        "notehead": (255, 0, 0),
        "stem": (200, 200, 200),
        "barline": (200, 200, 200),
        "unknown": (255, 255, 0),
    }

    # Group 3: clefs vs SFN/TMN accidentals
    contours3, _ = cv2.findContours(group3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours3:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        region = _crop_region(group3, (x, y, w, h), pad=2)
        if area >= args.clef_area_th:
            # Likely clef
            try:
                label = classifier.predict(region, "clef")
            except Exception:
                label = "clef"
            color = colors.get("clef")
        else:
            # SFN/TMN
            try:
                label = classifier.predict(region, "sfn")
            except Exception:
                label = "sfn"
            color = colors.get("sfn")
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Group 1: rests vs stems/barlines (filter by aspect ratio)
    contours1, _ = cv2.findContours(group1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours1:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = h / max(1, w)
        region = _crop_region(group1, (x, y, w, h), pad=2)
        if aspect >= args.stem_aspect_th:
            # Likely stem/barline; skip or mark lightly
            color = colors.get("stem")
            label = "stem"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
            cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            continue
        # Treat as rest
        try:
            label = classifier.predict(region, chosen_rest)
        except Exception:
            label = "rest"
        color = colors.get("rest")
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Group 2: noteheads (generic or classified if model exists)
    contours2, _ = cv2.findContours(group2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Not: notehead için model kullanılmıyor; tüm notehead'ler genel etiketlenir
    for c in contours2:
        x, y, w, h = cv2.boundingRect(c)
        region = _crop_region(group2, (x, y, w, h), pad=1)
        label = "notehead"
        color = colors.get("notehead")
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    print(f"Saved annotated image: {args.out}")


if __name__ == "__main__":
    main()
