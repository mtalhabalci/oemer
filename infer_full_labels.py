import argparse
import os
import cv2
import numpy as np
from PIL import Image

from oemer.inference import inference as seg_inference
from oemer.inference import resize_image as _infer_resize
from oemer.inference import predict as clf_predict


# Color palette per fine-grained type
COLORS = {
    "notehead": (255, 0, 0),          # red
    "barline": (250, 0, 200),         # pink
    "rest": (11, 163, 0),            # green
    "clef": (0, 180, 255),           # orange-blue
    "accidental": (53, 0, 168),      # purple
}


def classify_clef_or_accidental(region: np.ndarray) -> str:
    """Try clef first; if not a clef, fall back to accidental (sfn)."""
    try:
        label = clf_predict(region, "clef")
        if label in {"gclef", "fclef"}:
            return label
    except Exception:
        pass
    try:
        sfn = clf_predict(region, "sfn")
        return sfn  # sharp/flat/natural
    except Exception:
        return "unknown"


def classify_rest(region: np.ndarray) -> str:
    """Classify rest types using rests and rests_above8 models."""
    try:
        label = clf_predict(region, "rests")
        if "8th" in label:
            # refine 8th+ using the dedicated model
            label = clf_predict(region, "rests_above8")
        return label
    except Exception:
        return "rest"


def main():
    parser = argparse.ArgumentParser(description="Segment an image and draw bboxes with fine-grained English labels for all symbols.")
    parser.add_argument("--model-dir", required=True, help="Final model directory containing arch.json/weights.h5 or model.keras")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="annotated_full_labels.png", help="Output annotated image path")
    parser.add_argument("--step-size", type=int, default=128, help="Sliding window step size (default: 128)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    # Heuristic thresholds
    parser.add_argument("--barline-height-min", type=int, default=40, help="Minimum height in pixels to consider a barline")
    parser.add_argument("--barline_aspect_min", type=float, default=6.0, help="Min h/w aspect ratio to consider a barline")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Run TF-based inference using final model artifacts
    class_map, _ = seg_inference(
        model_path=args.model_dir,
        img_path=args.image,
        step_size=args.step_size,
        batch_size=args.batch_size,
        manual_th=None,
        use_tf=True,
    )

    # Use the same resizing logic as inference to align masks and drawing
    _pil = Image.open(args.image).convert("RGB")
    _pil_resized = _infer_resize(_pil)
    img_gray = np.array(_pil_resized.convert("L"))
    annotated = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    stats = {"notehead": 0, "barline": 0, "rest": 0, "clef": 0, "accidental": 0}

    # Masks
    mask_c1 = (class_map == 1).astype(np.uint8)  # stems/rests/barlines
    mask_c2 = (class_map == 2).astype(np.uint8)  # noteheads
    mask_c3 = (class_map == 3).astype(np.uint8)  # clefs/keys/accidentals

    # Noteheads
    contours, _ = cv2.findContours(mask_c2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), COLORS["notehead"], 2)
        cv2.putText(annotated, "notehead", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["notehead"], 1)
        stats["notehead"] += 1

    # C1 split into barlines vs rests using heuristics
    contours, _ = cv2.findContours(mask_c1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / max(w, 1)
        region = img_gray[y:y + h, x:x + w]
        if h >= args.barline_height_min and aspect >= args.barline_aspect_min:
            # barline
            cv2.rectangle(annotated, (x, y), (x + w, y + h), COLORS["barline"], 2)
            cv2.putText(annotated, "barline", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["barline"], 1)
            stats["barline"] += 1
        else:
            # rest
            label = classify_rest(region)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), COLORS["rest"], 2)
            cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["rest"], 1)
            stats["rest"] += 1

    # Clefs / accidentals (TMN) from C3
    # Light morphological cleanup to separate touching components
    ker = np.ones((3, 3), dtype=np.uint8)
    mask_c3_clean = cv2.morphologyEx(mask_c3, cv2.MORPH_OPEN, ker)
    contours, _ = cv2.findContours(mask_c3_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        region = img_gray[y:y + h, x:x + w]
        label = classify_clef_or_accidental(region)
        if label in {"gclef", "fclef"}:
            color = COLORS["clef"]
            stats["clef"] += 1
        elif label in {"sharp", "flat", "natural"}:
            color = COLORS["accidental"]
            stats["accidental"] += 1
        else:
            color = COLORS["accidental"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    print(f"Saved annotated image: {args.out}")
    print("Counts:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
