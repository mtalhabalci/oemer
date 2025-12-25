import argparse
import os
import cv2
import numpy as np
from PIL import Image

from oemer.inference import inference as seg_inference
from oemer.inference import resize_image as _infer_resize


CLASS_NAMES = {
    1: "Stems/Rests/Barlines",
    2: "Noteheads",
    3: "Clefs/Keys/Accidentals (TMN)",
}

CLASS_COLORS = {
    1: (0, 200, 0),      # green
    2: (255, 0, 0),      # red
    3: (0, 128, 255),    # orange-blue
}


def main():
    parser = argparse.ArgumentParser(description="Segment an image and draw bboxes with real class names for all classes.")
    parser.add_argument("--model-dir", required=True, help="Final model directory containing arch.json/weights.h5 or model.keras")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="annotated_all.png", help="Output annotated image path")
    parser.add_argument("--step-size", type=int, default=128, help="Sliding window step size (default: 128)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
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
    img_np = np.array(_pil_resized.convert("L"))
    annotated = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    total_counts = {}
    for cid in (1, 2, 3):
        binary_mask = (class_map == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        color = CLASS_COLORS.get(cid, (255, 255, 0))
        label = CLASS_NAMES.get(cid, f"Class {cid}")
        for (x, y, w, h) in bboxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            # Place readable label just above the bbox
            cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        total_counts[label] = len(bboxes)

    cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    print(f"Saved annotated image: {args.out}")
    print("Counts per class:")
    for label, cnt in total_counts.items():
        print(f"- {label}: {cnt}")


if __name__ == "__main__":
    main()
