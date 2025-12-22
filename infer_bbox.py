import argparse
import os
import cv2
import numpy as np
from PIL import Image

from oemer.inference import inference as seg_inference
from oemer.inference import resize_image as _infer_resize


def draw_bboxes_on_image(img_np: np.ndarray, bboxes, color=(0, 255, 0), thickness=2):
    img_draw = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness)
    return img_draw


def main():
    parser = argparse.ArgumentParser(description="Segment a single image and draw bounding boxes per class.")
    parser.add_argument("--model-dir", required=True, help="Final model directory containing arch.json and weights.h5")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--class-id", type=int, default=None, help="Target class id (1: stems/rests/barlines, 2: noteheads, 3: clefs/keys/accidentals incl. TMN)")
    parser.add_argument("--all", action="store_true", help="Process all classes and draw them with different colors")
    parser.add_argument("--out", default="annotated.png", help="Output annotated image path")
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

    # Colors per class
    class_colors = {
        1: (0, 200, 0),      # stems/rests/barlines
        2: (255, 0, 0),      # noteheads
        3: (0, 128, 255),    # clefs/keys/accidentals (incl TMN)
    }

    process_all = args.all or (args.class_id is None)
    if process_all:
        annotated = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        total_bboxes = 0
        for cid in (1, 2, 3):
            binary_mask = (class_map == cid).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(c) for c in contours]
            color = class_colors.get(cid, (255, 255, 0))
            for (x, y, w, h) in bboxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, f"C{cid}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            total_bboxes += len(bboxes)
        cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        print(f"Saved all-classes annotated image: {args.out} (total bboxes: {total_bboxes})")
        print("Class legend: C1=stems/rests/barlines, C2=noteheads, C3=clefs/keys/accidentals (TMN dahildir)")
    else:
        cid = int(args.class_id)
        binary_mask = (class_map == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        color = class_colors.get(cid, (0, 255, 0))
        annotated = draw_bboxes_on_image(img_np, bboxes, color=color)
        cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        print(f"Saved class {cid} annotated image: {args.out} (bboxes: {len(bboxes)})")


if __name__ == "__main__":
    main()
