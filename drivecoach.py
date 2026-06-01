"""
DriveCoach - Traffic Scene Analysis
Computer Vision (YOLOv8) + Natural Language Processing (FLAN-T5)

Usage:
    python drivecoach.py <image_path> [--mode template|lm|both] [--conf 0.4]

Example:
    python drivecoach.py test.jpg --mode both
"""

import argparse
import json

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ultralytics import YOLO

# --- Settings ---
YOLO_WEIGHTS = "yolov8n.pt"
LM_MODEL = "google/flan-t5-base"
CONFIDENCE_THRESHOLD = 0.4

VEHICLE_CLASSES = {"car", "truck", "bus"}
PEDESTRIAN_CLASSES = {"person"}
RIDER_CLASSES = {"bicycle", "motorcycle"}
ALL_CLASSES = VEHICLE_CLASSES | PEDESTRIAN_CLASSES | RIDER_CLASSES

YOLO_ID_TO_NAME = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ─────────────────────────────────────────────
# Computer Vision Module
# ─────────────────────────────────────────────

def run_detection(image_path: str, conf: float = CONFIDENCE_THRESHOLD) -> dict:
    """Run YOLOv8 on an image and return a structured scene summary as JSON."""
    model = YOLO(YOLO_WEIGHTS)
    results = model(image_path, conf=conf, verbose=False)[0]

    img_w, img_h = results.orig_shape[1], results.orig_shape[0]
    img_area = img_w * img_h

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = YOLO_ID_TO_NAME.get(cls_id)
        if cls_name not in ALL_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        detections.append({
            "class": cls_name,
            "confidence": round(float(box.conf), 3),
            "area_ratio": round(area / img_area, 4),
        })

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    num_vehicles   = sum(counts.get(c, 0) for c in VEHICLE_CLASSES)
    num_pedestrians = sum(counts.get(c, 0) for c in PEDESTRIAN_CLASSES)
    num_riders     = sum(counts.get(c, 0) for c in RIDER_CLASSES)
    total          = len(detections)

    if total == 0:
        density = "empty"
    elif total <= 3:
        density = "low"
    elif total <= 8:
        density = "medium"
    else:
        density = "high"

    dominant = max(counts, key=counts.get).upper() if counts else "NONE"
    closest  = max(detections, key=lambda d: d["area_ratio"])["class"].upper() if detections else "NONE"

    return {
        "num_vehicles":        num_vehicles,
        "num_pedestrians":     num_pedestrians,
        "num_riders":          num_riders,
        "total_objects":       total,
        "traffic_level":       density,
        "dominant_object_type": dominant,
        "closest_object_type": closest,
        "object_counts":       counts,
    }


# ─────────────────────────────────────────────
# NLP Module — Template baseline
# ─────────────────────────────────────────────

def generate_template(scene: dict) -> dict:
    """Deterministic rule-based description and question from scene JSON."""
    density = scene["traffic_level"]
    n_veh   = scene["num_vehicles"]
    n_ped   = scene["num_pedestrians"]
    n_rid   = scene["num_riders"]

    parts = []
    if n_veh > 0:
        parts.append(f"{n_veh} vehicle{'s' if n_veh > 1 else ''}")
    if n_ped > 0:
        parts.append(f"{n_ped} pedestrian{'s' if n_ped > 1 else ''}")
    if n_rid > 0:
        parts.append(f"{n_rid} rider{'s' if n_rid > 1 else ''}")

    if parts:
        joined = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0])
        description = f"The scene contains {joined} with {density} traffic density."
    else:
        description = "No traffic participants were detected in the scene."

    if density == "high":
        question = "What precautions should you take when driving in heavy traffic?"
    elif n_ped > 0:
        question = "How should you behave when pedestrians are present near the road?"
    elif n_rid > 0:
        question = "What should you be aware of when cyclists or riders are nearby?"
    elif density in ("low", "medium"):
        question = f"How does {density} traffic density affect your following distance and attention?"
    else:
        question = "What does an empty road mean for your speed and awareness?"

    return {"description": description, "question": question}


# ─────────────────────────────────────────────
# NLP Module — FLAN-T5 language model
# ─────────────────────────────────────────────

def generate_lm(scene: dict, model_name: str = LM_MODEL) -> dict:
    """Generate description and question using FLAN-T5 from structured scene JSON."""
    print(f"  Loading language model: {model_name} ...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    lm = T5ForConditionalGeneration.from_pretrained(model_name)
    lm.eval()

    prompt = (
        "You are DriveCoach, a driving instructor.\n"
        "You will receive ONLY JSON describing a driving scene.\n"
        "Generate learner-friendly output grounded ONLY in the JSON.\n\n"
        "Return two parts:\n"
        "Description: <one sentence>\n"
        "Question: <one question>\n\n"
        f"JSON:\n{json.dumps(scene, ensure_ascii=False)}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = lm.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    description, question = "", ""
    for line in raw.split("\n"):
        if line.startswith("Description:"):
            description = line.removeprefix("Description:").strip()
        elif line.startswith("Question:"):
            question = line.removeprefix("Question:").strip()
    if not description:
        description = raw.strip()

    return {"description": description, "question": question}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DriveCoach: CV + NLP traffic scene analysis")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--mode", choices=["template", "lm", "both"], default="both",
                        help="NLP generation mode (default: both)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"YOLOv8 confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--lm-model", default=LM_MODEL,
                        help=f"HuggingFace model name (default: {LM_MODEL})")
    parser.add_argument("--save-json", metavar="FILE",
                        help="Save structured CV output to a JSON file")
    args = parser.parse_args()

    # Step 1: Computer Vision
    print(f"\n[CV] Detecting objects in: {args.image}  (conf >= {args.conf})")
    scene = run_detection(args.image, conf=args.conf)
    print("\n[CV] Structured scene summary:")
    print(json.dumps(scene, indent=2))

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(scene, f, indent=2)
        print(f"[CV] JSON saved to: {args.save_json}")

    # Step 2: NLP
    print()
    if args.mode in ("template", "both"):
        print("=== TEMPLATE OUTPUT ===")
        out = generate_template(scene)
        print(f"Description: {out['description']}")
        print(f"Question:    {out['question']}\n")

    if args.mode in ("lm", "both"):
        print("=== LANGUAGE MODEL OUTPUT ===")
        out = generate_lm(scene, model_name=args.lm_model)
        print(f"Description: {out['description']}")
        print(f"Question:    {out['question']}\n")


if __name__ == "__main__":
    main()
