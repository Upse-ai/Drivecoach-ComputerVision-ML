# DriveCoach — Computer Vision + NLP Traffic Scene Analyzer

A modular AI pipeline that combines object detection and language generation to analyze traffic scenes and produce educational feedback for learner drivers.

Built as a group project for **AI508 & AI509** at SDU (Southern University of Denmark).

---

## How It Works

```
Input image
     │
     ▼
[Computer Vision]  YOLOv8n detects traffic participants
     │              → cars, trucks, buses, pedestrians, riders
     ▼
[Structured JSON]  Scene summary (object counts, traffic density, dominant object)
     │
     ▼
[NLP Module]       FLAN-T5 generates a description + reflective question
     │              OR rule-based template as baseline
     ▼
Output text for learner driver
```

---

## Models

| Module | Model | Purpose |
|--------|-------|---------|
| Computer Vision | YOLOv8n | Object detection on traffic images |
| NLP | FLAN-T5-base | Generate scene descriptions and driving questions |
| NLP Baseline | Rule-based template | Deterministic comparison baseline |

---

## Dataset

The CV module was evaluated on a subset of the **Berkeley DeepDrive (BDD100K)** dataset — real-world dashcam images with bounding box annotations.

The NLP module was fine-tuned on 44 manually annotated JSON-to-text pairs (structured scene → description + question).

---

## Detected Classes

| Category | Objects |
|----------|---------|
| Vehicles | car, truck, bus |
| Pedestrians | person |
| Riders | bicycle, motorcycle |

Traffic lights and road signs are intentionally excluded to keep the system geographically neutral.

---

## Requirements

```
torch
transformers
ultralytics
Pillow
sentencepiece
```

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Run both template and language model output
python drivecoach.py test.jpg --mode both

# Template only (no model download needed)
python drivecoach.py test.jpg --mode template

# Language model only
python drivecoach.py test.jpg --mode lm

# Adjust confidence threshold and save JSON output
python drivecoach.py test.jpg --conf 0.5 --save-json output.json
```

---

## Example Output

**CV — Structured JSON:**
```json
{
  "num_vehicles": 7,
  "num_pedestrians": 0,
  "num_riders": 1,
  "traffic_level": "medium",
  "dominant_object_type": "CAR",
  "closest_object_type": "BIKE"
}
```

**Template output:**
> Description: The scene contains 7 vehicles and 1 rider with medium traffic density.
> Question: What should you be aware of when cyclists or riders are nearby?

**FLAN-T5 output (fine-tuned):**
> Description: There are 7 vehicles at a speed of 7.5 mps. The closest object is a bike and the dominant object type is a car.
> Question: What should a driver focus on when sharing the road with cyclists?

---

## Key Findings

- YOLOv8n reliably detects vehicles but struggles with smaller objects (riders, bicycles) at high confidence thresholds
- Zero-shot FLAN-T5 cannot meaningfully interpret abstract JSON input
- Light fine-tuning (44 examples) significantly improves output coherence but hallucination remains a challenge
- The template baseline is rigid but factually reliable — a useful lower bound

---

## Authors

- **Emil Mygind Holm** — YOLO training, CV module
- **Benjamin Benedict Ahlefeldt-Laurvig** — NLP experiments and output evaluation
