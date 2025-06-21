from ultralytics import YOLO
import json

# Load the YOLOv8 model (you can change this to yolov8m.pt if using that)
model = YOLO('yolov8n.pt')
yolo_classes = list(model.names.values())

# Load your original danger_objects.json
with open('danger_objects.json', 'r') as f:
    danger_data = json.load(f)

danger_keys = list(danger_data.keys())

# Try to match with YOLO classes
matched = {}
aliases = {}

for key in danger_keys:
    if key in yolo_classes:
        matched[key] = danger_data[key]
    else:
        for yolo_name in yolo_classes:
            if key in yolo_name or yolo_name in key:
                aliases[yolo_name] = danger_data[key]
                break
        else:
            aliases[key] = danger_data[key]  # no match found

# Merge both mappings
final_mapped_data = {**matched, **aliases}

# Save new mapped JSON file
with open("danger_mapped_yolo.json", "w") as f:
    json.dump(final_mapped_data, f, indent=2)

print("✅ YOLO-mapped JSON saved as danger_mapped_yolo.json")
