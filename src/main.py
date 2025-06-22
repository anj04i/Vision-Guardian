"""
Core signal pipeline: Frame → Detection → Risk Assessment → Navigation Signal

SignalGenerator:
 - YOLO object detection + OpenCV face detection
 - Places365 scene classification
 - Spatial analysis (proximity, quadrants from bounding boxes)
 - Risk scoring (danger weights * proximity * scene context)
 - Movement calculation (safe direction based on clear regions)

NavigationSignal output:
 - risk_level: 0-1 overall danger
 - primary_threat: {object, location, proximity}
 - movement: {action, urgency, safe_regions}
 - scene_context: environment type
 - confidence: assessment certainty
"""

import json
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from torchvision import models
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    object_class: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    area: int


@dataclass
class ThreatInfo:
    object: str
    location: str  # "left", "center", "right", etc.
    proximity: float  # 0-1
    confidence: float


@dataclass
class MovementInfo:
    action: str  # "move_left", "move_right", "stop", "proceed"
    urgency: float  # 0-1
    safe_regions: List[str]


@dataclass
class NavigationSignal:
    risk_level: float
    primary_threat: Optional[ThreatInfo]
    movement: MovementInfo
    scene_context: str
    timestamp: float
    confidence: float


class SignalGenerator:
    def __init__(self):
        self.yolo = None
        self.scene_model = None
        self.scene_classes = []
        self.face_cascade = None
        self.danger_weights = {}
        self.scene_multipliers = {}

        self._load_models()
        self._load_danger_mappings()
        self._setup_scene_multipliers()

    def _load_models(self):
        """Load all AI models"""
        # YOLO for object detection
        self.yolo = YOLO("./data/yolov8n.pt")

        # Places365 for scene classification
        self.scene_model = models.resnet18(weights=None)
        self.scene_model.fc = torch.nn.Linear(self.scene_model.fc.in_features, 365)

        # Load Places365 weights if available
        try:
            checkpoint = torch.hub.load_state_dict_from_url(
                "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
                map_location="cpu",
            )
            self.scene_model.load_state_dict(checkpoint["state_dict"])
        except:
            print(
                "Warning: Could not load Places365 weights, using random initialization"
            )

        self.scene_model.eval()

        # Scene transform
        self.scene_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load scene class names
        self._load_scene_classes()

        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _load_scene_classes(self):
        """Load Places365 class names"""
        try:
            with open("./data/categories_places365.txt", "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) >= 1:
                        class_name = parts[0][3:]  # Remove /a/ prefix
                        self.scene_classes.append(class_name)
        except FileNotFoundError:
            # Fallback to common scene types
            self.scene_classes = [
                "street",
                "kitchen",
                "bedroom",
                "office",
                "park",
                "store",
                "restaurant",
                "corridor",
                "stairs",
                "parking_lot",
            ] + ["unknown"] * 355

    def _load_danger_mappings(self):
        """Load danger object weights from JSON"""
        try:
            with open("./data/danger_objects.json", "r") as f:
                danger_data = json.load(f)

            # Convert text warnings to numerical weights
            for obj, warning in danger_data.items():
                # Simple heuristic: more urgent warnings = higher weight
                if any(
                    word in warning.lower()
                    for word in ["immediately", "shelter", "avoid"]
                ):
                    weight = 0.9
                elif any(
                    word in warning.lower()
                    for word in ["danger", "stay back", "move away"]
                ):
                    weight = 0.7
                elif any(
                    word in warning.lower() for word in ["caution", "careful", "alert"]
                ):
                    weight = 0.5
                else:
                    weight = 0.3

                self.danger_weights[obj] = weight

        except FileNotFoundError:
            # Fallback danger weights
            self.danger_weights = {
                "car": 0.8,
                "truck": 0.9,
                "knife": 0.9,
                "fire": 1.0,
                "person": 0.2,
                "chair": 0.1,
                "dog": 0.6,
                "stairs": 0.5,
            }

    def _setup_scene_multipliers(self):
        """Scene-specific risk multipliers"""
        self.scene_multipliers = {
            "street": 1.3,
            "highway": 1.5,
            "parking_lot": 1.2,
            "kitchen": 1.1,
            "construction_site": 1.4,
            "stairs": 1.3,
            "escalator": 1.2,
            "bedroom": 0.7,
            "office": 0.8,
            "park": 0.9,
        }

    def process_frame(self, frame: np.ndarray) -> NavigationSignal:
        """Main signal generation pipeline"""
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()

        # Detection phase
        detections = self._detect_objects(frame)
        scene = self._classify_scene(frame)

        # Analysis phase
        spatial_data = self._analyze_spatial(detections, frame.shape)
        risk_assessment = self._assess_risk(spatial_data, scene)
        movement = self._calculate_movement(spatial_data, risk_assessment)

        # Primary threat identification
        primary_threat = self._identify_primary_threat(spatial_data, risk_assessment)

        # Overall confidence
        confidence = self._calculate_confidence(detections, scene)

        return NavigationSignal(
            risk_level=risk_assessment["total_risk"],
            primary_threat=primary_threat,
            movement=movement,
            scene_context=scene,
            timestamp=timestamp,
            confidence=confidence,
        )

    def _detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """YOLO + face detection"""
        detections = []

        # YOLO detection
        results = self.yolo(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                object_class = self.yolo.names[cls_id]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)

                detections.append(
                    Detection(
                        object_class=object_class,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=center,
                        area=area,
                    )
                )

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            detections.append(
                Detection(
                    object_class="face",
                    bbox=(x, y, x + w, y + h),
                    confidence=0.8,  # Haar cascade doesn't provide confidence
                    center=(x + w // 2, y + h // 2),
                    area=w * h,
                )
            )

        return detections

    def _classify_scene(self, frame: np.ndarray) -> str:
        """Places365 scene classification"""
        try:
            # Convert to PIL and apply transforms
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            input_tensor = self.scene_transform(pil_image).unsqueeze(0)

            with torch.no_grad():
                output = self.scene_model(input_tensor)
                _, predicted = torch.max(output, 1)
                scene_idx = predicted.item()

                if scene_idx < len(self.scene_classes):
                    return self.scene_classes[scene_idx]
                else:
                    return "unknown"

        except Exception:
            return "unknown"

    def _analyze_spatial(
        self, detections: List[Detection], frame_shape: Tuple[int, int, int]
    ) -> Dict:
        """Spatial analysis of detections"""
        h, w = frame_shape[:2]

        spatial_data = {
            "frame_quadrants": {"left": [], "center": [], "right": []},
            "proximity_scores": {},
            "blocked_regions": set(),
            "clear_regions": set(),
        }

        for detection in detections:
            # Quadrant mapping
            center_x = detection.center[0]
            if center_x < w * 0.33:
                quadrant = "left"
            elif center_x > w * 0.67:
                quadrant = "right"
            else:
                quadrant = "center"

            spatial_data["frame_quadrants"][quadrant].append(detection)

            # Proximity calculation (based on bounding box size)
            object_size = detection.area / (w * h)  # Normalized size
            proximity = min(1.0, object_size * 10)  # Scale factor

            spatial_data["proximity_scores"][detection] = proximity

            # Region blocking (high proximity objects block their quadrant)
            if proximity > 0.3:
                spatial_data["blocked_regions"].add(quadrant)

        # Identify clear regions
        all_regions = {"left", "center", "right"}
        spatial_data["clear_regions"] = all_regions - spatial_data["blocked_regions"]

        return spatial_data

    def _assess_risk(self, spatial_data: Dict, scene: str) -> Dict:
        """Calculate risk scores"""
        total_risk = 0.0
        object_risks = {}

        scene_multiplier = self.scene_multipliers.get(scene, 1.0)

        for detection, proximity in spatial_data["proximity_scores"].items():
            base_weight = self.danger_weights.get(detection.object_class, 0.1)

            # Risk = base_weight × proximity × scene_multiplier × confidence
            risk_score = (
                base_weight * proximity * scene_multiplier * detection.confidence
            )
            object_risks[detection] = risk_score
            total_risk += risk_score

        # Normalize total risk to 0-1
        total_risk = min(1.0, total_risk)

        return {
            "total_risk": total_risk,
            "object_risks": object_risks,
            "scene_multiplier": scene_multiplier,
        }

    def _calculate_movement(
        self, spatial_data: Dict, risk_assessment: Dict
    ) -> MovementInfo:
        """Calculate recommended movement"""
        clear_regions = list(spatial_data["clear_regions"])
        total_risk = risk_assessment["total_risk"]

        # Determine action
        if total_risk > 0.8:
            action = "stop"
            urgency = 1.0
        elif total_risk > 0.5:
            if "left" in clear_regions:
                action = "move_left"
            elif "right" in clear_regions:
                action = "move_right"
            else:
                action = "stop"
            urgency = 0.8
        elif total_risk > 0.2:
            if len(clear_regions) > 0:
                # Prefer center, then left, then right
                if "center" in clear_regions:
                    action = "proceed"
                elif "left" in clear_regions:
                    action = "move_left"
                else:
                    action = "move_right"
            else:
                action = "proceed"
            urgency = 0.4
        else:
            action = "proceed"
            urgency = 0.1

        return MovementInfo(action=action, urgency=urgency, safe_regions=clear_regions)

    def _identify_primary_threat(
        self, spatial_data: Dict, risk_assessment: Dict
    ) -> Optional[ThreatInfo]:
        """Find the highest risk object"""
        if not risk_assessment["object_risks"]:
            return None

        # Find highest risk detection
        primary_detection = max(
            risk_assessment["object_risks"].keys(),
            key=lambda d: risk_assessment["object_risks"][d],
        )

        # Determine location
        for quadrant, detections in spatial_data["frame_quadrants"].items():
            if primary_detection in detections:
                location = quadrant
                break
        else:
            location = "unknown"

        proximity = spatial_data["proximity_scores"][primary_detection]

        return ThreatInfo(
            object=primary_detection.object_class,
            location=location,
            proximity=proximity,
            confidence=primary_detection.confidence,
        )

    def _calculate_confidence(self, detections: List[Detection], scene: str) -> float:
        """Overall assessment confidence"""
        if not detections:
            return 0.5  # Neutral confidence with no detections

        # Average detection confidence
        avg_detection_conf = sum(d.confidence for d in detections) / len(detections)

        # Scene confidence (known scenes are more confident)
        scene_conf = 0.8 if scene != "unknown" else 0.5

        # Combined confidence
        return (avg_detection_conf + scene_conf) / 2


# Example usage
if __name__ == "__main__":
    generator = SignalGenerator()

    # Test with single image
    image_path = "./test/sample1.jpeg"

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load {image_path}")
        print("Please add sample1.jpg to ./test folder")
        exit()

    # Process image
    signal = generator.process_frame(frame)

    print("\n=== VISION ANALYSIS ===")
    print(f"Risk Level: {signal.risk_level:.2f}")
    print(f"Scene: {signal.scene_context}")
    print(f"Action: {signal.movement.action}")
    print(f"Urgency: {signal.movement.urgency:.2f}")
    print(f"Safe Regions: {signal.movement.safe_regions}")
    print(f"Confidence: {signal.confidence:.2f}")

    if signal.primary_threat:
        print("\n=== PRIMARY THREAT ===")
        print(f"Object: {signal.primary_threat.object}")
        print(f"Location: {signal.primary_threat.location}")
        print(f"Proximity: {signal.primary_threat.proximity:.2f}")
        print(f"Threat Confidence: {signal.primary_threat.confidence:.2f}")
    else:
        print("\n=== NO THREATS DETECTED ===")
