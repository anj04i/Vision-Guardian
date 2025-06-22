# Vision Guardian

Clean signal generator for navigation assistance. Input frame, get navigation signal.

## Architecture

```
Frame → Detection → Risk Assessment → Navigation Signal
```

## Components

**Detection Layer**

- YOLO: objects + bounding boxes
- Places365: scene classification
- OpenCV: face detection

**Risk Assessment**

- Spatial analysis (proximity from bounding boxes)
- Danger scoring (existing JSON mappings)
- Scene context multipliers

**Signal Output**

- Risk level (0-1)
- Primary threat (object + location)
- Movement direction (left/right/stop/proceed)
- Confidence score

## Usage

```python
generator = VisionSignalGenerator()
signal = generator.process(camera_frame)
```

## Signal Structure

```json
{
  "risk_level": 0.85,
  "primary_threat": {
    "object": "car",
    "location": "center_right",
    "proximity": 0.9
  },
  "movement": {
    "action": "move_left",
    "urgency": 0.8
  },
  "scene_context": "street",
  "confidence": 0.82
}
```

## Action Modules

Signal drives any output:

- TTS module
- Haptic feedback
- API endpoints
- UI displays
