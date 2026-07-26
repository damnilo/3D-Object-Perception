# Road Object 3D Mapping

Builds a 3D map of road objects (vehicles, pedestrians, traffic signs, etc.)
from monocular dashcam footage by combining:

- **YOLOv8** (pretrained on COCO) for 2D object detection
- **Depth Anything V2** for monocular depth estimation
- **Custom ORB-based visual odometry** for camera pose estimation (monocular VO with essential-matrix decomposition and frame-to-frame scale propagation)
- Custom fusion logic to back-project detections into 3D and de-duplicate
  repeated sightings of the same physical object

## Setup

1. Install Python dependencies:

```bash
   pip install -r requirements.txt
```

2. Place a dashcam video clip at `data/raw/drive_clip.mp4`
   (or update the path in `configs/pipeline.yaml`).

## Usage

**Step 1 — Extract frames from your video:**

```bash
python scripts/extract_frames.py --config configs/pipeline.yaml
```

**Step 2 — Run the full pipeline** (VO poses -> detection -> depth -> 3D fusion -> visualization):

```bash
python -m src.main --config configs/pipeline.yaml
```

This will:

- Estimate camera poses for each frame via ORB-based visual odometry
- Run YOLO + depth estimation on every frame
- Project 2D detections into 3D world coordinates
- Cluster repeated sightings into distinct map objects
- Save the result to `outputs/map.ply`
- Render a scene snapshot to `outputs/scene.png`

## Configuration

All tunable parameters live in `configs/pipeline.yaml`:

- Frame extraction rate and resolution
- Camera intrinsics (or let visual odometry estimate them)
- Detection model + confidence threshold + target classes
- Depth model
- Clustering sensitivity (how close sightings must be to merge into one object)

## Project layout

road-object-3d-map/
├── configs/ # pipeline.yaml - all tunable parameters
├── data/
│ ├── raw/ # put your dashcam video(s) here
│ └── processed/ # extracted frames (generated)
├── src/
│ ├── detection/ # YOLO wrapper
│ ├── depth/ # depth model wrapper
│ ├── slam/ # Visual Odometry implementation
│ ├── mapping/ # 2D->3D projection + clustering
│ ├── visualization/ # Open3D rendering
│ └── main.py # end-to-end orchestration
├── scripts/
│ └── extract_frames.py # video -> frame images
├── tests/ # pytest unit tests
└── outputs/ # generated .ply maps + scene renders (generated)

## Known limitations / things to revisit

- Depth Anything gives **relative**, not metric, depth. Object distances in
  the map are only as accurate as the visual odometry's scale recovery — if
  you need true metric accuracy, consider adding a known reference distance
  or switching to a stereo/LiDAR setup later.
- The ORB-based visual odometry can drift or fail to register frames with
  heavy motion blur, sudden turns, or low texture (e.g. featureless highway
  barriers). It also relies on frame-to-frame scale propagation rather than
  full bundle adjustment, so long clips are more prone to accumulated drift
  than a global SfM approach. Start with short, steady clips while
  validating the pipeline.
- `_parse_intrinsics` in `src/main.py` currently only handles the
  `PINHOLE`/`SIMPLE_PINHOLE`/`SIMPLE_RADIAL` camera models — extend it if
  your intrinsics setup uses a different model.
- No fine-tuning yet: detection is 100% pretrained COCO classes. Revisit if
  you need road-specific classes (cones, potholes, custom signage).
