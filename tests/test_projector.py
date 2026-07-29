import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of how/where this script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from src.slam.visual_odometry import VisualOdometry

vo = VisualOdometry(frames_dir="data/processed/frames")
vo.run()
poses = vo.read_poses()

sorted_poses = sorted(poses.values(), key=lambda p: p.frame_name)
xs = [p.translation[0] for p in sorted_poses]
zs = [p.translation[2] for p in sorted_poses]

plt.figure(figsize=(8, 8))
plt.plot(xs, zs, marker='o', markersize=2)
plt.xlabel("X"); plt.ylabel("Z"); plt.title("VO trajectory (top-down)")
plt.axis("equal")
plt.savefig("outputs/vo_debug.png")
print(f"Registered {len(poses)} poses")
print("Saved outputs/vo_debug.png")