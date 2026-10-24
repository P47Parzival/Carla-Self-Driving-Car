#!/usr/bin/env python3
"""Attaches a temporary chase camera to the currently-training vehicle and
saves frames to disk. Runs as a second CARLA client alongside training —
does NOT call world.tick(), so it doesn't interfere with the training env's
synchronous-mode stepping.
"""
import carla
import time
import sys
import os

OUT_DIR = os.path.expanduser("~/carla-rl/spectate_frames")
DURATION_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()

vehicles = []
for _ in range(20):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if vehicles:
        break
    time.sleep(0.5)
if not vehicles:
    print("No vehicles found in the world.")
    sys.exit(1)

# Heuristic: the RL agent's vehicle is respawned every episode reset, so it
# has the highest actor id among currently-alive vehicles (NPCs persist).
target = max(vehicles, key=lambda v: v.id)
print(f"Following vehicle {target.type_id} (id={target.id})")

bp_lib = world.get_blueprint_library()
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "960")
camera_bp.set_attribute("image_size_y", "540")
camera_bp.set_attribute("fov", "100")
camera_bp.set_attribute("sensor_tick", "0.1")

chase_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12))
camera = world.spawn_actor(camera_bp, chase_transform, attach_to=target)

frame_count = [0]
def on_image(image):
    image.save_to_disk(os.path.join(OUT_DIR, f"{frame_count[0]:06d}.png"))
    frame_count[0] += 1

camera.listen(on_image)
time.sleep(DURATION_SEC)
camera.stop()
camera.destroy()

print(f"Captured {frame_count[0]} frames to {OUT_DIR}")
