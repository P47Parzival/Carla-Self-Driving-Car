#!/usr/bin/env python3
"""Continuous MJPEG live-view server for a radar-style LIDAR bird's-eye view.
Connects as a secondary CARLA client, spawns its own LIDAR sensor (same
config as carla_env.py's forward 180-degree sensor) on whichever vehicle is
currently being trained, and renders a labeled radar plot (range rings,
FORWARD/LEFT/RIGHT labels, live min-distance readouts) as MJPEG over HTTP.
"""
import carla
import time
import threading
import http.server
import socketserver
import math
import numpy as np
import cv2

HOST, PORT = "0.0.0.0", 8083
MAX_RANGE_M = 25.0
CANVAS = 800
ORIGIN = (CANVAS // 2, CANVAS - 90)
SCALE = (CANVAS - 150) / MAX_RANGE_M  # px per meter

latest_jpeg = [None]
lock = threading.Lock()


def world_to_px(x, y):
    """x = forward (m), y = left (m) -> pixel coords."""
    px = ORIGIN[0] - y * SCALE
    py = ORIGIN[1] - x * SCALE
    return int(px), int(py)


def draw_label(img, text, center, bg=(50, 50, 50), fg=(255, 255, 255), scale=0.6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = center
    cv2.rectangle(img, (x - tw // 2 - 8, y - th // 2 - 6), (x + tw // 2 + 8, y + th // 2 + 6), bg, -1)
    cv2.putText(img, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)


def process_points(lidar_measurement):
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    if points.shape[0] == 0:
        return np.array([]), np.array([])

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    horizontal_dist = np.sqrt(x ** 2 + y ** 2)

    ground_threshold = -1.65
    ground_ray_z = -0.268 * horizontal_dist
    ground_margin = 0.3
    not_ground = (z > ground_threshold) & (z > ground_ray_z + ground_margin)
    not_self = horizontal_dist > 0.5
    upper_threshold = 1.2
    mask = not_self & not_ground & (z < upper_threshold) & (x > 0)
    return x[mask], y[mask]


def sector_min(obstacle_x, obstacle_y, lo_deg, hi_deg):
    if len(obstacle_x) == 0:
        return float("inf")
    angles = np.degrees(np.arctan2(obstacle_y, obstacle_x))
    dists = np.sqrt(obstacle_x ** 2 + obstacle_y ** 2)
    sector = (angles >= lo_deg) & (angles < hi_deg)
    return float(np.min(dists[sector])) if np.any(sector) else float("inf")


def fmt(d):
    return f"{d:.2f}m" if math.isfinite(d) else "inf m"


def render_frame(obstacle_x, obstacle_y):
    img = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)

    # Range rings + grid
    for r in range(5, int(MAX_RANGE_M) + 1, 5):
        cv2.circle(img, ORIGIN, int(r * SCALE), (210, 210, 210), 1, cv2.LINE_AA)
        if r % 10 == 0:
            cv2.putText(img, f"{r}m", (ORIGIN[0] + int(r * SCALE) + 4, ORIGIN[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(img, (ORIGIN[0], 0), (ORIGIN[0], CANVAS), (225, 225, 225), 1, cv2.LINE_AA)
    cv2.line(img, (0, ORIGIN[1]), (CANVAS, ORIGIN[1]), (225, 225, 225), 1, cv2.LINE_AA)

    # Direction labels
    draw_label(img, "FORWARD", (ORIGIN[0], 30))
    draw_label(img, "LEFT", (35, ORIGIN[1] - 35))
    draw_label(img, "RIGHT", (CANVAS - 50, ORIGIN[1] - 35))

    # Obstacle points
    for ox, oy in zip(obstacle_x, obstacle_y):
        px, py = world_to_px(ox, oy)
        if 0 <= px < CANVAS and 0 <= py < CANVAS:
            cv2.circle(img, (px, py), 2, (20, 20, 20), -1)

    # Sector min distances + clearance indicators
    front_min = sector_min(obstacle_x, obstacle_y, -30, 30)
    left_min = sector_min(obstacle_x, obstacle_y, 30, 150)
    right_min = sector_min(obstacle_x, obstacle_y, -150, -30)
    overall_min = min(front_min, left_min, right_min)

    if math.isfinite(front_min):
        end_px, end_py = world_to_px(min(front_min, MAX_RANGE_M), 0)
        cv2.line(img, ORIGIN, (end_px, end_py), (0, 0, 220), 2)
        draw_label(img, fmt(front_min), (end_px, max(end_py - 10, 40)), bg=(0, 215, 255), fg=(0, 0, 0), scale=0.5)
    if math.isfinite(right_min):
        end_px, end_py = world_to_px(0, -min(right_min, MAX_RANGE_M))
        cv2.line(img, ORIGIN, (end_px, end_py), (0, 0, 220), 2)
        draw_label(img, fmt(right_min), (min(end_px + 30, CANVAS - 60), end_py), bg=(0, 215, 255), fg=(0, 0, 0), scale=0.5)

    # Vehicle marker
    cv2.circle(img, ORIGIN, 8, (0, 0, 220), 2)
    cv2.drawMarker(img, ORIGIN, (0, 0, 220), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    # Title + stats box
    cv2.putText(img, "LIDAR Bird's Eye View", (CANVAS // 2 - 140, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    stats = [
        f"Points: {len(obstacle_x)}",
        f"Min Distance: {fmt(overall_min)}",
        f"Front: {fmt(front_min)}",
        f"Left: {fmt(left_min)}",
        f"Right: {fmt(right_min)}",
        "Rear: inf m (no coverage)",
    ]
    box_h = 22 * len(stats) + 16
    cv2.rectangle(img, (10, 10), (240, 10 + box_h), (255, 255, 255), -1)
    cv2.rectangle(img, (10, 10), (240, 10 + box_h), (120, 120, 120), 1)
    for i, line in enumerate(stats):
        cv2.putText(img, line, (18, 32 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def on_lidar(lidar_measurement):
    obstacle_x, obstacle_y = process_points(lidar_measurement)
    img = render_frame(obstacle_x, obstacle_y)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with lock:
            latest_jpeg[0] = buf.tobytes()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    frame = latest_jpeg[0]
                if frame is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def lidar_loop():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "25")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "60000")
    lidar_bp.set_attribute("upper_fov", "10.0")
    lidar_bp.set_attribute("lower_fov", "-15.0")
    lidar_bp.set_attribute("horizontal_fov", "180.0")
    lidar_transform = carla.Transform(carla.Location(x=0.5, z=1.8), carla.Rotation(pitch=0))

    lidar = None
    current_target_id = None
    while True:
        try:
            vehicles = list(world.get_actors().filter("vehicle.*"))
            if vehicles:
                target = max(vehicles, key=lambda v: v.id)
                if target.id != current_target_id:
                    if lidar is not None:
                        lidar.stop()
                        lidar.destroy()
                    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=target)
                    lidar.listen(on_lidar)
                    current_target_id = target.id
                    print(f"Now following {target.type_id} (id={target.id})")
        except Exception as e:
            print(f"lidar_loop error: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    threading.Thread(target=lidar_loop, daemon=True).start()
    print(f"Serving LIDAR radar MJPEG stream on http://{HOST}:{PORT}/")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((HOST, PORT), Handler)
    server.serve_forever()
