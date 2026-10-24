#!/usr/bin/env python3
"""Continuous MJPEG live-view server for the LIDAR bird's-eye-view occupancy
grid the agent actually trains on. Connects as a secondary CARLA client,
spawns its own LIDAR sensor (same config as carla_env.py) on whichever
vehicle is currently being trained, builds the same 64x64 front-facing BEV
grid, upscales it for visibility, and serves it as MJPEG over HTTP.
"""
import carla
import time
import threading
import http.server
import socketserver
import numpy as np
import cv2

HOST, PORT = "0.0.0.0", 8083
BEV_GRID_SIZE = 64
BEV_FORWARD_RANGE = 25.0
BEV_LATERAL_RANGE = 25.0
DISPLAY_SIZE = 512  # upscaled output resolution

latest_jpeg = [None]
lock = threading.Lock()


def build_bev_image(lidar_measurement):
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    bev_grid = np.zeros((BEV_GRID_SIZE, BEV_GRID_SIZE), dtype=np.uint8)
    if points.shape[0] == 0:
        return bev_grid

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    horizontal_dist = np.sqrt(x ** 2 + y ** 2)

    ground_threshold = -1.65
    ground_ray_z = -0.268 * horizontal_dist
    ground_margin = 0.3
    not_ground = (z > ground_threshold) & (z > ground_ray_z + ground_margin)
    not_self = horizontal_dist > 0.5
    upper_threshold = 1.2
    bev_mask = not_self & not_ground & (z < upper_threshold) & (x > 0)

    bev_x, bev_y = x[bev_mask], y[bev_mask]
    if len(bev_x) > 0:
        grid_row = ((BEV_FORWARD_RANGE - bev_x) / BEV_FORWARD_RANGE * (BEV_GRID_SIZE - 1)).astype(np.int32)
        grid_col = ((bev_y + BEV_LATERAL_RANGE) / (2 * BEV_LATERAL_RANGE) * (BEV_GRID_SIZE - 1)).astype(np.int32)
        grid_row = np.clip(grid_row, 0, BEV_GRID_SIZE - 1)
        grid_col = np.clip(grid_col, 0, BEV_GRID_SIZE - 1)
        bev_grid[grid_row, grid_col] = 255
    return bev_grid


def on_lidar(lidar_measurement):
    bev_grid = build_bev_image(lidar_measurement)
    img = cv2.resize(bev_grid, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    # Mark ego vehicle position (bottom center, row 63 -> bottom of image)
    cv2.drawMarker(img_color, (DISPLAY_SIZE // 2, DISPLAY_SIZE - 8), (0, 0, 255),
                    markerType=cv2.MARKER_TRIANGLE_UP, markerSize=16, thickness=2)
    ok, buf = cv2.imencode(".jpg", img_color, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
    print(f"Serving LIDAR BEV MJPEG stream on http://{HOST}:{PORT}/")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((HOST, PORT), Handler)
    server.serve_forever()
