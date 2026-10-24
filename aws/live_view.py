#!/usr/bin/env python3
"""Continuous MJPEG live-view server. Connects as a secondary CARLA client
(doesn't interfere with training's synchronous-mode stepping), keeps a chase
camera attached to whichever vehicle is currently being trained (re-attaches
whenever a new episode's vehicle appears), and serves the feed as MJPEG over
HTTP so it can be viewed in a plain browser tab via an SSH tunnel.
"""
import carla
import time
import threading
import http.server
import socketserver
import numpy as np
import cv2

HOST, PORT = "0.0.0.0", 8082
latest_jpeg = [None]
lock = threading.Lock()


def on_image(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
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


def camera_loop():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "960")
    camera_bp.set_attribute("image_size_y", "540")
    camera_bp.set_attribute("fov", "100")
    camera_bp.set_attribute("sensor_tick", "0.1")
    chase_transform = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12))

    camera = None
    current_target_id = None
    while True:
        try:
            vehicles = list(world.get_actors().filter("vehicle.*"))
            if vehicles:
                target = max(vehicles, key=lambda v: v.id)
                if target.id != current_target_id:
                    if camera is not None:
                        camera.stop()
                        camera.destroy()
                    camera = world.spawn_actor(camera_bp, chase_transform, attach_to=target)
                    camera.listen(on_image)
                    current_target_id = target.id
                    print(f"Now following {target.type_id} (id={target.id})")
        except Exception as e:
            print(f"camera_loop error: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print(f"Serving live MJPEG stream on http://{HOST}:{PORT}/")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer((HOST, PORT), Handler)
    server.serve_forever()
