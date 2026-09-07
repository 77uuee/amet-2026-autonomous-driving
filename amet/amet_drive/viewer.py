"""MYAPP 탭(포트 5000) 실시간 디버그 뷰어.

무엇을 보고 조향하는지 눈으로 봐야 튜닝이 된다.
주행 루프를 절대 막지 않도록 최신 프레임 하나만 들고 있는 구조.
대회 제출 시에는 config 의 debug_view: false 로 끄는 걸 권장 (CPU 절약).
"""
import socket
import threading

import cv2
import numpy as np

_PAGE = b"""<!doctype html><meta charset=utf-8><title>AMET debug</title>
<style>body{margin:0;background:#111;color:#eee;font:14px ui-monospace,monospace;
display:flex;flex-direction:column;align-items:center;gap:8px;padding:10px}
img{max-width:100%;image-rendering:pixelated;border:1px solid #333}</style>
<h3>AMET 2026 &mdash; live</h3><img src="/stream">
"""


class DebugViewer:
    def __init__(self, port=5000):
        self.port = port
        self._frame = None
        self._lock = threading.Lock()
        self._server = None
        self._started = False

    # ------------------------------------------------------------------
    def start(self):
        if self._started:
            return
        try:
            probe = socket.socket()
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("", self.port))
            probe.close()
        except OSError:
            print(f"[viewer] port {self.port} busy - running without debug view")
            return

        try:
            from flask import Flask, Response
            import logging

            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            app = Flask(__name__)

            @app.route("/")
            def index():
                return Response(_PAGE, mimetype="text/html")

            @app.route("/stream")
            def stream():
                def gen():
                    import time

                    while True:
                        with self._lock:
                            frame = self._frame
                        if frame is not None:
                            ok, buf = cv2.imencode(".jpg", frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ok:
                                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                                       + buf.tobytes() + b"\r\n")
                        time.sleep(0.08)

                return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

            from werkzeug.serving import make_server

            self._server = make_server("0.0.0.0", self.port, app, threaded=True)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            self._started = True
            print(f"[viewer] debug view on port {self.port} (MYAPP tab)")
        except Exception as exc:  # noqa: BLE001
            print(f"[viewer] disabled: {exc}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
        self._started = False

    # ------------------------------------------------------------------
    def publish(self, frame):
        if not self._started:
            return
        with self._lock:
            self._frame = frame


# ----------------------------------------------------------------------
def render(bgr, lane, light, obs, state, speed, steer, fps, roi_top, roi_bottom):
    """디버그 오버레이 이미지를 만든다. (원본은 건드리지 않음)"""
    img = bgr.copy()
    h, w = img.shape[:2]
    y1 = int(h * roi_top)
    y2 = int(h * roi_bottom)

    # 도로 마스크를 반투명 청록으로
    if lane is not None and lane.mask is not None:
        m = lane.mask
        sub = img[y1:y1 + m.shape[0]]
        if sub.shape[:2] == m.shape[:2]:
            tint = np.zeros_like(sub)
            tint[:, :, 0] = 255      # 청색 계열 - 잔디와 헷갈리지 않게
            tint[:, :, 1] = 170
            idx = m > 0
            sub[idx] = (0.75 * sub[idx] + 0.25 * tint[idx]).astype(np.uint8)

    cv2.rectangle(img, (0, y1), (w - 1, min(y2, h - 1)), (255, 200, 0), 1)
    cv2.line(img, (w // 2, y1), (w // 2, h - 1), (255, 255, 255), 1)

    # 주황 중앙선으로 인식된 픽셀
    if lane is not None and lane.orange is not None:
        om = lane.orange
        sub = img[y1:y1 + om.shape[0]]
        if sub.shape[:2] == om.shape[:2]:
            sub[om > 0] = (0, 140, 255)

    # 라바콘으로 인식해 도로에서 뺀 픽셀 (빨강). 주황 중앙선과 헷갈리지 않게.
    if lane is not None and getattr(lane, "cone", None) is not None:
        cm = lane.cone
        sub = img[y1:y1 + cm.shape[0]]
        if sub.shape[:2] == cm.shape[:2]:
            idx = cm > 0
            sub[idx] = (0.35 * sub[idx] + 0.65 * np.array([40, 40, 255])).astype(np.uint8)

    if lane is not None and lane.ok:
        for (ry, cx, x_lo, x_hi) in lane.rows:
            yy = y1 + ry
            cv2.line(img, (int(x_lo), yy), (int(x_hi), yy), (0, 200, 255), 1)

        # 슬라이딩 윈도우가 실제로 물고 간 점
        for (px, py) in lane.win_pts:
            cv2.circle(img, (int(px), y1 + int(py)), 3, (255, 255, 255), -1)

        # 피팅된 중앙선
        pts = [(int(px), y1 + int(py)) for (px, py) in lane.curve_pts]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (60, 255, 60), 2)

        ax, ay = int(lane.aim_x), y1 + int(lane.aim_y)
        cv2.circle(img, (ax, ay), 6, (255, 0, 255), -1)
        cv2.line(img, (w // 2, h - 2), (ax, ay), (255, 0, 255), 2)

    if light is not None and light.box is not None:
        x, y, bw, bh = light.box
        col = (0, 0, 255) if light.color == "red" else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 2)

    bar = np.zeros((52, w, 3), np.uint8)
    lines = [
        f"{state}  v={speed:.2f}m/s  steer={steer:+.1f}deg  {fps:.0f}fps",
        (f"src={lane.source} err={lane.error:+.2f} curv={lane.curvature:+.2f} "
         f"w={lane.width:.2f}" if (lane and lane.ok) else "LANE LOST"),
        (f"light={light.phase}/{light.stable}({light.area:.0f}) "
         f"obs={obs.distance:.2f}m"
         f"  cone={int((lane.cone > 0).sum()) if (lane is not None and getattr(lane, 'cone', None) is not None) else 0}"
         if (light and obs) else ""),
    ]
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (6, 15 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([img, bar])
