"""HSV / ROI 캘리브레이션 툴.

주행 노드를 끄고 이걸 돌리면 포트 5000에 튜닝 화면이 뜬다.
슬라이더로 아스팔트 마스크와 신호등 색 범위를 눈으로 맞춘 뒤,
표시되는 YAML 조각을 config.yaml 에 그대로 붙여넣으면 된다.

실행:  python3 tools/tune.py
"""
import io
import json
import math
import os
import sys

import cv2
import numpy as np
import requests

BASE_URL = "http://localhost"
CFG = os.environ.get("AMET_CONFIG",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "config.yaml"))

P = {
    "roi_top": 50, "roi_bottom": 98,
    "road_s_max": 90, "road_v_min": 25, "road_v_max": 255,
    "grass_h_lo": 30, "grass_h_hi": 95, "grass_s_min": 55,
    "tl_top": 2, "tl_bottom": 55,
    "red_h1": 8, "red_h2": 170, "red_s": 120, "red_v": 110,
    "grn_lo": 45, "grn_hi": 90, "grn_s": 110, "grn_v": 140,
    "mode": 0,   # 0 = road mask, 1 = traffic light
}

PAGE = """<!doctype html><meta charset=utf-8><title>AMET tuner</title>
<style>body{margin:0;background:#111;color:#ddd;font:13px ui-monospace,monospace;padding:10px}
.wrap{display:flex;gap:14px;flex-wrap:wrap}img{border:1px solid #333;max-width:640px}
label{display:flex;align-items:center;gap:8px;margin:3px 0}input[type=range]{width:180px}
b{color:#8cf}pre{background:#000;padding:10px;border:1px solid #333;white-space:pre-wrap}</style>
<div class=wrap>
<div><img id=v src="/stream"></div>
<div id=ctl></div>
</div>
<h4>config.yaml 에 붙여넣기</h4><pre id=out></pre>
<script>
const P=%s;
const ctl=document.getElementById('ctl');
const RANGES={roi_top:[0,100],roi_bottom:[0,100],road_s_max:[0,255],road_v_min:[0,255],
road_v_max:[0,255],grass_h_lo:[0,179],grass_h_hi:[0,179],grass_s_min:[0,255],
tl_top:[0,100],tl_bottom:[0,100],red_h1:[0,60],red_h2:[120,179],red_s:[0,255],red_v:[0,255],
grn_lo:[0,179],grn_hi:[0,179],grn_s:[0,255],grn_v:[0,255],mode:[0,1]};
for(const k in P){const r=RANGES[k]||[0,255];
 const l=document.createElement('label');
 l.innerHTML=`<b style="width:110px;display:inline-block">${k}</b>
 <input type=range min=${r[0]} max=${r[1]} value=${P[k]} id=s_${k}>
 <span id=t_${k}>${P[k]}</span>`;
 ctl.appendChild(l);
 l.querySelector('input').oninput=e=>{
   document.getElementById('t_'+k).textContent=e.target.value;
   fetch('/set?k='+k+'&v='+e.target.value).then(r=>r.text()).then(t=>{
     document.getElementById('out').textContent=t;});};}
fetch('/set?k=mode&v='+P.mode).then(r=>r.text()).then(t=>document.getElementById('out').textContent=t);
</script>"""


def camera():
    jpg = requests.get(f"{BASE_URL}/camera", timeout=2).content
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is not None and img.shape[1] != 320:
        s = 320 / img.shape[1]
        img = cv2.resize(img, (320, int(img.shape[0] * s)))
    return img


def yaml_snippet():
    return f"""lane:
  roi_top: {P['roi_top'] / 100:.2f}
  roi_bottom: {P['roi_bottom'] / 100:.2f}
  road_s_max: {P['road_s_max']}
  road_v_min: {P['road_v_min']}
  road_v_max: {P['road_v_max']}
  grass_h_lo: {P['grass_h_lo']}
  grass_h_hi: {P['grass_h_hi']}
  grass_s_min: {P['grass_s_min']}

traffic_light:
  roi_top: {P['tl_top'] / 100:.2f}
  roi_bottom: {P['tl_bottom'] / 100:.2f}
  red_hsv:  [[0, {P['red_s']}, {P['red_v']}], [{P['red_h1']}, 255, 255], \
[{P['red_h2']}, {P['red_s']}, {P['red_v']}], [179, 255, 255]]
  green_hsv: [[{P['grn_lo']}, {P['grn_s']}, {P['grn_v']}], [{P['grn_hi']}, 255, 255]]"""


def render():
    img = camera()
    if img is None:
        return np.zeros((240, 320, 3), np.uint8)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = img.copy()

    if P["mode"] == 0:
        y1, y2 = int(h * P["roi_top"] / 100), int(h * P["roi_bottom"] / 100)
        y2 = max(y1 + 2, y2)
        sub = hsv[y1:y2]
        mask = cv2.inRange(sub, (0, 0, P["road_v_min"]),
                           (179, P["road_s_max"], P["road_v_max"]))
        grass = cv2.inRange(sub, (P["grass_h_lo"], P["grass_s_min"], 30),
                            (P["grass_h_hi"], 255, 255))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(grass))
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        region = out[y1:y2]
        region[mask > 0] = (0.4 * region[mask > 0] + np.array([0, 140, 255]) * 0.6)
        cv2.rectangle(out, (0, y1), (w - 1, y2 - 1), (255, 200, 0), 1)
        cv2.line(out, (w // 2, y1), (w // 2, y2 - 1), (255, 255, 255), 1)
    else:
        y1, y2 = int(h * P["tl_top"] / 100), int(h * P["tl_bottom"] / 100)
        y2 = max(y1 + 2, y2)
        sub = hsv[y1:y2]
        red = cv2.inRange(sub, (0, P["red_s"], P["red_v"]), (P["red_h1"], 255, 255))
        red |= cv2.inRange(sub, (P["red_h2"], P["red_s"], P["red_v"]), (179, 255, 255))
        grn = cv2.inRange(sub, (P["grn_lo"], P["grn_s"], P["grn_v"]),
                          (P["grn_hi"], 255, 255))
        region = out[y1:y2]
        region[red > 0] = (0, 0, 255)
        region[grn > 0] = (0, 255, 0)
        cv2.rectangle(out, (0, y1), (w - 1, y2 - 1), (255, 200, 0), 1)
        for name, m in (("R", red), ("G", grn)):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a < 8:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.putText(out, f"{name}{int(a)}", (x, y1 + y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)


def main():
    from flask import Flask, Response, request
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(PAGE % json.dumps(P), mimetype="text/html")

    @app.route("/set")
    def setter():
        k, v = request.args.get("k"), request.args.get("v")
        if k in P:
            P[k] = int(v)
        return Response(yaml_snippet(), mimetype="text/plain")

    @app.route("/stream")
    def stream():
        import time

        def gen():
            while True:
                ok, buf = cv2.imencode(".jpg", render(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                time.sleep(0.12)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    print("tuner: MYAPP 탭 (port 5000) 을 열어라.  Ctrl-C 로 종료")
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
