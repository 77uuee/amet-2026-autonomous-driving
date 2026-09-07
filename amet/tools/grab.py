"""시뮬레이터에서 카메라 프레임을 뭉텅이로 녹화한다 (오프라인 튜닝용 데이터 수집).

주행 노드를 띄울 필요 없다. HTTP API 로 카메라만 받아온다.
CONTROL 패널의 방향키로 직접 몰면서 돌리면, 코스 전 구간의 대표 프레임이 모인다.
특히 **분기점, 급코너, 신호등 앞** 을 꼭 지나면서 녹화할 것.

  python3 amet/tools/grab.py --n 400 --hz 10 --out capture

수집이 끝나면 capture/ 를 zip 으로 받아서 네 노트북에서 replay.py 로 튜닝한다.
"""
import argparse
import os
import time

import cv2
import numpy as np
import requests

BASE_URL = os.environ.get("PHYSICAR_URL", "http://localhost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="받을 프레임 수")
    ap.add_argument("--hz", type=float, default=10.0, help="초당 몇 장")
    ap.add_argument("--out", default="capture", help="저장 폴더")
    ap.add_argument("--width", type=int, default=320, help="저장 폭 (주행 코드와 동일해야 함)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    period = 1.0 / max(args.hz, 0.5)
    saved, fails = 0, 0
    print(f"녹화 시작: {args.n} 프레임 @ {args.hz}Hz -> {args.out}/")
    print("지금 CONTROL 패널 방향키로 코스를 몰아라. 분기점/코너/신호등을 꼭 지날 것.")

    t_end = time.time()
    for i in range(args.n):
        t0 = time.time()
        try:
            jpg = requests.get(f"{BASE_URL}/camera", timeout=2).content
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            fails += 1
            if fails > 20:
                print(f"카메라를 못 읽는다: {exc}")
                break
            continue
        if img is None:
            fails += 1
            continue
        if args.width and img.shape[1] != args.width:
            s = args.width / float(img.shape[1])
            img = cv2.resize(img, (args.width, max(1, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.out, f"f{i:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved += 1
        if saved % 50 == 0:
            print(f"  {saved}/{args.n}")
        t_end = t0 + period
        time.sleep(max(0.0, t_end - time.time()))

    print(f"완료: {saved}장 저장 (실패 {fails})")
    print(f"이제 이렇게 받아가라:  cd ~/physicar_ws && zip -qr {args.out}.zip {args.out}")


if __name__ == "__main__":
    main()
