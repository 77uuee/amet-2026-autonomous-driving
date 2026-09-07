"""오프라인 리플레이 — 시뮬레이터 없이 인지 파이프라인을 튜닝한다.

녹화된 프레임(grab.py 또는 config 의 log_dir 산출물)에 대해 실제 주행과
**똑같은 코드**를 돌리고, 어노테이션 영상과 통계를 뽑는다.
노트북에서 돌아가므로 시뮬 크레딧도, 브라우저 터미널 지연도 없다.

기본 실행:
    python3 amet/tools/replay.py --frames capture

    -> replay_out.mp4  (오버레이 영상)
    -> replay_out.csv  (프레임별 수치)
    -> 콘솔에 요약 통계

파라미터 스윕 (어떤 값이 제일 좋은지 자동 비교):
    python3 amet/tools/replay.py --frames capture \
        --sweep centerline.window=0.10,0.13,0.16,0.20

좋은 설정의 기준:
  * center%   높을수록 좋다 (주황 중앙선을 물고 있는 비율)
  * lost%     낮을수록 좋다 (도로 자체를 놓친 비율)
  * jitter    낮을수록 좋다 (프레임 간 조향 오차 변화량 = 떨림)
"""
import argparse
import copy
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amet_drive.config import Config          # noqa: E402
from amet_drive.control import Controller     # noqa: E402
from amet_drive.lane import LaneDetector      # noqa: E402
from amet_drive.lights import TrafficLightDetector  # noqa: E402
from amet_drive.obstacle import ObstacleResult      # noqa: E402
from amet_drive import viewer as vm           # noqa: E402


def load_frames(path, width):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jpg")) +
                       glob.glob(os.path.join(path, "*.png")))
        if not files:
            raise SystemExit(f"프레임이 없다: {path}")
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            yield resize(img, width)
    else:
        cap = cv2.VideoCapture(path)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield resize(img, width)
        cap.release()


def resize(img, width):
    if width and img.shape[1] != width:
        s = width / float(img.shape[1])
        img = cv2.resize(img, (width, max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def set_path(data, dotted, value):
    """'centerline.window' 같은 경로에 값을 넣는다."""
    sec, key = dotted.split(".", 1)
    try:
        value = int(value) if value.strip().lstrip("-").isdigit() else float(value)
    except ValueError:
        pass
    data.setdefault(sec, {})[key] = value


def run(cfg, frames, out_video=None, out_csv=None, verbose=True):
    lane_det = LaneDetector(cfg)
    light_det = TrafficLightDetector(cfg)
    ctrl = Controller(cfg)
    dt = 1.0 / float(cfg.get("general", "control_hz", 20.0))

    writer, rows = None, []
    n = n_center = n_lost = 0
    prev_err = None
    jitter = []
    moved = False
    first_go = None

    for img in frames:
        n += 1
        lane = lane_det.process(img)
        light = light_det.process(img, lane.mask, lane.roi_y0)
        obs = ObstacleResult()

        if lane.ok and lane.source.startswith("center"):
            n_center += 1
        if not lane.ok:
            n_lost += 1

        if lane.ok:
            if prev_err is not None:
                jitter.append(abs(lane.error - prev_err))
            prev_err = lane.error

        if light.stop:
            state, steer = ("WAIT_LIGHT" if light.phase == "WAIT" else "STOP_LIGHT",
                            ctrl.steer_deg)
            speed = ctrl.ramp_speed(0.0, dt)
        elif not lane.ok:
            state, steer = "LANE_LOST", ctrl.steer_deg
            speed = ctrl.ramp_speed(float(cfg.get("drive", "v_min", 0.26)) * 0.7, dt)
        else:
            state = "DRIVE"
            steer = ctrl.steering(lane.error, dt, lane.curvature)
            speed = ctrl.ramp_speed(ctrl.target_speed(lane.curvature, lane.error), dt)
            if first_go is None:
                first_go = n
        if speed > 0.02:
            moved = True
            light_det.note_moved()

        rows.append((n, state, lane.source, round(lane.error, 4),
                     round(lane.curvature, 4), round(speed, 3), round(steer, 2),
                     light.phase, light.stable, int(light.area)))

        if out_video:
            frame = vm.render(img, lane, light, obs, state, speed, steer, 1.0 / dt,
                              float(cfg.get("lane", "roi_top", 0.5)),
                              float(cfg.get("lane", "roi_bottom", 0.98)))
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                         1.0 / dt, (w, h))
            writer.write(frame)

    if writer:
        writer.release()
    if out_csv:
        with open(out_csv, "w") as f:
            f.write("frame,state,src,err,curv,speed,steer,light_phase,light,light_area\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    stats = {
        "frames": n,
        "center%": 100.0 * n_center / max(n, 1),
        "lost%": 100.0 * n_lost / max(n, 1),
        "jitter": float(np.mean(jitter)) if jitter else float("nan"),
        "first_go": first_go,
        "moved": moved,
    }
    if verbose:
        print(f"  프레임 {stats['frames']}  center {stats['center%']:.1f}%  "
              f"lost {stats['lost%']:.1f}%  jitter {stats['jitter']:.4f}  "
              f"첫 출발 프레임 {stats['first_go']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="프레임 폴더 또는 동영상 파일")
    ap.add_argument("--config", default=None, help="config.yaml 경로")
    ap.add_argument("--out", default="replay_out", help="출력 파일 접두사")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--sweep", default=None,
                    help="예: centerline.window=0.10,0.13,0.16")
    args = ap.parse_args()

    cfg = Config(args.config) if args.config else Config()
    base = copy.deepcopy(cfg.data)

    if args.sweep:
        dotted, values = args.sweep.split("=", 1)
        print(f"스윕: {dotted}")
        results = []
        for v in values.split(","):
            cfg.data = copy.deepcopy(base)
            set_path(cfg.data, dotted.strip(), v)
            print(f"\n{dotted} = {v}")
            st = run(cfg, load_frames(args.frames, args.width), None, None)
            results.append((v, st))
        print("\n=== 요약 (center% 높고 lost%/jitter 낮은 게 좋다) ===")
        print(f"{'value':>10} {'center%':>9} {'lost%':>8} {'jitter':>9}")
        for v, st in results:
            print(f"{v:>10} {st['center%']:>9.1f} {st['lost%']:>8.1f} {st['jitter']:>9.4f}")
        best = max(results, key=lambda r: r[1]["center%"] - r[1]["lost%"])
        print(f"\n추천: {dotted} = {best[0]}")
        return

    video = None if args.no_video else args.out + ".mp4"
    print(f"리플레이: {args.frames}")
    run(cfg, load_frames(args.frames, args.width), video, args.out + ".csv")
    if video:
        print(f"  -> {video}")
    print(f"  -> {args.out}.csv")


if __name__ == "__main__":
    main()
