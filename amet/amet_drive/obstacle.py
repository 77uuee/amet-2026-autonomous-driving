"""전방 장애물(앞차) 감지 - LiDAR.

카메라 기반 차량 인식보다 훨씬 싸고 안정적이다. 충돌 페널티는 5초.
전방 좁은 부채꼴의 하위 퍼센타일 거리를 대표값으로 쓴다 (튀는 값 방어).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ObstacleResult:
    distance: float = float("inf")
    stop: bool = False
    scale: float = 1.0        # 속도 배율 0~1


class ObstacleDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._clear_n = 0
        self._stopped = False

    def process(self, scan) -> ObstacleResult:
        c = self.cfg
        if scan is None or not c.get("obstacle", "enabled", True):
            return ObstacleResult()

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        n = ranges.size
        if n == 0:
            return ObstacleResult()

        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        angles = np.arctan2(np.sin(angles), np.cos(angles))  # -pi..pi 정규화

        off = np.radians(float(c.get("obstacle", "angle_offset_deg", 0.0)))
        half = np.radians(float(c.get("obstacle", "fov_deg", 26.0))) * 0.5
        rel = np.arctan2(np.sin(angles - off), np.cos(angles - off))

        min_valid = float(c.get("obstacle", "min_valid", 0.05))
        sel = (np.abs(rel) <= half) & np.isfinite(ranges) & (ranges > min_valid)
        if scan.range_max > 0:
            sel &= ranges < scan.range_max * 0.99

        if not sel.any():
            self._clear_n += 1
            if self._clear_n >= int(c.get("obstacle", "clear_frames", 4)):
                self._stopped = False
            return ObstacleResult()

        pct = float(c.get("obstacle", "percentile", 20))
        dist = float(np.percentile(ranges[sel], pct))

        stop_d = float(c.get("obstacle", "stop_dist", 0.32))
        slow_d = float(c.get("obstacle", "slow_dist", 0.85))

        if dist <= stop_d:
            self._stopped = True
            self._clear_n = 0
        elif dist > stop_d * 1.25:
            self._clear_n += 1
            if self._clear_n >= int(c.get("obstacle", "clear_frames", 4)):
                self._stopped = False
        else:
            self._clear_n = 0

        if self._stopped:
            return ObstacleResult(distance=dist, stop=True, scale=0.0)

        if dist >= slow_d:
            scale = 1.0
        else:
            span = max(1e-3, slow_d - stop_d)
            scale = float(np.clip((dist - stop_d) / span, 0.0, 1.0))
        return ObstacleResult(distance=dist, stop=False, scale=scale)
