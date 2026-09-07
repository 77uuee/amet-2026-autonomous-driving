"""신호등 인지.

채점표가 비대칭이다:
    빨간불 출발 = +10초 (최대 페널티)  /  괜히 멈춤 = 시간 손해뿐
따라서 **의심스러우면 선다**. 그리고 출발 전에는 반드시 판정이 끝날 때까지 정지한다.

이전 버전이 빨간불에 출발한 원인 두 가지를 고쳤다:
  (1) 판정에 3프레임이 필요한데 그 사이에 이미 출발해버렸다
      -> 시작 상태를 WAIT 로 두고, 초록 확인 or 신호등 없음 확인 전에는 절대 안 움직인다.
  (2) 노란색 범위(H 18~33)를 빨강에 더했는데 그게 도로의 주황 점선 색이었다
      -> 기본값에서 제거. 대신 신호등 ROI 에서 **도로 픽셀을 통째로 제외**한다.
         신호등은 절대 아스팔트 위에 있지 않다. 이게 가장 강력한 오검출 필터.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

RED, GREEN, NONE = "red", "green", None


@dataclass
class LightResult:
    color: str = None        # 이번 프레임 원시 판정
    stable: str = None       # 디바운스 통과 판정
    area: float = 0.0
    box: tuple = None
    stop: bool = True        # 최종 결론 (초기값은 정지 = 안전측)
    phase: str = "WAIT"      # WAIT | GO | STOP


class TrafficLightDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._red_n = 0
        self._green_n = 0
        self._none_n = 0
        self._stable = None
        self._red_latch = False
        self._armed = False          # 한 번이라도 "출발해도 된다"가 확정되었는가
        self._t0 = time.time()
        self._moved = False

    def note_moved(self):
        self._moved = True

    # ------------------------------------------------------------------
    def _best_blob(self, mask, cfg):
        min_a = float(cfg.get("traffic_light", "min_area_px", 14))
        max_a = float(cfg.get("traffic_light", "max_area_px", 4000))
        min_fill = float(cfg.get("traffic_light", "min_fill", 0.50))
        asp_tol = float(cfg.get("traffic_light", "aspect_tol", 2.2))

        best = None
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 2 or bh < 2:
                continue
            if area / float(bw * bh) < min_fill:
                continue
            asp = bw / float(bh)
            if asp > asp_tol or asp < 1.0 / asp_tol:
                continue
            if best is None or area > best[0]:
                best = (area, (x, y, bw, bh))
        return best

    @staticmethod
    def _ranges_mask(hsv, ranges):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for i in range(0, len(ranges) - 1, 2):
            m |= cv2.inRange(hsv, np.array(ranges[i], np.uint8),
                             np.array(ranges[i + 1], np.uint8))
        return m

    # ------------------------------------------------------------------
    def process(self, bgr, road_mask=None, road_y0=0) -> LightResult:
        c = self.cfg
        if not c.get("traffic_light", "enabled", True):
            self._armed = True
            return LightResult(stop=False, phase="GO")

        h, w = bgr.shape[:2]
        y1 = int(h * float(c.get("traffic_light", "roi_top", 0.0)))
        y2 = int(h * float(c.get("traffic_light", "roi_bottom", 0.75)))
        y1 = max(0, min(y1, h - 2))
        y2 = max(y1 + 2, min(y2, h))
        roi = bgr[y1:y2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 도로 위 픽셀 제외 (주황 점선/노면 반사 오검출 차단)
        block = None
        if road_mask is not None and c.get("traffic_light", "exclude_road", True):
            block = np.zeros((h, w), np.uint8)
            rh = road_mask.shape[0]
            y0 = max(0, min(road_y0, h - 1))
            block[y0:y0 + rh] = cv2.dilate(road_mask, np.ones((5, 5), np.uint8))
            block = block[y1:y2]

        red_r = list(c.get("traffic_light", "red_hsv",
                           [[0, 130, 140], [8, 255, 255], [170, 130, 140], [179, 255, 255]]))
        if c.get("traffic_light", "yellow_treated_as_red", False):
            red_r = red_r + [[20, 150, 170], [32, 255, 255]]
        grn_r = list(c.get("traffic_light", "green_hsv", [[45, 110, 130], [92, 255, 255]]))

        red_m = self._ranges_mask(hsv, red_r)
        grn_m = self._ranges_mask(hsv, grn_r)
        if block is not None:
            inv = cv2.bitwise_not(block)
            red_m = cv2.bitwise_and(red_m, inv)
            grn_m = cv2.bitwise_and(grn_m, inv)
        k = np.ones((3, 3), np.uint8)
        red_m = cv2.morphologyEx(red_m, cv2.MORPH_OPEN, k)
        grn_m = cv2.morphologyEx(grn_m, cv2.MORPH_OPEN, k)

        red = self._best_blob(red_m, c)
        grn = self._best_blob(grn_m, c)

        act = float(c.get("traffic_light", "act_area_px", 30))
        color, area, box = NONE, 0.0, None
        if red is not None and (grn is None or red[0] >= grn[0] * 0.5):
            color, area, box = RED, red[0], red[1]       # 동률이면 무조건 빨강 편
        elif grn is not None:
            color, area, box = GREEN, grn[0], grn[1]
        if box is not None:
            box = (box[0], box[1] + y1, box[2], box[3])

        judged = color if area >= act else NONE

        # ---------------- 디바운스 ----------------
        if judged == RED:
            self._red_n += 1
            self._green_n = 0
            self._none_n = 0
        elif judged == GREEN:
            self._green_n += 1
            self._red_n = 0
            self._none_n = 0
        else:
            self._none_n += 1
            self._red_n = 0
            self._green_n = 0

        confirm = int(c.get("traffic_light", "confirm_frames", 3))
        release = int(c.get("traffic_light", "release_frames", 5))
        red_release = int(c.get("traffic_light", "red_release_frames", 14))
        no_light = int(c.get("traffic_light", "no_light_frames", 20))

        if self._red_n >= confirm:
            self._stable, self._red_latch = RED, True
        elif self._green_n >= confirm:
            self._stable = GREEN
            self._armed = True
            if self._green_n >= release:
                self._red_latch = False
        elif self._none_n >= release:
            self._stable = NONE
            # 빨강을 봤다가 사라진 경우: 충분히 오래 안 보여야 해제
            if self._red_latch and self._none_n >= red_release:
                self._red_latch = False
            # 출발선에 신호등이 아예 없는 코스일 수도 있다
            if not self._armed and self._none_n >= no_light and not self._red_latch:
                self._armed = True

        stop = self._red_latch or (not self._armed)
        phase = "STOP" if self._red_latch else ("WAIT" if not self._armed else "GO")

        # 어떤 이유로든 계속 서 있으면 완주 자체가 불가능하다 -> 최후의 안전장치
        if stop and not self._moved:
            if time.time() - self._t0 > float(c.get("traffic_light", "start_max_wait", 30.0)):
                stop, phase = False, "GO"
                self._armed, self._red_latch = True, False

        return LightResult(color=color, stable=self._stable, area=area,
                           box=box, stop=stop, phase=phase)
