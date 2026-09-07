"""차선 인지 - 주황 중앙 점선 추종 (슬라이딩 윈도우).

왜 이 방식인가:
  * 트랙에 분기(갈림길)가 있다. "가장 넓은 아스팔트"를 쫓으면 분기점에서 옆으로 튄다.
  * 주황 점선은 코스가 지정한 **정답 경로**다. 이걸 따라가면 분기 문제가 사라진다.
  * 점선이라 끊기는 문제는 슬라이딩 윈도우 + 다항식 피팅으로 해결한다.
    (밴드마다 이전 위치 ±window 안에서만 찾으므로 갈라지는 가짜 선도 무시된다)

파이프라인:
  ROI -> 아스팔트 마스크 -> 차량 바로 앞과 연결된 성분만 남김(= 우리가 달리는 도로)
      -> 그 안의 주황 픽셀만 추출 -> 아래에서 위로 슬라이딩 윈도우 추적
      -> 2차 다항식 피팅 -> 근/중/원거리 조준점 + 곡률

주황선을 못 찾으면 도로 중앙 추종으로 자동 폴백한다.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class LaneResult:
    ok: bool = False
    error: float = 0.0          # -1(목표가 왼쪽) ~ +1(오른쪽)
    curvature: float = 0.0      # 원거리-근거리 차이. 코너 강도
    width: float = 0.0          # 근거리 도로 폭 (화면 폭 비율)
    source: str = "none"        # 'center'(주황선) | 'road'(폴백)
    aim_x: float = 0.0
    aim_y: float = 0.0
    curve_pts: tuple = ()       # 디버그: 피팅된 중앙선 [(x, y), ...]
    win_pts: tuple = ()         # 디버그: 슬라이딩 윈도우가 실제로 찾은 점
    rows: tuple = ()            # 디버그: [(y, cx, x_lo, x_hi), ...]
    mask: Optional[np.ndarray] = None       # 도로 마스크 (ROI 좌표)
    orange: Optional[np.ndarray] = None     # 주황 마스크 (ROI 좌표)
    cone: Optional[np.ndarray] = None       # 라바콘 마스크 (ROI 좌표)
    roi_y0: int = 0


def _longest_run(row_bool):
    if not row_bool.any():
        return None
    padded = np.concatenate(([False], row_bool, [False]))
    d = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._curv = 0.0
        self._prev_base = None      # 직전 프레임의 중앙선 하단 x (연속성 유지)
        self._last_coef = None
        self._last_span = None

    # ------------------------------------------------------------------
    def _road_mask(self, bgr, hsv):
        c = self.cfg
        s_max = int(c.get("lane", "road_s_max", 90))
        v_min = int(c.get("lane", "road_v_min", 25))
        v_max = int(c.get("lane", "road_v_max", 255))
        mask = cv2.inRange(hsv, (0, 0, v_min), (179, s_max, v_max))

        g_lo = int(c.get("lane", "grass_h_lo", 30))
        g_hi = int(c.get("lane", "grass_h_hi", 95))
        g_s = int(c.get("lane", "grass_s_min", 55))
        mask = cv2.bitwise_and(
            mask, cv2.bitwise_not(cv2.inRange(hsv, (g_lo, g_s, 30), (g_hi, 255, 255))))

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    def _cone_mask(self, hsv):
        """초록 라바콘. 도로에서 빼면 _longest_run 이 자동으로 빈 쪽을 고른다.

        실측: 콘 H 59~61 / S 229~255,  잔디 H 32~35 / S 95~148 -> S 로 확실히 갈린다.
        별도의 회피 제어를 만들지 않고 기존 조준 로직이 그대로 피하게 하는 게 요점이다.
        """
        c = self.cfg
        if not c.get("obstacle", "cone_avoid", True):
            return None
        m = cv2.inRange(hsv,
                        (int(c.get("obstacle", "cone_h_lo", 50)),
                         int(c.get("obstacle", "cone_s_min", 170)),
                         int(c.get("obstacle", "cone_v_min", 120))),
                        (int(c.get("obstacle", "cone_h_hi", 75)), 255, 255))
        if int(np.count_nonzero(m)) < int(c.get("obstacle", "cone_min_px", 40)):
            return None
        pad = int(c.get("obstacle", "cone_pad_px", 9))
        if pad > 1:
            m = cv2.dilate(m, np.ones((pad, pad), np.uint8))
        return m

    @staticmethod
    def _own_road(mask):
        """차량 바로 앞(하단 중앙)과 연결된 성분만 남긴다.
        분기로 뻗은 다른 도로나 배경의 도로가 섞이는 것을 막는다."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        h, w = mask.shape
        band = labels[int(h * 0.90):, int(w * 0.30):int(w * 0.70)]
        vals, counts = np.unique(band[band > 0], return_counts=True)
        if vals.size:
            lab = int(vals[int(np.argmax(counts))])
        else:  # 하단 중앙에 도로가 없다 -> 가장 큰 성분
            lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == lab, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    def _track_centerline(self, orange, rw, rh):
        """아래에서 위로 밴드를 훑으며 주황 점선을 추적한다."""
        c = self.cfg
        bands = int(c.get("centerline", "bands", 10))
        win = float(c.get("centerline", "window", 0.16)) * rw
        min_px = int(c.get("centerline", "min_band_px", 12))

        band_h = max(2, rh // bands)
        pts = []
        x_ref = self._prev_base

        for b in range(bands):
            y2 = rh - b * band_h
            y1 = max(0, y2 - band_h)
            if y1 >= y2:
                break
            sub = orange[y1:y2]
            ys, xs = np.nonzero(sub)
            if xs.size < min_px:
                continue
            if x_ref is not None:
                sel = np.abs(xs - x_ref) <= win
                if sel.sum() < min_px:
                    continue
                xs, ys = xs[sel], ys[sel]
            cx = float(xs.mean())
            cy = float(y1 + ys.mean())
            pts.append((cy, cx))
            x_ref = cx
            # 위로 갈수록 원근으로 좁아지므로 탐색창도 살짝 줄인다
            win = max(win * 0.88, 0.05 * rw)

        return pts

    # ------------------------------------------------------------------
    def process(self, bgr) -> LaneResult:
        c = self.cfg
        h, w = bgr.shape[:2]

        y1 = int(h * float(c.get("lane", "roi_top", 0.50)))
        y2 = int(h * float(c.get("lane", "roi_bottom", 0.98)))
        y1 = max(0, min(y1, h - 4))
        y2 = max(y1 + 4, min(y2, h))
        roi = bgr[y1:y2]
        rh, rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        road = self._own_road(self._road_mask(roi, hsv))
        # 라바콘은 도로가 아니다. 스캔용 마스크에서만 빼서 조준점이 옆으로 비켜가게 한다.
        # (주황선 게이팅과 신호등 exclude_road 는 원래 road 를 그대로 쓴다)
        cone = self._cone_mask(hsv)
        road_free = cv2.bitwise_and(road, cv2.bitwise_not(cone)) if cone is not None else road

        # ---- 주황 중앙선: 도로 위에 있는 주황 픽셀만 ----
        o_lo = int(c.get("centerline", "h_lo", 5))
        o_hi = int(c.get("centerline", "h_hi", 28))
        o_s = int(c.get("centerline", "s_min", 110))
        o_v = int(c.get("centerline", "v_min", 110))
        orange = cv2.inRange(hsv, (o_lo, o_s, o_v), (o_hi, 255, 255))
        road_dil = cv2.dilate(road, np.ones((7, 7), np.uint8), iterations=1)
        orange = cv2.bitwise_and(orange, road_dil)
        orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        # ---- 스캔 행: 도로 폭 측정 + 폴백용 ----
        fracs = (float(c.get("lane", "look_near", 0.90)),
                 float(c.get("lane", "look_mid", 0.62)),
                 float(c.get("lane", "look_far", 0.30)))
        min_width = float(c.get("lane", "min_road_width", 0.10)) * rw
        rows = []
        for f in fracs:
            y = int(np.clip(rh * f, 0, rh - 1))
            band = road_free[max(0, y - 1):min(rh, y + 2)].max(axis=0) > 0
            run = _longest_run(band)
            if run is None or (run[1] - run[0]) < min_width:
                rows.append(None)
            else:
                rows.append((y, (run[0] + run[1]) * 0.5, run[0], run[1]))

        near, mid, far = rows
        if near is None:
            near = mid or far
        if near is None:
            self._prev_base = None
            return LaneResult(ok=False, mask=road, orange=orange, cone=cone, roi_y0=y1)
        if mid is None:
            mid = near
        if far is None:
            far = mid

        # ---- 중앙선 추적 + 피팅 ----
        pts = self._track_centerline(orange, rw, rh)
        source, coef, span = "road", None, None
        if len(pts) >= int(c.get("centerline", "min_points", 3)):
            ys = np.array([p[0] for p in pts], np.float32)
            xs = np.array([p[1] for p in pts], np.float32)
            deg = 2 if (ys.max() - ys.min()) > rh * 0.35 and len(pts) >= 4 else 1
            try:
                coef = np.polyfit(ys, xs, deg)
                source = "center"
                span = (float(ys.min()), float(ys.max()))
            except Exception:  # noqa: BLE001
                coef = None
        if coef is None and self._last_coef is not None and \
                c.get("centerline", "hold_last", True):
            # 한두 프레임 놓친 것뿐이면 직전 피팅을 잠깐 유지
            coef, source, span = self._last_coef, "center_hold", self._last_span
        if source == "center":
            self._last_coef, self._last_span = coef, span

        offset = float(np.clip(c.get("lane", "lane_offset", 0.0), -1.0, 1.0))
        margin = float(c.get("lane", "edge_margin", 0.10))

        # 피팅은 점이 있던 구간에서만 유효하다. 그 밖으로 외삽하면 90도 코너에서
        # "직선이 계속된다"는 거짓 신호가 나와 곡률이 0 이 되고 감속/조향을 놓친다.
        # 지지구간 밖에서는 눈에 보이는 도로 중심을 쓴다 -> 코너가 곡률로 드러난다.
        extra = float(c.get("centerline", "extrapolate", 0.08)) * rh

        def target_at(row):
            ry, road_cx, x_lo, x_hi = row
            half = (x_hi - x_lo) * 0.5
            if coef is not None and span is not None and \
                    (span[0] - extra) <= ry <= (span[1] + extra):
                cx = float(np.polyval(coef, ry))
            else:
                cx = road_cx
            cx += offset * half
            # 무슨 일이 있어도 도로 밖을 조준하지 않는다
            return float(np.clip(cx, x_lo + half * margin, x_hi - half * margin))

        blend = float(np.clip(c.get("lane", "aim_blend", 0.55), 0.0, 1.0))
        aim_x = (1 - blend) * target_at(near) + blend * target_at(mid)
        aim_y = (1 - blend) * near[0] + blend * mid[0]

        center = rw * 0.5
        error = float((aim_x - center) / center)
        raw_curv = float((target_at(far) - target_at(near)) / center)
        lpf = float(np.clip(c.get("control", "curve_lpf", 0.35), 0.01, 1.0))
        self._curv = (1 - lpf) * self._curv + lpf * raw_curv

        # 다음 프레임 추적 시작점
        if coef is not None and span is not None:
            # 외삽 금지: 지지구간 안쪽의 가장 가까운 y 에서 평가한다
            self._prev_base = float(np.polyval(coef, float(np.clip(near[0], span[0], span[1]))))
        else:
            self._prev_base = near[1]

        curve_pts = ()
        if coef is not None:
            yy = np.linspace(0, rh - 1, 12)
            curve_pts = tuple((float(np.polyval(coef, v)), float(v)) for v in yy)

        return LaneResult(
            ok=True,
            error=float(np.clip(error, -1.5, 1.5)),
            curvature=self._curv,
            width=(near[3] - near[2]) / rw,
            source=source,
            aim_x=aim_x, aim_y=aim_y,
            curve_pts=curve_pts,
            win_pts=tuple((x, y) for y, x in pts),
            rows=tuple(r for r in (near, mid, far) if r is not None),
            mask=road, orange=orange, cone=cone, roi_y0=y1,
        )
