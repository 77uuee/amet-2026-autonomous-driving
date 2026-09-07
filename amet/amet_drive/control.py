"""조향/속도 제어 + 슬루레이트 제한.

조향: PD (오차는 정규화된 화면 좌표)
속도: 곡률 기반 스케줄링 -> 직선에서 밟고 코너에서 미리 줄인다.
      랩타임의 대부분이 여기서 갈린다.
"""
import time

import numpy as np


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_err = 0.0
        self._integral = 0.0
        self._prev_t = time.time()
        self.steer_deg = 0.0
        self.speed = 0.0

    def reset(self):
        self._prev_err = 0.0
        self._integral = 0.0
        self.steer_deg = 0.0
        self.speed = 0.0

    # ------------------------------------------------------------------
    def steering(self, error, dt, curvature=0.0):
        c = self.cfg
        kp = float(c.get("control", "kp", 26.0))
        kd = float(c.get("control", "kd", 4.5))
        ki = float(c.get("control", "ki", 0.0))
        kc = float(c.get("control", "curve_steer", 0.0))
        smax = float(c.get("drive", "steer_max_deg", 20.0))
        sign = float(c.get("drive", "steer_sign", 1.0))

        d = (error - self._prev_err) / max(dt, 1e-3)
        self._prev_err = error
        self._integral = float(np.clip(self._integral + error * dt, -2.0, 2.0)) if ki else 0.0

        # 도로가 오른쪽(error>0)이면 오른쪽으로 = 음의 조향각(+ 가 좌측이므로)
        # curvature 피드포워드: 90도 코너에서는 점선이 화면에서 수평이 되어 x=f(y)
        # 다항식으로 표현이 안 되고 error 가 거의 0 으로 나온다. 반면 도로 마스크의
        # 원거리 중심은 코너를 정확히 보여주므로(=curvature) 그걸 직접 조향에 더한다.
        # error 와 부호 규약이 같다(좌회전 = 음수).
        raw = -sign * (kp * error + kd * d + ki * self._integral + kc * curvature)
        target = float(np.clip(raw, -smax, smax))

        rate = float(c.get("drive", "steer_rate_deg", 240.0)) * dt
        self.steer_deg = float(np.clip(target, self.steer_deg - rate, self.steer_deg + rate))
        return self.steer_deg

    # ------------------------------------------------------------------
    def target_speed(self, curvature, error):
        c = self.cfg
        v_max = float(c.get("drive", "v_max", 0.75))
        v_min = float(c.get("drive", "v_min", 0.28))
        gain = float(c.get("control", "curve_gain", 1.9))

        # 코너 강도 = 예측 곡률 + 현재 횡오차 (둘 다 크면 위험)
        severity = abs(curvature) * gain + abs(error) * 0.55
        v = v_max - (v_max - v_min) * float(np.clip(severity, 0.0, 1.0))
        return float(np.clip(v, v_min, v_max))

    # ------------------------------------------------------------------
    def ramp_speed(self, target, dt):
        c = self.cfg
        a_up = float(c.get("drive", "accel_limit", 1.2)) * dt
        a_dn = float(c.get("drive", "decel_limit", 3.0)) * dt
        if target > self.speed:
            self.speed = min(target, self.speed + a_up)
        else:
            self.speed = max(target, self.speed - a_dn)
        if self.speed < 1e-3:
            self.speed = 0.0
        return self.speed
