"""LiDAR 점검 - "라이다 켜져 있나?" 를 30초 안에 확정한다.

주행 노드를 끄고 단독 실행:
    python3 amet/tools/lidar_check.py

확인하는 것:
  1) 토픽이 실제로 오는가 (config 의 obstacle.topic 그대로 사용)
  2) 유효한 거리값이 있는가 (전부 inf/nan 이면 센서가 죽은 것)
  3) **0도가 진짜 정면인가** - 이게 제일 자주 틀린다.
     섹터별 최단거리를 찍어주므로, 차 앞에 물체를 놓고 어느 섹터가
     반응하는지 보면 angle_offset_deg 를 얼마로 줘야 하는지 바로 나온다.
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amet_drive.config import Config  # noqa: E402


class Check(Node):
    def __init__(self):
        super().__init__("lidar_check")
        cfg = Config()
        self.topic = str(cfg.get("obstacle", "topic", "/scan_filtered"))
        self.fov = float(cfg.get("obstacle", "fov_deg", 26.0))
        self.off = float(cfg.get("obstacle", "angle_offset_deg", 0.0))
        self.stop_d = float(cfg.get("obstacle", "stop_dist", 0.32))
        self.n = 0
        self.create_subscription(LaserScan, self.topic, self._cb,
                                 qos_profile_sensor_data)
        print(f"구독: {self.topic}   (config 의 obstacle.topic)")
        print("5초 안에 아무것도 안 나오면 토픽 이름이나 센서를 의심하라.\n")

    def _cb(self, msg):
        self.n += 1
        if self.n % 10 != 1:          # 너무 빨리 찍히지 않게
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(r.size, dtype=np.float32) * msg.angle_increment
        ang = np.degrees(np.arctan2(np.sin(ang), np.cos(ang)))
        ok = np.isfinite(r) & (r > 0.05)
        if msg.range_max > 0:
            ok &= r < msg.range_max * 0.99

        print(f"[{self.n:>4}] 포인트 {r.size}  유효 {int(ok.sum())} "
              f"({100.0 * ok.mean():.0f}%)  각도 {ang.min():+.0f}~{ang.max():+.0f}deg "
              f"range_max={msg.range_max:.1f}")
        if not ok.any():
            print("      -> 유효값이 하나도 없다. 센서가 꺼져 있거나 토픽이 비었다.")
            return

        # 섹터별 최단거리: 어느 방향이 정면인지 눈으로 확인하라
        line = "      "
        for lo in range(-180, 180, 45):
            sel = ok & (ang >= lo) & (ang < lo + 45)
            d = f"{np.min(r[sel]):.2f}" if sel.any() else "  -  "
            line += f"[{lo:+4d}..{lo + 45:+4d}] {d}   "
        print(line)

        # 현재 설정이 실제로 보는 값
        half = self.fov * 0.5
        rel = ang - self.off
        rel = (rel + 180.0) % 360.0 - 180.0
        sel = ok & (np.abs(rel) <= half)
        if sel.any():
            d = float(np.percentile(r[sel], 20))
            flag = "  <-- STOP 판정" if d <= self.stop_d else ""
            print(f"      현재 설정(fov={self.fov:.0f}deg, offset={self.off:.0f}deg) "
                  f"전방거리 = {d:.2f} m{flag}")
        else:
            print(f"      현재 설정 부채꼴 안에 유효값 없음 "
                  f"-> angle_offset_deg 가 틀렸을 가능성이 크다")


def main():
    rclpy.init()
    node = Check()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
