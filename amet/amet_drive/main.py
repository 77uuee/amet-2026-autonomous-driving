"""AMET 2026 - 단일 프로세스 자율주행 노드.

파이프라인 (제어 루프 1회):
    최신 카메라 프레임 -> 차선 인지 -> 신호등 인지
                        -> LiDAR 전방 거리
                        -> FSM 판단 -> PD 조향 + 곡률 기반 속도
                        -> /speed, /steering 발행 (워치독 갱신)

설계 메모:
 * /cmd_vel(Twist) 대신 /speed + /steering 을 직접 쓴다.
   Ackermann 역변환을 추측할 필요가 없고 조향각을 도(deg)로 직접 다룰 수 있다.
 * 드라이브 명령은 ~1초 후 만료되므로 매 루프 재발행한다.
 * 종료 시 반드시 0을 보낸다 (차가 계속 달리면 안 됨).
"""
import os
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float64

from amet_drive.config import Config
from amet_drive.control import Controller
from amet_drive.lane import LaneDetector
from amet_drive.lights import TrafficLightDetector
from amet_drive.obstacle import ObstacleDetector
from amet_drive import viewer as viewer_mod

# FSM 상태
S_WAIT_LIGHT = "WAIT_LIGHT"
S_DRIVE = "DRIVE"
S_STOP_LIGHT = "STOP_LIGHT"
S_STOP_OBSTACLE = "STOP_OBS"
S_LANE_LOST = "LANE_LOST"


class AmetDriver(Node):
    def __init__(self):
        super().__init__("amet_driver")
        self.cfg = Config()
        self.lane = LaneDetector(self.cfg)
        self.light = TrafficLightDetector(self.cfg)
        self.obst = ObstacleDetector(self.cfg)
        self.ctrl = Controller(self.cfg)

        self._frame = None
        self._frame_t = 0.0
        self._scan = None
        self._last_good_t = time.time()
        self._last_t = time.time()
        self._fps = 0.0
        self._state = S_WAIT_LIGHT
        self._log_n = 0
        self._t_start = time.time()

        self.speed_pub = self.create_publisher(Float64, "/speed", 10)
        self.steer_pub = self.create_publisher(Float64, "/steering", 10)

        self.create_subscription(
            CompressedImage, "/camera/image_raw/compressed",
            self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, str(self.cfg.get("obstacle", "topic", "/scan_filtered")),
            self._on_scan, qos_profile_sensor_data)

        self.viewer = viewer_mod.DebugViewer()
        if self.cfg.get("general", "debug_view", True):
            self.viewer.start()

        self._log_dir = str(self.cfg.get("general", "log_dir", "") or "")
        self._log_f = None
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)
            self._log_f = open(os.path.join(self._log_dir, "drive_log.csv"), "w")
            self._log_f.write("t,state,err,curv,width,speed,steer,light,obs\n")

        hz = float(self.cfg.get("general", "control_hz", 20.0))
        self.create_timer(1.0 / max(hz, 1.0), self._loop)
        self.get_logger().info("amet_driver ready")

    # ------------------------------------------------------------------
    def _on_image(self, msg):
        try:
            buf = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return
        if img is None:
            return
        target_w = int(self.cfg.get("general", "proc_width", 320))
        if target_w and img.shape[1] != target_w:
            scale = target_w / float(img.shape[1])
            img = cv2.resize(img, (target_w, max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        self._frame = img
        self._frame_t = time.time()

    def _on_scan(self, msg):
        self._scan = msg

    # ------------------------------------------------------------------
    def _publish(self, speed, steer_deg):
        self.speed_pub.publish(Float64(data=float(speed)))
        self.steer_pub.publish(Float64(data=float(np.radians(steer_deg))))

    def stop_now(self):
        self.ctrl.reset()
        for _ in range(3):
            self._publish(0.0, 0.0)
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def _loop(self):
        now = time.time()
        dt = max(1e-3, min(0.2, now - self._last_t))
        self._last_t = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self.cfg.reload()

        frame = self._frame
        # 카메라가 끊기면 무조건 정지 (제어 불가 상태로 달리면 이탈 확정)
        if frame is None or (now - self._frame_t) > 1.0:
            self._state = S_LANE_LOST
            self._publish(0.0, self.ctrl.steer_deg)
            return

        lane = self.lane.process(frame)
        # 신호등 판정에 도로 마스크를 넘겨 아스팔트 위 픽셀을 제외시킨다
        light = self.light.process(frame, lane.mask, lane.roi_y0)
        obs = self.obst.process(self._scan)

        # ---------------- FSM ----------------
        if lane.ok:
            self._last_good_t = now

        lost_for = now - self._last_good_t
        lost_timeout = float(self.cfg.get("lane", "lost_timeout", 0.6))

        if light.stop:
            state = S_WAIT_LIGHT if light.phase == "WAIT" else S_STOP_LIGHT
        elif obs.stop:
            state = S_STOP_OBSTACLE
        elif not lane.ok and lost_for > lost_timeout:
            state = S_LANE_LOST
        else:
            state = S_DRIVE
        if self._state in (S_WAIT_LIGHT, S_STOP_LIGHT) and state == S_DRIVE:
            self.get_logger().info(f"GO  (t+{now - self._t_start:.1f}s)")
        elif self._state != state and state in (S_WAIT_LIGHT, S_STOP_LIGHT):
            self.get_logger().info(f"HOLD at light  ({light.stable}, area={light.area:.0f})")
        self._state = state

        # ---------------- 제어 ----------------
        if state in (S_WAIT_LIGHT, S_STOP_LIGHT, S_STOP_OBSTACLE):
            # 정지 중에도 조향은 마지막 값을 유지 (재출발 시 방향 유지)
            steer = self.ctrl.steer_deg
            speed = self.ctrl.ramp_speed(0.0, dt)
            self.ctrl._prev_err = lane.error if lane.ok else self.ctrl._prev_err
        elif state == S_LANE_LOST:
            # 도로를 놓쳤다: 마지막 조향을 유지한 채 아주 천천히 기어간다.
            # 그 자리에 서면 완주 자체가 불가능하므로 멈추지는 않는다.
            steer = float(np.clip(self.ctrl.steer_deg, -18.0, 18.0))
            speed = self.ctrl.ramp_speed(float(self.cfg.get("drive", "v_min", 0.28)) * 0.7, dt)
        else:
            steer = self.ctrl.steering(lane.error, dt, lane.curvature)
            target = self.ctrl.target_speed(lane.curvature, lane.error)
            target *= obs.scale
            # 출발 직후 램프업
            if now - self._t_start < 1.0:
                target = min(target, float(self.cfg.get("drive", "v_start", 0.30)))
            speed = self.ctrl.ramp_speed(target, dt)

        if speed > 0.02:
            self.light.note_moved()

        self._publish(speed, steer)

        # ---------------- 디버그 / 로그 ----------------
        if self.viewer._started:
            try:
                overlay = viewer_mod.render(
                    frame, lane, light, obs, state, speed, steer, self._fps,
                    float(self.cfg.get("lane", "roi_top", 0.5)),
                    float(self.cfg.get("lane", "roi_bottom", 0.98)))
                self.viewer.publish(overlay)
            except Exception:
                pass

        if self._log_f is not None:
            self._log_n += 1
            every = max(1, int(self.cfg.get("general", "log_every", 3)))
            if self._log_n % every == 0:
                self._log_f.write(
                    f"{now - self._t_start:.3f},{state},{lane.error:.4f},"
                    f"{lane.curvature:.4f},{lane.width:.3f},{speed:.3f},{steer:.2f},"
                    f"{light.stable},{obs.distance:.3f}\n")
                cv2.imwrite(os.path.join(self._log_dir, f"f{self._log_n:06d}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 65])

    def destroy_node(self):
        try:
            self.stop_now()
            self.viewer.stop()
            if self._log_f:
                self._log_f.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = AmetDriver()

    def _bye(*_):
        node.stop_now()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
