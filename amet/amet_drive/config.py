"""YAML 설정 로더 - 주행 중 자동 리로드.

현장에서 10분짜리 세션 동안 코드를 재시작하지 않고 값만 바꿔서
즉시 반영하기 위한 모듈. 파싱에 실패하면 직전 값을 유지한다.
"""
import os
import time

import yaml


class Config:
    def __init__(self, path=None, reload_period=0.5):
        self.path = path or os.environ.get(
            "AMET_CONFIG",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        )
        self._reload_period = reload_period
        self._mtime = 0.0
        self._next_check = 0.0
        self.data = {}
        self.reload(force=True)

    # ------------------------------------------------------------------
    def reload(self, force=False):
        """변경되었으면 다시 읽는다. 반환값: 실제로 리로드했는지."""
        now = time.time()
        if not force and now < self._next_check:
            return False
        self._next_check = now + self._reload_period
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return False
        if not force and mtime <= self._mtime:
            return False
        try:
            with open(self.path, "r") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError("config root must be a mapping")
        except Exception as exc:  # noqa: BLE001 - 잘못된 편집으로 죽으면 안 됨
            print(f"[config] reload FAILED, keeping previous values: {exc}")
            self._mtime = mtime
            return False
        self.data = data
        self._mtime = mtime
        if not force:
            print("[config] reloaded")
        return True

    # ------------------------------------------------------------------
    def get(self, section, key, default=None):
        try:
            val = self.data[section][key]
        except (KeyError, TypeError):
            return default
        return default if val is None else val

    def section(self, name):
        val = self.data.get(name)
        return val if isinstance(val, dict) else {}
