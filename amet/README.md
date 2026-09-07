# AMET 2026 주행 코드

## 설치

`_amet_bundle.zip` 을 **physicar_ws 루트**에서 풀면 이렇게 된다:

```
physicar_ws/
├── run.sh                  <- 기존 파일을 덮어씀 (Evaluation 진입점)
└── amet/
    ├── config.yaml         <- 튜닝은 전부 여기서
    ├── README.md
    ├── amet_drive/
    │   ├── main.py         제어 루프 + FSM
    │   ├── lane.py         아스팔트 영역 인지
    │   ├── lights.py       신호등
    │   ├── obstacle.py     LiDAR 전방 차량
    │   ├── control.py      PD 조향 + 곡률 속도 스케줄링
    │   ├── config.py       YAML 라이브 리로드
    │   └── viewer.py       MYAPP 디버그 뷰어
    └── tools/tune.py       HSV/ROI 캘리브레이션 툴
```

터미널에서:

```bash
cd ~/physicar_ws && unzip -o _amet_bundle.zip && chmod +x run.sh
```

## 실행

Evaluation 창의 Run command 는 그대로 두면 된다:

```
source /home/physicar/physicar_ws/run.sh
```

터미널에서 직접 돌려볼 때도 같은 명령. **Ctrl-C 로 끄면 차가 반드시 정지**한다.

## 튜닝 워크플로

1. `run.sh` 실행 → **MYAPP 탭(포트 5000)** 에서 차가 무엇을 보고 있는지 확인
   - 청록 반투명 = 도로로 인식한 영역
   - 노란 가로선 3개 = 근/중/원거리 스캔 행
   - 자홍 원 = 조준점. 이게 도로 중앙에 있어야 정상
2. 이상하면 **주행을 멈추지 않고** `amet/config.yaml` 을 편집 → 저장 → 약 0.5초 후 자동 반영
3. 색이 안 잡히면 노드를 끄고 `python3 amet/tools/tune.py` 실행 → 슬라이더로 맞춘 뒤
   화면에 나오는 YAML 조각을 `config.yaml` 에 붙여넣기

## 튜닝 순서 (이 순서를 지킬 것)

| 증상 | 건드릴 값 |
|---|---|
| 도로를 아예 못 찾음 (LANE LOST) | `lane.road_v_min` / `road_s_max` / `grass_h_*` |
| 직선에서 좌우로 떨림 | `control.kp` ↓, `control.kd` ↑ |
| 코너 진입이 늦어 밖으로 나감 | `lane.aim_blend` ↑, `control.curve_gain` ↑ |
| 코너에서 너무 느림 | `control.curve_gain` ↓, `drive.v_min` ↑ |
| 전체적으로 느림 | `drive.v_max` ↑ (한 번에 0.05씩만) |
| 빨간불에 출발함 | `traffic_light.confirm_frames` ↓, `act_area_px` ↓ |
| 초록불인데 안 감 | `traffic_light.release_frames` ↓, `green_hsv` 재조정 |
| 앞차에 부딪힘 | `obstacle.stop_dist` ↑, `slow_dist` ↑ |

## 오프라인 튜닝 (노트북에서 — 시뮬레이터 없이)

인지 튜닝의 대부분은 시뮬레이터가 필요 없다. 프레임 한 뭉치면 된다.
브라우저 터미널 지연도, 크레딧 소모도, CPU 경쟁도 없다.

**1) 시뮬에서 프레임 녹화** (CONTROL 패널 방향키로 직접 몰면서)

```bash
python3 amet/tools/grab.py --n 400 --hz 10 --out capture
zip -qr capture.zip capture      # 탐색기에서 Download
```

분기점 · 급코너 · 신호등 앞을 **반드시 지나면서** 녹화할 것.

**2) 노트북에서 리플레이**

```bash
pip install opencv-python numpy pyyaml
python3 amet/tools/replay.py --frames capture
# -> replay_out.mp4 (오버레이 영상), replay_out.csv, 콘솔 통계
```

**3) 파라미터 자동 비교**

```bash
python3 amet/tools/replay.py --frames capture --sweep centerline.window=0.10,0.13,0.16,0.20
python3 amet/tools/replay.py --frames capture --sweep centerline.h_hi=20,24,28,32
python3 amet/tools/replay.py --frames capture --sweep traffic_light.act_area_px=20,30,45,60
```

판단 기준: **center% 높을수록 / lost% 낮을수록 / jitter 낮을수록** 좋다.
`center%` 는 주황 중앙선을 물고 있는 프레임 비율이다. 95% 미만이면 색범위부터 잡아라.

좋은 값을 찾으면 `config.yaml` 에 반영하고, 그때만 시뮬에서 실주행으로 확인한다.

## 대회 제출 전 체크

- [ ] `general.debug_view: false` (CPU 절약)
- [ ] `general.log_dir: ""` (디스크 쓰기 제거)
- [ ] `drive.steer_max_deg` 가 20.0 을 넘지 않는지
- [ ] Ctrl-C 후 차가 완전히 정지하는지 확인
