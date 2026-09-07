#!/bin/bash
# ============================================================
#  AMET 2026 - entrypoint
#  Evaluation "Run command" 가 이 파일을 source 한다.
#  실제 로직은 amet/amet_drive/ 아래 .py 파일에 있음.
#  여기는 절대 건드리지 말 것 (튜닝은 amet/config.yaml 에서).
# ============================================================

AMET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/amet"

source /opt/ros/jazzy/setup.bash
[ -f "$AMET_DIR/../install/setup.bash" ] && source "$AMET_DIR/../install/setup.bash"

export PYTHONPATH="$AMET_DIR:$PYTHONPATH"
export AMET_CONFIG="${AMET_CONFIG:-$AMET_DIR/config.yaml}"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo " AMET 2026 autonomous run"
echo " config : $AMET_CONFIG"
echo " camera : /camera/image_raw/compressed"
echo " control: /speed + /steering"
echo "=============================================="

python3 -m amet_drive.main
