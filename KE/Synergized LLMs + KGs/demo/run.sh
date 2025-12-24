#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KG_DIR="$(cd "${DEMO_DIR}/.." && pwd)"
KE_DIR="$(cd "${KG_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${KE_DIR}/.." && pwd)"

VENV_DIR="${REPO_ROOT}/.venv"
VENV_PY="${VENV_DIR}/bin/python"

PORT="${PORT:-8000}"
ACCOUNT_ID="${ACCOUNT_ID:-A0250}"
WINDOW_MINUTES="${WINDOW_MINUTES:-2880}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-demo-neo4j}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

COLOR="${COLOR:-1}"
if [[ "${COLOR}" -eq 1 ]] && [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  if [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_RESET="$(tput sgr0)"
    C_BOLD="$(tput bold)"
    C_RED="$(tput setaf 1)"
    C_GREEN="$(tput setaf 2)"
    C_YELLOW="$(tput setaf 3)"
    C_BLUE="$(tput setaf 4)"
    C_CYAN="$(tput setaf 6)"
  else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
  fi
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
fi

log_step() { printf "%s==>%s %s%s%s\n" "${C_BLUE}${C_BOLD}" "${C_RESET}" "${C_BOLD}" "$*" "${C_RESET}"; }
log_info() { printf "%s[INFO]%s %s\n" "${C_CYAN}" "${C_RESET}" "$*"; }
log_ok() { printf "%s[ OK ]%s %s\n" "${C_GREEN}" "${C_RESET}" "$*"; }
log_warn() { printf "%s[WARN]%s %s\n" "${C_YELLOW}" "${C_RESET}" "$*"; }
log_err() { printf "%s[ERR ]%s %s\n" "${C_RED}" "${C_RESET}" "$*" >&2; }

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  COMPOSE_CMD=(docker compose)
fi

log_step "0/10 环境信息"
log_info "REPO_ROOT=${REPO_ROOT}"
log_info "DEMO_DIR=${DEMO_DIR}"
log_info "PORT=${PORT} ACCOUNT_ID=${ACCOUNT_ID} WINDOW_MINUTES=${WINDOW_MINUTES}"
log_info "Neo4j=${NEO4J_CONTAINER} (${NEO4J_USER}/******)"
log_info "Compose=${COMPOSE_CMD[*]}"

if [[ ! -x "${VENV_PY}" ]]; then
  log_step "1/10 创建虚拟环境"
  python3 -m venv "${VENV_DIR}"
  log_ok "Created venv: ${VENV_DIR}"
else
  log_ok "Reuse venv: ${VENV_DIR}"
fi

log_step "2/10 安装依赖"
"${VENV_PY}" -m pip install -U pip >/dev/null
"${VENV_PY}" -m pip install -r "${DEMO_DIR}/requirements.txt" >/dev/null
log_ok "Dependencies installed"

log_step "3/10 启动 Neo4j 容器"
cd "${DEMO_DIR}"
"${COMPOSE_CMD[@]}" up -d neo4j

neo4j_ready=0
for _ in $(seq 1 60); do
  if docker exec "${NEO4J_CONTAINER}" cypher-shell -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" "RETURN 1" >/dev/null 2>&1; then
    neo4j_ready=1
    break
  fi
  sleep 2
done
if [[ "${neo4j_ready}" -ne 1 ]]; then
  log_err "Neo4j not ready"
  log_err "Check logs: docker logs ${NEO4J_CONTAINER} --tail 100"
  exit 1
fi
log_ok "Neo4j ready: bolt://localhost:7687 http://localhost:7474"

log_step "4/10 生成合成数据"
"${VENV_PY}" "${DEMO_DIR}/etl/generate_data.py"

log_step "5/10 导入图谱到 Neo4j"
"${VENV_PY}" "${DEMO_DIR}/etl/load_to_neo4j.py"

log_step "6/10 验证图谱落地"
docker exec "${NEO4J_CONTAINER}" cypher-shell -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" \
  "MATCH (n) RETURN count(n) AS nodes"

api_started=0
api_pid=""
cleanup() {
  if [[ "${api_started}" -eq 1 ]] && [[ -n "${api_pid}" ]]; then
    kill "${api_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  api_started=0
  log_ok "Reuse existing API: http://127.0.0.1:${PORT}"
else
  log_step "7/10 启动 API"
  cd "${KG_DIR}"
  "${VENV_PY}" -m uvicorn demo.api.server:app --port "${PORT}" >/dev/null 2>&1 &
  api_pid="$!"
  api_started=1
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  log_ok "API started: http://127.0.0.1:${PORT} (pid=${api_pid})"
fi

log_step "8/10 规则判分（/agent/verdict）"
curl -s -X POST "http://127.0.0.1:${PORT}/agent/verdict" \
  -H 'Content-Type: application/json' \
  -d "{\"account_id\":\"${ACCOUNT_ID}\",\"window_minutes\":${WINDOW_MINUTES}}" \
  | "${VENV_PY}" -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'

log_step "9/10 子图取证规模（/graph/subgraph）"
curl -s "http://127.0.0.1:${PORT}/graph/subgraph?account_id=${ACCOUNT_ID}&hops=2&window_minutes=${WINDOW_MINUTES}" \
  | "${VENV_PY}" -c 'import json,sys; o=json.load(sys.stdin); print(json.dumps({"nodes":len(o.get("nodes",[])),"edges":len(o.get("edges",[]))}, ensure_ascii=False))'

log_step "10/10 LLM 判决（/agent/verdict_llm）"
if curl -sf -X POST "http://127.0.0.1:${PORT}/agent/verdict_llm" \
  -H 'Content-Type: application/json' \
  -d "{\"account_id\":\"${ACCOUNT_ID}\",\"window_minutes\":${WINDOW_MINUTES}}" \
  | "${VENV_PY}" -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'; then
  log_ok "LLM verdict completed"
else
  log_warn "Skip LLM verdict: set MOONSHOT_API_KEY via repo root .env.local or demo/configs/.env.local"
fi

log_ok "Demo finished"
