#!/usr/bin/env bash
# scripts/ci.sh — Local CI runner for Nexus Agent
#
# Mirrors the GitHub Actions pipeline so you can verify locally before pushing.
#
# Usage:
#   bash scripts/ci.sh                    # PR profile (stages 1-5)
#   bash scripts/ci.sh --profile commit   # Stages 1-2 only
#   bash scripts/ci.sh --profile pr       # Stages 1-5
#   bash scripts/ci.sh --profile release  # All 8 stages
#   bash scripts/ci.sh --stage 3          # Run a single stage by number
#   bash scripts/ci.sh --help

set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ─── Defaults ─────────────────────────────────────────────────────────────────
PROFILE="pr"
SINGLE_STAGE=""
START_TIME=$(date +%s)
declare -A STAGE_STATUS=()

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)  PROFILE="$2"; shift 2 ;;
    --stage)    SINGLE_STAGE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/ci.sh [--profile commit|pr|release] [--stage N]"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
banner() {
  local n="$1" title="$2"
  echo ""
  echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${CYAN}${BOLD}  STAGE $n — $title${RESET}"
  echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

run_stage() {
  local n="$1" title="$2"
  shift 2

  # If a single stage was requested, skip everything else
  if [[ -n "$SINGLE_STAGE" && "$SINGLE_STAGE" != "$n" ]]; then
    return 0
  fi

  banner "$n" "$title"

  local stage_start
  stage_start=$(date +%s)

  if "$@"; then
    local elapsed=$(( $(date +%s) - stage_start ))
    echo -e "\n${GREEN}PASS${RESET} Stage $n finished in ${elapsed}s"
    STAGE_STATUS[$n]="PASS"
  else
    local elapsed=$(( $(date +%s) - stage_start ))
    echo -e "\n${RED}FAIL${RESET} Stage $n failed after ${elapsed}s"
    STAGE_STATUS[$n]="FAIL"
    # Print summary and exit immediately
    summary
    exit 1
  fi
}

summary() {
  local total_elapsed=$(( $(date +%s) - START_TIME ))
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  CI SUMMARY (profile: $PROFILE) — ${total_elapsed}s total${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  for n in 1 2 3 4 5 6 7 8; do
    if [[ -v STAGE_STATUS[$n] ]]; then
      local s="${STAGE_STATUS[$n]}"
      if [[ "$s" == "PASS" ]]; then
        echo -e "  Stage $n: ${GREEN}PASS${RESET}"
      else
        echo -e "  Stage $n: ${RED}FAIL${RESET}"
      fi
    fi
  done
  echo ""
}

# ─── Stage functions ──────────────────────────────────────────────────────────
stage1_lint() {
  echo ">>> ruff check nexus/ tests/"
  ruff check nexus/ tests/
  echo ">>> mypy nexus/"
  mypy nexus/
}

stage2_unit() {
  echo ">>> pytest tests/unit/ --cov=nexus --cov-report=term-missing --cov-report=json --cov-fail-under=80"
  pytest tests/unit/ -v \
    --cov=nexus \
    --cov-report=term-missing \
    --cov-report=json \
    --cov-fail-under=80 \
    --timeout=60
}

stage3_integration() {
  echo ">>> pytest tests/integration/ -v --timeout=60 (appending coverage)"
  pytest tests/integration/ -v \
    --cov=nexus --cov-append \
    --cov-report=json \
    --timeout=60
}

stage4_property() {
  echo ">>> pytest tests/property/ -v"
  pytest tests/property/ -v
}

stage5_adversarial() {
  echo ">>> pytest tests/adversarial/ -v (appending coverage)"
  pytest tests/adversarial/ -v \
    --cov=nexus --cov-append \
    --cov-report=json
  echo ">>> python scripts/check_coverage.py  (combined: unit+integration+adversarial)"
  python scripts/check_coverage.py
}

stage6_benchmarks() {
  echo ">>> pytest tests/benchmarks/ -v --timeout=120"
  pytest tests/benchmarks/ -v --timeout=120
}

stage7_security() {
  echo ">>> bandit -r nexus/ -ll -q"
  bandit -r nexus/ -ll -q
  echo ">>> safety check --full-report"
  safety check --full-report
}

stage8_dead_code() {
  echo ">>> vulture nexus/ --min-confidence 80"
  vulture nexus/ --min-confidence 80
}

# ─── Profile dispatch ─────────────────────────────────────────────────────────
echo -e "${BOLD}Nexus Agent — Local CI Runner${RESET}"
echo -e "Profile : ${YELLOW}${PROFILE}${RESET}"
echo -e "Time    : $(date)"
echo ""

case "$PROFILE" in
  commit)
    run_stage 1 "Lint"       stage1_lint
    run_stage 2 "Unit Tests" stage2_unit
    ;;
  pr)
    run_stage 1 "Lint"             stage1_lint
    run_stage 2 "Unit Tests"       stage2_unit
    run_stage 3 "Integration"      stage3_integration
    run_stage 4 "Property Tests"   stage4_property
    run_stage 5 "Adversarial"      stage5_adversarial
    ;;
  release)
    run_stage 1 "Lint"             stage1_lint
    run_stage 2 "Unit Tests"       stage2_unit
    run_stage 3 "Integration"      stage3_integration
    run_stage 4 "Property Tests"   stage4_property
    run_stage 5 "Adversarial"      stage5_adversarial
    run_stage 6 "Benchmarks"       stage6_benchmarks
    run_stage 7 "Security"         stage7_security
    run_stage 8 "Dead Code"        stage8_dead_code
    ;;
  *)
    echo "Unknown profile: $PROFILE (use commit | pr | release)"
    exit 1
    ;;
esac

summary
