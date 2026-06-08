#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_live.sh <provider> <api_key> [query] [--base-url <url>] [--model <model>] [--allow-network-failure]

Examples:
  scripts/verify_live.sh ds sk-...
  scripts/verify_live.sh openai sk-... "scientific discovery agents"
  scripts/verify_live.sh oa sk-... --base-url https://example.com/v1 --model gpt-4o-mini
  scripts/verify_live.sh deepseek sk-... --allow-network-failure

This script creates a temporary project, runs:
  1. ra init
  2. provider setup with the shortest CLI command
  3. ra a
  4. ra "<query>" --file seeds/demo_papers.jsonl --ideas 3

Use --allow-network-failure when you only want to confirm that the CLI
reaches the provider's network layer in a restricted environment.
EOF
}

if [[ $# -ge 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

provider_arg="$1"
api_key="$2"
shift 2

query="scientific discovery agents"
allow_network_failure=0
base_url=""
model=""

while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --allow-network-failure)
      allow_network_failure=1
      shift
      ;;
    --base-url)
      if [[ $# -lt 2 ]]; then
        echo "--base-url requires a URL." >&2
        exit 1
      fi
      base_url="$2"
      shift 2
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "--model requires a model name." >&2
        exit 1
      fi
      model="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      query="$arg"
      shift
      ;;
  esac
done

case "$provider_arg" in
  ds|deepseek)
    provider_cli="ds"
    provider_name="deepseek"
    ;;
  oa|openai)
    provider_cli="oa"
    provider_name="openai"
    ;;
  *)
    echo "Unknown provider: $provider_arg. Use ds/deepseek or oa/openai." >&2
    exit 1
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
project_dir="$(mktemp -d)"

cleanup() {
  if [[ -n "${KEEP_VERIFY_PROJECT:-}" ]]; then
    echo "Keeping temp project: $project_dir"
  else
    rm -rf "$project_dir"
  fi
}
trap cleanup EXIT

run_and_capture() {
  local output
  set +e
  output="$("$@" 2>&1)"
  local status=$?
  set -e
  printf '%s' "$output"
  return "$status"
}

accept_network_failure() {
  local label="$1"
  local output="$2"
  if [[ "$allow_network_failure" -eq 1 ]] && [[ "$output" == *"network, DNS, or proxy"* ]]; then
    echo "$output"
    echo
    echo "$label reached the provider network layer, but this environment cannot complete the outbound request."
    echo "Temp project: $project_dir"
    echo "Set KEEP_VERIFY_PROJECT=1 if you want to inspect the generated project next time."
    return 0
  fi
  return 1
}

echo "Creating temp project: $project_dir"
(
  cd "$project_dir"
  "$repo_root/ra" init >/dev/null
)

cd "$project_dir"
if [[ -n "$base_url" ]]; then
  llm_args=(llm "$provider_cli" --api-key "$api_key" --base-url "$base_url")
else
  llm_args=("$provider_cli" "$api_key")
fi
if [[ -n "$model" ]]; then
  llm_args+=(--model "$model")
fi
./ra "${llm_args[@]}" >/dev/null

echo "Running smoke check with $provider_name..."
smoke_output="$(run_and_capture ./ra a --provider "$provider_cli")" || {
  if accept_network_failure "Smoke check" "$smoke_output"; then
    exit 0
  fi
  echo "$smoke_output" >&2
  exit 1
}
echo "$smoke_output"

echo
echo "Running demo pipeline with query: $query"
run_output="$(run_and_capture ./ra "$query" --file seeds/demo_papers.jsonl --ideas 3 --provider "$provider_cli")" || {
  if accept_network_failure "Demo pipeline" "$run_output"; then
    exit 0
  fi
  echo "$run_output" >&2
  exit 1
}
echo "$run_output"

echo
echo "Live verification finished successfully."
echo "Temp project: $project_dir"
echo "Set KEEP_VERIFY_PROJECT=1 next time if you want to inspect the generated project after the script exits."
