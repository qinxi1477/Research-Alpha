#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_local.sh

Runs an offline local verification in a temporary project:
  1. ra init
  2. import a local Gold seed file with local HTML full-text links
  3. fetch full-text sections
  4. build extractive Genome and Pattern cards
  5. refresh stale Genome cards
  6. run evidence, doctor, and strict audit
  7. run a mocked GUI HTTP smoke for new idea, current-session follow-up, and hidden review-loop output

Set KEEP_VERIFY_PROJECT=1 to keep the temporary project for inspection.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

run() {
  echo "+ $*"
  "$@"
}

echo "Creating temp project: $project_dir"
(
  cd "$project_dir"
  run "$repo_root/ra" init >/dev/null
  mkdir -p local_papers seeds
  cat > local_papers/gold-a.html <<'HTML'
<html><body>
<h1>Introduction</h1><p>Local Gold A studies hidden assumptions in research-agent evaluation and motivates a stricter evidence story.</p>
<h1>Related Work</h1><p>Prior work scores task success but leaves failure boundaries underspecified.</p>
<h1>Methods</h1><p>The method reframes evaluation around assumption stress tests and traceable evidence.</p>
<h1>Experiments</h1><p>Experiments compare ordinary rankings with assumption-boundary audits and ablations.</p>
<h1>Limitations</h1><p>The approach fails when the target domain has no hidden assumption to expose.</p>
<h1>Conclusion</h1><p>The paper turns benchmark success into an evidence-backed failure-boundary story.</p>
</body></html>
HTML
  cat > local_papers/gold-b.html <<'HTML'
<html><body>
<h1>Introduction</h1><p>Local Gold B studies scalable route learning for scientific agents.</p>
<h1>Related Work</h1><p>Existing systems rely on broad summaries instead of structured paper logic.</p>
<h1>Approach</h1><p>The approach separates domain knowledge from scoring standards.</p>
<h1>Evaluation</h1><p>Evaluation checks whether generated ideas preserve provenance and novelty boundaries.</p>
<h1>Limitations</h1><p>The method is limited by paper parsing quality and source coverage.</p>
<h1>Conclusion</h1><p>The story emphasizes controlled migration from strong papers to a new domain.</p>
</body></html>
HTML
  cat > local_papers/gold-c.html <<'HTML'
<html><body>
<h1>Introduction</h1><p>Local Gold C frames a reviewer loop as a way to catch weak novelty claims.</p>
<h1>Related Work</h1><p>Prior idea-generation tools often stop at first-pass novelty claims.</p>
<h1>Method</h1><p>The method uses repeated critique and constrained revision.</p>
<h1>Experiments</h1><p>Experiments inspect whether revisions improve clarity and defensibility across multiple reviewer perspectives.</p>
<h1>Limitations</h1><p>The loop can still fail when source evidence is shallow or the novelty boundary is underspecified.</p>
<h1>Conclusion</h1><p>The work makes the final idea more auditable.</p>
</body></html>
HTML
  python3 - <<'PY'
import json
from pathlib import Path
rows = []
for idx, (title, award) in enumerate([
    ("Local Verification Gold A", "best_paper"),
    ("Local Verification Gold B", "oral"),
    ("Local Verification Gold C", "outstanding_paper"),
], start=1):
    rows.append({
        "title": title,
        "venue": "ICLR",
        "year": 2026,
        "award": award,
        "citation_count": 1000 - idx,
        "influential_citation_count": 100 - idx,
        "external_ref": str(Path("local_papers") / f"gold-{chr(96 + idx)}.html"),
        "abstract": "Local verification abstract for evidence pipeline smoke testing."
    })
Path("seeds/local_gold.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
PY
  run ./ra gold-seed --file seeds/local_gold.jsonl --fulltext --extractive-genomes
  run ./ra gb --refresh-stale --extractive --limit 10
  run ./ra pb --limit 3 --extractive
  run ./ra evidence --papers 10
  echo "+ ./ra doctor --papers 10"
  ./ra doctor --papers 10 || true
  echo "+ ./ra audit --papers 10 --skip-run"
  ./ra audit --papers 10 --skip-run || true
  python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("outputs/evidence/evidence_report.json").read_text(encoding="utf-8"))
counts = payload["counts"]
assert payload["status"] == "partially_ready", payload["status"]
assert counts["high_weight_papers"] == 3, counts
assert counts["paper_like_full_text_papers"] >= 2, counts
assert counts["stale_genome_cards"] == 0, counts
assert counts["stale_pattern_cards"] == 0, counts
audit = json.loads(Path("outputs/evidence/evidence_audit.json").read_text(encoding="utf-8"))
assert audit["ok"] is False, audit
assert any(item.get("code") == "evidence_not_ready" for item in audit.get("failures", [])), audit
print("Local verification counts:", json.dumps(counts, sort_keys=True))
PY
  echo "+ GUI HTTP smoke"
  PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from research_alpha.db import add_idea_session_turn, get_idea_session, update_idea_session_context
from research_alpha.gui import create_gui_http_server


def post_json(base: str, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def fake_chat(self, prompt, **kwargs):
    text = (
        "标题：稳定 GUI smoke idea\n\n"
        "核心想法：把对话稳定性当成科研 agent 的第一等研究对象，先保证用户的追问写回当前会话，再把审稿作为隐式质量层。"
    )
    return type("Resp", (), {"text": text, "usage": {}, "estimated_cost_usd": 0})()


def fake_gui_command(command, root_dir, on_output=None):
    if command and command[0] == "reviewer":
        session_id = int(command[command.index("--session-id") + 1])
        db_path = Path(root_dir) / "data" / "research_alpha.db"
        session = get_idea_session(db_path, session_id)
        context = json.loads(session["context_json"]) if session else {}
        history = context.get("review_history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "decision": "rethink_required",
                "summary": "单轮审稿认为 novelty 边界还需要收紧。",
                "fatal_flaws": ["单轮审稿 raw flaw 不应作为长输出暴露。"],
                "required_rethink": ["保留短审稿意见即可。"],
            }
        )
        context["review_history"] = history[-5:]
        update_idea_session_context(db_path, session_id, json.dumps(context, ensure_ascii=False))
        return {
            "exit_code": 2,
            "stdout": (
                "Reviewer decision: rethink_required\n"
                "Summary: 单轮审稿认为 novelty 边界还需要收紧。\n"
                "JSON: outputs/reviews/review-smoke.json\n"
            ),
            "stderr": "内部审稿 raw trace 不应进入对话",
        }
    if not command or command[0] != "review-loop":
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}
    session_id = int(command[command.index("--session-id") + 1])
    db_path = Path(root_dir) / "data" / "research_alpha.db"
    add_idea_session_turn(
        db_path,
        session_id,
        "根据第 1/4 轮顶会审稿意见改写当前 idea。\n内部长审稿指令",
        "partial",
        "循环审稿后的最终 idea",
        json.dumps({"source": "review_loop_refinement", "hidden_in_gui_chat": True}, ensure_ascii=False),
    )
    session = get_idea_session(db_path, session_id)
    context = json.loads(session["context_json"]) if session else {}
    context["review_history"] = [
        {
            "summary": "新颖性边界需要更硬。",
            "fatal_flaws": ["像方法硬迁移。"],
            "required_rethink": ["先抽故事线再迁移。"],
        }
    ]
    update_idea_session_context(db_path, session_id, json.dumps(context, ensure_ascii=False))
    return {
        "exit_code": 0,
        "stdout": 'Review-loop summary JSON: {"rounds_completed":1,"requested_rounds":4,"review_count":1,"final_idea":"循环审稿后的最终 idea"}',
        "stderr": "",
    }


def wait_for(predicate, label: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {label}: {last!r}")


root_dir = Path.cwd()
(root_dir / ".env").write_text("DEEPSEEK_API_KEY=test-deepseek-key\nRA_LLM_PROVIDER=deepseek\n", encoding="utf-8")
with patch("research_alpha.llm.LLMClient.chat", new=fake_chat), patch(
    "research_alpha.gui.run_app_command_streaming",
    side_effect=fake_gui_command,
):
    server = create_gui_http_server(root_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = post_json(
            base,
            "/api/ideate",
            {"direction": "科研 agent 稳定对话", "ui_scope": "draft:smoke", "provider": "deepseek", "lang": "zh"},
        )
        assert status == 200, body
        session_id = wait_for(
            lambda: next(
                (
                    int(job["created_session_id"])
                    for job in get_json(base, "/api/state?draft=1").get("jobs", [])
                    if job.get("kind") == "ideate" and job.get("created_session_id")
                ),
                0,
            ),
            "created GUI idea session",
        )
        state = get_json(base, f"/api/state?session_id={session_id}")
        assert state["chat"]["active"] is True, state["chat"]
        status, body = post_json(
            base,
            "/api/step",
            {
                "session_id": session_id,
                "ui_scope": f"session:{session_id}",
                "instruction": "继续写入当前对话，不要新建",
                "provider": "deepseek",
            },
        )
        assert status == 200, body
        state = wait_for(
            lambda: (
                data
                if any("继续写入当前对话" in message.get("text", "") for message in data["chat"]["messages"])
                else None
            )
            if (data := get_json(base, f"/api/state?session_id={session_id}"))
            else None,
            "current-session follow-up",
        )
        assert len(state["sessions"]) == 1, state["sessions"]
        status, body = post_json(
            base,
            "/api/ideate",
            {
                "direction": "这个请求带着当前 session scope，必须写回当前对话",
                "session_id": None,
                "ui_scope": f"session:{session_id}",
                "provider": "deepseek",
                "lang": "zh",
            },
        )
        assert status == 200, body
        assert body.get("routed_to_session_step") is True, body
        assert body.get("job", {}).get("kind") == "step", body
        state = wait_for(
            lambda: (
                data
                if any("当前 session scope" in message.get("text", "") for message in data["chat"]["messages"])
                else None
            )
            if (data := get_json(base, f"/api/state?session_id={session_id}"))
            else None,
            "misrouted ideate writes current session",
        )
        assert len(state["sessions"]) == 1, state["sessions"]
        status, body = post_json(
            base,
            "/api/review",
            {"session_id": session_id, "ui_scope": f"session:{session_id}", "provider": "deepseek"},
        )
        assert status == 200, body
        state = wait_for(
            lambda: (
                data
                if "审稿意见：单轮审稿认为 novelty 边界还需要收紧。" in "\n".join(message.get("text", "") for message in data["chat"]["messages"])
                else None
            )
            if (data := get_json(base, f"/api/state?session_id={session_id}"))
            else None,
            "single reviewer summary",
        )
        joined = "\n".join(message.get("text", "") for message in state["chat"]["messages"])
        assert "内部审稿 raw trace" not in joined, joined
        assert len(state["sessions"]) == 1, state["sessions"]
        status, body = post_json(
            base,
            "/api/review-loop",
            {"session_id": session_id, "ui_scope": f"session:{session_id}", "rounds": 4, "provider": "deepseek"},
        )
        assert status == 200, body
        state = wait_for(
            lambda: (
                data
                if "审稿意见：新颖性边界需要更硬。" in "\n".join(message.get("text", "") for message in data["chat"]["messages"])
                else None
            )
            if (data := get_json(base, f"/api/state?session_id={session_id}"))
            else None,
            "hidden review-loop summary",
        )
        joined = "\n".join(message.get("text", "") for message in state["chat"]["messages"])
        assert "内部长审稿指令" not in joined, joined
        assert "根据第 1/4 轮顶会审稿意见" not in joined, joined
        print(f"GUI HTTP smoke OK: session={session_id}, messages={len(state['chat']['messages'])}, covers=ideate+step+misroute+review+review-loop")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
PY
)

echo
echo "Local verification finished successfully."
echo "Temp project: $project_dir"
