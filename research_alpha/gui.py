from __future__ import annotations

import json
import os
import ipaddress
import subprocess
import sys
import threading
import webbrowser
import queue
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import re

from research_alpha.config import load_config, provider_api_key_env
from research_alpha.db import (
    count_idea_cards,
    count_pattern_cards,
    count_scored_papers,
    delete_candidate_idea,
    delete_idea_session,
    delete_user_library_paper,
    get_candidate_idea,
    get_idea_session,
    get_paper_by_id,
    init_db,
    list_candidate_ideas,
    list_frontier_papers,
    list_idea_sessions,
    list_idea_session_turns,
    list_pattern_cards,
    list_runs,
    list_top_papers,
    list_user_library_papers,
    upsert_user_library_paper,
    table_counts,
)
from research_alpha.layout import ensure_layout


TAG_ALIASES = {
    "fields": {
        "科研智能体": ["科研智能体", "research agent", "research agents", "science agent", "scientific agent"],
        "大模型智能体": ["大模型智能体", "llm agent", "llm agents", "language agent", "tool agent"],
        "多智能体协作": ["多智能体", "multi-agent", "multi agent", "agent collaboration", "协作"],
        "长上下文与记忆": ["长上下文", "long context", "memory", "记忆", "上下文缓存", "缓存压缩"],
        "多模态推理": ["多模态", "multimodal", "vision-language", "vlm", "视觉语言"],
        "计算机视觉": ["计算机视觉", "computer vision", "视觉", "目标检测", "object detection", "检测", "camouflaged", "camouflage", "伪装"],
        "AI for Science": ["ai for science", "ai4science", "科学发现", "实验闭环", "hypothesis generation"],
        "具身智能": ["具身", "embodied", "robot", "robotics", "机器人"],
        "高效训练与适配": ["高效", "efficient", "adapter", "lora", "微调", "适配"],
    },
    "hotspots": {
        "可靠评测": ["可靠评测", "可靠性", "evaluation", "benchmark", "评测", "基准"],
        "失败发现": ["失败发现", "failure", "failure mode", "失败模式", "stress test", "压力测试"],
        "可验证推理": ["可验证", "verifiable", "verification", "推理", "reasoning"],
        "记忆压缩": ["记忆压缩", "context compression", "memory compression", "缓存压缩", "compression"],
        "事实性与可控性": ["事实性", "hallucination", "faithfulness", "controllability", "可控性"],
        "实验闭环": ["实验闭环", "closed-loop", "closed loop", "lab automation", "实验自动化"],
        "安全边界": ["安全边界", "safety boundary", "safety", "risk", "风险边界"],
        "数据合成与选择": ["数据合成", "synthetic data", "data selection", "数据选择", "curation"],
        "伪装目标检测": ["伪装目标检测", "camouflaged object detection", "camouflage object detection", "camouflaged object", "伪装检测"],
    },
    "constraints": {
        "重点利用近半年优秀论文中明确写出的局限性": ["近半年", "半年", "recent limitation", "limitations", "局限"],
        "优先输出能被顶会审稿人验证的实验路径": ["可验证实验", "实验路径", "reviewer", "审稿人", "defensible"],
        "强调 prior-art 差异边界，重复则重新思考": ["prior-art", "已有工作", "有没有人做过", "重复", "novelty boundary"],
        "提取最佳论文的完整故事线而不是摘要关键词": ["完整逻辑", "故事线", "storyline", "logic line", "不是关键词"],
    },
}


SEARCH_TERMS = {
    "科研智能体": ["scientific research agents"],
    "大模型智能体": ["LLM agents", "tool use"],
    "多智能体协作": ["multi-agent collaboration"],
    "长上下文与记忆": ["long context", "memory compression"],
    "多模态推理": ["multimodal reasoning"],
    "计算机视觉": ["computer vision", "object detection"],
    "AI for Science": ["AI for Science", "hypothesis generation"],
    "具身智能": ["embodied AI", "robotics"],
    "高效训练与适配": ["efficient adaptation"],
    "可靠评测": ["evaluation", "benchmark", "reliability"],
    "失败发现": ["failure modes", "stress testing"],
    "可验证推理": ["verifiable reasoning"],
    "记忆压缩": ["memory compression", "context compression"],
    "事实性与可控性": ["faithfulness", "controllability"],
    "实验闭环": ["closed-loop experimentation"],
    "安全边界": ["safety boundary"],
    "数据合成与选择": ["data selection", "synthetic data"],
    "伪装目标检测": ["camouflaged object detection", "camouflage object segmentation", "concealed object detection"],
    "重点利用近半年优秀论文中明确写出的局限性": ["recent limitations"],
}


VENUE_GROUPS = [
    {
        "id": "core",
        "label": "AI / ML Core",
        "venues": [
            {"value": "ICLR", "label": "ICLR", "rank": "核心"},
            {"value": "NeurIPS", "label": "NeurIPS", "rank": "核心"},
            {"value": "ICML", "label": "ICML", "rank": "核心"},
            {"value": "TPAMI", "label": "TPAMI", "rank": "核心"},
            {"value": "AAAI", "label": "AAAI", "rank": "趋势辅助"},
            {"value": "IJCAI", "label": "IJCAI", "rank": "趋势辅助"},
        ],
    },
    {
        "id": "nlp",
        "label": "NLP",
        "venues": [
            {"value": "ACL", "label": "ACL", "rank": "趋势辅助"},
            {"value": "EMNLP", "label": "EMNLP", "rank": "趋势辅助"},
            {"value": "NAACL", "label": "NAACL", "rank": "趋势辅助"},
        ],
    },
    {
        "id": "vision",
        "label": "Vision",
        "venues": [
            {"value": "CVPR", "label": "CVPR", "rank": "核心"},
            {"value": "ICCV", "label": "ICCV", "rank": "核心"},
            {"value": "ECCV", "label": "ECCV", "rank": "趋势辅助"},
        ],
    },
    {
        "id": "systems",
        "label": "Systems / Data / HCI",
        "venues": [
            {"value": "SIGIR", "label": "SIGIR", "rank": "趋势辅助"},
            {"value": "KDD", "label": "KDD", "rank": "趋势辅助"},
            {"value": "WWW", "label": "WWW", "rank": "趋势辅助"},
            {"value": "CHI", "label": "CHI", "rank": "趋势辅助"},
            {"value": "CoRL", "label": "CoRL", "rank": "趋势辅助"},
            {"value": "RSS", "label": "RSS", "rank": "趋势辅助"},
        ],
    },
]


VENUE_PRESETS = [
    {"id": "core_all", "label": "核心全量", "venues": ["ICLR", "NeurIPS", "ICML", "CVPR", "ICCV", "TPAMI"]},
    {"id": "ml_core", "label": "AI/ML 核心", "venues": ["ICLR", "NeurIPS", "ICML", "TPAMI"]},
    {"id": "vision_core", "label": "视觉核心", "venues": ["CVPR", "ICCV", "TPAMI", "NeurIPS", "ICML"]},
    {"id": "aux_agents", "label": "核心+Agent趋势", "venues": ["ICLR", "NeurIPS", "ICML", "ACL", "EMNLP"]},
    {"id": "aux_frontier", "label": "核心+跨域趋势", "venues": ["ICLR", "NeurIPS", "ICML", "KDD", "SIGIR", "WWW", "CHI"]},
]


DIRECTION_PRESETS = [
    {"value": "科研智能体可靠评测", "label": "科研智能体可靠评测"},
    {"value": "多智能体协作中的可验证失败发现", "label": "多智能体可靠性"},
    {"value": "长上下文智能体的记忆、压缩与可追溯评测", "label": "长上下文记忆"},
    {"value": "AI for Science 中的假设生成与实验闭环", "label": "AI for Science"},
    {"value": "具身智能中的开放环境泛化与安全边界", "label": "具身智能"},
    {"value": "多模态模型的事实性、可控性与评测", "label": "多模态可靠性"},
]


FIELD_OPTIONS = [
    {"value": "科研智能体", "label": "科研智能体"},
    {"value": "大模型智能体", "label": "大模型智能体"},
    {"value": "多智能体协作", "label": "多智能体"},
    {"value": "长上下文与记忆", "label": "长上下文"},
    {"value": "多模态推理", "label": "多模态"},
    {"value": "AI for Science", "label": "AI4Science"},
    {"value": "具身智能", "label": "具身智能"},
    {"value": "高效训练与适配", "label": "高效适配"},
]


HOTSPOT_OPTIONS = [
    {"value": "可靠评测", "label": "可靠评测"},
    {"value": "失败发现", "label": "失败发现"},
    {"value": "可验证推理", "label": "可验证推理"},
    {"value": "记忆压缩", "label": "记忆压缩"},
    {"value": "事实性与可控性", "label": "事实可控"},
    {"value": "实验闭环", "label": "实验闭环"},
    {"value": "安全边界", "label": "安全边界"},
    {"value": "数据合成与选择", "label": "数据选择"},
]


CONSTRAINT_OPTIONS = [
    {"value": "重点利用近半年优秀论文中明确写出的局限性", "label": "近半年局限性"},
    {"value": "优先输出能被顶会审稿人验证的实验路径", "label": "可验证实验"},
    {"value": "强调 prior-art 差异边界，重复则重新思考", "label": "新颖性边界"},
    {"value": "提取最佳论文的完整故事线而不是摘要关键词", "label": "完整故事线"},
]


REMOTE_OPTIONS = [
    {"value": "openalex", "label": "OpenAlex"},
    {"value": "s2", "label": "Semantic Scholar"},
]

CORE_GOLD_GUI_VENUES = {"iclr", "neurips", "nips", "icml", "cvpr", "iccv", "tpami", "t-pami"}
GUI_JOB_RESULT_TAIL_CHARS = 8000
GUI_JOB_LIVE_TAIL_CHARS = 12000
GUI_JOB_MAX_STORED = 80
GUI_JOB_RECENT_COMPLETED_KEEP = 50
GUI_CONVERSATION_JOB_KINDS = {"ideate", "step", "review", "review-loop"}
GUI_EVIDENCE_JOB_KINDS = {"gold-build", "quality-enrich", "cleanup"}
GUI_MAX_JSON_BODY_BYTES = 128 * 1024
GUI_MAX_DIRECTION_CHARS = 4000
GUI_MAX_INSTRUCTION_CHARS = 6000
GUI_MAX_GOLD_QUERY_CHARS = 1000
GUI_MAX_USER_LIBRARY_URL_CHARS = 2000
GUI_MAX_USER_LIBRARY_TITLE_CHARS = 300
GUI_MAX_USER_LIBRARY_NOTE_CHARS = 2000
GUI_MAX_SCOPE_CHARS = 160
GUI_MAX_VENUES = 40
GUI_MAX_VENUE_CHARS = 80
GUI_USER_LIBRARY_METADATA_TIMEOUT = 4
VENUE_CANONICAL_NAMES = {
    "iclr": "ICLR",
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "icml": "ICML",
    "cvpr": "CVPR",
    "iccv": "ICCV",
    "tpami": "TPAMI",
    "t-pami": "TPAMI",
    "t_pami": "TPAMI",
    "aaai": "AAAI",
    "ijcai": "IJCAI",
    "acl": "ACL",
    "emnlp": "EMNLP",
    "naacl": "NAACL",
    "eccv": "ECCV",
    "sigir": "SIGIR",
    "kdd": "KDD",
    "www": "WWW",
    "chi": "CHI",
    "corl": "CoRL",
    "rss": "RSS",
}


PROVIDER_OPTIONS = [
    {"value": "deepseek", "label": "DeepSeek"},
    {"value": "openai", "label": "OpenAI"},
]


LANGUAGE_OPTIONS = [
    {"value": "zh", "label": "中文"},
    {"value": "en", "label": "English"},
    {"value": "auto", "label": "自动"},
]


class GuiState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: list[dict[str, Any]] = []
        self.active_session_id: int | None = None
        self.deleting_session_ids: set[int] = set()

    def add_job(
        self,
        name: str,
        target: Callable[[], dict[str, Any]],
        *,
        kind: str = "generic",
        session_id: int | None = None,
        ui_scope: str = "",
        group_id: str = "",
        pass_job: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            job = self._create_job_locked(
                name,
                kind=kind,
                session_id=session_id,
                ui_scope=ui_scope,
                group_id=group_id,
            )
        self._start_job_runner(job, target, pass_job=pass_job)
        return job

    def add_job_unless_running(
        self,
        name: str,
        target: Callable[[], dict[str, Any]],
        *,
        dedupe: Callable[[dict[str, Any]], bool],
        kind: str = "generic",
        session_id: int | None = None,
        ui_scope: str = "",
        group_id: str = "",
        pass_job: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        with self.lock:
            self._prune_jobs_locked()
            for existing in self.jobs:
                if existing.get("status") == "running" and dedupe(existing):
                    return dict(existing), False
            job = self._create_job_locked(
                name,
                kind=kind,
                session_id=session_id,
                ui_scope=ui_scope,
                group_id=group_id,
                prune=False,
            )
        self._start_job_runner(job, target, pass_job=pass_job)
        return job, True

    def add_jobs_unless_running(
        self,
        specs: list[dict[str, Any]],
        *,
        dedupe: Callable[[dict[str, Any]], bool],
    ) -> tuple[list[dict[str, Any]], bool]:
        with self.lock:
            self._prune_jobs_locked()
            for existing in self.jobs:
                if existing.get("status") == "running" and dedupe(existing):
                    return [dict(existing)], False
            jobs = [
                self._create_job_locked(
                    str(spec.get("name", "")),
                    kind=str(spec.get("kind", "generic") or "generic"),
                    session_id=spec.get("session_id"),
                    ui_scope=str(spec.get("ui_scope", "") or ""),
                    group_id=str(spec.get("group_id", "") or ""),
                    prune=False,
                )
                for spec in specs
            ]
        for job, spec in zip(jobs, specs):
            target = spec["target"]
            self._start_job_runner(job, target, pass_job=bool(spec.get("pass_job", False)))
        return jobs, True

    def _create_job_locked(
        self,
        name: str,
        *,
        kind: str = "generic",
        session_id: int | None = None,
        ui_scope: str = "",
        group_id: str = "",
        prune: bool = True,
    ) -> dict[str, Any]:
        if prune:
            self._prune_jobs_locked()
        created_at = time.time()
        job = {
            "id": max([int(item.get("id", 0) or 0) for item in self.jobs] + [0]) + 1,
            "name": name,
            "kind": kind,
            "session_id": int(session_id) if session_id else None,
            "ui_scope": ui_scope.strip() or (f"session:{int(session_id)}" if session_id else ""),
            "group_id": group_id.strip(),
            "created_session_id": None,
            "status": "running",
            "result": None,
            "error": "",
            "live_stdout": "",
            "live_stderr": "",
            "last_line": "",
            "created_at": created_at,
            "updated_at": created_at,
            "last_output_at": None,
            "completed_at": None,
        }
        self.jobs.insert(0, job)
        return job

    def _start_job_runner(
        self,
        job: dict[str, Any],
        target: Callable[[], dict[str, Any]],
        *,
        pass_job: bool = False,
    ) -> None:
        def runner() -> None:
            try:
                result = target(job) if pass_job else target()
                created_session_id = extract_created_session_id(result)
                exit_code = command_exit_code(result)
                with self.lock:
                    finished_at = time.time()
                    job["result"] = compact_job_result(result)
                    job["updated_at"] = finished_at
                    job["completed_at"] = finished_at
                    if exit_code and not (job.get("kind") == "review" and exit_code == 2):
                        job["status"] = "failed"
                        job["error"] = command_error_message(result)
                        return
                    job["status"] = "completed"
                    if created_session_id:
                        job["created_session_id"] = created_session_id
            except Exception as exc:  # pragma: no cover - surfaced through GUI
                with self.lock:
                    finished_at = time.time()
                    job["status"] = "failed"
                    job["error"] = public_gui_text(exc)
                    job["updated_at"] = finished_at
                    job["completed_at"] = finished_at

        threading.Thread(target=runner, daemon=True).start()

    def _prune_jobs_locked(self) -> None:
        if len(self.jobs) <= GUI_JOB_MAX_STORED:
            return
        running = [job for job in self.jobs if job.get("status") == "running"]
        completed = [job for job in self.jobs if job.get("status") != "running"]
        completed.sort(
            key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0),
            reverse=True,
        )
        kept_completed = completed[:GUI_JOB_RECENT_COMPLETED_KEEP]
        keep_group_ids = {
            str(job.get("group_id", "") or "").strip()
            for job in kept_completed
            if str(job.get("group_id", "") or "").strip()
        }
        if keep_group_ids:
            kept_ids = {id(job) for job in kept_completed}
            for job in completed[GUI_JOB_RECENT_COMPLETED_KEEP:]:
                if str(job.get("group_id", "") or "").strip() in keep_group_ids and id(job) not in kept_ids:
                    kept_completed.append(job)
                    kept_ids.add(id(job))
        self.jobs = running + kept_completed

    def append_job_output(self, job: dict[str, Any], stream_name: str, line: str) -> None:
        cleaned = sanitize_user_facing_text(line).strip()
        if not cleaned:
            return
        key = "live_stderr" if stream_name == "stderr" else "live_stdout"
        with self.lock:
            now = time.time()
            job[key] = (str(job.get(key, "") or "") + cleaned + "\n")[-GUI_JOB_LIVE_TAIL_CHARS:]
            job["last_line"] = cleaned[-500:]
            job["last_output_at"] = now
            job["updated_at"] = now

    def running_job(
        self,
        *,
        kind: str = "",
        ui_scope: str = "",
        session_id: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_scope = str(ui_scope or "").strip()
        normalized_kind = str(kind or "").strip()
        with self.lock:
            for job in self.jobs:
                if job.get("status") != "running":
                    continue
                if normalized_kind and job.get("kind") != normalized_kind:
                    continue
                if normalized_scope and job.get("ui_scope") != normalized_scope:
                    continue
                if session_id is not None and job.get("session_id") != int(session_id):
                    continue
                return dict(job)
        return None

    def running_jobs_for_session(self, session_id: int) -> list[dict[str, Any]]:
        target_session = int(session_id)
        target_scope = f"session:{target_session}"
        with self.lock:
            return [
                dict(job)
                for job in self.jobs
                if job.get("status") == "running"
                and (job.get("session_id") == target_session or job.get("ui_scope") == target_scope)
            ]

    def begin_session_delete(self, session_id: int) -> tuple[bool, list[dict[str, Any]]]:
        target_session = int(session_id)
        target_scope = f"session:{target_session}"
        with self.lock:
            if target_session in self.deleting_session_ids:
                return False, []
            running_jobs = [
                dict(job)
                for job in self.jobs
                if job.get("status") == "running"
                and (job.get("session_id") == target_session or job.get("ui_scope") == target_scope)
            ]
            if running_jobs:
                return False, running_jobs
            self.deleting_session_ids.add(target_session)
            return True, []

    def finish_session_delete(self, session_id: int) -> None:
        with self.lock:
            self.deleting_session_ids.discard(int(session_id))

    def is_session_deleting(self, session_id: int) -> bool:
        with self.lock:
            return int(session_id) in self.deleting_session_ids

    def running_conversation_job_for_session(self, session_id: int) -> dict[str, Any] | None:
        target_session = int(session_id)
        target_scope = f"session:{target_session}"
        with self.lock:
            for job in self.jobs:
                if job.get("status") != "running":
                    continue
                if job.get("kind") not in GUI_CONVERSATION_JOB_KINDS:
                    continue
                if job.get("session_id") == target_session or job.get("ui_scope") == target_scope:
                    return dict(job)
        return None

    def running_evidence_job_for_scope(self, ui_scope: str) -> dict[str, Any] | None:
        normalized_scope = str(ui_scope or "").strip()
        if not normalized_scope:
            return None
        with self.lock:
            for job in self.jobs:
                if job.get("status") != "running":
                    continue
                if job.get("kind") not in GUI_EVIDENCE_JOB_KINDS:
                    continue
                if job.get("ui_scope") == normalized_scope:
                    return dict(job)
        return None

    def snapshot(self, limit: int = 12) -> list[dict[str, Any]]:
        with self.lock:
            selected: list[dict[str, Any]] = []
            selected_ids: set[int] = set()

            def add(job: dict[str, Any]) -> None:
                try:
                    job_id = int(job.get("id", 0) or 0)
                except (TypeError, ValueError):
                    job_id = 0
                if not job_id or job_id in selected_ids:
                    return
                selected.append(public_job_snapshot(job))
                selected_ids.add(job_id)

            for job in self.jobs:
                if job.get("status") == "running":
                    add(job)
            for job in self.jobs:
                if len(selected) >= max(1, int(limit)):
                    break
                if job.get("status") == "running":
                    continue
                if job.get("kind") in GUI_CONVERSATION_JOB_KINDS:
                    add(job)
            for job in self.jobs:
                if len(selected) >= max(1, int(limit)):
                    break
                if job.get("kind") in GUI_CONVERSATION_JOB_KINDS:
                    continue
                add(job)
            selected_group_ids = {
                str(job.get("group_id", "") or "").strip()
                for job in selected
                if str(job.get("group_id", "") or "").strip()
            }
            if selected_group_ids:
                for job in self.jobs:
                    if str(job.get("group_id", "") or "").strip() in selected_group_ids:
                        add(job)
            return selected


def create_gui_http_server(
    root_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    state: GuiState | None = None,
) -> ThreadingHTTPServer:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    state = state or GuiState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_index_html())
                return
            if parsed.path == "/api/state":
                params = parse_qs(parsed.query)
                draft_mode = params.get("draft", ["0"])[0] == "1"
                try:
                    requested_session_id = parse_gui_state_session_id(params)
                except ValueError as exc:
                    code = "missing_direction" if str(exc) == "direction is required" else ""
                    self._send_json(gui_error_response(str(exc), status=400, code=code), status=400)
                    return
                session_id = resolve_gui_state_session_id(
                    config.db_path,
                    state,
                    requested_session_id=requested_session_id,
                    draft_mode=draft_mode,
                )
                snapshot = build_gui_snapshot(
                    config.root_dir,
                    state,
                    session_id=session_id,
                    draft_mode=draft_mode or requested_session_id is None,
                )
                self._send_json(snapshot)
                return
            self._send_json(gui_error_response("not found", status=404), status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
            except GuiRequestError as exc:
                self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                return
            except ValueError as exc:
                self._send_json(gui_error_response(str(exc), status=400), status=400)
                return
            if parsed.path == "/api/ideate":
                try:
                    direction = parse_gui_text(
                        payload.get("direction", ""),
                        name="direction",
                        required=True,
                        max_chars=GUI_MAX_DIRECTION_CHARS,
                        missing_message="direction is required",
                    )
                    context_session_id = parse_gui_int(
                        payload.get("session_id", 0),
                        name="session_id",
                        default=0,
                        minimum=0,
                    )
                    ideas = parse_gui_int(payload.get("ideas", 3), name="idea 数", default=3, minimum=1, maximum=8)
                    limit = parse_gui_int(payload.get("limit", 12), name="近期文章检索上限", default=12, minimum=4, maximum=50)
                    remote = parse_gui_choice(payload.get("remote", "openalex"), name="近期趋势源", default="openalex", allowed={"openalex", "s2"})
                    provider = parse_gui_choice(payload.get("provider", "deepseek"), name="模型", default="deepseek", allowed={"deepseek", "openai"})
                    lang = parse_gui_choice(payload.get("lang", "zh"), name="输出语言", default="zh", allowed={"zh", "en", "auto"})
                    ui_scope = parse_gui_scope(payload.get("ui_scope", "draft"), default="draft")
                except ValueError as exc:
                    code = "missing_direction" if str(exc) == "direction is required" else ""
                    self._send_json(gui_error_response(str(exc), status=400, code=code), status=400)
                    return
                try:
                    require_gui_llm_provider(config.root_dir, provider)
                except GuiRequestError as exc:
                    self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                    return
                context_messages = (
                    build_chat_snapshot(config.root_dir, session_id=context_session_id).get("messages", [])
                    if context_session_id
                    else []
                )
                brief = infer_research_brief(direction, context_messages)
                job, created = state.add_job_unless_running(
                    f"ideate: {direction}",
                    lambda job: run_app_command_streaming(
                        [
                            "ideate",
                            str(brief["brief_direction"]),
                            "--display-direction",
                            direction,
                            "--frontier-query",
                            str(brief["frontier_query"]),
                            "--ideas",
                            str(ideas),
                            "--remote",
                            remote,
                            "--limit",
                            str(limit),
                            "--session",
                            "--review-loop",
                            "--review-rounds",
                            "4",
                            "--provider",
                            provider,
                            "--lang",
                            lang,
                        ],
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="ideate",
                    ui_scope=ui_scope,
                    dedupe=lambda existing, ui_scope=ui_scope: (
                        existing.get("kind") == "ideate" and existing.get("ui_scope") == ui_scope
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            if parsed.path == "/api/step":
                try:
                    instruction = parse_gui_text(
                        payload.get("instruction", ""),
                        name="instruction",
                        required=True,
                        max_chars=GUI_MAX_INSTRUCTION_CHARS,
                        missing_message="instruction is required",
                    )
                    session_id = parse_gui_int(payload.get("session_id", 0), name="session_id", default=0, minimum=0)
                    provider = parse_gui_choice(payload.get("provider", "deepseek"), name="模型", default="deepseek", allowed={"deepseek", "openai"})
                    ui_scope = parse_gui_scope(payload.get("ui_scope", ""), default="")
                except ValueError as exc:
                    code = "missing_instruction" if str(exc) == "instruction is required" else ""
                    self._send_json(gui_error_response(str(exc), status=400, code=code), status=400)
                    return
                try:
                    require_gui_llm_provider(config.root_dir, provider)
                except GuiRequestError as exc:
                    self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                    return
                resolved_session = resolve_gui_session(config.db_path, state, session_id, ui_scope=ui_scope)
                if not resolved_session["ok"]:
                    self._send_json(gui_error_response(resolved_session["error"], status=int(resolved_session["status"]), code=str(resolved_session.get("code") or "session_error")), status=int(resolved_session["status"]))
                    return
                resolved_session_id = int(resolved_session["session_id"])
                session_scope = f"session:{resolved_session_id}"
                job, created = state.add_job_unless_running(
                    "session step",
                    lambda job: run_app_command_streaming(
                        ["step", "--session-id", str(resolved_session_id), instruction, "--provider", provider],
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="step",
                    session_id=resolved_session_id,
                    ui_scope=session_scope,
                    dedupe=lambda existing, resolved_session_id=resolved_session_id, session_scope=session_scope: (
                        existing.get("kind") in GUI_CONVERSATION_JOB_KINDS
                        and (existing.get("session_id") == resolved_session_id or existing.get("ui_scope") == session_scope)
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            if parsed.path == "/api/review":
                try:
                    session_id = parse_gui_int(payload.get("session_id", 0), name="session_id", default=0, minimum=0)
                    provider = parse_gui_choice(payload.get("provider", "deepseek"), name="模型", default="deepseek", allowed={"deepseek", "openai"})
                    ui_scope = parse_gui_scope(payload.get("ui_scope", ""), default="")
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                try:
                    require_gui_llm_provider(config.root_dir, provider)
                except GuiRequestError as exc:
                    self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                    return
                resolved_session = resolve_gui_session(config.db_path, state, session_id, ui_scope=ui_scope)
                if not resolved_session["ok"]:
                    self._send_json(gui_error_response(resolved_session["error"], status=int(resolved_session["status"]), code=str(resolved_session.get("code") or "session_error")), status=int(resolved_session["status"]))
                    return
                resolved_session_id = int(resolved_session["session_id"])
                session_scope = f"session:{resolved_session_id}"
                job, created = state.add_job_unless_running(
                    "review current idea",
                    lambda job: run_app_command_streaming(
                        ["reviewer", "--session-id", str(resolved_session_id), "--provider", provider],
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="review",
                    session_id=resolved_session_id,
                    ui_scope=session_scope,
                    dedupe=lambda existing, resolved_session_id=resolved_session_id, session_scope=session_scope: (
                        existing.get("kind") in GUI_CONVERSATION_JOB_KINDS
                        and (existing.get("session_id") == resolved_session_id or existing.get("ui_scope") == session_scope)
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            if parsed.path == "/api/review-loop":
                try:
                    session_id = parse_gui_int(payload.get("session_id", 0), name="session_id", default=0, minimum=0)
                    rounds = parse_gui_int(payload.get("rounds", 4), name="审稿循环轮数", default=4, minimum=3, maximum=5)
                    provider = parse_gui_choice(payload.get("provider", "deepseek"), name="模型", default="deepseek", allowed={"deepseek", "openai"})
                    ui_scope = parse_gui_scope(payload.get("ui_scope", ""), default="")
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                try:
                    require_gui_llm_provider(config.root_dir, provider)
                except GuiRequestError as exc:
                    self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                    return
                resolved_session = resolve_gui_session(config.db_path, state, session_id, ui_scope=ui_scope)
                if not resolved_session["ok"]:
                    self._send_json(gui_error_response(resolved_session["error"], status=int(resolved_session["status"]), code=str(resolved_session.get("code") or "session_error")), status=int(resolved_session["status"]))
                    return
                resolved_session_id = int(resolved_session["session_id"])
                session_scope = f"session:{resolved_session_id}"
                job, created = state.add_job_unless_running(
                    f"review loop {rounds} rounds",
                    lambda job: run_app_command_streaming(
                        ["review-loop", "--session-id", str(resolved_session_id), "--rounds", str(rounds), "--provider", provider],
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="review-loop",
                    session_id=resolved_session_id,
                    ui_scope=session_scope,
                    dedupe=lambda existing, resolved_session_id=resolved_session_id, session_scope=session_scope: (
                        existing.get("kind") in GUI_CONVERSATION_JOB_KINDS
                        and (existing.get("session_id") == resolved_session_id or existing.get("ui_scope") == session_scope)
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            if parsed.path == "/api/session/delete":
                try:
                    session_id = parse_gui_int(payload.get("session_id", 0), name="session_id", default=0, minimum=0)
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                if not session_id:
                    self._send_json(gui_error_response("session_id is required", status=400, code="missing_session_id"), status=400)
                    return
                if not get_idea_session(config.db_path, session_id):
                    if state.active_session_id == session_id:
                        state.active_session_id = None
                    self._send_json(gui_error_response("session not found", status=404, code="session_not_found"), status=404)
                    return
                delete_started, running_session_jobs = state.begin_session_delete(session_id)
                if running_session_jobs:
                    names = ", ".join(str(job.get("name", "") or job.get("kind", "任务")) for job in running_session_jobs[:3])
                    suffix = " 等任务" if len(running_session_jobs) > 3 else ""
                    self._send_json(
                        {
                            **gui_error_response(
                                f"这条对话仍有任务在运行：{names}{suffix}。请等待完成或刷新状态后再删除。",
                                status=409,
                            ),
                            "jobs": jobs_response(running_session_jobs),
                        },
                        status=409,
                    )
                    return
                if not delete_started:
                    self._send_json(gui_error_response("session delete is already in progress", status=409, code="session_delete_in_progress"), status=409)
                    return
                try:
                    delete_idea_session(config.db_path, session_id)
                    if state.active_session_id == session_id:
                        state.active_session_id = None
                    self._send_json({"ok": True})
                finally:
                    state.finish_session_delete(session_id)
                return
            if parsed.path == "/api/idea/delete":
                try:
                    idea_id = parse_gui_int(payload.get("idea_id", 0), name="idea_id", default=0, minimum=0)
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                if not idea_id:
                    self._send_json(gui_error_response("idea_id is required", status=400, code="missing_idea_id"), status=400)
                    return
                if not get_candidate_idea(config.db_path, idea_id):
                    self._send_json(gui_error_response("idea not found", status=404, code="idea_not_found"), status=404)
                    return
                delete_candidate_idea(config.db_path, idea_id)
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/user-library/add":
                try:
                    raw_url = parse_gui_text(
                        payload.get("url", ""),
                        name="论文链接",
                        required=True,
                        max_chars=GUI_MAX_USER_LIBRARY_URL_CHARS,
                        missing_message="论文链接不能为空",
                    )
                    url = validate_user_library_url(raw_url)
                    title = parse_gui_text(
                        payload.get("title", ""),
                        name="论文标题",
                        required=False,
                        max_chars=GUI_MAX_USER_LIBRARY_TITLE_CHARS,
                    )
                    note = parse_gui_text(
                        payload.get("note", ""),
                        name="学习备注",
                        required=False,
                        max_chars=GUI_MAX_USER_LIBRARY_NOTE_CHARS,
                    )
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400, code="invalid_user_library_link"), status=400)
                    return
                metadata = user_library_metadata_from_url(url, title=title, note=note)
                paper_id = upsert_user_library_paper(
                    config.db_path,
                    title=metadata["title"],
                    venue=metadata["venue"],
                    year=metadata["year"],
                    abstract=metadata["abstract"],
                    external_ref=url,
                )
                self._send_json({"ok": True, "paper_id": paper_id, "paper": paper_row_to_gui_card(get_user_library_paper(config.db_path, paper_id))})
                return
            if parsed.path == "/api/user-library/delete":
                try:
                    paper_id = parse_gui_int(payload.get("paper_id", 0), name="paper_id", default=0, minimum=0)
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                if not paper_id:
                    self._send_json(gui_error_response("paper_id is required", status=400, code="missing_paper_id"), status=400)
                    return
                row = get_paper_by_id(config.db_path, paper_id)
                if not row or str(row["source_kind"]).strip() != "user_library":
                    self._send_json(gui_error_response("user library paper not found", status=404, code="user_library_paper_not_found"), status=404)
                    return
                delete_user_library_paper(config.db_path, paper_id)
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/session/new":
                state.active_session_id = None
                self._send_json({"ok": True, "chat": build_draft_chat_snapshot()})
                return
            if parsed.path == "/api/gold-build":
                try:
                    venues = parse_venue_payload(payload.get("venues", ["ICLR", "NeurIPS", "ICML"]))
                    from_year = parse_gui_int(payload.get("from_year", 2024), name="起始年份", default=2024, minimum=1990, maximum=current_gui_year())
                    to_year = parse_gui_int(payload.get("to_year", from_year), name="结束年份", default=from_year, minimum=1990, maximum=current_gui_year())
                    per = parse_gui_int(payload.get("per_venue_year", 30), name="每个会议年份最多核验", default=30, minimum=2, maximum=40)
                    query = parse_gui_text(
                        payload.get("query", ""),
                        name="Gold query",
                        required=False,
                        max_chars=GUI_MAX_GOLD_QUERY_CHARS,
                    )
                    ui_scope = parse_gui_scope(payload.get("ui_scope", "draft"), default="draft")
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                if to_year < from_year:
                    self._send_json(gui_error_response("结束年份不能早于起始年份", status=400, code="invalid_year_range"), status=400)
                    return
                try:
                    extractive_genomes = parse_gui_bool(
                        payload.get("extractive_genomes", False),
                        name="是否生成可审计逻辑卡",
                        default=False,
                    )
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                if not venues:
                    self._send_json(gui_error_response("select at least one venue", status=400, code="missing_venues"), status=400)
                    return
                core_venues = core_gold_venues(venues)
                auxiliary_venues = auxiliary_gold_venues(venues)
                if not core_venues:
                    self._send_json(
                        gui_error_response(
                            (
                                "核心优秀论文库只接受 ICLR、NeurIPS/NIPS、ICML、CVPR、ICCV、TPAMI。"
                                f" 已选择的辅助来源仅作趋势参考：{', '.join(auxiliary_venues)}。"
                            ),
                            status=400,
                            code="no_core_gold_venues",
                        ),
                        status=400,
                    )
                    return
                remote_groups = split_gold_venues_by_remote(core_venues)
                metadata_only_venues = metadata_only_gold_venues(remote_groups)
                source_plan = describe_gold_source_plan(remote_groups)
                group_id = f"gold-build:{ui_scope}:{int(time.time() * 1000)}"
                specs = []
                for remote, remote_venues in remote_groups.items():
                    source_label = {"icml_awards": "ICML 官方奖项", "neurips_awards": "NeurIPS 官方奖项", "cvf_awards": "CVF 官方奖项", "openreview": "OpenReview Oral", "openalex": "OpenAlex 元数据"}.get(remote, remote)
                    specs.append(
                        {
                            "name": f"gold-build {source_label}: " + ", ".join(remote_venues),
                            "target": lambda job, remote=remote, remote_venues=remote_venues: run_app_command_streaming(
                                [
                                    "gold-build",
                                    "--remote",
                                    remote,
                                    "--venues",
                                    ",".join(remote_venues),
                                    "--from-year",
                                    str(from_year),
                                    "--to-year",
                                    str(to_year),
                                    "--per-venue-year",
                                    str(per),
                                ]
                                + (["--query", query] if query else [])
                                + (["--extractive-genomes"] if extractive_genomes else []),
                                config.root_dir,
                                on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                            ),
                            "kind": "gold-build",
                            "ui_scope": ui_scope,
                            "group_id": group_id,
                            "pass_job": True,
                        }
                    )
                jobs, created = state.add_jobs_unless_running(
                    specs,
                    dedupe=lambda existing, ui_scope=ui_scope: (
                        existing.get("kind") in GUI_EVIDENCE_JOB_KINDS and existing.get("ui_scope") == ui_scope
                    ),
                )
                if not created:
                    self._send_json(
                        {
                            "jobs": jobs_response(jobs),
                            "job": job_response(jobs[0] if jobs else None),
                            "deduped": True,
                            "auxiliary_venues": auxiliary_venues,
                            "metadata_only_venues": metadata_only_venues,
                            "source_plan": source_plan,
                        }
                    )
                    return
                self._send_json({"jobs": jobs_response(jobs), "job": job_response(jobs[0] if jobs else None), "auxiliary_venues": auxiliary_venues, "metadata_only_venues": metadata_only_venues, "source_plan": source_plan})
                return
            if parsed.path == "/api/quality-enrich":
                try:
                    limit = parse_gui_int(payload.get("limit", 30), name="质量核验数量", default=30, minimum=1, maximum=80)
                    provider = parse_gui_choice(payload.get("provider", "deepseek"), name="模型", default="deepseek", allowed={"deepseek", "openai"})
                    ui_scope = parse_gui_scope(payload.get("ui_scope", "draft"), default="draft")
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                try:
                    require_gui_llm_provider(config.root_dir, provider)
                except GuiRequestError as exc:
                    self._send_json(gui_error_response(str(exc), status=exc.status, code=exc.code), status=exc.status)
                    return
                job, created = state.add_job_unless_running(
                    "quality-enrich: metadata audit",
                    lambda job: run_app_command_streaming(
                        [
                            "quality-enrich",
                            "--limit",
                            str(max(1, limit)),
                            "--apply",
                            "--provider",
                            provider,
                        ],
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="quality-enrich",
                    ui_scope=ui_scope,
                    dedupe=lambda existing, ui_scope=ui_scope: (
                        existing.get("kind") in GUI_EVIDENCE_JOB_KINDS and existing.get("ui_scope") == ui_scope
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            if parsed.path == "/api/cleanup":
                try:
                    apply_cleanup = parse_gui_bool(payload.get("apply", False), name="是否执行清理", default=False)
                    ui_scope = parse_gui_scope(payload.get("ui_scope", "draft"), default="draft")
                except ValueError as exc:
                    self._send_json(gui_error_response(str(exc), status=400), status=400)
                    return
                job, created = state.add_job_unless_running(
                    "cleanup evidence" + (" apply" if apply_cleanup else " dry-run"),
                    lambda job: run_app_command_streaming(
                        ["cleanup"] + (["--apply"] if apply_cleanup else []),
                        config.root_dir,
                        on_output=lambda stream, line, job=job: state.append_job_output(job, stream, line),
                    ),
                    kind="cleanup",
                    ui_scope=ui_scope,
                    dedupe=lambda existing, ui_scope=ui_scope: (
                        existing.get("kind") in GUI_EVIDENCE_JOB_KINDS and existing.get("ui_scope") == ui_scope
                    ),
                    pass_job=True,
                )
                if not created:
                    self._send_json({"job": job_response(job), "deduped": True})
                    return
                self._send_json({"job": job_response(job)})
                return
            self._send_json(gui_error_response("not found", status=404), status=404)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError as exc:
                raise ValueError("Content-Length 必须是整数") from exc
            if length <= 0:
                return {}
            if length > GUI_MAX_JSON_BODY_BYTES:
                raise GuiRequestError(
                    f"请求体不能超过 {GUI_MAX_JSON_BODY_BYTES} bytes",
                    status=413,
                    code="request_too_large",
                )
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"请求 JSON 解析失败：{exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON object")
            return payload

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            write_http_response(self, status=status, content_type="application/json; charset=utf-8", body=body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            write_http_response(self, status=200, content_type="text/html; charset=utf-8", body=body)

    return ThreadingHTTPServer((host, int(port)), Handler)


def run_gui(root_dir: Path, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    server = create_gui_http_server(root_dir, host=host, port=port)
    url = f"http://{host}:{int(port)}"
    print(f"Research Alpha GUI: {url}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped GUI.")
    finally:
        server.server_close()
    return 0


def write_http_response(handler: Any, *, status: int, content_type: str, body: bytes) -> bool:
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return False


def current_gui_year() -> int:
    return date.today().year


def validate_user_library_url(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("论文链接只支持 http 或 https")
    if not parsed.netloc or "@" in parsed.netloc:
        raise ValueError("论文链接格式不正确")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("论文链接缺少域名")
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        raise ValueError("论文链接不能指向本机或内网地址")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast):
        raise ValueError("论文链接不能指向本机或内网地址")
    red_flags = ("token=", "api_key=", "apikey=", "access_token=", "secret=", "password=")
    if any(flag in parsed.query.lower() for flag in red_flags):
        raise ValueError("论文链接不能包含 token、key 或密码参数")
    return parsed.geturl()


def arxiv_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host.endswith("arxiv.org"):
        return ""
    match = re.search(r"/(?:abs|pdf|html)/([^?#]+)", parsed.path)
    return normalize_arxiv_id_from_path(match.group(1) if match else "")


def normalize_arxiv_id_from_path(value: str) -> str:
    candidate = unquote(str(value or "").strip()).strip("/")
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"v[0-9]+$", "", candidate, flags=re.IGNORECASE)
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}$", candidate):
        return candidate
    if re.match(r"^[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7}$", candidate, re.IGNORECASE):
        return candidate
    return ""


def doi_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path or "")
    if host in {"doi.org", "dx.doi.org"}:
        candidate = path.lstrip("/")
    else:
        match = re.search(r"\b(10\.\d{4,9}/[^\s?#]+)", unquote(url), re.IGNORECASE)
        candidate = match.group(1) if match else ""
    candidate = candidate.strip().rstrip(".,;)")
    return candidate if re.match(r"^10\.\d{4,9}/\S+$", candidate, re.IGNORECASE) else ""


def openreview_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host.endswith("openreview.net"):
        return ""
    query = parse_qs(parsed.query)
    for key in ("id", "forum"):
        values = query.get(key, [])
        if values and str(values[0]).strip():
            return str(values[0]).strip()
    match = re.search(r"/(?:pdf|forum|attachment)\?id=([^&#]+)", url)
    return unquote(match.group(1)).strip() if match else ""


def normalize_arxiv_atom_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def year_from_arxiv_id(arxiv_id: str) -> int:
    value = str(arxiv_id or "").strip()
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}$", value):
        prefix_year = int(value[:2])
        return 2000 + prefix_year if prefix_year <= 80 else 1900 + prefix_year
    legacy = re.match(r"^[a-z-]+(?:\.[A-Z]{2})?/([0-9]{2})[0-9]{5}$", value, re.IGNORECASE)
    if not legacy:
        return current_gui_year()
    prefix_year = int(legacy.group(1))
    return 2000 + prefix_year if prefix_year <= 80 else 1900 + prefix_year


def fetch_arxiv_metadata(arxiv_id: str, *, timeout: int = GUI_USER_LIBRARY_METADATA_TIMEOUT) -> dict[str, Any]:
    if not arxiv_id:
        return {}
    query_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    request = Request(query_url, headers={"User-Agent": "ResearchAlpha/0.1 (user-library metadata)"})
    with urlopen(request, timeout=max(1, int(timeout))) as response:
        raw = response.read(1024 * 1024)
    root = ET.fromstring(raw)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return {}
    title = normalize_arxiv_atom_text(entry.findtext("atom:title", default="", namespaces=ns))
    summary = normalize_arxiv_atom_text(entry.findtext("atom:summary", default="", namespaces=ns))
    published = normalize_arxiv_atom_text(entry.findtext("atom:published", default="", namespaces=ns))
    year = year_from_arxiv_id(arxiv_id)
    if re.match(r"^[0-9]{4}-", published):
        try:
            year = int(published[:4])
        except ValueError:
            pass
    result = {
        "title": title,
        "abstract": summary,
        "year": year,
        "metadata_source": "arxiv_atom",
    }
    return {key: value for key, value in result.items() if value}


def user_library_metadata_from_url(url: str, *, title: str = "", note: str = "") -> dict[str, Any]:
    arxiv_id = arxiv_id_from_url(url)
    doi = doi_from_url(url)
    openreview_id = openreview_id_from_url(url)
    fetched: dict[str, Any] = {}
    if arxiv_id:
        try:
            fetched = fetch_arxiv_metadata(arxiv_id)
        except Exception:
            fetched = {}
    parsed = urlparse(url)
    fallback_title = parsed.netloc
    if arxiv_id:
        fallback_title = f"arXiv:{arxiv_id}"
    elif doi:
        fallback_title = f"DOI:{doi}"
    elif openreview_id:
        fallback_title = f"OpenReview:{openreview_id}"
    cleaned_title = title.strip() or str(fetched.get("title", "")).strip() or fallback_title
    year = int(fetched.get("year", 0) or 0) or (year_from_arxiv_id(arxiv_id) if arxiv_id else current_gui_year())
    fetched_abstract = str(fetched.get("abstract", "")).strip()
    ref_note = ""
    venue = "User library"
    metadata_source = str(fetched.get("metadata_source", "")).strip()
    if arxiv_id:
        venue = "arXiv"
    elif doi:
        venue = "DOI"
        metadata_source = "doi_url"
        ref_note = f"DOI: {doi}"
    elif openreview_id:
        venue = "OpenReview"
        metadata_source = "openreview_url"
        ref_note = f"OpenReview id: {openreview_id}"
    abstract_parts = [part for part in [note.strip(), ref_note, fetched_abstract] if part]
    return {
        "title": cleaned_title,
        "venue": venue,
        "year": year,
        "abstract": "\n\n".join(abstract_parts),
        "metadata_source": metadata_source,
    }


def get_user_library_paper(db_path: Path, paper_id: int) -> Any:
    row = get_paper_by_id(db_path, int(paper_id))
    if not row:
        raise ValueError("user library paper was not saved")
    return row


def redact_local_paths(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"(?m)^(\s*(?:Manifest|Report|Dossier|Output|Database|Env example|Demo seeds):\s*)(/[^ \n\r]+)", r"\1[local path]", text)
    text = re.sub(r"(?m)^(\s*Generated run artifact:\s*)(/[^ \n\r]+)", r"\1[local path]", text)
    text = re.sub(r"/(?:Users|private|var|tmp)/[^ \n\r\"'<>]+", "[local path]", text)
    return text


def redact_secrets(value: object) -> str:
    text = str(value or "")
    key_name = r"(?:[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD))"
    text = re.sub(
        rf"(?i)\b({key_name})\s*([=:])\s*([^\s\"'<>]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{' ' if match.group(2) == ':' else ''}[redacted]",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{6,}\b", "sk-[redacted]", text)
    return text


def compact_text_tail(value: Any, limit: int, *, public: bool = False) -> str:
    text = sanitize_user_facing_text(str(value or ""))
    if public:
        text = redact_secrets(redact_local_paths(text))
    max_chars = max(1, int(limit))
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def compact_job_result(result: Any, *, public: bool = False, omit_streams: bool = False) -> Any:
    if not isinstance(result, dict):
        return result
    if public:
        compacted: dict[str, Any] = {}
        if "exit_code" in result:
            compacted["exit_code"] = command_exit_code(result)
        if omit_streams:
            return compacted
        for key in ("stdout", "stderr"):
            if key in result:
                compacted[key] = compact_text_tail(result.get(key, ""), GUI_JOB_RESULT_TAIL_CHARS, public=True)
        return compacted
    compacted = dict(result)
    for key in ("stdout", "stderr"):
        if key in compacted:
            compacted[key] = compact_text_tail(compacted.get(key, ""), GUI_JOB_RESULT_TAIL_CHARS, public=public)
    return compacted


def json_line_from_text(text: str, label: str) -> dict[str, Any] | None:
    escaped = re.escape(label)
    match = re.search(rf"{escaped}:\s*(\{{[^\n]+\}})", str(text or ""))
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def first_line_match(text: str, pattern: str) -> str:
    match = re.search(pattern, str(text or ""), re.IGNORECASE)
    return str(match.group(1)).strip() if match else ""


def conversation_result_text(result: Any, error: str = "") -> str:
    if not isinstance(result, dict):
        return str(error or "").strip()
    return "\n".join(
        str(result.get(key, "") or "").strip()
        for key in ("stdout", "stderr")
        if str(result.get(key, "") or "").strip()
    ).strip() or str(error or "").strip()


def build_public_conversation_summary(job: dict[str, Any]) -> dict[str, Any]:
    kind = str(job.get("kind", "") or "")
    result = job.get("result")
    text = conversation_result_text(result, str(job.get("error", "") or ""))
    summary: dict[str, Any] = {
        "kind": kind,
        "status": str(job.get("status", "") or ""),
        "exit_code": command_exit_code(result),
    }
    if kind == "ideate":
        idea_summary = json_line_from_text(text, "Idea-session summary JSON")
        failure = json_line_from_text(text, "Idea-generation failure JSON")
        if idea_summary:
            auto_loop = idea_summary.get("auto_review_loop") if isinstance(idea_summary, dict) else None
            auto_loop_status = str(auto_loop.get("status", "")).strip() if isinstance(auto_loop, dict) else ""
            if auto_loop_status == "completed":
                summary["type"] = "idea_session_review_ready"
            elif auto_loop_status:
                summary["type"] = "idea_session_review_incomplete"
            else:
                summary["type"] = "idea_session_created"
            summary["idea_session"] = sanitize_public_payload(idea_summary)
        elif failure:
            summary["type"] = "idea_generation_not_ready"
            summary["failure"] = sanitize_public_payload(failure)
        elif job.get("created_session_id"):
            summary["type"] = "idea_session_created"
            summary["idea_session"] = {"session_id": int(job.get("created_session_id") or 0)}
        else:
            summary["type"] = "idea_generation_not_ready"
            summary["failure"] = build_public_ideate_fallback_failure(text, command_exit_code(result))
    elif kind == "step":
        failure = json_line_from_text(text, "Session-step failure JSON")
        memory = json_line_from_text(text, "Session memory JSON")
        decision = first_line_match(text, r"Turn\s+\d+\s+decision:\s*([^\n]+)")
        revised_idea = first_line_match(text, r"Revised idea:\s*([^\n]+)")
        current_best = first_line_match(text, r"Current best:\s*([^\n]+)")
        trend_alignment = first_line_match(text, r"Trend check:\s*([^\n]+)")
        dossier = first_line_match(text, r"Dossier:\s*([^\n]+)")
        has_step_result = bool(decision or revised_idea or current_best or trend_alignment or dossier or memory)
        summary["type"] = "session_step_failed" if failure or not has_step_result else "session_step_completed"
        if failure:
            summary["failure"] = sanitize_public_payload(failure)
        elif not has_step_result:
            summary["failure"] = build_public_session_step_fallback_failure(command_exit_code(result))
        summary["decision"] = decision
        summary["revised_idea"] = revised_idea
        summary["current_best"] = current_best
        summary["trend_alignment"] = trend_alignment
        summary["dossier"] = dossier
        if memory:
            summary["memory"] = sanitize_public_payload(memory)
    elif kind == "review":
        failure = json_line_from_text(text, "Reviewer failure JSON")
        decision = first_line_match(text, r"Reviewer decision:\s*([^\n]+)") or first_line_match(text, r"Decision:\s*([^\n]+)")
        review_summary = first_line_match(text, r"Summary:\s*([^\n]+)")
        review_path = first_line_match(text, r"JSON:\s*([^\n]+)") or first_line_match(text, r"Saved critique #\d+:\s*([^\n]+)")
        has_review_result = bool(decision or review_summary or review_path)
        summary["type"] = "reviewer_failed" if failure or not has_review_result else "review_completed"
        if failure:
            summary["failure"] = sanitize_public_payload(failure)
        elif not has_review_result:
            summary["failure"] = build_public_reviewer_fallback_failure(command_exit_code(result))
        summary["decision"] = decision
        summary["summary"] = review_summary
        summary["review_path"] = review_path
    elif kind == "review-loop":
        loop_summary = json_line_from_text(text, "Review-loop summary JSON")
        loop_failure = json_line_from_text(text, "Review-loop failure JSON")
        if loop_summary and str(job.get("status", "") or "") == "completed" and command_exit_code(result) == 0:
            summary["type"] = "review_loop_completed"
            summary["review_loop"] = sanitize_public_payload(
                {
                    "session_id": loop_summary.get("session_id"),
                    "rounds_completed": loop_summary.get("rounds_completed"),
                    "requested_rounds": loop_summary.get("requested_rounds"),
                    "review_count": loop_summary.get("review_count"),
                    "final_idea": loop_summary.get("final_idea"),
                }
            )
        else:
            summary["type"] = "review_loop_failed"
            summary["failure"] = sanitize_public_payload(loop_failure) if loop_failure else build_public_review_loop_fallback_failure(command_exit_code(result))
            loop_source = loop_failure if loop_failure else loop_summary
            if loop_source:
                summary["review_loop"] = sanitize_public_payload(
                    {
                        "session_id": loop_source.get("session_id"),
                        "rounds_completed": loop_source.get("rounds_completed"),
                        "requested_rounds": loop_source.get("requested_rounds"),
                        "review_count": loop_source.get("review_count"),
                        "final_idea": loop_source.get("final_idea"),
                        "state_policy": loop_source.get("state_policy"),
                        "recovery_strategy": loop_source.get("recovery_strategy"),
                        "safe_resume_command": loop_source.get("safe_resume_command"),
                        "next_action": loop_source.get("next_action"),
                    }
                )
    else:
        summary["type"] = "conversation_job"
    return sanitize_public_payload({key: value for key, value in summary.items() if value not in ("", None, [], {})})


def evidence_result_text(job: dict[str, Any]) -> str:
    return conversation_result_text(job.get("result"), str(job.get("error", "") or ""))


def public_summary_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def parse_gold_build_summary_text(text: str) -> dict[str, Any] | None:
    payload = json_line_from_text(text, "Gold-build summary JSON")
    if payload:
        return payload
    summary_match = re.search(
        r"Gold-build summary:\s*fetched=(\d+);\s*candidate_imported=(\d+);\s*excellent_imported=(\d+);\s*poster_skipped=(\d+);\s*openreview_non_excellent_skipped=(\d+);\s*unqualified_skipped=(\d+);\s*failures=(\d+)",
        str(text or ""),
    )
    imported_match = re.search(r"优秀论文库 imported/updated (\d+) records", str(text or ""))
    if not summary_match and not imported_match:
        return None
    return {
        "fetched": int(summary_match.group(1)) if summary_match else None,
        "candidate_imported": int(summary_match.group(2)) if summary_match else None,
        "excellent_imported": int(summary_match.group(3)) if summary_match else int(imported_match.group(1)),
        "poster_skipped": int(summary_match.group(4)) if summary_match else None,
        "openreview_non_excellent_skipped": int(summary_match.group(5)) if summary_match else None,
        "unqualified_skipped": int(summary_match.group(6)) if summary_match else None,
        "failures": int(summary_match.group(7)) if summary_match else None,
        "source_failures": [],
        "source_empty_results": [],
        "next_actions": [],
    }


def public_source_issue(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    issue = {
        "year": public_summary_int(item.get("year")),
        "venue": compact_text_tail(item.get("venue", ""), 80, public=True),
        "remote": compact_text_tail(item.get("remote", ""), 80, public=True),
        "error": compact_text_tail(item.get("error", ""), 220, public=True),
        "recommended_action": compact_text_tail(item.get("recommended_action", ""), 260, public=True),
    }
    return {key: value for key, value in issue.items() if value not in ("", None, [], {})}


def public_next_action(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    action = {
        "label": compact_text_tail(item.get("label", ""), 80, public=True),
        "reason": compact_text_tail(item.get("reason", ""), 260, public=True),
        "command": compact_text_tail(item.get("command", ""), 320, public=True),
    }
    return {key: value for key, value in action.items() if value not in ("", None, [], {})}


def build_public_evidence_failure_summary(job: dict[str, Any]) -> dict[str, Any]:
    kind = str(job.get("kind", "") or "")
    text = evidence_result_text(job)
    base: dict[str, Any] = {
        "kind": kind,
        "status": str(job.get("status", "") or ""),
        "reason_code": "evidence_job_failed",
        "hint": "证据任务失败；可重试或缩小检索范围。",
        "next_step": "调整检索设置后重试。",
    }
    if kind == "gold-build":
        summary = parse_gold_build_summary_text(text) or {}
        source_failures = [
            public_source_issue(item)
            for item in (summary.get("source_failures") if isinstance(summary.get("source_failures"), list) else [])
        ]
        source_failures = [item for item in source_failures if item][:3]
        empty_results = [
            public_source_issue(item)
            for item in (summary.get("source_empty_results") if isinstance(summary.get("source_empty_results"), list) else [])
        ]
        empty_results = [item for item in empty_results if item][:4]
        next_actions = [
            public_next_action(item)
            for item in (summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else [])
        ]
        next_actions = [item for item in next_actions if item][:4]
        failure_count = public_summary_int(summary.get("failures"))
        candidate_imported = public_summary_int(summary.get("candidate_imported"))
        excellent_imported = public_summary_int(summary.get("excellent_imported"))
        if source_failures or failure_count:
            reason_code = "source_failure"
            hint = "部分检索源失败；建议重试失败源，或收窄年份/会议范围。"
        elif empty_results:
            reason_code = "source_empty_results"
            hint = "检索源请求成功但没有返回论文；建议更换来源或放宽主题过滤。"
        elif (candidate_imported or 0) > 0 and (excellent_imported or 0) == 0:
            reason_code = "zero_gold_imported"
            hint = "有趋势论文写入，但没有记录通过 Best/Nominee/Oral 等核心 Gold 标准。"
        else:
            reason_code = "gold_build_failed"
            hint = "核心库扩充没有完成；建议调整检索设置后重试。"
        base.update(
            {
                "reason_code": reason_code,
                "hint": hint,
                "next_step": next_actions[0].get("label", "") if next_actions else "调整检索设置后重试。",
                "fetched": public_summary_int(summary.get("fetched")),
                "candidate_imported": candidate_imported,
                "excellent_imported": excellent_imported,
                "poster_skipped": public_summary_int(summary.get("poster_skipped")),
                "openreview_non_excellent_skipped": public_summary_int(summary.get("openreview_non_excellent_skipped")),
                "unqualified_skipped": public_summary_int(summary.get("unqualified_skipped")),
                "failures": failure_count,
                "source_failures": source_failures,
                "source_empty_results": empty_results,
                "next_actions": next_actions,
            }
        )
    elif kind == "quality-enrich":
        base.update(
            {
                "reason_code": "quality_enrich_failed",
                "hint": "质量标签核验失败；趋势论文仍保留在趋势区，不会进入 Gold 标准。",
                "next_step": "重新核验质量标签。",
            }
        )
    elif kind == "cleanup":
        base.update(
            {
                "reason_code": "cleanup_failed",
                "hint": "证据上下文清理失败；现有证据未被自动降级或删除。",
                "next_step": "调整检索设置或稍后重试清理。",
            }
        )
    return sanitize_public_payload({key: value for key, value in base.items() if value not in ("", None, [], {})})


def build_public_ideate_fallback_failure(text: str, exit_code: int) -> dict[str, Any]:
    cleaned = public_gui_text(str(text or "")).strip()
    lower = cleaned.lower()
    if "current-hotspot evidence is not ready" in lower:
        primary = "frontier_evidence_not_ready"
        reason = "当前方向的近期趋势证据不足，系统没有写入新 idea。"
        next_action = "补充与当前方向匹配的近期趋势论文，然后重新生成。"
    elif "strict evidence is not ready" in lower or "strict idea generation is not ready" in lower:
        primary = "gold_evidence_not_ready"
        reason = "核心优秀论文、Genome 或 Pattern 严格证据尚未就绪，系统没有写入新 idea。"
        next_action = "补充核心优秀论文、Genome 逻辑卡和 Pattern 卡后重新生成。"
    elif exit_code:
        primary = "idea_generation_not_ready"
        reason = "idea 生成任务失败，系统没有写入新对话。"
        next_action = "查看失败原因，补证据或收窄研究方向后重新生成。"
    else:
        primary = "idea_generation_not_ready"
        reason = "idea 生成任务结束但没有保存新对话；系统已按未生成处理。"
        next_action = "重新生成前先确认核心优秀论文、Genome、Pattern 和近期趋势证据都已就绪。"
    return {
        "status": "idea_generation_not_ready",
        "primary_reason": primary,
        "requested": 0,
        "accepted": 0,
        "rejected_count": 0,
        "reason": reason,
        "next_action": next_action,
    }


def build_public_session_step_fallback_failure(exit_code: int) -> dict[str, Any]:
    return {
        "status": "session_step_failed",
        "reason_code": "session_step_failed",
        "reason": "本轮追问任务结束但没有返回可写入会话的结构化修改结果；系统已按未写入处理。",
        "exit_code": int(exit_code),
        "next_action": "请重试、缩短指令，或先补足当前会话的 Pattern/Genome 证据。",
    }


def build_public_reviewer_fallback_failure(exit_code: int) -> dict[str, Any]:
    return {
        "status": "reviewer_failed",
        "reason_code": "reviewer_failed",
        "reason": "审稿任务结束但没有返回可保存的审稿结论；系统已按未写入处理。",
        "exit_code": int(exit_code),
        "next_action": "请重试、缩短当前 idea，或先补足核心证据上下文。",
    }


def build_public_review_loop_fallback_failure(exit_code: int) -> dict[str, Any]:
    return {
        "status": "review_loop_failed",
        "reason_code": "review_loop_failed",
        "reason": "审稿循环没有完成，当前 idea 未被最终覆盖。",
        "exit_code": int(exit_code),
        "next_action": "请重试循环审稿；如果连续失败，先补足 Pattern/Genome/Gold 证据或缩短当前 idea。",
    }


def sanitize_public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_public_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_payload(item) for item in value]
    if isinstance(value, str):
        return compact_text_tail(value, GUI_JOB_RESULT_TAIL_CHARS, public=True)
    return value


def public_job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    kind = str(job.get("kind", "") or "")
    conversation_job = kind in GUI_CONVERSATION_JOB_KINDS
    snapshot: dict[str, Any] = {
        "id": job.get("id"),
        "name": compact_text_tail(job.get("name", ""), GUI_MAX_SCOPE_CHARS, public=True),
        "kind": compact_text_tail(kind, GUI_MAX_SCOPE_CHARS, public=True),
        "session_id": job.get("session_id"),
        "ui_scope": compact_text_tail(job.get("ui_scope", ""), GUI_MAX_SCOPE_CHARS, public=True),
        "group_id": compact_text_tail(job.get("group_id", ""), GUI_MAX_SCOPE_CHARS, public=True),
        "created_session_id": job.get("created_session_id"),
        "status": str(job.get("status", "") or ""),
        "result": compact_job_result(job.get("result"), public=True, omit_streams=conversation_job),
        "error": "",
        "live_stdout": "",
        "live_stderr": "",
        "last_line": "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "last_output_at": job.get("last_output_at"),
        "completed_at": job.get("completed_at"),
    }
    for key in ("name", "kind", "ui_scope", "group_id"):
        snapshot[key] = str(snapshot.get(key, "") or "")
    if conversation_job:
        snapshot["result_summary"] = build_public_conversation_summary(job)
        if job.get("error") and not (
            isinstance(snapshot.get("result_summary"), dict)
            and snapshot["result_summary"].get("failure")
        ):
            snapshot["error"] = compact_text_tail(job.get("error", ""), 280, public=True)
        else:
            snapshot["error"] = ""
    else:
        snapshot["live_stdout"] = compact_text_tail(job.get("live_stdout", ""), GUI_JOB_LIVE_TAIL_CHARS, public=True)
        snapshot["live_stderr"] = compact_text_tail(job.get("live_stderr", ""), GUI_JOB_LIVE_TAIL_CHARS, public=True)
        snapshot["last_line"] = compact_text_tail(job.get("last_line", ""), 500, public=True)
        if job.get("error"):
            snapshot["error"] = compact_text_tail(job.get("error", ""), 1800, public=True)
    return snapshot


def job_response(job: dict[str, Any] | None) -> dict[str, Any] | None:
    return public_job_snapshot(job) if job else None


def jobs_response(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [public_job_snapshot(job) for job in jobs]


class GuiRequestError(ValueError):
    def __init__(self, message: str, *, status: int = 400, code: str = "request_error") -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code or "request_error")


def gui_error_response(message: Any, *, status: int = 400, code: str = "") -> dict[str, Any]:
    text = public_gui_text(str(message or "request failed"))
    resolved_code = code or {
        400: "validation_error",
        404: "not_found",
        409: "conflict",
    }.get(int(status), "request_error")
    retryable_by_code = {
        "conflict": True,
        "session_delete_in_progress": True,
        "request_error": True,
        "session_error": True,
    }
    retryable = bool(retryable_by_code.get(resolved_code, int(status) in {408, 409, 429, 500, 502, 503, 504}))
    next_actions = {
        "missing_direction": "请输入研究方向后重试。",
        "missing_instruction": "请输入追问或修改要求后重试。",
        "missing_session_id": "从左侧历史重新选择对话，或新建一个对话。",
        "session_not_found": "从左侧历史重新选择，或新建对话。",
        "session_scope_mismatch": "刷新状态后重新选择当前对话。",
        "missing_idea_id": "刷新证据库后重新选择要删除的 idea。",
        "idea_not_found": "刷新证据库后确认该 idea 是否仍存在。",
        "missing_paper_id": "刷新用户论文库后重新选择论文。",
        "user_library_paper_not_found": "刷新用户论文库后重新选择论文。",
        "invalid_user_library_link": "请输入 arXiv、DOI 或 OpenReview 链接。",
        "request_too_large": "缩短输入内容后重试。",
        "invalid_year_range": "调整年份范围后重试。",
        "missing_venues": "至少选择一个核心会议后重试。",
        "no_core_gold_venues": "选择 ICLR、NeurIPS/NIPS、ICML、CVPR、ICCV 或 TPAMI 后重试。",
        "missing_provider_key": "先配置当前模型的 API key，或切换到已配置的模型。",
        "validation_error": "按提示修正输入后重试。",
        "not_found": "刷新状态后重新选择对象。",
        "conflict": "等待当前任务完成，或刷新状态后重试。",
        "session_delete_in_progress": "等待删除完成后刷新状态。",
        "session_error": "刷新状态后重新选择当前对话。",
        "request_error": "刷新状态后重试；如果连续失败，检查本地服务是否仍在运行。",
    }
    return {
        "ok": False,
        "error": text,
        "code": resolved_code,
        "status": int(status),
        "retryable": retryable,
        "next_action": next_actions.get(resolved_code, next_actions["request_error"]),
    }


def require_gui_llm_provider(root_dir: Path, provider: str) -> None:
    fresh_config = load_config(root_dir)
    selected_provider = str(provider or fresh_config.llm.provider or "").strip().lower()
    env_name = provider_api_key_env(selected_provider)
    generic_key = os.environ.get("RA_LLM_API_KEY", "").strip()
    provider_key = os.environ.get(env_name, "").strip()
    if provider_key or generic_key:
        return
    raise GuiRequestError(
        f"模型 {selected_provider or 'unknown'} 未配置 API key。请先在顶部模型设置中配置 {env_name}，或设置 RA_LLM_API_KEY。",
        status=400,
        code="missing_provider_key",
    )


def parse_gui_state_session_id(params: dict[str, list[str]]) -> int | None:
    values = params.get("session_id", [])
    if not values:
        return None
    raw = str(values[0]).strip()
    if raw == "":
        return None
    return parse_gui_int(raw, name="session_id", default=0, minimum=1)


def resolve_gui_state_session_id(
    db_path: Path,
    state: GuiState,
    *,
    requested_session_id: int | None,
    draft_mode: bool,
) -> int | None:
    if draft_mode or requested_session_id is None:
        return None
    if get_idea_session(db_path, int(requested_session_id)):
        state.active_session_id = int(requested_session_id)
        return int(requested_session_id)
    if state.active_session_id == int(requested_session_id):
        state.active_session_id = None
    return int(requested_session_id)


def parse_gui_int(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None or value == "":
        parsed = int(default)
    elif isinstance(value, bool):
        raise ValueError(f"{name}必须是整数")
    else:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是整数") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name}不能小于 {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name}不能大于 {maximum}")
    return parsed


def parse_gui_bool(value: Any, *, name: str, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{name}必须是布尔值")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"{name}必须是布尔值")


def parse_gui_text(
    value: Any,
    *,
    name: str,
    required: bool = False,
    max_chars: int = 1000,
    missing_message: str | None = None,
) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(missing_message or f"{name}不能为空")
    if len(text) > int(max_chars):
        raise ValueError(f"{name}不能超过 {int(max_chars)} 个字符")
    return text


def parse_gui_scope(value: Any, *, default: str = "draft") -> str:
    parsed = sanitize_user_facing_text(str(value or "")).strip()
    fallback = sanitize_user_facing_text(str(default or "")).strip()
    if len(parsed) > GUI_MAX_SCOPE_CHARS:
        raise ValueError(f"ui_scope不能超过 {GUI_MAX_SCOPE_CHARS} 个字符")
    if len(fallback) > GUI_MAX_SCOPE_CHARS:
        raise ValueError(f"ui_scope默认值不能超过 {GUI_MAX_SCOPE_CHARS} 个字符")
    return parsed or fallback


def parse_gui_choice(
    value: Any,
    *,
    name: str,
    default: str,
    allowed: set[str],
) -> str:
    parsed = str(value or default).strip().lower()
    if parsed not in allowed:
        choices = "、".join(sorted(allowed))
        raise ValueError(f"{name}必须是以下之一：{choices}")
    return parsed


def resolve_gui_session(
    db_path: Path,
    state: GuiState,
    session_id: int,
    *,
    ui_scope: str = "",
) -> dict[str, Any]:
    if not session_id:
        return {"ok": False, "status": 400, "code": "missing_session_id", "error": "session_id is required"}
    expected_scope = f"session:{int(session_id)}"
    provided_scope = str(ui_scope or "").strip()
    if provided_scope and provided_scope != expected_scope:
        return {
            "ok": False,
            "status": 409,
            "code": "session_scope_mismatch",
            "error": f"session scope mismatch: expected {expected_scope}, got {provided_scope}",
        }
    if state.is_session_deleting(int(session_id)):
        return {
            "ok": False,
            "status": 409,
            "code": "session_delete_in_progress",
            "error": "session delete is in progress",
        }
    if not get_idea_session(db_path, int(session_id)):
        if state.active_session_id == int(session_id):
            state.active_session_id = None
        return {"ok": False, "status": 404, "code": "session_not_found", "error": "session not found"}
    state.active_session_id = int(session_id)
    return {"ok": True, "status": 200, "session_id": int(session_id)}


def run_app_command(argv: list[str], root_dir: Path) -> dict[str, Any]:
    return run_app_command_streaming(argv, root_dir)


def run_app_command_streaming(
    argv: list[str],
    root_dir: Path,
    on_output: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        "-c",
        "import sys; from research_alpha.app import main; raise SystemExit(main(sys.argv[1:]))",
        *argv,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(root_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def pump(stream_name: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((stream_name, line))
        finally:
            output_queue.put((stream_name, None))

    threads = []
    for stream_name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        thread = threading.Thread(target=pump, args=(stream_name, stream), daemon=True)
        thread.start()
        threads.append(thread)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    done_streams = 0
    while done_streams < len(threads):
        stream_name, line = output_queue.get()
        if line is None:
            done_streams += 1
            continue
        if stream_name == "stderr":
            stderr_parts.append(line)
        else:
            stdout_parts.append(line)
        if on_output:
            on_output(stream_name, line)
    return_code = process.wait()
    for thread in threads:
        thread.join(timeout=0.2)
    return {
        "command": "ra " + " ".join(argv),
        "exit_code": int(return_code),
        "stdout": sanitize_user_facing_text("".join(stdout_parts)),
        "stderr": sanitize_user_facing_text("".join(stderr_parts)),
    }


def extract_created_session_id(result: dict[str, Any] | None) -> int | None:
    if not isinstance(result, dict):
        return None
    text = "\n".join(
        str(result.get(key, "") or "")
        for key in ("stdout", "stderr")
    )
    for pattern in (
        r"Auto session:\s*#(\d+)",
        r"Created idea session #(\d+)",
        r"Research Alpha chat session #(\d+)",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    manifest_match = re.search(r"Manifest:\s*(.+)", text)
    if manifest_match:
        manifest_path = Path(manifest_match.group(1).strip())
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        auto_session = payload.get("auto_session") if isinstance(payload, dict) else None
        if isinstance(auto_session, dict):
            try:
                session_id = int(auto_session.get("session_id", 0) or 0)
            except (TypeError, ValueError):
                session_id = 0
            if session_id > 0:
                return session_id
    return None


def command_exit_code(result: object) -> int:
    if not isinstance(result, dict):
        return 0
    try:
        return int(result.get("exit_code", 0) or 0)
    except (TypeError, ValueError):
        return 1


def command_error_message(result: object) -> str:
    if not isinstance(result, dict):
        return "Command failed."
    text = "\n".join(str(result.get(key, "") or "").strip() for key in ("stderr", "stdout") if str(result.get(key, "") or "").strip())
    exit_code = command_exit_code(result)
    if text:
        return public_gui_text(text)[-1800:]
    return f"Command exited with code {exit_code}."


def run_session_review_command(root_dir: Path, *, session_id: int, provider: str) -> dict[str, Any]:
    import contextlib
    import io
    import os

    from research_alpha.app import (
        build_session_reviewer_payload,
        cmd_reviewer_gate_for_payload,
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    old_cwd = Path.cwd()
    try:
        os.chdir(root_dir)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cmd_reviewer_gate_for_payload(
                root_dir,
                idea_payload=build_session_reviewer_payload(root_dir, int(session_id)),
                source_label=f"session-{int(session_id)}",
                provider=provider,
                model="",
                pattern_limit=5,
                genome_limit=5,
            )
    finally:
        os.chdir(old_cwd)
    return {
        "command": f"ra reviewer session-{int(session_id)}",
        "exit_code": int(code),
        "stdout": sanitize_user_facing_text(stdout.getvalue()),
        "stderr": sanitize_user_facing_text(stderr.getvalue()),
    }


def build_gui_snapshot(root_dir: Path, state: GuiState, session_id: int | None = None, draft_mode: bool = False) -> dict[str, Any]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    counts = {name: count for name, count in table_counts(config.db_path, ["papers", "idea_cards", "pattern_cards", "candidate_ideas", "idea_sessions", "runs"])}
    from research_alpha.app import compute_strict_evidence_readiness

    strict_evidence = compute_strict_evidence_readiness(
        config.db_path,
        paper_limit=max(
            12,
            count_scored_papers(config.db_path),
            count_idea_cards(config.db_path),
            count_pattern_cards(config.db_path),
        ),
    )
    strict_counts = strict_evidence.get("counts", {}) if isinstance(strict_evidence.get("counts"), dict) else {}
    readiness = {
        "high_weight_papers": int(strict_counts.get("high_weight_papers", 0) or 0),
        "genome_cards": int(strict_counts.get("strict_genome_cards", 0) or 0),
        "pattern_cards": int(strict_counts.get("strict_pattern_cards", 0) or 0),
        "raw_genome_cards": int(strict_counts.get("genome_cards", 0) or 0),
        "raw_pattern_cards": int(strict_counts.get("pattern_cards", 0) or 0),
        "strict_evidence_status": str(strict_evidence.get("status", "") or ""),
    }
    jobs = state.snapshot()
    health = build_gui_health_snapshot(
        project_configured=bool(config.llm.api_key),
        readiness=readiness,
        counts=counts,
        jobs=jobs,
    )
    readiness["evidence_status"] = str(health.get("evidence_status", "") or "")
    readiness["evidence_ready"] = bool(health.get("evidence_ready"))
    readiness["missing_requirements"] = health.get("missing_requirements", [])
    return {
        "project": {
            "provider": config.llm.provider,
            "model": config.llm.resolved_model,
            "api_key_configured": bool(config.llm.api_key),
        },
        "options": {
            "venue_groups": VENUE_GROUPS,
            "venue_presets": VENUE_PRESETS,
            "direction_presets": DIRECTION_PRESETS,
            "field_options": FIELD_OPTIONS,
            "hotspot_options": HOTSPOT_OPTIONS,
            "constraint_options": CONSTRAINT_OPTIONS,
            "remote_options": REMOTE_OPTIONS,
            "provider_options": PROVIDER_OPTIONS,
            "language_options": LANGUAGE_OPTIONS,
            "default_gold_venues": ["ICLR", "NeurIPS", "ICML", "CVPR", "ICCV", "TPAMI"],
            "default_from_year": 2024,
            "default_to_year": 2026,
        },
        "counts": counts,
        "readiness": readiness,
        "health": health,
        "top_papers": [
            paper_row_to_gui_card(row)
            for row in list_top_papers(config.db_path, limit=80)
            if float(row["paper_weight"] or 0) > 0
        ][:40],
        "frontier_papers": [paper_row_to_gui_card(row) for row in list_frontier_papers(config.db_path, limit=30)],
        "user_papers": [paper_row_to_gui_card(row) for row in list_user_library_papers(config.db_path, limit=30)],
        "sessions": [session_row_to_card(row) for row in list_idea_sessions(config.db_path, limit=30)],
        "ideas": [idea_row_to_card(row) for row in list_candidate_ideas(config.db_path, limit=6)],
        "patterns": [pattern_row_to_card(row) for row in list_pattern_cards(config.db_path, limit=6)],
        "runs": [run_row_to_card(row) for row in list_runs(config.db_path, limit=6)],
        "chat": build_draft_chat_snapshot() if draft_mode else build_chat_snapshot(config.root_dir, session_id=session_id),
        "jobs": jobs,
    }


def build_gui_health_snapshot(
    *,
    project_configured: bool,
    readiness: dict[str, Any],
    counts: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = {
        "high_weight_papers": 3,
        "genome_cards": 2,
        "pattern_cards": 1,
    }
    normalized_readiness = {
        key: max(0, int(readiness.get(key, 0) or 0))
        for key in thresholds
    }
    missing_requirements = [
        {
            "key": key,
            "current": normalized_readiness[key],
            "required": required,
        }
        for key, required in thresholds.items()
        if normalized_readiness[key] < required
    ]
    running_jobs = [job for job in jobs if job.get("status") == "running"]
    finished_evidence_jobs = [
        job
        for job in jobs
        if job.get("status") != "running" and job.get("kind") in GUI_EVIDENCE_JOB_KINDS
    ]
    finished_evidence_jobs.sort(
        key=lambda item: float(item.get("completed_at") or item.get("updated_at") or item.get("created_at") or 0),
        reverse=True,
    )
    latest_finished_evidence_job = finished_evidence_jobs[0] if finished_evidence_jobs else None
    failed_evidence_jobs = [job for job in finished_evidence_jobs if job.get("status") == "failed"]
    failed_evidence_jobs.sort(
        key=lambda item: float(item.get("completed_at") or item.get("updated_at") or item.get("created_at") or 0),
        reverse=True,
    )
    latest_failed_evidence_job = (
        latest_finished_evidence_job
        if latest_finished_evidence_job and latest_finished_evidence_job.get("status") == "failed"
        else None
    )
    latest_failure = None
    if latest_failed_evidence_job:
        public_failed_job = public_job_snapshot(latest_failed_evidence_job)
        latest_failure = {
            "id": int(public_failed_job.get("id", 0) or 0),
            "kind": str(public_failed_job.get("kind", "") or ""),
            "name": str(public_failed_job.get("name", "") or ""),
            "ui_scope": str(public_failed_job.get("ui_scope", "") or ""),
            "completed_at": public_failed_job.get("completed_at"),
            "error": compact_text_tail(public_failed_job.get("error", ""), 500, public=True),
            "failure_summary": build_public_evidence_failure_summary(latest_failed_evidence_job),
        }
    evidence_ready = not missing_requirements
    if not project_configured:
        evidence_status = "missing_provider_key"
    elif evidence_ready:
        evidence_status = "ready"
    elif any(job.get("kind") in GUI_EVIDENCE_JOB_KINDS for job in running_jobs):
        evidence_status = "building"
    elif latest_failed_evidence_job:
        evidence_status = "evidence_failed"
    else:
        evidence_status = "needs_evidence"
    return {
        "project_configured": bool(project_configured),
        "evidence_ready": evidence_ready,
        "evidence_status": evidence_status,
        "missing_requirements": missing_requirements,
        "failed_evidence_jobs": len(failed_evidence_jobs),
        "latest_failed_evidence_job": latest_failure,
        "running_jobs": len(running_jobs),
        "blocking_jobs": sum(1 for job in running_jobs if job.get("kind") in GUI_CONVERSATION_JOB_KINDS),
        "evidence_jobs": sum(1 for job in running_jobs if job.get("kind") in GUI_EVIDENCE_JOB_KINDS),
        "session_count": max(0, int(counts.get("idea_sessions", 0) or 0)),
    }


def row_to_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def public_external_ref(value: object) -> str:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    return public_gui_text(parsed.geturl())


def paper_row_to_gui_card(row: Any) -> dict[str, Any]:
    payload = row_to_dict(row)
    status, detail = describe_paper_library_status(payload)
    return {
        "id": int(payload.get("id", 0) or 0),
        "title": public_gui_text(payload.get("title", "")),
        "venue": public_gui_text(payload.get("venue", "")),
        "year": int(payload.get("year", 0) or 0),
        "source_kind": public_gui_text(payload.get("source_kind", "")),
        "source_label": describe_paper_source(payload.get("source_kind", "")),
        "external_ref": public_external_ref(payload.get("external_ref", "")),
        "award": public_gui_text(payload.get("award", "")),
        "citation_count": int(payload.get("citation_count", 0) or 0),
        "influential_citation_count": int(payload.get("influential_citation_count", 0) or 0),
        "paper_weight": round(float(payload.get("paper_weight", 0) or 0), 3),
        "publication_date": public_gui_text(payload.get("publication_date", "")),
        "abstract_excerpt": compact_text_tail(payload.get("abstract", ""), 360, public=True),
        "quality_reason": describe_paper_quality_reason(payload),
        "library_status": status,
        "library_status_detail": detail,
    }


def describe_paper_library_status(paper: dict[str, Any]) -> tuple[str, str]:
    source_kind = str(paper.get("source_kind", "") or "").strip().lower()
    award = str(paper.get("award", "") or "").strip().lower()
    try:
        paper_weight = float(paper.get("paper_weight", 0) or 0)
    except (TypeError, ValueError):
        paper_weight = 0.0
    if paper_weight > 0:
        if award in {"best_paper", "outstanding_paper", "oral", "test_of_time"}:
            return "Gold", "参与 idea 生成和证据驱动评分"
        return "需补优秀信号", "已有权重，但建议补全 Best/Nominee/Oral 等明确优秀信号"
    if source_kind == "user_library":
        return "用户库", "只作为领域知识/路线学习，不参与 idea 评分标准"
    if source_kind.startswith("frontier_openalex") or source_kind.startswith("frontier_s2"):
        return "待核验趋势", "先进入趋势区；需 quality-enrich 核验优秀信号后才可进入 Gold"
    if source_kind.startswith("frontier_"):
        return "趋势参考", "不参与 idea 评分标准"
    if source_kind.startswith("gold_"):
        return "辅助/降级", "未满足当前核心 Gold 标准，不参与 idea 评分标准"
    return "待评分", "需要评分或质量核验后才能参与生成依据"


def describe_paper_source(source_kind: object) -> str:
    value = str(source_kind or "").strip()
    labels = {
        "gold_openreview": "OpenReview 官方接收类型",
        "gold_icml_awards": "ICML 官方奖项页",
        "gold_neurips_awards": "NeurIPS 官方奖项页",
        "gold_cvf_awards": "CVF 官方奖项页",
        "gold_acl_awards": "ACL 官方奖项页（辅助）",
        "gold_eccv_awards": "ECCV 官方奖项页（辅助）",
        "frontier_openreview": "OpenReview 趋势参考",
        "frontier_openalex": "OpenAlex 趋势参考",
        "frontier_s2": "Semantic Scholar 趋势参考",
        "frontier_acl_awards": "ACL 官方奖项页（辅助）",
        "frontier_eccv_awards": "ECCV 官方奖项页（辅助）",
        "frontier_icml_awards": "ICML 官方奖项页（趋势）",
        "frontier_neurips_awards": "NeurIPS 官方奖项页（趋势）",
        "frontier_cvf_awards": "CVF 官方奖项页（趋势）",
        "user_library": "用户自建论文库",
        "seed": "本地导入",
        "manual": "本地导入",
    }
    return labels.get(value, value or "本地导入")


def describe_paper_quality_reason(paper: dict[str, Any]) -> str:
    award = str(paper.get("award", "") or "").strip().lower()
    source_kind = str(paper.get("source_kind", "") or "").strip().lower()
    try:
        paper_weight = float(paper.get("paper_weight", 0) or 0)
    except (TypeError, ValueError):
        paper_weight = 0.0
    if paper_weight <= 0:
        if source_kind == "user_library":
            return "领域知识，不参与 Gold 标准/评分"
        if source_kind.startswith("frontier_"):
            return "仅作为趋势参考，不参与 idea 评分标准"
        if award:
            return "有奖项/接收类型元数据，但未通过核心 Gold 标准；仅作辅助趋势参考，不参与 idea 评分标准"
        return "未通过核心优秀论文标准；仅作为辅助趋势参考，不参与 idea 评分标准"
    reason_by_award = {
        "best_paper": "最佳论文/奖项论文，直接进入优秀依据库",
        "outstanding_paper": "提名、Honorable Mention 或 Outstanding 论文，直接进入优秀依据库",
        "oral": "官方 Oral 接收类型，作为高权重优秀依据",
        "spotlight": "Spotlight 仅作辅助质量线索，不作为当前核心 Gold 入库标准",
        "test_of_time": "Test-of-Time 类奖项，作为历史高影响参考",
        "high_citation": "高引用仅作发现辅助线索，不单独进入核心 Gold 库",
    }
    if award in reason_by_award:
        return reason_by_award[award]
    score_notes = str(paper.get("score_notes", "") or "").strip()
    if "citation_percentile_bonus" in score_notes:
        return "同会议年份引用分位较高，作为补充优秀依据"
    if source_kind.startswith("gold_"):
        return "来自官方优秀论文源，等待补全奖项标签"
    if source_kind.startswith("frontier_"):
        return "仅作为趋势参考，不参与 idea 评分标准"
    return "本地论文记录，需通过评分后才作为优秀依据"


def sanitize_user_facing_text(value: object) -> str:
    text = str(value or "")
    replacements = [
        (r"（请用中文输出idea标题、解释、风险和实验计划）", ""),
        (r"\(Please return idea titles, explanations, risks, and experiment plans in English\.\)", ""),
        (r"(?m)^Inferred evidence-matching tags:.*(?:\n|$)", ""),
        (r"(?m)^Inferred evidence emphasis:.*(?:\n|$)", ""),
        (r"(?m)^Use these inferred tags only to retrieve and match evidence; do not treat them as a fixed idea template\.\s*", ""),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def public_gui_text(value: object) -> str:
    return redact_secrets(redact_local_paths(sanitize_user_facing_text(value))).strip()


def session_row_to_card(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": public_gui_text(row["name"]),
        "initial_idea": public_gui_text(row["initial_idea"]),
        "current_idea": public_gui_text(row["current_idea"]),
        "status": public_gui_text(row["status"]),
        "created_at": public_gui_text(row["created_at"]),
        "updated_at": public_gui_text(row["updated_at"]),
    }


def run_row_to_card(row: Any) -> dict[str, Any]:
    manifest_path = str(row["manifest_path"] or "") if "manifest_path" in row.keys() else ""
    return {
        "id": row["id"],
        "query": public_gui_text(row["query"]),
        "status": public_gui_text(row["status"]),
        "created_at": public_gui_text(row["created_at"]),
        "manifest_available": bool(manifest_path),
    }


def pattern_row_to_card(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "pattern_key": public_gui_text(row["pattern_key"]),
        "created_at": public_gui_text(row["created_at"]),
    }


def idea_row_to_card(row: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(row["content_json"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    prior_gate = payload.get("prior_art_gate", {})
    if not isinstance(prior_gate, dict):
        prior_gate = {}
    score = payload.get("evidence_grounded_score", {})
    if not isinstance(score, dict):
        score = {}
    why_not = prior_gate.get("why_not_done_yet", [])
    if not isinstance(why_not, list):
        why_not = []
    return {
        "id": row["id"],
        "query": public_gui_text(row["query"]),
        "title": public_gui_text(row["title"]),
        "created_at": public_gui_text(row["created_at"]),
        "prior_art_gate_decision": public_gui_text(prior_gate.get("decision", "")),
        "prior_art_gate_summary": public_gui_text(prior_gate.get("summary", "")),
        "why_not_done_yet": [public_gui_text(value) for value in why_not if public_gui_text(value)][:2],
        "evidence_score": score.get("total", ""),
    }


def parse_venue_payload(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None or isinstance(value, str):
        raw_items = str(value or "").split(",")
    else:
        raise ValueError("venues必须是逗号分隔字符串或字符串数组")
    if len(raw_items) > GUI_MAX_VENUES:
        raise ValueError(f"venues不能超过 {GUI_MAX_VENUES} 个")
    seen: set[str] = set()
    venues: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            raise ValueError("venues数组只能包含字符串")
        item = raw_item.strip()
        if not item:
            continue
        if len(item) > GUI_MAX_VENUE_CHARS:
            raise ValueError(f"venue名称不能超过 {GUI_MAX_VENUE_CHARS} 个字符")
        venue = canonicalize_venue_name(item)
        key = venue.lower()
        if key in seen:
            continue
        seen.add(key)
        venues.append(venue)
    return venues


def canonicalize_venue_name(value: object) -> str:
    cleaned = str(value or "").strip()
    return VENUE_CANONICAL_NAMES.get(cleaned.lower(), cleaned)


def dedupe_venues(venues: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for venue in venues:
        cleaned = canonicalize_venue_name(venue)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
    return items


def split_gold_venues_by_remote(venues: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for cleaned in dedupe_venues(venues):
        lower = cleaned.lower()
        if lower == "icml":
            remote = "icml_awards"
        elif lower in {"neurips", "nips"}:
            remote = "neurips_awards"
        elif lower in {"cvpr", "iccv"}:
            remote = "cvf_awards"
        elif lower in {"iclr"}:
            remote = "openreview"
        else:
            remote = "openalex"
        groups.setdefault(remote, []).append(cleaned)
    return groups


def metadata_only_gold_venues(remote_groups: dict[str, list[str]]) -> list[str]:
    """Core venues whose current connector can only collect trend metadata."""
    return list(remote_groups.get("openalex", []))


def describe_gold_source_plan(remote_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    labels = {
        "icml_awards": "ICML 官方奖项页",
        "neurips_awards": "NeurIPS 官方奖项页",
        "cvf_awards": "CVF 官方奖项页",
        "openreview": "OpenReview Best/Nominee/Oral",
        "openalex": "OpenAlex 元数据趋势",
    }
    modes = {
        "icml_awards": "strict_gold",
        "neurips_awards": "strict_gold",
        "cvf_awards": "strict_gold",
        "openreview": "strict_gold",
        "openalex": "metadata_only",
    }
    plan: list[dict[str, Any]] = []
    for remote, venues in remote_groups.items():
        plan.append(
            {
                "remote": remote,
                "label": labels.get(remote, remote),
                "venues": list(venues),
                "mode": modes.get(remote, "metadata_only"),
            }
        )
    return plan


def is_core_gold_gui_venue(venue: object) -> bool:
    return canonicalize_venue_name(venue).lower() in CORE_GOLD_GUI_VENUES


def core_gold_venues(venues: list[str]) -> list[str]:
    return dedupe_venues([venue for venue in venues if is_core_gold_gui_venue(venue)])


def auxiliary_gold_venues(venues: list[str]) -> list[str]:
    return dedupe_venues([venue for venue in venues if not is_core_gold_gui_venue(venue)])


def infer_research_brief(user_text: str, messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cleaned = " ".join(str(user_text or "").split())
    history = " ".join(
        str(item.get("text", ""))
        for item in (messages or [])[-6:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
    )
    haystack = f"{history} {cleaned}".lower()
    fields = infer_tag_values(haystack, "fields")
    hotspots = infer_tag_values(haystack, "hotspots")
    constraints = infer_tag_values(haystack, "constraints")
    search_query = build_inferred_search_query(cleaned, fields + hotspots + constraints)
    return {
        "user_direction": cleaned,
        "inferred_fields": fields,
        "inferred_hotspots": hotspots,
        "inferred_constraints": constraints,
        "frontier_query": search_query,
        "brief_direction": build_brief_direction(cleaned, fields, hotspots, constraints),
    }


def infer_tag_values(haystack: str, group: str) -> list[str]:
    values: list[str] = []
    for value, aliases in TAG_ALIASES.get(group, {}).items():
        if any(alias.lower() in haystack for alias in aliases):
            values.append(value)
    return values


def build_inferred_search_query(user_text: str, tags: list[str]) -> str:
    terms: list[str] = []
    translated_terms = translate_user_query_terms(user_text)
    terms.extend(translated_terms)
    if not any("\u4e00" <= char <= "\u9fff" for char in user_text):
        terms.extend(user_text.split())
    for tag in tags:
        terms.extend(SEARCH_TERMS.get(tag, []))
    if not terms:
        terms = [user_text.strip()] if user_text.strip() else ["AI", "research"]
    return " ".join(dedupe_keep_order([str(term).strip() for term in terms if str(term).strip()]))


def translate_user_query_terms(user_text: str) -> list[str]:
    text = str(user_text or "").lower()
    terms: list[str] = []
    if any(item in text for item in ["伪装目标检测", "伪装检测", "camouflaged object", "camouflage object"]):
        terms.extend(["camouflaged object detection", "camouflage object segmentation", "concealed object detection"])
    elif "目标检测" in text or "object detection" in text:
        terms.extend(["object detection", "visual detection"])
    if any(item in text for item in ["遥感", "remote sensing"]):
        terms.append("remote sensing")
    if any(item in text for item in ["小目标", "small object"]):
        terms.append("small object detection")
    return terms


def build_brief_direction(user_text: str, fields: list[str], hotspots: list[str], constraints: list[str]) -> str:
    lines = [user_text.strip()]
    if fields or hotspots:
        lines.append("Inferred evidence-matching tags: " + "; ".join(
            item for item in [
                "fields=" + ", ".join(fields) if fields else "",
                "frontier_signals=" + ", ".join(hotspots) if hotspots else "",
            ]
            if item
        ))
    if constraints:
        lines.append("Inferred evidence emphasis: " + "；".join(constraints))
    lines.append("Use these inferred tags only to retrieve and match evidence; do not treat them as a fixed idea template.")
    return "\n".join(line for line in lines if line.strip())


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(item)
    return values


def build_chat_snapshot(root_dir: Path, session_id: int | None = None) -> dict[str, Any]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    sessions = []
    if session_id is not None:
        session = get_idea_session(config.db_path, int(session_id))
        if session:
            sessions = [session]
        else:
            return {
                "active": False,
                "session": None,
                "missing_session_id": int(session_id),
                "messages": [
                    {
                        "role": "assistant",
                        "kind": "missing",
                        "text": "没有找到这条历史对话。可以新建对话，或从左侧历史记录重新选择。",
                        "meta": "session missing",
                    }
                ],
            }
    if session_id is None and not sessions:
        sessions = list_idea_sessions(config.db_path, limit=1)
    if not sessions:
        return {
            "active": False,
            "session": None,
            "messages": [
                {
                    "role": "assistant",
                    "kind": "empty",
                    "text": "直接输入你想研究的领域、问题或一个粗略方向。我会从对话里自动识别研究标签，检索近期证据，再给出可继续追问和打磨的 idea。",
                }
            ],
        }
    session = sessions[0]
    session_id = int(session["id"])
    turns = list_idea_session_turns(config.db_path, session_id)
    session_context = load_gui_session_context(session)
    seed_text = public_gui_text(session["initial_idea"]) or public_gui_text(session["current_idea"])
    evidence_text = build_evidence_message_for_session(session_context)
    if evidence_text:
        seed_text = f"{seed_text}\n\n{evidence_text}"
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "kind": "seed",
            "text": seed_text,
            "meta": f"Session #{session_id} · {public_gui_text(session['name'])}",
        }
    ]
    for turn in turns:
        messages.append(
            {
                "role": "user",
                "kind": "instruction",
                "text": public_gui_text(turn["user_instruction"]),
                "meta": f"turn {turn['turn_index']}",
            }
        )
        assistant_text = public_gui_text(turn["revised_idea"])
        try:
            payload = json.loads(str(turn["content_json"]))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            verdict = str(payload.get("verdict_summary", "")).strip()
            coach = str(payload.get("coach_message", "")).strip()
            if verdict or coach:
                assistant_text = "\n\n".join(public_gui_text(item) for item in [assistant_text, verdict, coach] if item)
        messages.append(
            {
                "role": "assistant",
                "kind": "revision",
                "text": assistant_text,
                "meta": f"decision: {public_gui_text(turn['decision'])}",
            }
        )
    if not turns and str(session["current_idea"]).strip() != str(session["initial_idea"]).strip():
        messages.append(
            {
                "role": "assistant",
                "kind": "current",
                "text": public_gui_text(session["current_idea"]),
                "meta": "current best",
            }
        )
    return {
        "active": True,
        "session": session_row_to_card(session),
        "messages": messages,
    }


def build_draft_chat_snapshot() -> dict[str, Any]:
    return {
        "active": False,
        "session": None,
        "draft": True,
        "messages": [
            {
                "role": "assistant",
                "kind": "draft",
                "text": "已准备好新对话。直接输入你想研究的方向，我会从论文逻辑库和近期证据里生成新 idea。",
                "meta": "new workspace",
            }
        ],
    }


def load_gui_session_context(session: Any) -> dict[str, Any]:
    try:
        raw = str(session["context_json"]).strip()
    except (KeyError, IndexError):
        raw = ""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_evidence_message_for_session(context: dict[str, Any]) -> str:
    if not context:
        return ""
    lines: list[str] = []
    basis = []
    explicit_basis = context.get("evidence_basis", [])
    if isinstance(explicit_basis, list):
        for item in explicit_basis[:4]:
            if not isinstance(item, dict):
                continue
            title = public_gui_text(item.get("source_title", ""))
            source_type = public_gui_text(item.get("source_type", ""))
            source_id = public_gui_text(item.get("source_id", ""))
            borrowed = public_gui_text(item.get("borrowed_standard", ""))
            used_for = public_gui_text(item.get("used_for", ""))
            label = title or source_id or source_type
            if label:
                detail = borrowed or "用于约束生成和评分标准"
                suffix = f"；用于：{used_for}" if used_for else ""
                basis.append(f"参考优秀论文/逻辑资产：{label} -> {detail}{suffix}")
    paper_angle = public_gui_text(context.get("paper_angle", ""))
    historical_pattern = public_gui_text(context.get("historical_pattern", ""))
    if paper_angle:
        basis.append(f"参考优秀论文思路：{paper_angle}")
    if historical_pattern:
        basis.append(f"迁移的逻辑模式：{historical_pattern}")
    if basis:
        lines.append("依据：\n" + "\n".join(f"- {item}" for item in basis[:3]))

    problem_items = []
    frontier_gap = public_gui_text(context.get("frontier_gap", ""))
    trend_support = public_gui_text(context.get("trend_support", ""))
    if frontier_gap:
        problem_items.append(f"当前方法/趋势的问题：{frontier_gap}")
    if trend_support:
        problem_items.append(f"近期趋势依据：{trend_support}")
    if problem_items:
        lines.append("为什么不是拍脑袋：\n" + "\n".join(f"- {item}" for item in problem_items[:3]))

    defense_items = []
    novelty = public_gui_text(context.get("novelty", ""))
    why_now = public_gui_text(context.get("why_now", ""))
    key_risk = public_gui_text(context.get("key_risk", ""))
    if novelty:
        defense_items.append(f"新颖性边界：{novelty}")
    if why_now:
        defense_items.append(f"为什么现在能做：{why_now}")
    if key_risk:
        defense_items.append(f"最可能被审稿人攻击的点：{key_risk}")
    why_not = context.get("why_not_done_yet", [])
    if isinstance(why_not, list) and why_not:
        defense_items.append("为什么此前没人做/没做成：" + "；".join(public_gui_text(item) for item in why_not[:2] if public_gui_text(item)))
    defense_items.extend(review_defense_items(context))
    if defense_items:
        lines.append("审稿防御要点：\n" + "\n".join(f"- {item}" for item in defense_items[:6]))

    score_text = build_gui_score_message(context.get("evidence_grounded_score", {}))
    if score_text:
        lines.append(score_text)

    return "\n\n".join(lines)


def review_defense_items(context: dict[str, Any]) -> list[str]:
    history = context.get("review_history", [])
    if not isinstance(history, list) or not history:
        return []
    items: list[str] = []
    for review in history[-2:]:
        if not isinstance(review, dict):
            continue
        decision = public_gui_text(review.get("decision", ""))
        fatal = first_public_list_item(review.get("fatal_flaws"))
        attack = first_public_list_item(review.get("main_attacks"))
        rethink = first_public_list_item(review.get("required_rethink"))
        missing = first_public_list_item(review.get("missing_logic_links"))
        risk = fatal or attack or missing
        if risk:
            prefix = f"审稿人 {decision} 攻击：" if decision else "审稿人攻击："
            items.append(prefix + risk)
        if rethink:
            items.append("审稿后下一步：" + rethink)
    return items[-4:]


def first_public_list_item(value: object) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        cleaned = public_gui_text(item)
        if cleaned:
            return cleaned
    return ""


def build_gui_score_message(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    total = value.get("total", "")
    scale = value.get("scale", "0-10")
    rubric_source = sanitize_user_facing_text(value.get("rubric_source", ""))
    dimensions = value.get("dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
    labels = [
        ("innovation", "创新性"),
        ("logic", "逻辑性"),
        ("feasibility", "可行性"),
        ("value", "价值"),
        ("defensibility", "抗审稿攻击性"),
    ]
    lines = []
    if total != "":
        lines.append(f"总分：{total}/{scale}")
    if rubric_source:
        lines.append(f"评分标准来源：{rubric_source}")
    for key, label in labels:
        item = dimensions.get(key, {})
        if not isinstance(item, dict):
            continue
        score = item.get("score", "")
        explicit = "有明确证据" if item.get("explicit_evidence_covered", False) else "证据不足"
        evidence = item.get("evidence", [])
        evidence_note = ""
        if isinstance(evidence, list) and evidence:
            first = evidence[0]
            if isinstance(first, dict):
                source_type = sanitize_user_facing_text(first.get("source_type", ""))
                source_id = sanitize_user_facing_text(first.get("source_id", ""))
                if source_type or source_id:
                    evidence_note = f"；依据：{source_type}:{source_id}".strip()
        if score != "":
            lines.append(f"{label}：{score}/10（{explicit}{evidence_note}）")
    if not lines:
        return ""
    return "证据驱动评分：\n" + "\n".join(f"- {line}" for line in lines[:8])


def render_index_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Research Alpha</title>
  <style>
    :root {
      color-scheme: light;
      --ink:#111111;
      --muted:#6b7280;
      --line:#e5e7eb;
      --soft:#f8f7fb;
      --panel:#ffffff;
      --accent:#4b5563;
      --accent-strong:#111827;
      --accent-soft:#f3f4f6;
      --bad:#b42318;
      --warn:#a16207;
      --shadow:0 14px 34px rgba(17,17,17,.06);
      --shadow-soft:0 1px 2px rgba(17,17,17,.05);
    }
    * { box-sizing:border-box; }
    html, body { height:100%; }
    body { margin:0; font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#fbfbfd; overflow:hidden; }
    button, input, textarea, select { font:inherit; }
    button { border:0; border-radius:8px; min-height:38px; padding:9px 12px; font-weight:760; cursor:pointer; background:var(--accent); color:#fff; transition:background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease; }
    button:hover { filter:none; }
    button:disabled { cursor:not-allowed; opacity:.72; filter:none; }
    button.ghost { background:#fff; color:#171717; border:1px solid var(--line); }
    button.black { background:var(--accent); color:#fff; }
    button.icon { width:38px; padding:0; display:grid; place-items:center; }
    button.subtle { background:var(--accent-soft); color:#374151; }
    button.status-pill { border-radius:999px; min-height:28px; padding:4px 9px; background:#fff; color:#374151; border:1px solid var(--line); font-size:12px; font-weight:760; box-shadow:none; position:relative; }
    button.status-pill::after, .nav-button::after { content:"打开"; margin-left:6px; color:#374151; font-size:10px; font-weight:900; opacity:0; transition:opacity .14s ease; }
    button.status-pill:hover::after, button.status-pill:focus-visible::after, .nav-button:hover::after, .nav-button:focus-visible::after { opacity:.88; }
    button.status-pill:hover, button.status-pill:focus-visible, .nav-button:hover, .nav-button:focus-visible { border-color:rgba(17,24,39,.14); background:#f9fafb; box-shadow:inset 0 0 0 1px rgba(17,24,39,.08); outline:none; }
    button.status-pill.ok { border-color:rgba(17,24,39,.12); background:var(--accent-soft); color:#374151; }
    button.status-pill.bad { border-color:#f6c2bd; background:#fff1f0; color:var(--bad); }
    input, textarea, select { width:100%; border:1px solid var(--line); border-radius:8px; padding:10px 11px; background:#fff; color:var(--ink); outline:none; }
    input:focus, textarea:focus, select:focus { border-color:rgba(17,24,39,.22); box-shadow:0 0 0 3px rgba(17,24,39,.10); }
    textarea { resize:none; line-height:1.5; min-height:56px; max-height:170px; }
    h1 { margin:0; font-size:17px; font-weight:820; letter-spacing:0; }
    h2 { margin:0; font-size:12px; text-transform:uppercase; letter-spacing:0; color:#4b5563; }
    h3 { margin:0 0 8px; font-size:14px; }
    label { display:block; margin:10px 0 5px; font-size:12px; color:var(--muted); }
    header { height:58px; display:flex; justify-content:space-between; align-items:center; padding:0 18px; background:#fff; border-bottom:1px solid var(--line); }
    main { height:calc(100vh - 58px); display:grid; grid-template-columns:292px minmax(560px, 1fr) 356px; min-height:0; }
    aside, section { min-height:0; overflow:auto; }
    .brand-mark { display:flex; align-items:center; gap:10px; }
    .brand-copy { display:flex; flex-direction:column; gap:1px; }
    .brand-copy span { color:var(--muted); font-size:11px; font-weight:650; }
    .brand-dot { width:32px; height:32px; display:grid; place-items:center; border-radius:9px; background:var(--accent); color:#fff; font-weight:850; }
    .toolbar { display:flex; align-items:center; gap:8px; }
    .pill { display:inline-flex; align-items:center; gap:5px; border:1px solid var(--line); border-radius:999px; padding:4px 9px; background:#fff; color:#374151; font-size:12px; }
    .pill.ok { border-color:rgba(17,24,39,.12); background:var(--accent-soft); color:#374151; }
    .pill.bad { border-color:#f6c2bd; background:#fff1f0; color:var(--bad); }
    .left-rail { border-right:1px solid var(--line); background:#fff; padding:14px 12px; display:flex; flex-direction:column; gap:12px; overflow:auto; min-height:0; }
    .rail-header { display:flex; gap:8px; align-items:center; }
    .rail-header button:first-child { flex:1; }
    .section-title { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
    .history-panel { flex:1 1 auto; min-height:160px; display:flex; flex-direction:column; overflow:hidden; }
    .history-list { flex:1; min-height:0; overflow:auto; display:flex; flex-direction:column; gap:6px; padding-right:2px; }
    .session-row { display:grid; grid-template-columns:1fr 32px; gap:4px; align-items:stretch; }
    .session-item { width:100%; text-align:left; border:1px solid transparent; border-radius:8px; background:#fff; color:#111; padding:9px 10px; min-height:54px; position:relative; transition:border-color .15s ease, background .15s ease, box-shadow .15s ease; }
    .session-item:hover, .session-item:focus-visible { border-color:rgba(17,24,39,.12); background:#f9fafb; box-shadow:inset 3px 0 0 rgba(17,24,39,.18); outline:none; }
    .session-item.active { border-color:rgba(17,24,39,.14); background:var(--accent-soft); }
    .delete-session { min-height:54px; width:32px; padding:0; border:1px solid transparent; border-radius:8px; background:#fff; color:#9ca3af; font-size:17px; transition:border-color .15s ease, background .15s ease, color .15s ease; }
    .delete-session:hover { border-color:#f6c2bd; background:#fff1f0; color:var(--bad); }
    .session-title { font-size:13px; font-weight:760; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .session-preview { margin-top:4px; font-size:12px; color:var(--muted); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; line-height:1.35; }
    .session-item.pending { border-color:rgba(17,24,39,.12); background:#f9fafb; }
    .session-status { margin-top:6px; display:inline-flex; align-items:center; gap:6px; color:#374151; font-size:11px; font-weight:760; }
    .session-status .mini-dot { width:6px; height:6px; border-radius:999px; background:var(--accent); animation:typingPulse 1.15s infinite ease-in-out; }
    .evidence-card { border:1px solid var(--line); background:#fff; border-radius:8px; padding:12px; box-shadow:var(--shadow-soft); }
    .metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfbfd; }
    .metric-button { width:100%; min-height:82px; text-align:left; color:var(--ink); position:relative; cursor:pointer; overflow:visible; display:grid; grid-template-columns:1fr auto; align-items:center; gap:10px; border-color:rgba(17,24,39,.10); background:#fff; box-shadow:none; transition:border-color .15s ease, background .15s ease, box-shadow .15s ease; }
    .metric-button::before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--accent); opacity:.18; transition:opacity .15s ease; }
    .metric-button::after { content:attr(data-tooltip); position:absolute; left:10px; right:auto; bottom:calc(100% + 8px); max-width:230px; padding:8px 9px; border:1px solid rgba(17,24,39,.12); border-radius:10px; background:#fff; color:#374151; box-shadow:0 6px 18px rgba(17,17,17,.08); opacity:0; pointer-events:none; transition:opacity .14s ease; font-size:12px; font-weight:650; line-height:1.4; z-index:20; }
    .clickable-row, .status-link, .empty-action, .section-link { --action-label:"查看"; cursor:pointer; }
    .action-panel { cursor:pointer; }
    .empty-action { --action-label:"开始"; }
    .clickable-row::after, .status-link::after, .empty-action::after, .section-link::after { content:var(--action-label); position:absolute; right:10px; top:9px; color:#374151; opacity:0; transition:opacity .14s ease; font-size:11px; font-weight:900; }
    .clickable-row::before, .status-link::before, .section-link::before, .empty-action::before { content:""; position:absolute; inset:0 auto 0 0; width:3px; border-radius:inherit; background:var(--accent); opacity:0; transition:opacity .14s ease; }
    .metric-button:hover, .metric-button:focus-visible, .clickable-row:hover, .clickable-row:focus-visible, .status-link:hover, .status-link:focus-visible, .empty-action:hover, .empty-action:focus-visible, .section-link:hover, .section-link:focus-visible { border-color:rgba(17,24,39,.14); background:#f9fafb; box-shadow:inset 0 0 0 1px rgba(17,24,39,.08); outline:none; }
    .metric-button:hover::before, .metric-button:focus-visible::before { opacity:.72; }
    .metric-button:hover::after, .metric-button:focus-visible::after { opacity:1; }
    .clickable-row:hover::before, .clickable-row:focus-visible::before, .status-link:hover::before, .status-link:focus-visible::before, .section-link:hover::before, .section-link:focus-visible::before, .empty-action:hover::before, .empty-action:focus-visible::before { opacity:.5; }
    .clickable-row:hover::after, .clickable-row:focus-visible::after, .status-link:hover::after, .status-link:focus-visible::after, .empty-action:hover::after, .empty-action:focus-visible::after, .section-link:hover::after, .section-link:focus-visible::after { opacity:.88; }
    .metric-button:active, .clickable-row:active, .status-link:active, .empty-action:active, .section-link:active { background:#f3f4f6; box-shadow:inset 0 0 0 1px rgba(17,24,39,.10); }
    .metric-top { display:flex; align-items:center; justify-content:space-between; gap:8px; position:relative; z-index:1; }
    .metric strong { display:block; font-size:22px; line-height:1; position:relative; z-index:1; }
    .metric-open { flex:0 0 auto; display:grid; place-items:center; width:28px; height:28px; border:1px solid rgba(17,24,39,.12); border-radius:999px; background:#fff; color:#374151; font-size:16px; font-weight:900; line-height:1; transition:background .15s ease, border-color .15s ease; }
    .metric span { color:var(--muted); font-size:12px; position:relative; z-index:1; }
    .metric .metric-open { color:#374151; }
    .metric-label { display:block; margin-top:4px; color:var(--muted); font-size:12px; position:relative; z-index:1; }
    .metric-button:hover .metric-open, .metric-button:focus-visible .metric-open { background:var(--accent-soft); border-color:rgba(17,24,39,.14); }
    .metric-button:hover strong, .metric-button:focus-visible strong { color:#111827; }
    .nav-button { width:100%; display:flex; align-items:center; justify-content:space-between; gap:8px; background:#fff; color:#111; border:1px solid var(--line); position:relative; }
    .nav-button.active { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:#374151; }
    .library-view { display:none; overflow:auto; padding:26px 24px 34px; background:#fbfbfd; }
    .library-view.active { display:block; }
    .library-grid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; max-width:980px; margin:0 auto; }
    .library-panel { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; box-shadow:var(--shadow-soft); min-width:0; }
    .library-panel.flash { border-color:rgba(17,24,39,.18); box-shadow:0 8px 22px rgba(17,24,39,.05); }
    .user-library-main { grid-column:1 / -1; }
    .library-panel.collapsed { align-self:start; }
    .library-panel.collapsed .library-body { display:none; }
    .library-toggle { width:100%; min-height:auto; padding:9px 10px; display:flex; align-items:center; justify-content:space-between; gap:8px; background:#fbfbfd; color:inherit; border:1px solid var(--line); border-radius:8px; text-align:left; }
    .library-toggle:hover, .library-toggle:focus-visible { border-color:rgba(17,24,39,.14); background:#f9fafb; box-shadow:none; outline:none; }
    .library-toggle h2 { position:relative; z-index:1; }
    .library-toggle-icon { width:24px; height:24px; border:1px solid var(--line); border-radius:999px; display:grid; place-items:center; color:#6b7280; background:#fff; font-size:12px; line-height:1; }
    .library-body { margin-top:10px; }
    .artifact-row { display:grid; grid-template-columns:1fr 30px; gap:6px; align-items:stretch; margin:8px 0; }
    .artifact-row .artifact-card { margin:0; }
    .delete-artifact { min-height:auto; width:30px; padding:0; border:1px solid transparent; border-radius:6px; background:#fff; color:#9ca3af; font-size:16px; opacity:.64; }
    .artifact-row:hover .delete-artifact, .delete-artifact:focus-visible { opacity:1; }
    .delete-artifact:hover, .delete-artifact:focus-visible { background:#fff1f0; border-color:rgba(239,68,68,.18); color:var(--bad); }
    .section-link { width:100%; min-height:auto; padding:9px 30px 9px 10px; margin-top:10px; display:flex; align-items:center; justify-content:space-between; gap:8px; position:relative; background:#fbfbfd; color:inherit; border:1px solid var(--line); border-radius:8px; text-align:left; }
    .section-link h2 { position:relative; z-index:1; }
    .section-link .pill { position:relative; z-index:1; }
    .section-right { display:flex; align-items:center; gap:7px; position:relative; z-index:1; }
    .section-action { color:#374151; font-size:11px; font-weight:850; opacity:.72; }
    .section-link:hover .section-action, .section-link:focus-visible .section-action { opacity:1; }
    .section-link:hover h2, .section-link:focus-visible h2 { color:#111827; }
    .section-link.flash, .metric-button.flash { border-color:#4b5563; box-shadow:inset 0 0 0 1px rgba(17,24,39,.10); }
    .paper-list { display:flex; flex-direction:column; gap:8px; margin-top:10px; }
    .paper-row { border:1px solid var(--line); border-radius:8px; padding:10px 28px 10px 10px; position:relative; transition:border-color .15s ease, box-shadow .15s ease, background .15s ease; }
    a.paper-row, a.artifact-card { display:block; color:inherit; text-decoration:none; }
    button.paper-row { width:100%; min-height:unset; display:block; text-align:left; color:inherit; cursor:pointer; background:#fff; font-weight:inherit; }
    .paper-title-line { display:flex; align-items:flex-start; gap:8px; justify-content:space-between; }
    .paper-title { font-size:12px; font-weight:720; line-height:1.35; }
    .paper-status { flex:0 0 auto; border:1px solid rgba(17,24,39,.12); border-radius:999px; padding:2px 7px; background:var(--accent-soft); color:#374151; font-size:10px; font-weight:860; line-height:1.4; }
    .paper-status.frontier { border-color:#e5e7eb; background:#f9fafb; color:#4b5563; }
    .paper-status.pending { border-color:#fde68a; background:#fffbeb; color:#92400e; }
    .paper-meta { color:var(--muted); font-size:11px; margin-top:3px; }
    .paper-reason { color:#374151; font-size:11px; margin-top:4px; line-height:1.35; }
    .paper-row:hover .paper-title, .paper-row:focus-visible .paper-title, .artifact-card:hover .title, .artifact-card:focus-visible .title { color:#111827; }
    .paper-row:hover .paper-status, .paper-row:focus-visible .paper-status { border-color:rgba(17,24,39,.14); }
    .paper-reason a { color:#374151; text-decoration:none; font-weight:760; }
    .user-library-block { margin-top:14px; padding-top:14px; border-top:1px solid var(--line); }
    .user-library-form { display:grid; gap:8px; margin-top:10px; padding:12px; border:1px solid rgba(17,24,39,.08); border-radius:8px; background:rgba(255,255,255,.62); backdrop-filter:saturate(145%) blur(14px); -webkit-backdrop-filter:saturate(145%) blur(14px); }
    .user-library-form .row { gap:8px; }
    .user-library-form label { margin:0 0 4px; }
    .user-library-form button { min-height:38px; border-radius:6px; background:#111827; color:#fff; }
    .user-library-note { color:var(--muted); font-size:11px; line-height:1.5; margin-top:8px; }
    .user-library-error { color:var(--bad); font-size:12px; min-height:18px; }
    .chat-section { display:grid; grid-template-rows:1fr auto; background:#fbfbfd; overflow:hidden; position:relative; }
    .chat-section.hidden { display:none; }
    .chat-head { background:#fff; border-bottom:1px solid var(--line); padding:16px 20px 13px; }
    .chat-headline { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .chat-title { margin-top:4px; font-size:19px; font-weight:820; }
    .chat-subtitle { color:var(--muted); font-size:12px; margin-top:3px; }
    .message-list { overflow:auto; padding:42px 24px 36px; scroll-behavior:smooth; }
    .message-list::before { content:""; display:block; max-width:900px; height:1px; margin:0 auto; background:transparent; }
    .msg { display:flex; gap:10px; max-width:900px; margin:0 auto 18px; }
    .msg.user { flex-direction:row-reverse; }
    .avatar { width:30px; height:30px; flex:0 0 auto; border-radius:8px; display:grid; place-items:center; border:1px solid var(--line); background:#fff; font-size:12px; font-weight:850; color:#111; }
    .msg.assistant .avatar { background:var(--accent); color:#fff; border-color:var(--accent); }
    .msg.user .avatar { background:var(--accent-soft); color:#374151; border-color:rgba(17,24,39,.12); }
    .bubble { max-width:min(760px, calc(100% - 42px)); border:1px solid var(--line); border-radius:8px; padding:12px 13px; background:#fff; box-shadow:var(--shadow-soft); }
    .msg.user .bubble { background:var(--accent-soft); border-color:rgba(17,24,39,.12); }
    .typing .bubble { min-width:112px; }
    .bubble-actions { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; padding-top:9px; border-top:1px solid var(--line); }
    .bubble-action { min-height:30px; padding:5px 10px; border:1px solid rgba(17,24,39,.12); border-radius:999px; background:#fff; color:#374151; font-size:12px; font-weight:820; box-shadow:none; }
    .bubble-action:hover, .bubble-action:focus-visible { background:var(--accent-soft); border-color:rgba(17,24,39,.14); outline:none; }
    .thinking-row { display:flex; flex-direction:column; gap:7px; align-items:flex-start; min-height:42px; }
    .thinking-pulse { display:flex; gap:9px; align-items:center; height:20px; }
    .thinking-note { color:var(--muted); font-size:12px; line-height:1.4; }
    .bubble-spinner { display:inline-block; width:14px; height:14px; border:2px solid rgba(17,24,39,.12); border-top-color:var(--accent); border-radius:999px; animation:spin .8s linear infinite; }
    .typing-dots { display:flex; gap:5px; align-items:center; height:18px; }
    .typing-dots span { width:6px; height:6px; border-radius:999px; background:var(--accent); animation:typingPulse 1.15s infinite ease-in-out; }
    .typing-dots span:nth-child(2) { animation-delay:.16s; }
    .typing-dots span:nth-child(3) { animation-delay:.32s; }
    @keyframes typingPulse { 0%, 80%, 100% { opacity:.25; transform:translateY(0); } 40% { opacity:1; transform:translateY(-4px); } }
    .spinner { display:inline-block; width:14px; height:14px; border:2px solid rgba(255,255,255,.45); border-top-color:#fff; border-radius:999px; animation:spin .8s linear infinite; vertical-align:-2px; margin-right:6px; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .bubble-meta { font-size:11px; color:var(--muted); margin-bottom:6px; }
    .bubble-text { white-space:pre-wrap; line-height:1.56; font-size:14px; overflow-wrap:anywhere; }
    .composer { border-top:1px solid var(--line); background:#fff; padding:14px 22px 18px; }
    .live-progress { max-width:900px; min-height:20px; margin:0 auto 8px; color:var(--muted); font-size:12px; line-height:1.35; display:flex; align-items:center; gap:7px; }
    .live-progress.active { color:#374151; }
    .live-progress .mini-dot { width:6px; height:6px; border-radius:999px; background:var(--accent); animation:typingPulse 1.15s infinite ease-in-out; flex:0 0 auto; }
    .composer-box { max-width:900px; margin:0 auto; display:grid; grid-template-columns:auto auto 1fr auto; gap:8px; align-items:end; border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; box-shadow:var(--shadow); }
    .composer-box textarea { border-color:transparent; padding:8px 6px; min-height:42px; max-height:180px; overflow-y:auto; }
    .composer-box textarea:focus { border-color:transparent; box-shadow:none; }
    .tool-button { background:#fff; color:#111; border:1px solid var(--line); }
    .tool-button:hover { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:#111827; }
    .side-rail { border-left:1px solid var(--line); background:#fff; padding:14px 12px; overflow:auto; }
    .workspace-kicker { color:var(--muted); font-size:11px; font-weight:750; text-transform:uppercase; letter-spacing:0; }
    .workspace-title { margin-top:2px; font-size:18px; line-height:1.2; font-weight:850; }
    .tool-panel { border:1px solid var(--line); border-radius:8px; padding:13px; margin-bottom:12px; background:#fff; box-shadow:var(--shadow-soft); }
    .action-panel { position:relative; cursor:pointer; transition:border-color .15s ease, box-shadow .15s ease, background .15s ease; }
    .action-panel:hover .title, .action-panel:focus-visible .title, .action-panel:hover .workspace-title, .action-panel:focus-visible .workspace-title { color:#111827; }
    .action-panel:hover .panel-jump, .action-panel:focus-visible .panel-jump { opacity:1; }
    .action-panel .panel-actions, .action-panel .metric-grid, .action-panel details, .action-panel button, .action-panel input, .action-panel select, .action-panel textarea { position:relative; z-index:2; }
    .action-panel-link { display:inline-flex; align-items:center; gap:6px; margin-top:14px; color:#6b7280; font-size:13px; font-weight:650; opacity:.72; transition:opacity .14s ease, color .14s ease; }
    .action-panel-link::after { content:"→"; }
    .action-panel:hover .action-panel-link, .action-panel:focus-visible .action-panel-link { opacity:.96; }
    .tool-panel.accent { border-color:var(--line); background:#fff; box-shadow:var(--shadow-soft); }
    .tool-panel .title { font-size:15px; font-weight:820; margin:2px 0 5px; }
    .tooltip-anchor { position:relative; display:inline-grid; place-items:center; width:20px; height:20px; border:1px solid rgba(17,24,39,.12); border-radius:999px; background:rgba(255,255,255,.66); color:#6b7280; font-size:12px; font-weight:760; cursor:help; }
    .tooltip-anchor::after { content:attr(data-tooltip); position:absolute; top:calc(100% + 8px); right:0; width:min(280px, 72vw); padding:9px 10px; border:1px solid rgba(17,24,39,.10); border-radius:10px; background:rgba(255,255,255,.92); color:#374151; box-shadow:0 8px 22px rgba(17,24,39,.06); backdrop-filter:saturate(160%) blur(12px); -webkit-backdrop-filter:saturate(160%) blur(12px); opacity:0; pointer-events:none; transition:opacity .14s ease; font-size:12px; font-weight:650; line-height:1.45; text-transform:none; letter-spacing:0; z-index:30; }
    .tooltip-anchor:hover::after, .tooltip-anchor:focus-visible::after { opacity:1; }
    .panel-jump { flex:0 0 auto; display:inline-grid; place-items:center; width:28px; height:28px; padding:0; color:#6b7280; font-size:16px; line-height:1; font-weight:650; opacity:.78; }
    button.panel-jump { min-height:28px; padding:0; border:1px solid rgba(17,24,39,.10); border-radius:999px; background:#fff; box-shadow:none; }
    button.panel-jump:hover, button.panel-jump:focus-visible { opacity:1; border-color:rgba(17,24,39,.16); background:#f9fafb; outline:none; }
    .panel-actions { display:grid; gap:8px; margin-top:10px; }
    .mini-kicker { color:var(--muted); font-size:11px; text-transform:uppercase; margin-bottom:4px; }
    .meta { color:var(--muted); font-size:12px; line-height:1.45; }
    .row { display:flex; gap:8px; align-items:center; }
    .row > * { flex:1; }
    .chips, .tabs { display:flex; flex-wrap:wrap; gap:7px; }
    .chip, .tab { border:1px solid var(--line); background:#fff; color:#111; border-radius:999px; padding:7px 10px; min-height:34px; font-size:12px; }
    .tab { border-radius:8px; }
    .chip.active, .tab.active { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:#374151; }
    .chip small { color:var(--muted); margin-left:4px; }
    .selected-strip { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; min-height:24px; }
    .selected-strip span { border:1px solid rgba(17,24,39,.12); background:var(--accent-soft); color:#374151; border-radius:999px; padding:3px 8px; font-size:12px; }
    .switchline { display:flex; align-items:center; gap:8px; margin:8px 0; color:#374151; font-size:13px; }
    .switchline input { width:auto; }
    details.settings { border:1px solid var(--line); border-radius:8px; padding:11px 12px; background:#fff; margin-bottom:12px; box-shadow:var(--shadow-soft); transition:border-color .15s ease, box-shadow .15s ease, background .15s ease; }
    details.settings:hover, details.settings:focus-within { border-color:rgba(17,24,39,.14); box-shadow:inset 0 0 0 1px rgba(17,24,39,.08); background:#f9fafb; }
    details.settings summary { cursor:pointer; font-weight:820; font-size:13px; display:flex; align-items:center; justify-content:space-between; gap:8px; border-radius:10px; padding:2px 2px 8px; list-style:none; }
    details.settings summary::-webkit-details-marker { display:none; }
    details.settings summary::after { content:"设置"; flex:0 0 auto; color:#374151; border:1px solid rgba(17,24,39,.12); border-radius:999px; padding:3px 8px; font-size:11px; font-weight:850; background:var(--accent-soft); }
    details.settings[open] summary::after { content:"收起"; }
    .empty { color:var(--muted); padding:12px; border:1px dashed var(--line); border-radius:10px; background:#fff; font-size:13px; }
    .empty-action { width:100%; min-height:unset; display:block; text-align:left; color:var(--muted); padding:12px 28px 12px 12px; border:1px dashed var(--line); border-radius:10px; background:#fff; font-size:13px; font-weight:650; position:relative; transition:border-color .15s ease, box-shadow .15s ease, background .15s ease; }
    .artifact-card { border:1px solid var(--line); border-radius:8px; padding:10px 28px 10px 10px; margin:8px 0; background:#fff; position:relative; transition:border-color .15s ease, box-shadow .15s ease, background .15s ease; }
    button.artifact-card { width:100%; text-align:left; color:inherit; cursor:pointer; }
    .artifact-card .title { font-size:13px; font-weight:750; margin-bottom:5px; }
    .status-mini { border:1px solid var(--line); border-radius:8px; padding:9px 32px 9px 10px; background:#fbfbfd; font-size:12px; color:#374151; }
    button.status-mini { width:100%; min-height:auto; text-align:left; position:relative; cursor:pointer; }
    .query-card { max-width:900px; margin:0 auto 22px; border:1px solid rgba(17,24,39,.12); border-radius:8px; background:#fff; box-shadow:var(--shadow-soft); padding:16px; }
    .query-card-title { font-size:18px; font-weight:850; margin-bottom:5px; }
    .query-card-text { color:var(--muted); font-size:13px; line-height:1.55; }
    .quick-prompts { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .quick-prompt { border:1px solid var(--line); background:#fff; color:#111; border-radius:999px; min-height:32px; padding:7px 10px; font-size:12px; font-weight:720; }
    .quick-prompt:hover { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:#374151; }
    .rail-caption { color:var(--muted); font-size:12px; line-height:1.45; margin:-3px 0 2px; }
    .control-note { border:1px solid rgba(17,24,39,.12); background:var(--accent-soft); color:#374151; border-radius:8px; padding:9px 10px; font-size:12px; line-height:1.45; margin-top:10px; }
    .status-mini.running { border-color:rgba(17,24,39,.12); background:var(--accent-soft); color:#374151; }
    body::before { content:""; position:fixed; inset:0; pointer-events:none; background:linear-gradient(90deg, rgba(17,24,39,.025), transparent 28%, rgba(17,24,39,.018)); z-index:-1; }
    header { height:64px; padding:0 22px; background:rgba(255,255,255,.88); backdrop-filter:blur(18px); }
    main { height:calc(100vh - 64px); grid-template-columns:316px minmax(560px, 1fr) 372px; }
    .brand-dot { width:36px; height:36px; border-radius:10px; box-shadow:0 10px 22px rgba(17,24,39,.10); }
    .brand-copy span { font-size:12px; }
    .pill { min-height:28px; font-weight:760; }
    .left-rail { padding:18px 14px; gap:14px; background:rgba(255,255,255,.9); backdrop-filter:blur(16px); }
    .rail-header { display:grid; grid-template-columns:1fr 42px; gap:10px; }
    .rail-header button:first-child { min-height:48px; border-radius:14px; font-size:17px; box-shadow:0 12px 28px rgba(17,24,39,.10); }
    button.icon { width:42px; min-height:48px; border-radius:14px; font-size:18px; }
    .section-title { margin-bottom:10px; }
    h2 { color:#4b5563; font-size:12px; font-weight:900; letter-spacing:.04em; }
    .rail-caption { margin:0 0 10px; font-size:12px; color:#6b7280; }
    .history-panel { flex:1 1 auto; min-height:180px; max-height:none; }
    .history-list { gap:8px; padding:1px 2px 8px 0; }
    .session-row { grid-template-columns:1fr 34px; gap:6px; }
    .session-item { min-height:72px; border-color:transparent; border-radius:14px; padding:13px 14px; background:transparent; }
    .session-item:hover { background:#f9fafb; border-color:rgba(17,24,39,.12); box-shadow:inset 3px 0 0 rgba(17,24,39,.18); }
    .session-item.active, .session-item.pending { border-color:rgba(17,24,39,.12); background:#f9fafb; box-shadow:inset 3px 0 0 var(--accent); }
    .session-title { font-size:14px; font-weight:850; letter-spacing:0; }
    .session-preview { font-size:12px; line-height:1.45; }
    .delete-session { min-height:72px; width:34px; border-radius:12px; font-size:20px; background:transparent; opacity:.72; }
    .delete-session:hover { opacity:1; }
    .session-status { font-size:12px; margin-top:8px; }
    .nav-button { border-radius:14px; min-height:44px; padding:10px 12px; }
    .evidence-card, .tool-panel, details.settings, .library-panel { border-radius:14px; }
    .evidence-card.action-panel { background:#fff; border-color:var(--line); }
    .evidence-card.action-panel .section-title { padding-right:42px; }
    .action-panel:hover .title, .action-panel:focus-visible .title { color:#111827; }
    .action-panel:hover .panel-jump, .action-panel:focus-visible .panel-jump { opacity:1; background:#f9fafb; }
    .metric { border-radius:12px; background:#fff; }
    .metric-button { min-height:78px; padding:12px 13px; }
    .metric strong { font-size:28px; }
    .chat-section { background:linear-gradient(180deg,#fbfbfd 0%,#fff 62%,#fbfbfd 100%); }
    .message-list { padding:46px 30px 42px; }
    .msg { max-width:920px; margin-bottom:22px; animation:messageIn .18s ease-out; }
    @keyframes messageIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    .avatar { width:40px; height:40px; border-radius:12px; font-size:13px; box-shadow:0 1px 2px rgba(17,17,17,.05); }
    .msg.assistant .avatar { box-shadow:0 1px 2px rgba(17,24,39,.10); }
    .bubble { max-width:min(780px, calc(100% - 54px)); border-radius:16px; padding:15px 16px; background:rgba(255,255,255,.96); box-shadow:0 1px 2px rgba(17,17,17,.045); }
    .msg.user .bubble { background:#f3f4f6; }
    .bubble-meta { font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.03em; }
    .bubble-text { font-size:14px; line-height:1.68; }
    .typing .bubble { min-width:220px; }
    .thinking-row { gap:9px; }
    .thinking-note { font-size:12px; color:#374151; }
    .query-card { max-width:920px; margin-top:6px; border-radius:20px; padding:24px; background:#fff; box-shadow:0 1px 2px rgba(17,17,17,.045); }
    .query-card-title { font-size:24px; letter-spacing:0; }
    .query-card-text { max-width:680px; font-size:14px; }
    .quick-prompts { margin-top:18px; }
    .quick-prompt { border-radius:12px; min-height:38px; padding:9px 12px; }
    .composer { padding:12px 26px 22px; background:rgba(255,255,255,.9); backdrop-filter:blur(18px); }
    .live-progress { max-width:920px; margin-bottom:10px; padding:0 4px; font-weight:760; }
    .composer-box { max-width:920px; grid-template-columns:42px 42px 1fr 74px; gap:9px; border-radius:18px; padding:9px; box-shadow:0 1px 2px rgba(17,17,17,.06); }
    .composer-box textarea { min-height:46px; font-size:14px; }
    .tool-button { width:42px; min-height:42px; padding:0; border-radius:13px; overflow:hidden; font-size:12px; letter-spacing:0; }
    #sendButton { min-height:42px; border-radius:13px; }
    .side-rail { padding:18px 14px; background:rgba(255,255,255,.88); backdrop-filter:blur(16px); }
    .workspace-title { font-size:20px; letter-spacing:0; }
    .tool-panel { padding:15px; box-shadow:0 1px 2px rgba(17,17,17,.04); }
    .tool-panel.accent { box-shadow:0 1px 2px rgba(17,17,17,.04); }
    .panel-actions { gap:9px; }
    .panel-actions button { border-radius:12px; }
    input, textarea, select { border-radius:12px; }
    .chip, .tab { border-radius:12px; font-weight:760; }
    .chip.active, .tab.active { box-shadow:inset 0 0 0 1px rgba(17,24,39,.08); }
    details.settings { padding:13px 14px; }
    details.settings summary { font-size:14px; }
    .status-mini { border-radius:12px; line-height:1.45; }
    .library-view { padding:34px 28px 40px; }
    .library-grid { max-width:1060px; gap:16px; }
    .paper-row, .artifact-card { border-radius:12px; }
    .paper-row.clickable-row { --action-label:"打开"; }
    .section-link { border-radius:12px; background:linear-gradient(180deg,#fff,#fbfbfd); }
    :root {
      --ink:#111827;
      --muted:#6b7280;
      --line:rgba(17,24,39,.08);
      --soft:#f3f4f6;
      --panel:#ffffff;
      --accent:#4b5563;
      --accent-strong:#111827;
      --accent-soft:#f3f4f6;
      --bad:#ef4444;
      --warn:#a16207;
      --shadow:0 4px 6px -1px rgba(17,24,39,.05), 0 2px 4px -1px rgba(17,24,39,.03);
      --shadow-soft:0 1px 2px rgba(17,24,39,.035);
      --focus-ring:0 0 0 3px rgba(17,24,39,.10);
      --ok:#10b981;
    }
    body { background:#fbfbfc; color:var(--ink); -webkit-font-smoothing:antialiased; }
    body::before { display:none; }
    button { border-radius:6px; background:var(--accent); font-weight:650; box-shadow:none; transition:background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease, opacity .15s ease; }
    button:hover { background:var(--accent-strong); }
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible { outline:none; box-shadow:var(--focus-ring); border-color:var(--accent); }
    button.ghost, .tool-button, .nav-button, .chip, .tab, .bubble-action, button.panel-jump { background:#fff; color:#374151; border:1px solid var(--line); }
    button.ghost:hover, .tool-button:hover, .nav-button:hover, .chip:hover, .tab:hover, .bubble-action:hover, button.panel-jump:hover { background:#f3f4f6; color:#111827; border-color:rgba(17,24,39,.14); }
    button.black { background:#111827; color:#fff; box-shadow:var(--shadow-soft); }
    button.black:hover { background:#1f2937; }
    input, textarea, select { border-color:var(--line); border-radius:6px; background:#fff; }
    input:focus, textarea:focus, select:focus { border-color:var(--accent); box-shadow:var(--focus-ring); }
    header { height:56px; padding:0 24px; background:rgba(255,255,255,.86); border-bottom:1px solid var(--line); backdrop-filter:saturate(180%) blur(12px); -webkit-backdrop-filter:saturate(180%) blur(12px); z-index:50; }
    main { height:calc(100vh - 56px); grid-template-columns:260px minmax(520px, 1fr) 280px; background:#fbfbfc; }
    .brand-dot { width:30px; height:30px; border-radius:8px; background:#111827; box-shadow:none; font-size:12px; }
    .brand-copy h1 { font-size:15px; font-weight:720; }
    .brand-copy span { color:#6b7280; font-size:11px; font-weight:560; }
    .toolbar { gap:10px; }
    .status-cluster { display:flex; align-items:center; gap:8px; }
    .status-cluster::before { content:""; width:1px; height:16px; margin-right:2px; background:var(--line); }
    .health-dot { width:8px; height:8px; border-radius:999px; background:var(--ok); box-shadow:0 0 0 2px rgba(16,185,129,.18); flex:0 0 auto; }
    .health-dot.bad { background:var(--bad); box-shadow:0 0 0 2px rgba(239,68,68,.16); }
    .health-dot.building { background:var(--accent); box-shadow:0 0 0 2px rgba(17,24,39,.10); animation:typingPulse 1.15s infinite ease-in-out; }
    button.status-pill, .pill { min-height:26px; padding:4px 8px; border-color:var(--line); border-radius:999px; background:#fff; color:#6b7280; font-size:12px; font-weight:650; box-shadow:none; }
    button.status-pill::after, .nav-button::after { color:var(--accent); }
    button.status-pill:hover, button.status-pill:focus-visible, .nav-button:hover, .nav-button:focus-visible { border-color:rgba(17,24,39,.14); background:var(--accent-soft); box-shadow:none; }
    button.status-pill.ok, .pill.ok { border-color:rgba(16,185,129,.22); background:rgba(16,185,129,.08); color:#047857; }
    button.status-pill.bad, .pill.bad { border-color:rgba(239,68,68,.22); background:rgba(239,68,68,.08); color:#b42318; }
    .left-rail, .side-rail { padding:20px 16px; background:#fff; backdrop-filter:none; }
    .left-rail { border-right:1px solid var(--line); gap:18px; }
    .side-rail { border-left:1px solid var(--line); padding:18px 16px; }
    .rail-header { grid-template-columns:1fr 38px; gap:8px; }
    .rail-header button:first-child { min-height:40px; border-radius:6px; font-size:13px; box-shadow:var(--shadow-soft); }
    button.icon { width:38px; min-height:40px; border-radius:6px; font-size:16px; }
    h2 { color:#6b7280; font-size:11px; font-weight:720; text-transform:uppercase; letter-spacing:.04em; }
    .section-title { margin-bottom:12px; }
    .history-list { gap:2px; padding:0; }
    .session-row { grid-template-columns:1fr 28px; gap:4px; }
    .session-item { min-height:52px; padding:8px 10px; border-radius:6px; background:#fff; border-color:transparent; box-shadow:none; }
    .session-item:hover, .session-item:focus-visible { background:#f3f4f6; border-color:transparent; box-shadow:none; }
    .session-item.active, .session-item.pending { background:var(--accent-soft); border-color:transparent; color:var(--accent); box-shadow:none; }
    .session-title { font-size:13px; font-weight:650; }
    .session-preview { margin-top:3px; font-size:12px; line-height:1.35; }
    .session-status { margin-top:5px; color:var(--accent); font-size:11px; font-weight:650; }
    .session-status .mini-dot, .live-progress .mini-dot, .typing-dots span { background:var(--accent); }
    .delete-session { min-height:52px; width:28px; border-radius:6px; background:#fff; font-size:17px; opacity:0; }
    .session-row:hover .delete-session { opacity:1; }
    .delete-session:hover { background:#fff1f0; border-color:rgba(239,68,68,.18); color:var(--bad); }
    .nav-button { min-height:38px; padding:8px 10px; border-radius:6px; font-size:13px; }
    .nav-button.active { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:var(--accent); }
    .evidence-card, .tool-panel, details.settings, .library-panel, .query-card { border-color:var(--line); border-radius:8px; background:#fff; box-shadow:var(--shadow-soft); }
    .evidence-card.action-panel { border-color:var(--line); }
    .tool-panel { padding:20px 18px; margin-bottom:16px; background:rgba(255,255,255,.66); border-color:rgba(17,24,39,.09); box-shadow:0 1px 2px rgba(17,24,39,.018); backdrop-filter:saturate(155%) blur(16px); -webkit-backdrop-filter:saturate(155%) blur(16px); }
    .tool-panel.accent { border-left:1px solid rgba(17,24,39,.09); box-shadow:0 1px 2px rgba(17,24,39,.018); }
    .tool-panel .title { font-size:17px; line-height:1.45; font-weight:660; color:#111827; }
    .workspace-kicker, .mini-kicker { color:#6b7280; font-size:11px; font-weight:650; letter-spacing:.04em; }
    .workspace-title { font-size:18px; font-weight:760; }
    .panel-actions button { border-radius:8px; min-height:42px; font-size:14px; font-weight:680; }
    .action-panel-link, .panel-jump, .section-action { color:#6b7280; }
    .metric-grid { gap:8px; }
    .metric-button { min-height:68px; padding:10px; grid-template-columns:minmax(0,1fr) 24px; border-color:var(--line); background:#fbfbfc; border-radius:6px; }
    .metric-button::before, .clickable-row::before, .status-link::before, .section-link::before, .empty-action::before { background:var(--accent); }
    .metric-button::after, .clickable-row::after, .status-link::after, .empty-action::after, .section-link::after { color:var(--accent); }
    .metric-button:hover, .metric-button:focus-visible, .clickable-row:hover, .clickable-row:focus-visible, .status-link:hover, .status-link:focus-visible, .action-panel:hover, .action-panel:focus-visible, .empty-action:hover, .empty-action:focus-visible, .section-link:hover, .section-link:focus-visible { border-color:rgba(17,24,39,.14); background:#f9fafb; box-shadow:none; }
    .metric-button:hover::before, .metric-button:focus-visible::before, .clickable-row:hover::before, .clickable-row:focus-visible::before, .status-link:hover::before, .status-link:focus-visible::before, .section-link:hover::before, .section-link:focus-visible::before, .empty-action:hover::before, .empty-action:focus-visible::before { opacity:.42; }
    .metric-top > span { min-width:0; }
    .metric strong { font-size:22px; color:#111827; }
    .metric-label { white-space:nowrap; font-size:11px; }
    .metric-open { width:24px; height:24px; border-color:var(--line); color:var(--accent); }
    .metric-button:hover .metric-open, .metric-button:focus-visible .metric-open { background:var(--accent-soft); border-color:rgba(17,24,39,.14); }
    .metric-button:hover strong, .metric-button:focus-visible strong, .paper-row:hover .paper-title, .paper-row:focus-visible .paper-title, .artifact-card:hover .title, .artifact-card:focus-visible .title, .action-panel:hover .title, .action-panel:focus-visible .title, .action-panel:hover .workspace-title, .action-panel:focus-visible .workspace-title { color:#111827; }
    .chat-section { background:#fbfbfc; }
    .message-list { padding:32px; }
    .msg { max-width:900px; margin-bottom:24px; animation:messageIn .16s ease-out; }
    .avatar { width:30px; height:30px; border-radius:6px; background:#fff; border-color:var(--line); color:#374151; box-shadow:none; }
    .msg.assistant .avatar { background:var(--accent); border-color:var(--accent); color:#fff; box-shadow:none; }
    .msg.user .avatar { background:#fff; border-color:var(--line); color:#374151; }
    .bubble { max-width:min(760px, calc(100% - 42px)); padding:14px 18px; border-radius:12px; background:#fff; border-color:var(--line); box-shadow:var(--shadow-soft); }
    .msg.user .bubble { background:#fff; border-color:var(--line); border-bottom-right-radius:4px; }
    .msg.assistant .bubble { background:transparent; border-color:transparent; box-shadow:none; padding:4px 0 0; }
    .bubble-meta { color:#6b7280; font-size:11px; font-weight:700; text-transform:none; letter-spacing:0; }
    .msg.assistant .bubble-meta { color:var(--accent); }
    .bubble-text { font-size:14px; line-height:1.62; }
    .thinking-note { color:var(--accent); }
    .bubble-spinner { border-color:rgba(17,24,39,.10); border-top-color:var(--accent); }
    .composer { padding:0 32px 32px; background:transparent; border-top:0; backdrop-filter:none; }
    .live-progress { max-width:920px; min-height:22px; margin:0 auto 10px; color:#6b7280; font-weight:600; }
    .live-progress.active { color:var(--accent); }
    .composer-box { max-width:920px; grid-template-columns:40px 40px 1fr 88px; gap:8px; padding:12px 16px; border-radius:12px; border-color:var(--line); background:#fff; box-shadow:var(--shadow); align-items:end; }
    .composer-box:focus-within { border-color:var(--accent); box-shadow:var(--focus-ring); }
    .composer-box textarea { min-height:44px; border:0; border-radius:6px; padding:7px 4px; }
    .composer-box textarea:focus { box-shadow:none; }
    .tool-button { width:40px; min-height:40px; border-radius:6px; font-size:12px; }
    #sendButton { width:88px; min-width:88px; min-height:40px; padding:0 8px; border-radius:6px; background:#111827; color:#fff; display:inline-flex; align-items:center; justify-content:center; gap:6px; white-space:nowrap; }
    #sendButton:hover, #sendButton:focus-visible { background:#1f2937; border-color:rgba(17,24,39,.12); }
    .query-card { max-width:920px; margin-top:0; padding:22px; border-radius:12px; }
    .query-card-title { font-size:22px; font-weight:760; }
    .query-card-text { max-width:680px; color:#6b7280; }
    .quick-prompts { margin-top:16px; }
    .quick-prompt, .chip, .tab { min-height:32px; padding:7px 10px; border-radius:6px; font-size:12px; font-weight:650; }
    .quick-prompt:hover { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:var(--accent); }
    .chip.active, .tab.active { border-color:rgba(17,24,39,.14); background:var(--accent-soft); color:var(--accent); box-shadow:none; }
    details.settings { padding:12px 13px; }
    details.settings:hover, details.settings:focus-within { border-color:rgba(17,24,39,.14); background:#fff; box-shadow:none; }
    details.settings summary { font-size:13px; font-weight:720; }
    details.settings summary::after, .selected-strip span { color:var(--accent); border-color:rgba(17,24,39,.14); background:var(--accent-soft); }
    .control-note { border-color:rgba(17,24,39,.10); background:rgba(255,255,255,.68); color:#6b7280; border-radius:6px; backdrop-filter:saturate(150%) blur(10px); -webkit-backdrop-filter:saturate(150%) blur(10px); }
    .tooltip-anchor { color:#6b7280; border-color:rgba(17,24,39,.12); }
    .tooltip-anchor::after, .metric-button::after { border-color:var(--line); border-radius:8px; box-shadow:var(--shadow); }
    .library-view { background:#fff; padding:32px; }
    .library-grid { max-width:1060px; gap:16px; }
    .library-panel { padding:14px; border-radius:8px; }
    .user-library-form { background:rgba(255,255,255,.58); border-color:rgba(17,24,39,.08); }
    .section-link { border-radius:6px; background:#fbfbfc; }
    .paper-row, .artifact-card, .status-mini, .empty-action, .empty { border-radius:6px; border-color:var(--line); }
    .paper-status { border-color:rgba(16,185,129,.22); background:rgba(16,185,129,.08); color:#047857; }
    .paper-status.frontier, .paper-status.pending { border-color:var(--line); background:#f9fafb; color:#4b5563; }
    .paper-reason a { color:var(--accent); }
    .status-mini.running { border-color:rgba(17,24,39,.14); background:#f9fafb; color:#374151; }
    .spinner { width:13px; height:13px; margin-right:0; border-color:rgba(255,255,255,.42); border-top-color:#fff; flex:0 0 auto; }
    .action-panel::before, .action-panel::after { content:none !important; display:none !important; }
    .tool-panel.action-panel:hover, .tool-panel.action-panel:focus-visible { border-color:rgba(17,24,39,.13); background:rgba(255,255,255,.78); box-shadow:0 10px 28px rgba(17,24,39,.035); outline:none; }
    .tool-panel.action-panel:hover .title, .tool-panel.action-panel:focus-visible .title, .tool-panel.action-panel:hover .workspace-title, .tool-panel.action-panel:focus-visible .workspace-title { color:#111827; }
    .tool-panel .mini-kicker { display:none; }
    .tool-panel .section-title { margin-bottom:20px; align-items:flex-start; gap:14px; }
    .tool-panel .action-panel-link { display:none; }
    .tool-panel .tooltip-anchor { width:24px; height:24px; margin-top:1px; border-color:rgba(17,24,39,.12); color:#6b7280; background:rgba(255,255,255,.58); font-size:13px; font-weight:680; }
    .tool-panel .tooltip-anchor:hover, .tool-panel .tooltip-anchor:focus-visible { border-color:rgba(17,24,39,.18); color:#374151; background:rgba(255,255,255,.88); }
    .tool-panel .panel-actions { gap:10px; margin-top:22px; }
    .tool-panel .panel-actions button { width:100%; min-height:42px; border:1px solid rgba(17,24,39,.10); background:#111827; color:#fff; box-shadow:none; }
    .tool-panel .panel-actions button:hover, .tool-panel .panel-actions button:focus-visible { background:#1f2937; border-color:rgba(17,24,39,.10); box-shadow:none; }
    .tool-panel .panel-actions button.ghost { background:rgba(255,255,255,.72); color:#374151; border-color:rgba(17,24,39,.12); }
    .tool-panel .panel-actions button.ghost:hover, .tool-panel .panel-actions button.ghost:focus-visible { background:rgba(255,255,255,.92); color:#111827; border-color:rgba(17,24,39,.16); }
    .side-rail details.settings summary::after { content:none; display:none; }
    .side-rail details.settings:hover, .side-rail details.settings:focus-within { border-color:rgba(17,24,39,.13); background:rgba(255,255,255,.72); box-shadow:none; }
    :root {
      --accent:#374151;
      --accent-strong:#111827;
      --accent-soft:rgba(17,24,39,.045);
      --focus-ring:0 0 0 3px rgba(17,24,39,.075);
    }
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible { border-color:rgba(17,24,39,.18); box-shadow:var(--focus-ring); }
    input:focus, textarea:focus, select:focus { border-color:rgba(17,24,39,.18); box-shadow:var(--focus-ring); }
    button.status-pill::after, .nav-button::after, .clickable-row::after, .status-link::after, .empty-action::after, .section-link::after { content:none; display:none; }
    .clickable-row::before, .status-link::before, .section-link::before, .empty-action::before, .metric-button::before { content:none; display:none; }
    button.status-pill:hover, button.status-pill:focus-visible, .nav-button:hover, .nav-button:focus-visible, .metric-button:hover, .metric-button:focus-visible, .clickable-row:hover, .clickable-row:focus-visible, .status-link:hover, .status-link:focus-visible, .action-panel:hover, .action-panel:focus-visible, .empty-action:hover, .empty-action:focus-visible, .section-link:hover, .section-link:focus-visible { border-color:rgba(17,24,39,.13); background:rgba(255,255,255,.74); box-shadow:none; }
    button.status-pill.ok, .pill.ok { border-color:rgba(16,185,129,.18); background:rgba(16,185,129,.07); color:#047857; }
    .nav-button.active, .session-item.active, .session-item.pending, .chip.active, .tab.active { border-color:rgba(17,24,39,.12); background:rgba(17,24,39,.045); color:#111827; box-shadow:none; }
    .session-status, .live-progress.active, .msg.assistant .bubble-meta, .thinking-note { color:#374151; }
    .msg.assistant .avatar { background:#111827; border-color:#111827; }
    .health-dot.building, .session-status .mini-dot, .live-progress .mini-dot, .typing-dots span { background:#374151; box-shadow:0 0 0 2px rgba(17,24,39,.10); }
    .side-rail { background:rgba(255,255,255,.72); backdrop-filter:saturate(140%) blur(18px); -webkit-backdrop-filter:saturate(140%) blur(18px); }
    .tool-panel, .side-rail details.settings, .control-note { background:rgba(255,255,255,.58); border-color:rgba(17,24,39,.08); box-shadow:0 1px 2px rgba(17,24,39,.018); backdrop-filter:saturate(145%) blur(16px); -webkit-backdrop-filter:saturate(145%) blur(16px); }
    .tool-panel.accent { border-left:1px solid rgba(17,24,39,.08); }
    .tool-panel.action-panel:hover, .tool-panel.action-panel:focus-visible, .side-rail details.settings:hover, .side-rail details.settings:focus-within { border-color:rgba(17,24,39,.12); background:rgba(255,255,255,.68); box-shadow:0 8px 24px rgba(17,24,39,.028); }
    .tool-panel .section-title { display:grid; grid-template-columns:minmax(0,1fr) 24px; align-items:start; gap:12px; margin-bottom:18px; }
    .tool-panel .title { min-width:0; font-size:16px; line-height:1.6; font-weight:640; overflow-wrap:anywhere; }
    .tool-panel .tooltip-anchor { width:22px; height:22px; margin-top:2px; font-size:12px; border-color:rgba(17,24,39,.11); background:rgba(255,255,255,.54); }
    .tool-panel .panel-actions { margin-top:18px; gap:9px; }
    .section-action { display:none; }
    .section-link { padding:10px 12px; }
    .section-right { gap:8px; }
    .metric-button { grid-template-columns:1fr; align-items:start; min-height:78px; gap:0; padding:14px 12px; background:rgba(255,255,255,.58); }
    .metric-top { align-items:flex-start; }
    .metric-top > span { display:flex; min-width:0; flex-direction:column; gap:7px; }
    .metric-label { margin-top:0; line-height:1.25; white-space:normal; }
    .metric-open { display:none; }
    .metric-button:hover .metric-open, .metric-button:focus-visible .metric-open { background:rgba(255,255,255,.9); border-color:rgba(17,24,39,.16); }
    .metric-button:hover strong, .metric-button:focus-visible strong, .section-link:hover h2, .section-link:focus-visible h2, .quick-prompt:hover, .paper-reason a { color:#111827; }
    .selected-strip span, .paper-status, .paper-status.frontier, .paper-status.pending { border-color:rgba(17,24,39,.10); background:rgba(17,24,39,.045); color:#374151; }
    .bubble-action { border-color:rgba(17,24,39,.10); background:#fff; color:#374151; border-radius:6px; }
    .bubble-action:hover, .bubble-action:focus-visible, .quick-prompt:hover { border-color:rgba(17,24,39,.14); background:#f3f4f6; color:#111827; }
    .query-card { border-color:rgba(17,24,39,.08); }
    .side-rail, .tool-panel, .tool-panel .section-title { overflow:visible; }
    .side-rail { display:flex; flex-direction:column; min-height:0; overflow-y:auto; gap:18px; padding-bottom:30px; }
    .tool-panel { position:relative; }
    .tool-panel:has(.tooltip-anchor:hover), .tool-panel:has(.tooltip-anchor:focus-visible) { z-index:80; }
    .tooltip-anchor::after { content:none; display:none; }
    .tooltip-popover { position:fixed; left:0; top:0; z-index:2000; max-width:min(292px, calc(100vw - 24px)); padding:10px 11px; border:1px solid rgba(17,24,39,.10); border-radius:8px; background:rgba(255,255,255,.76); color:#374151; box-shadow:0 14px 34px rgba(17,24,39,.10); backdrop-filter:saturate(160%) blur(18px); -webkit-backdrop-filter:saturate(160%) blur(18px); font-size:12px; font-weight:620; line-height:1.55; opacity:0; pointer-events:none; transform:translate3d(0,0,0); transition:opacity .12s ease; }
    .tooltip-popover.visible { opacity:1; }
    .side-rail > * { flex:0 0 auto; margin-bottom:0; }
    .side-rail > .section-title { margin-bottom:0; }
    .user-library-tab { order:2; min-height:64px; padding:0; display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:0; border:1px solid rgba(17,24,39,.08); border-radius:8px; background:rgba(255,255,255,.50); color:#374151; cursor:pointer; box-shadow:0 1px 2px rgba(17,24,39,.012); backdrop-filter:saturate(145%) blur(16px); -webkit-backdrop-filter:saturate(145%) blur(16px); }
    .user-library-tab:focus-visible { outline:none; border-color:rgba(17,24,39,.18); box-shadow:var(--focus-ring); }
    .user-library-tab:hover { border-color:rgba(17,24,39,.13); background:rgba(255,255,255,.72); color:#111827; }
    #searchSettings { order:3; margin-top:0; min-height:112px; }
    .side-rail > .tool-panel.action-panel { order:1; }
    .side-rail .tool-panel.action-panel:last-of-type { order:4; margin-top:clamp(10px, 2.4vh, 24px); }
    .user-library-tab-label { min-width:0; display:flex; align-items:center; gap:8px; padding:10px 0 10px 12px; font-size:13px; font-weight:680; line-height:1.2; }
    .user-library-tab .pill { flex:0 0 auto; }
    .user-library-tab-arrow { flex:0 0 auto; padding:0 12px 0 0; color:#6b7280; font-size:15px; line-height:1; }
    .user-library-tab:hover .user-library-tab-arrow, .user-library-tab:focus-visible .user-library-tab-arrow { color:#111827; }
    @media (max-width:1180px) { main { grid-template-columns:260px minmax(480px, 1fr); } .side-rail { display:none; } }
    @media (max-width:1080px) { body { overflow:auto; } main { height:auto; grid-template-columns:1fr; } aside, section { border:0; border-bottom:1px solid var(--line); max-height:none; } .left-rail { overflow:auto; } .history-panel { flex:0 0 260px; } .chat-section { min-height:72vh; } .composer-box { grid-template-columns:40px 40px 1fr 76px; } .library-grid { grid-template-columns:1fr; } .side-rail { display:block; } }
    @media (max-width:640px) { header { padding:0 14px; } main { grid-template-columns:1fr; } .brand-copy span, #provider { display:none; } .toolbar { gap:6px; } .status-cluster::before { display:none; } .pill { padding:4px 8px; } .message-list { padding:26px 14px 30px; } .avatar { width:34px; height:34px; } .bubble { max-width:calc(100% - 44px); } .composer { padding:10px 12px 16px; } .composer-box { grid-template-columns:1fr 64px; padding:9px; } .tool-button { min-height:34px; width:auto; padding:0 10px; } .tool-button:nth-child(1), .tool-button:nth-child(2) { grid-row:2; } }
  </style>
</head>
<body>
  <header>
    <div class="brand-mark"><div class="brand-dot">Rα</div><div class="brand-copy"><h1>Research Alpha</h1><span>Deep research for paper-grounded ideas</span></div></div>
    <div class="toolbar">
      <div class="status-cluster">
        <span id="healthDot" class="health-dot bad" aria-hidden="true"></span>
        <button id="provider" class="status-pill" onclick="metricAction('settings')" title="查看模型与检索设置">加载中</button>
        <button id="strictStatus" class="status-pill" onclick="metricAction('excellent')" title="查看核心优秀论文库">检查中</button>
      </div>
    </div>
  </header>
  <main>
    <aside class="left-rail">
      <div class="rail-header">
        <button class="black" onclick="newChat()">New search</button>
        <button class="ghost icon" onclick="refresh()" title="刷新">↻</button>
      </div>
      <div class="history-panel">
        <div class="section-title"><h2>Searches</h2><span id="sessionCount" class="pill">0</span></div>
        <div id="sessionHistory" class="history-list"></div>
      </div>
      <button id="libraryNav" class="nav-button" onclick="showLibrary()" title="查看论文、逻辑卡、idea 和运行记录">Evidence library<span id="libraryCount" class="pill">0</span></button>
      <div id="corpusStatusPanel" class="evidence-card action-panel" role="button" tabindex="0" onclick="panelNavigate(event, 'excellent')" onkeydown="activatePanel(event, 'excellent')" title="查看核心优秀论文库状态">
        <div class="section-title"><h2>Corpus status</h2><button class="panel-jump" onclick="metricAction('library')" title="打开证据库" aria-label="打开证据库">›</button></div>
        <div id="metrics"></div>
      </div>
    </aside>
    <section id="chatSection" class="chat-section">
      <div id="chatMessages" class="message-list"></div>
      <div class="composer">
        <div id="liveProgressLine" class="live-progress"></div>
        <div class="composer-box">
          <button class="tool-button" onclick="review()" title="用审稿人视角攻击当前 idea">审稿</button>
          <button class="tool-button" onclick="reviewLoop()" title="自动经过 3-5 轮审稿专家意见调整">循环</button>
          <textarea id="instruction" placeholder="说一个研究方向，或继续追问当前 idea"></textarea>
          <button id="sendButton" onclick="sendChat()">发送</button>
        </div>
      </div>
    </section>
    <section id="libraryView" class="library-view">
      <div style="max-width:980px;margin:0 auto 16px">
        <div class="chat-headline">
          <div>
            <h2>Evidence library</h2>
            <div class="chat-title">Paper-grounded assets</div>
            <div class="chat-subtitle">优秀论文、趋势论文、逻辑卡和运行记录集中在这里；Poster 不会进入依据库。</div>
          </div>
          <button class="ghost" onclick="showChat()">返回对话</button>
        </div>
      </div>
      <div class="library-grid">
        <div id="userLibraryPanel" class="library-panel user-library-main"></div>
        <div id="papers" class="library-panel"></div>
        <div id="ideas" class="library-panel"></div>
        <div id="patterns" class="library-panel"></div>
        <div id="runs" class="library-panel"></div>
      </div>
    </section>
    <section class="side-rail">
      <div class="section-title"><div><div class="workspace-kicker">Run setup</div><div class="workspace-title">Evidence</div></div><button class="ghost" onclick="refresh()">刷新</button></div>
      <div class="tool-panel accent action-panel" role="button" tabindex="0" onclick="panelNavigate(event, 'excellent')" onkeydown="activatePanel(event, 'excellent')" title="查看核心优秀论文库">
        <div class="mini-kicker">Evidence</div>
        <div class="section-title"><div class="title">优秀论文逻辑库</div><span class="tooltip-anchor" tabindex="0" data-tooltip="核心参考只使用 CVPR、ICCV、ICLR、NeurIPS/NIPS、ICML、TPAMI 这类高水平来源的 Best、Nominee 和 Oral；其他会议只作辅助来源，高引用不单独入库。">?</span></div>
        <div class="action-panel-link">核心库</div>
        <div class="panel-actions">
          <button onclick="goldBuild()">扩充核心库</button>
          <button class="ghost" onclick="reviewLoop()">循环审稿</button>
        </div>
      </div>
      <div id="sideUserLibrary" class="user-library-tab" role="button" tabindex="0" onclick="panelNavigate(event, 'user-library')" onkeydown="activatePanel(event, 'user-library')"></div>
      <details id="searchSettings" class="settings">
        <summary>检索设置</summary>
        <label>会议组合</label>
        <div id="venuePresets" class="chips"></div>
        <details style="margin-top:10px">
          <summary>细选会议</summary>
          <label>会议分组</label><div id="venueTabs" class="tabs"></div>
          <label>会议选项</label><div id="venueGroups"></div>
        </details>
        <div id="selectedVenueStrip" class="selected-strip"></div>
        <label>主题过滤</label><input id="goldQuery" placeholder="可选，例如 research agents / multimodal reasoning" />
        <div class="row"><div><label>起始年份</label><input id="fromYear" type="number" value="2024" /></div><div><label>结束年份</label><input id="toYear" type="number" value="2026" /></div></div>
        <label>每个会议年份最多核验</label><input id="perVenue" type="number" min="2" max="40" value="8" />
        <label class="switchline"><input id="extractiveGenomes" type="checkbox" checked /> 同时生成可审计逻辑卡</label>
        <div class="row">
          <div><label>近期趋势源</label><select id="remote"></select></div>
          <div><label>输出语言</label><select id="lang"></select></div>
        </div>
        <div class="row">
          <div><label>模型</label><select id="providerSelect"></select></div>
          <div><label>idea 数</label><input id="ideaCount" type="number" min="1" max="8" value="3" /></div>
        </div>
        <label>近期文章检索上限</label><input id="limit" type="number" min="4" max="50" value="12" />
        <div class="control-note"><span class="tooltip-anchor" tabindex="0" data-tooltip="生成 idea 时会从对话抽取方向与约束；核心来源会路由到官方奖项页、OpenReview 或元数据核验源。">?</span> 自动抽取</div>
      </details>
      <div class="tool-panel action-panel" role="button" tabindex="0" onclick="panelNavigate(event, 'chat')" onkeydown="activatePanel(event, 'chat')" title="回到对话查看任务进度">
        <div class="section-title"><h2>Activity</h2><span id="jobCount" class="pill">0</span></div>
        <div class="action-panel-link">进度</div>
        <button id="jobFeed" class="status-mini status-link" onclick="metricAction('chat')" title="任务结果只会出现在中间对话">空闲</button>
      </div>
    </section>
  </main>
  <script>
    let appOptions = null;
    let selectedVenues = new Set(['ICLR', 'NeurIPS', 'ICML']);
    let selectedVenueGroup = 'core';
    let selectedVenuePreset = 'ml_core';
    let chatActive = false;
    let activeSessionId = null;
    let newChatMode = false;
    let activeView = 'chat';
    let lastJobStatuses = new Map();
    let transientMessagesByScope = {};
    let activityMessagesByScope = {};
    let draftScopeId = `draft:${Date.now()}`;
    let pendingAutoScroll = false;
    let chatRenderKey = '';
    let metricsRenderKey = '';
    let pendingNewSession = false;
    let pendingNewSessionJobId = null;
    let pendingNewSessionScope = null;
    let pendingIdeateJobsByScope = {};
    let pendingSearchesByScope = {};
    let progressByScope = {};
    let isBusy = false;
    let awaitingJobStart = false;
    let awaitingJobStartByScope = {};
    let instructionDraftsByScope = {};
    let libraryCollapsed = {};
    let refreshSeq = 0;
    let viewToken = 0;
    let refreshTimer = null;
    let lastStateError = '';
    const CORE_GOLD_UI_VENUES = new Set(['iclr', 'neurips', 'nips', 'icml', 'cvpr', 'iccv', 'tpami', 't-pami']);

    async function postJson(path, body) {
      const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body || {})});
      let json = {};
      try { json = await res.json(); } catch (_) { json = {}; }
      if (!res.ok) {
        throw requestErrorFromResponse(json, res.status);
      }
      return json;
    }
    async function fetchStateJson(path) {
      const res = await fetch(path);
      let json = {};
      try { json = await res.json(); } catch (_) { json = {}; }
      if (!res.ok) throw requestErrorFromResponse(json, res.status, '状态刷新失败');
      return json;
    }
    function requestErrorFromResponse(json={}, status=0, fallback='Request failed') {
      const error = new Error(json.error || fallback);
      error.status = Number(json.status || status || 0);
      error.code = json.code || '';
      error.retryable = Boolean(json.retryable);
      error.nextAction = json.next_action || '';
      error.details = json;
      return error;
    }
    function requestErrorMessage(error) {
      const message = error && error.message ? error.message : '请求失败，但服务没有返回详情。';
      const status = error && error.status ? `HTTP ${error.status}` : '';
      const code = error && error.code ? error.code : '';
      const prefix = [status, code].filter(Boolean).join(' · ');
      return prefix ? `${prefix}：${message}` : message;
    }
    function handleUiFailure(scope, label, error, options={}) {
      const message = cleanUiText(requestErrorMessage(error));
      const nextAction = error && error.nextAction ? cleanUiText(error.nextAction) : (error && error.retryable ? '刷新状态后重试。' : '按提示修正输入后重试。');
      setAwaitingJobStart(scope, false);
      clearProgress(scope);
      setBusy(false);
      if (options.clearIdeate) clearPendingIdeate(scope);
      if (options.clearSearch) removePendingSearch(scope);
      if (options.restoreText) restoreSubmittedInstruction(scope, options.restoreText);
      setScopeMessages(transientMessagesByScope, scope, scopeMessages(transientMessagesByScope, scope).filter(m => !m.thinking));
      pushActivityMessage(`${label}没有提交成功：${message}\n\n下一步：${nextAction}`, '请求失败', scope);
      renderSessionHistory(window.lastData || {});
      renderChat((window.lastData || {}).chat || {active:chatActive, messages:[]});
      renderLiveProgress(scope);
    }
    function bumpViewToken() { viewToken += 1; return viewToken; }
    function currentViewToken() { return viewToken; }
    function viewStillCurrent(token, scope='') {
      return Number(token) === Number(viewToken) && (!scope || scope === currentScope());
    }
    function renderChatIfCurrent(token, scope, chat) {
      if (!viewStillCurrent(token, scope)) return false;
      renderChat(chat);
      return true;
    }
    async function refreshIfCurrent(token, scope='') {
      if (!viewStillCurrent(token, scope)) {
        scheduleRefresh(700);
        return false;
      }
      await refresh();
      return true;
    }
    function handleUiFailureIfCurrent(token, scope, label, error, options={}) {
      if (!viewStillCurrent(token, scope)) {
        setAwaitingJobStart(scope, false);
        clearProgress(scope);
        if (options.clearIdeate) clearPendingIdeate(scope);
        if (options.clearSearch) removePendingSearch(scope);
        if (options.restoreText) restoreSubmittedInstruction(scope, options.restoreText);
        removeTransientThinking(scope);
        const nextAction = error && error.nextAction ? cleanUiText(error.nextAction) : '刷新状态后重试。';
        pushActivityMessage(`${label}没有提交成功：${cleanUiText(requestErrorMessage(error))}\n\n下一步：${nextAction}`, '请求失败', scope);
        scheduleRefresh(700);
        return;
      }
      handleUiFailure(scope, label, error, options);
    }
    function currentScope() { return activeSessionId && !newChatMode ? `session:${activeSessionId}` : draftScopeId; }
    function instructionEl() { return document.getElementById('instruction'); }
    function autosizeInstruction() {
      const instruction = instructionEl();
      if (!instruction) return;
      instruction.style.height = 'auto';
      instruction.style.height = `${Math.min(instruction.scrollHeight, 180)}px`;
    }
    function saveInstructionDraft(scope=currentScope()) {
      const instruction = instructionEl();
      if (!instruction || !scope) return;
      instructionDraftsByScope[scope] = instruction.value;
    }
    function restoreInstructionDraft(scope=currentScope()) {
      const instruction = instructionEl();
      if (!instruction) return;
      instruction.value = instructionDraftsByScope[scope] || '';
      autosizeInstruction();
    }
    function clearInstructionDraft(scope=currentScope()) {
      if (scope) delete instructionDraftsByScope[scope];
      if (scope === currentScope()) {
        const instruction = instructionEl();
        if (instruction) instruction.value = '';
        autosizeInstruction();
      }
    }
    function takeInstructionForSubmit(scope=currentScope()) {
      const instruction = instructionEl();
      const text = instruction ? instruction.value.trim() : '';
      if (!text) return '';
      clearInstructionDraft(scope);
      return text;
    }
    function restoreSubmittedInstruction(scope, text) {
      if (!scope || !text) return;
      instructionDraftsByScope[scope] = text;
      if (scope === currentScope()) restoreInstructionDraft(scope);
    }
    function nowTs() { return Date.now(); }
    function progressStepsForKind(kind='ideate') {
      if (kind === 'gold-build') return ['提交联网任务', '访问官方优秀论文源', '过滤 Poster 与普通论文', '写入优秀依据库', '生成可审计逻辑卡'];
      if (kind === 'quality-enrich') return ['提交质量核验', '读取趋势论文元数据', '核验 Best / Nominee / Oral 信号', '更新证据标签'];
      if (kind === 'review') return ['载入当前 idea', '审稿人攻击 novelty', '检查证据链与可行性', '形成修改或重想建议'];
      if (kind === 'review-loop') return ['载入当前 idea', '审稿专家攻击', '按意见改写', '重复 3-5 轮', '写入最终 idea'];
      if (kind === 'step') return ['读取会话记忆', '检索相关证据上下文', '推理用户修改意图', '更新当前 idea'];
      if (kind === 'cleanup') return ['读取证据上下文', '检查缺失依据', '核对不合格证据', '形成体检结果'];
      return ['提交研究问题', '抽取领域与约束', '检索优秀论文逻辑', '检索近期热点与局限性', '生成 idea 与证据评分', '等待写入新对话'];
    }
    function startProgress(scope, kind='ideate', label='') {
      if (!scope) return;
      progressByScope[scope] = {kind, label, startedAt: nowTs(), updatedAt: nowTs(), steps: progressStepsForKind(kind)};
      pendingAutoScroll = true;
      renderLiveProgress(scope);
    }
    function clearProgress(scope='') {
      if (scope) delete progressByScope[scope];
      renderLiveProgress(currentScope());
    }
    function progressText(scope) {
      const item = progressByScope[scope];
      if (!item) return '';
      const elapsed = Math.max(0, Math.floor((nowTs() - item.startedAt) / 1000));
      const steps = item.steps || progressStepsForKind(item.kind);
      const index = Math.min(steps.length - 1, Math.floor(elapsed / 8));
      const prefix = elapsed < 4 ? '刚刚开始' : `已运行 ${elapsed}s`;
      return `${prefix} · ${steps[index]}`;
    }
    function jobScope(job) {
      return job ? (job.ui_scope || (job.session_id ? `session:${job.session_id}` : '')) : '';
    }
    function isConversationBlockingJob(job) {
      return ['ideate', 'step', 'review', 'review-loop'].includes(String(job && job.kind || ''));
    }
    function runningJobsForScope(jobs, scope) { return (jobs || []).filter(job => job.status === 'running' && jobScope(job) === scope); }
    function runningConversationJobsForScope(jobs, scope) { return runningJobsForScope(jobs, scope).filter(isConversationBlockingJob); }
    function jobTsMs(job, key) {
      const raw = Number(job && job[key] || 0);
      if (!raw) return 0;
      return raw < 100000000000 ? raw * 1000 : raw;
    }
    function runningJobSilenceText(scope) {
      const jobs = ((window.lastData || {}).jobs || []);
      const running = runningJobsForScope(jobs, scope);
      if (!running.length) return '';
      const newest = running.reduce((best, item) => Math.max(best, jobTsMs(item, 'last_output_at') || jobTsMs(item, 'updated_at') || jobTsMs(item, 'created_at')), 0);
      if (!newest) return '';
      const silentSeconds = Math.max(0, Math.floor((nowTs() - newest) / 1000));
      if (silentSeconds < 18) return '';
      const lead = running.length > 1 ? `${running.length} 个任务仍在运行` : `${displayJobName(running[0])} 仍在运行`;
      return `${lead} · ${silentSeconds}s 没有新输出，正在等待远端模型或论文源返回`;
    }
    function liveJobLineForScope(scope) {
      return '';
    }
    function visibleProgressText(scope) {
      const silence = runningJobSilenceText(scope);
      if (silence) return cleanUiText(silence);
      const jobs = ((window.lastData || {}).jobs || []);
      const running = runningJobsForScope(jobs, scope);
      if (running.length > 1) return `${running.length} 个任务运行中`;
      if (running.length === 1) return '任务运行中';
      return cleanUiText(progressText(scope));
    }
    function sidebarProgressText(scope) {
      const jobs = ((window.lastData || {}).jobs || []);
      const running = runningJobsForScope(jobs, scope);
      if (running.length > 1) return `${running.length} 个任务运行中`;
      if (running.length === 1) return '任务运行中';
      return cleanUiText(progressText(scope));
    }
    function renderLiveProgress(scope=currentScope()) {
      const el = document.getElementById('liveProgressLine');
      if (!el) return;
      const text = visibleProgressText(scope);
      const busy = scopeIsBusy(scope);
      el.className = `live-progress ${busy || text ? 'active' : ''}`;
      el.innerHTML = text ? `<span class="mini-dot"></span><span>${esc(text)}</span>` : '';
    }
    function addPendingSearch(scope, title) {
      if (!scope) return;
      pendingSearchesByScope[scope] = {title: compact(title || 'New search', 80), createdAt: nowTs()};
    }
    function removePendingSearch(scope='') {
      if (scope) delete pendingSearchesByScope[scope];
    }
    function scopeMessages(store, scope) { return store[scope] || []; }
    function setScopeMessages(store, scope, messages) { store[scope] = messages; }
    function clearScopeMessages(scope) { delete transientMessagesByScope[scope]; delete activityMessagesByScope[scope]; clearProgress(scope); removePendingSearch(scope); }
    function pushUserMessage(text, scope=currentScope()) { setScopeMessages(transientMessagesByScope, scope, scopeMessages(transientMessagesByScope, scope).concat([{role:'user', meta:'你', text}])); pendingAutoScroll = true; }
    function pushThinking(label, scope=currentScope()) { setScopeMessages(transientMessagesByScope, scope, scopeMessages(transientMessagesByScope, scope).concat([{role:'assistant', meta:label || 'Research Alpha', text:'', thinking:true}])); pendingAutoScroll = true; }
    function pushActivityMessage(text, meta='任务', scope=currentScope()) { setScopeMessages(activityMessagesByScope, scope, scopeMessages(activityMessagesByScope, scope).concat([{role:'assistant', meta, text}]).slice(-8)); pendingAutoScroll = true; }
    function clearRefreshFailureMessages(scope='') {
      const shouldKeep = item => !(item && item.meta === '请求失败' && String(item.text || '').startsWith('状态刷新失败'));
      if (scope) {
        setScopeMessages(activityMessagesByScope, scope, scopeMessages(activityMessagesByScope, scope).filter(shouldKeep));
        return;
      }
      for (const key of Object.keys(activityMessagesByScope)) {
        setScopeMessages(activityMessagesByScope, key, scopeMessages(activityMessagesByScope, key).filter(shouldKeep));
      }
    }
    function removeTransientThinking(scope=currentScope()) { setScopeMessages(transientMessagesByScope, scope, scopeMessages(transientMessagesByScope, scope).filter(m => !m.thinking)); }
    function handleDedupedJob(scope, job, message='这个范围已有任务在运行，本次没有重复提交。') {
      if (job) mergeJobIntoLastData(job);
      setAwaitingJobStart(scope, false);
      removeTransientThinking(scope);
      pushActivityMessage(message + ' 我会继续显示已有任务的实时进度和最终结果。', '任务 · 继续等待', scope);
      renderLiveProgress(scope);
    }
    function isCoreGoldVenue(value) { return CORE_GOLD_UI_VENUES.has(String(value || '').trim().toLowerCase()); }
    function coreGoldVenueValues(values) { return (values || []).filter(isCoreGoldVenue); }
    function auxiliaryGoldVenueValues(values) { return (values || []).filter(value => !isCoreGoldVenue(value)); }
    function sourcePlanNote(plan) {
      const items = Array.isArray(plan) ? plan : [];
      const lines = items.map(item => {
        if (!item || typeof item !== 'object') return '';
        const venues = Array.isArray(item.venues) ? item.venues.join('、') : '';
        const label = item.label || item.remote || '来源';
        const mode = item.mode === 'metadata_only' ? '先入趋势区，quality-enrich 核验后才可进 Gold' : '直接按 Best/Nominee/Oral 严格入库';
        return venues ? `${venues} → ${label}（${mode}）` : '';
      }).filter(Boolean);
      return lines.length ? `\n\n核心优秀论文源路由：\n${lines.join('\n')}` : '';
    }
    function mergeJobIntoLastData(job) {
      if (!job || !job.id) return;
      const data = window.lastData || {};
      const jobs = (data.jobs || []).filter(item => Number(item.id) !== Number(job.id));
      window.lastData = {...data, jobs:[job].concat(jobs)};
    }
    function setPendingIdeate(scope, jobId=null) {
      if (!scope) return;
      pendingIdeateJobsByScope[scope] = {jobId: jobId ? Number(jobId) : null};
      pendingNewSession = true;
      pendingNewSessionJobId = jobId ? Number(jobId) : null;
      pendingNewSessionScope = scope;
    }
    function clearPendingIdeate(scope='') {
      if (scope) delete pendingIdeateJobsByScope[scope];
      if (!scope || pendingNewSessionScope === scope) {
        const entries = Object.entries(pendingIdeateJobsByScope);
        if (entries.length) {
          pendingNewSession = true;
          pendingNewSessionScope = entries[entries.length - 1][0];
          pendingNewSessionJobId = entries[entries.length - 1][1].jobId;
        } else {
          pendingNewSession = false;
          pendingNewSessionJobId = null;
          pendingNewSessionScope = null;
        }
      }
    }
    function pendingIdeateForScope(scope) { return scope ? pendingIdeateJobsByScope[scope] || null : null; }
    function setAwaitingJobStart(scope, next) {
      if (!scope) return;
      if (next) awaitingJobStartByScope[scope] = true;
      else delete awaitingJobStartByScope[scope];
      awaitingJobStart = Object.keys(awaitingJobStartByScope).length > 0;
    }
    function scopeAwaitingJobStart(scope) { return Boolean(scope && awaitingJobStartByScope[scope]); }
    function scopeIsBusy(scope) {
      const jobs = ((window.lastData || {}).jobs || []);
      return scopeAwaitingJobStart(scope) || Boolean(pendingIdeateForScope(scope)) || runningConversationJobsForScope(jobs, scope).length > 0;
    }
    function scopeHasRunningEvidenceJob(scope) {
      const jobs = ((window.lastData || {}).jobs || []);
      return runningJobsForScope(jobs, scope).some(job => !isConversationBlockingJob(job));
    }
    function reconcilePendingIdeates(jobs, excludeScope='') {
      const jobList = jobs || [];
      for (const [scope, pending] of Object.entries(pendingIdeateJobsByScope)) {
        if (scope === excludeScope) continue;
        if (scopeAwaitingJobStart(scope)) continue;
        const job = jobList.find(item => item.kind === 'ideate' && item.ui_scope === scope && (!pending.jobId || Number(item.id) === Number(pending.jobId)));
        if (job && job.status !== 'running') clearPendingIdeate(scope);
        if (!job && !runningJobsForScope(jobList, scope).length) clearPendingIdeate(scope);
      }
    }
    function setBusy(next) {
      isBusy = Boolean(next);
      const sendButton = document.getElementById('sendButton');
      const instruction = document.getElementById('instruction');
      if (!sendButton || !instruction) return;
      sendButton.disabled = isBusy;
      instruction.disabled = isBusy;
      sendButton.innerHTML = isBusy ? '<span class="spinner"></span>思考中' : (chatActive ? '发送' : '生成');
    }
    function clearPendingNewSession() {
      clearPendingIdeate();
    }
    function clearAwaitingUiState(scope=currentScope()) {
      setAwaitingJobStart(scope, false);
      setBusy(false);
    }
    function handleMissingActiveSession(data, previousScope) {
      const missingId = data && data.chat && data.chat.missing_session_id ? Number(data.chat.missing_session_id) : 0;
      if (!missingId || Number(activeSessionId || 0) !== missingId) return false;
      const draftText = (instructionEl() && instructionEl().value) || instructionDraftsByScope[previousScope] || '';
      clearAwaitingUiState(previousScope);
      clearScopeMessages(previousScope);
      clearInstructionDraft(previousScope);
      activeSessionId = null;
      newChatMode = true;
      draftScopeId = `draft:${Date.now()}`;
      transientMessagesByScope[draftScopeId] = [];
      activityMessagesByScope[draftScopeId] = [{
        role:'assistant',
        meta:'session missing',
        text:'这条历史对话已经不存在。我已切到新的对话草稿；可以直接重新输入，或从左侧选择其他历史。'
      }];
      if (draftText) instructionDraftsByScope[draftScopeId] = draftText;
      chatRenderKey = '';
      pendingAutoScroll = true;
      return true;
    }
    function isPendingIdeateJob(job) {
      const pending = job && job.ui_scope ? pendingIdeateForScope(job.ui_scope) : null;
      return Boolean(job && job.kind === 'ideate' && pending && (!pending.jobId || Number(job.id) === Number(pending.jobId)));
    }
    function isCreatedIdeateForScope(job, scope) {
      return Boolean(job && job.kind === 'ideate' && job.created_session_id && scope === `session:${Number(job.created_session_id)}`);
    }
    function finalizedJobKey(job) { return `final:${Number(job && job.id || 0)}`; }
    function adoptDraftMessagesToSession(sessionId, sourceScope=draftScopeId) {
      const newScope = `session:${Number(sessionId)}`;
      const draftTransient = scopeMessages(transientMessagesByScope, sourceScope).filter(m => !m.thinking).map(m => ({...m, adoptedFromDraft:true}));
      if (draftTransient.length) {
        setScopeMessages(transientMessagesByScope, newScope, scopeMessages(transientMessagesByScope, newScope).concat(draftTransient));
      }
      delete transientMessagesByScope[sourceScope];
      delete activityMessagesByScope[sourceScope];
      clearProgress(sourceScope);
      removePendingSearch(sourceScope);
    }
    function finalizeCompletedIdeateJob(job, activeScope='', options={}) {
      if (!job || job.kind !== 'ideate' || job.status === 'running') return false;
      const key = finalizedJobKey(job);
      if (lastJobStatuses.get(key)) return false;
      const jobScope = job.ui_scope || '';
      const pending = pendingIdeateForScope(jobScope);
      const createdScope = job.created_session_id ? `session:${Number(job.created_session_id)}` : '';
      const shouldHandle = Boolean(pending || createdScope || jobScope === activeScope);
      if (!shouldHandle) return false;
      setAwaitingJobStart(jobScope, false);
      clearPendingIdeate(jobScope);
      clearProgress(jobScope);
      removePendingSearch(jobScope);
      if (createdScope && jobScope) adoptDraftMessagesToSession(Number(job.created_session_id), jobScope);
      if (!createdScope) {
        const targetScope = jobScope;
        const activity = jobActivity(job);
        setScopeMessages(activityMessagesByScope, targetScope, scopeMessages(activityMessagesByScope, targetScope).filter(m => m.job_id !== job.id).concat([activity]).slice(-8));
      }
      setScopeMessages(transientMessagesByScope, jobScope, scopeMessages(transientMessagesByScope, jobScope).filter(m => !m.thinking));
      lastJobStatuses.set(key, job.status);
      lastJobStatuses.set(job.id, job.status);
      if (options.activate && createdScope) {
        activeSessionId = Number(job.created_session_id);
        newChatMode = false;
        chatRenderKey = '';
      }
      return true;
    }
    function finalizeCompletedIdeateJobs(jobs, activeScope='') {
      for (const job of jobs || []) {
        if (!job || job.kind !== 'ideate' || job.status === 'running') continue;
        const jobScope = job.ui_scope || '';
        const activate = Boolean(job.created_session_id && activeScope && jobScope === activeScope);
        finalizeCompletedIdeateJob(job, activeScope, {activate});
      }
    }
    function showChat() {
      activeView = 'chat';
      document.getElementById('chatSection').classList.remove('hidden');
      document.getElementById('libraryView').classList.remove('active');
      document.getElementById('libraryNav').classList.remove('active');
    }
    function showLibrary() {
      activeView = 'library';
      document.getElementById('chatSection').classList.add('hidden');
      document.getElementById('libraryView').classList.add('active');
      document.getElementById('libraryNav').classList.add('active');
    }
    async function ideate(direction) {
      const scope = currentScope();
      const token = currentViewToken();
      if (scopeHasRunningEvidenceJob(scope)) {
        restoreSubmittedInstruction(scope, direction);
        pushActivityMessage('已有证据任务在后台运行，本次不重复提交；你可以继续在对话框里补充研究方向。', '任务 · 继续等待', scope);
        renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]});
        return;
      }
      setBusy(true);
      setAwaitingJobStart(scope, true);
      addPendingSearch(scope, direction);
      startProgress(scope, 'ideate', direction);
      pushUserMessage(direction, scope);
      pushThinking('正在检索论文、生成 idea，并自动进行审稿循环', scope);
      setPendingIdeate(scope);
      renderSessionHistory(window.lastData || {});
      renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:false, messages:[]});
      try {
        const res = await postJson('/api/ideate', {direction, session_id:null, ui_scope:scope, remote:val('remote'), ideas:num('ideaCount'), limit:num('limit'), provider:val('providerSelect'), lang:val('lang')});
        if (!viewStillCurrent(token, scope)) {
          mergeJobIntoLastData(res && res.job);
          setAwaitingJobStart(scope, false);
          if (res && res.job) setPendingIdeate(scope, Number(res.job.id));
          scheduleRefresh(700);
          return;
        }
        if (res && res.deduped) {
          handleDedupedJob(scope, res.job, '当前新对话已经有 idea 生成任务在运行，本次没有重复提交。');
          await refreshIfCurrent(token, scope);
          return;
        }
        mergeJobIntoLastData(res && res.job);
        setPendingIdeate(scope, res && res.job ? Number(res.job.id) : null);
        pushActivityMessage('任务已提交：会先生成候选 idea，再自动经过审稿专家循环，最后写入新对话。', '任务 · 已开始', scope);
        setAwaitingJobStart(scope, false);
        renderLiveProgress(scope);
        await refreshIfCurrent(token, scope);
      } catch (error) {
        handleUiFailureIfCurrent(token, scope, 'idea 生成任务', error, {clearIdeate:true, clearSearch:true, restoreText:direction});
      }
    }
    async function step(instruction) {
      const scope = currentScope();
      const token = currentViewToken();
      if (!instruction) return;
      if (scopeIsBusy(scope)) {
        restoreSubmittedInstruction(scope, instruction);
        pushActivityMessage('当前对话已有回复在运行，我已保留这次输入；等上一轮完成后可以直接发送。', '任务 · 继续等待', scope);
        renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:true, messages:[]});
        return;
      }
      setBusy(true);
      setAwaitingJobStart(scope, true);
      startProgress(scope, 'step', instruction);
      pushUserMessage(instruction, scope);
      pushThinking('正在思考', scope);
      renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:true, messages:[]});
      try {
        const res = await postJson('/api/step', {instruction, session_id:activeSessionId, ui_scope:scope, provider:val('providerSelect')});
        if (!viewStillCurrent(token, scope)) {
          mergeJobIntoLastData(res && res.job);
          setAwaitingJobStart(scope, false);
          scheduleRefresh(700);
          return;
        }
        if (res && res.deduped) {
          handleDedupedJob(scope, res.job, '当前对话已有一轮回复在运行，本次没有重复提交。');
          await refreshIfCurrent(token, scope);
          return;
        }
        mergeJobIntoLastData(res && res.job);
        pushActivityMessage('任务已提交：正在基于当前会话记忆和证据上下文继续思考。', '任务 · 已开始', scope);
        setAwaitingJobStart(scope, false);
        renderLiveProgress(scope);
        await refreshIfCurrent(token, scope);
      } catch (error) {
        handleUiFailureIfCurrent(token, scope, '继续对话任务', error, {restoreText:instruction});
      }
    }
    async function sendChat() {
      const scope = currentScope();
      const text = takeInstructionForSubmit(scope);
      if (!text) return;
      showChat();
      if (chatActive && !newChatMode) await step(text);
      else await ideate(text);
    }
    async function review() { const scope=currentScope(); const token=currentViewToken(); if (!activeSessionId || scopeIsBusy(scope)) return; setBusy(true); setAwaitingJobStart(scope, true); startProgress(scope, 'review', '审稿人视角'); pushThinking('审稿人正在攻击这个 idea', scope); renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]}); try { const res = await postJson('/api/review', {session_id:activeSessionId, ui_scope:scope, provider:val('providerSelect')}); if (!viewStillCurrent(token, scope)) { mergeJobIntoLastData(res && res.job); setAwaitingJobStart(scope, false); scheduleRefresh(700); return; } if (res && res.deduped) { handleDedupedJob(scope, res.job, '当前对话已有审稿任务在运行，本次没有重复提交。'); await refreshIfCurrent(token, scope); return; } mergeJobIntoLastData(res && res.job); pushActivityMessage('任务已提交：审稿人视角会先攻击 novelty、证据链和可行性。', '任务 · 已开始', scope); setAwaitingJobStart(scope, false); renderLiveProgress(scope); await refreshIfCurrent(token, scope); } catch (error) { handleUiFailureIfCurrent(token, scope, '审稿人任务', error); } }
    async function reviewLoop() { const scope=currentScope(); const token=currentViewToken(); if (!activeSessionId || scopeIsBusy(scope)) return; setBusy(true); setAwaitingJobStart(scope, true); startProgress(scope, 'review-loop', '审稿循环'); pushThinking('正在进行多轮审稿和自动改写', scope); renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]}); try { const res = await postJson('/api/review-loop', {session_id:activeSessionId, ui_scope:scope, rounds:4, provider:val('providerSelect')}); if (!viewStillCurrent(token, scope)) { mergeJobIntoLastData(res && res.job); setAwaitingJobStart(scope, false); scheduleRefresh(700); return; } if (res && res.deduped) { handleDedupedJob(scope, res.job, '当前对话已有审稿循环在运行，本次没有重复提交。'); await refreshIfCurrent(token, scope); return; } mergeJobIntoLastData(res && res.job); pushActivityMessage('任务已提交：会自动经过 3-5 轮审稿专家攻击和改写，再保留最终 idea。', '任务 · 已开始', scope); setAwaitingJobStart(scope, false); renderLiveProgress(scope); await refreshIfCurrent(token, scope); } catch (error) { handleUiFailureIfCurrent(token, scope, '审稿循环任务', error); } }
    async function goldBuild() {
      showChat();
      const scope = currentScope();
      const token = currentViewToken();
      if (scopeIsBusy(scope)) return;
      const selected = [...selectedVenues];
      const coreVenues = coreGoldVenueValues(selected);
      const auxiliaryVenues = auxiliaryGoldVenueValues(selected);
      if (!coreVenues.length) {
        pushActivityMessage(`核心优秀论文库只接受 ICLR、NeurIPS/NIPS、ICML、CVPR、ICCV、TPAMI。\n\n已选择的辅助来源仅作趋势参考：${auxiliaryVenues.join('、') || '无'}。`, '请求失败', scope);
        renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]});
        return;
      }
      setBusy(true);
      setAwaitingJobStart(scope, true);
      startProgress(scope, 'gold-build', '扩充核心优秀论文库');
      pushThinking('正在联网扩充核心优秀论文库', scope);
      renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]});
      try {
        const res = await postJson('/api/gold-build', {venues:coreVenues, query:val('goldQuery'), from_year:num('fromYear'), to_year:num('toYear'), per_venue_year:num('perVenue'), extractive_genomes:document.getElementById('extractiveGenomes').checked, ui_scope:scope});
        if (!viewStillCurrent(token, scope)) {
          (res && res.jobs || []).forEach(mergeJobIntoLastData);
          setAwaitingJobStart(scope, false);
          scheduleRefresh(700);
          return;
        }
        if (res && res.deduped) {
          handleDedupedJob(scope, res.job, '当前范围已有优秀论文库扩充任务在运行，本次没有重复提交。');
          await refreshIfCurrent(token, scope);
          return;
        }
        (res && res.jobs || []).forEach(mergeJobIntoLastData);
        const count = res && res.jobs ? res.jobs.length : 1;
        const serverAuxiliary = res && res.auxiliary_venues ? res.auxiliary_venues : [];
        const auxiliary = auxiliaryVenues.concat(serverAuxiliary).filter((value, index, arr) => arr.indexOf(value) === index);
        const auxiliaryNote = auxiliary.length ? `\n\n未提交到核心 Gold：${auxiliary.join('、')} 只作为趋势参考，不参与 idea 评分标准。` : '';
        const metadataOnly = res && res.metadata_only_venues ? res.metadata_only_venues : [];
        const metadataOnlyNote = metadataOnly.length ? `\n\n${metadataOnly.join('、')} 当前通过 OpenAlex 元数据收集：先写入趋势区；只有后续质量核验确认 Best / Nominee / Oral 等明确优秀信号后，才会进入核心优秀依据库。` : '';
        const planNote = sourcePlanNote(res && res.source_plan);
        pushActivityMessage(`任务已提交：${count} 个官方优秀论文源正在联网拉取。Poster 会被硬过滤，普通论文只进入趋势区。${planNote}${auxiliaryNote}${metadataOnlyNote}`, '任务 · 已开始', scope);
        setAwaitingJobStart(scope, false);
        renderLiveProgress(scope);
        await refreshIfCurrent(token, scope);
      } catch (error) {
        handleUiFailureIfCurrent(token, scope, '核心优秀论文库扩充任务', error);
      }
    }
    async function qualityEnrich() {
      showChat();
      const scope = currentScope();
      const token = currentViewToken();
      if (scopeIsBusy(scope)) return;
      setBusy(true);
      setAwaitingJobStart(scope, true);
      startProgress(scope, 'quality-enrich', '证据质量审查');
      pushThinking('正在核验趋势论文质量标签', scope);
      renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]});
      try {
        const res = await postJson('/api/quality-enrich', {limit:Math.max(10, num('perVenue')), provider:val('providerSelect'), ui_scope:scope});
        if (!viewStillCurrent(token, scope)) {
          mergeJobIntoLastData(res && res.job);
          setAwaitingJobStart(scope, false);
          scheduleRefresh(700);
          return;
        }
        if (res && res.deduped) {
          handleDedupedJob(scope, res.job, '当前范围已有证据质量审查在运行，本次没有重复提交。');
          await refreshIfCurrent(token, scope);
          return;
        }
        mergeJobIntoLastData(res && res.job);
        pushActivityMessage('任务已提交：正在核验趋势论文是否有核心最佳、提名或 Oral 信号；Spotlight/高引用只作辅助发现线索。', '任务 · 已开始', scope);
        setAwaitingJobStart(scope, false);
        renderLiveProgress(scope);
        await refreshIfCurrent(token, scope);
      } catch (error) {
        handleUiFailureIfCurrent(token, scope, '证据质量审查任务', error);
      }
    }
    async function cleanup(apply) { const scope=currentScope(); const token=currentViewToken(); if (scopeHasRunningEvidenceJob(scope)) { pushActivityMessage('已有证据任务在后台运行，本次体检不重复提交；中间对话仍可继续使用。', '任务 · 继续等待', scope); renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]}); return; } setBusy(true); setAwaitingJobStart(scope, true); startProgress(scope, 'cleanup', apply ? '清理证据上下文' : '检查证据上下文'); pushThinking(apply ? '正在清理证据上下文' : '正在检查证据上下文', scope); renderChatIfCurrent(token, scope, (window.lastData || {}).chat || {active:chatActive, messages:[]}); try { const res = await postJson('/api/cleanup', {apply, ui_scope:scope}); if (!viewStillCurrent(token, scope)) { mergeJobIntoLastData(res && res.job); setAwaitingJobStart(scope, false); scheduleRefresh(700); return; } if (res && res.deduped) { handleDedupedJob(scope, res.job, apply ? '当前范围已有证据清理任务在运行，本次没有重复提交。' : '当前范围已有证据体检任务在运行，本次没有重复提交。'); await refreshIfCurrent(token, scope); return; } mergeJobIntoLastData(res && res.job); pushActivityMessage(apply ? '任务已提交：正在清理不合格证据上下文。' : '任务已提交：正在检查证据上下文完整性。', '任务 · 已开始', scope); setAwaitingJobStart(scope, false); renderLiveProgress(scope); await refreshIfCurrent(token, scope); } catch (error) { handleUiFailureIfCurrent(token, scope, apply ? '证据清理任务' : '证据体检任务', error); } }
    async function newChat() {
      showChat();
      const token = bumpViewToken();
      const previousScope = currentScope();
      saveInstructionDraft(previousScope);
      clearAwaitingUiState(previousScope);
      activeSessionId = null;
      newChatMode = true;
      draftScopeId = `draft:${Date.now()}`;
      transientMessagesByScope[draftScopeId] = [];
      activityMessagesByScope[draftScopeId] = [];
      chatRenderKey = '';
      try {
        await postJson('/api/session/new', {});
      } catch (error) {
        handleUiFailureIfCurrent(token, draftScopeId, '新对话', error);
        return;
      }
      if (!viewStillCurrent(token, draftScopeId)) return;
      restoreInstructionDraft(draftScopeId);
      renderChat({active:false, session:null, draft:true, messages:[{role:'assistant', meta:'new workspace', text:'已准备好新对话。直接输入你想研究的方向，我会从论文逻辑库和近期证据里生成新 idea。'}]});
      renderSessionHistory(window.lastData || {});
    }
    function selectSession(id) {
      showChat();
      bumpViewToken();
      const previousScope = currentScope();
      saveInstructionDraft(previousScope);
      clearAwaitingUiState(previousScope);
      activeSessionId = Number(id);
      newChatMode = false;
      chatRenderKey = '';
      restoreInstructionDraft(currentScope());
      refresh();
    }
    async function deleteSession(id, event) {
      event.stopPropagation();
      if (!confirm('删除这条历史对话？相关多轮记录也会一起删除。')) return;
      const deletingActive = Number(activeSessionId) === Number(id);
      const token = deletingActive ? bumpViewToken() : currentViewToken();
      const res = await fetch('/api/session/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:Number(id)})});
      const json = await res.json();
      if (!res.ok) {
        const error = requestErrorFromResponse(json, res.status, '删除失败');
        const jobs = Array.isArray(json.jobs) ? json.jobs : [];
        jobs.forEach(mergeJobIntoLastData);
        const runningHint = jobs.length ? `\n\n仍在运行：${jobs.slice(0, 3).map(displayJobName).join('、')}` : '';
        pushActivityMessage(`删除失败：${requestErrorMessage(error)}${runningHint}`, '请求失败', currentScope());
        renderChatIfCurrent(token, currentScope(), (window.lastData || {}).chat || {active:chatActive, messages:[]});
        await refreshIfCurrent(token, currentScope());
        return;
      }
      if (deletingActive) {
        const deletedScope = currentScope();
        clearAwaitingUiState(deletedScope);
        activeSessionId = null;
        newChatMode = true;
        draftScopeId = `draft:${Date.now()}`;
        restoreInstructionDraft(draftScopeId);
      }
      clearScopeMessages(`session:${Number(id)}`);
      clearInstructionDraft(`session:${Number(id)}`);
      await refreshIfCurrent(token, currentScope());
    }
    async function deleteIdea(id, event) {
      event.stopPropagation();
      if (!confirm('删除这条想法记录？')) return;
      const token = currentViewToken();
      const res = await fetch('/api/idea/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({idea_id:Number(id)})});
      const json = await res.json();
      if (!res.ok) {
        const error = requestErrorFromResponse(json, res.status, '删除失败');
        pushActivityMessage(`删除失败：${requestErrorMessage(error)}`, '请求失败', currentScope());
        renderChatIfCurrent(token, currentScope(), (window.lastData || {}).chat || {active:chatActive, messages:[]});
        await refreshIfCurrent(token, currentScope());
        return;
      }
      await refreshIfCurrent(token, currentScope());
    }
    async function addUserLibraryPaper(event) {
      event.preventDefault();
      const errorEl = document.getElementById('userLibraryError');
      if (errorEl) errorEl.textContent = '';
      const button = document.getElementById('addUserLibraryButton');
      const original = button ? button.textContent : '';
      if (button) { button.disabled = true; button.textContent = '添加中'; }
      try {
        await postJson('/api/user-library/add', {
          url: val('userLibraryUrl'),
          title: val('userLibraryTitle'),
          note: val('userLibraryNote'),
        });
        ['userLibraryUrl', 'userLibraryTitle', 'userLibraryNote'].forEach(id => {
          const node = document.getElementById(id);
          if (node) node.value = '';
        });
        await refreshIfCurrent(currentViewToken(), currentScope());
      } catch (error) {
        if (errorEl) errorEl.textContent = cleanUiText(requestErrorMessage(error));
      } finally {
        if (button) { button.disabled = false; button.textContent = original || '添加'; }
      }
    }
    async function deleteUserLibraryPaper(id, event) {
      event.preventDefault();
      event.stopPropagation();
      if (!confirm('删除这篇用户库论文？')) return;
      const token = currentViewToken();
      try {
        await postJson('/api/user-library/delete', {paper_id:Number(id)});
        await refreshIfCurrent(token, currentScope());
      } catch (error) {
        const errorEl = document.getElementById('userLibraryError');
        if (errorEl) errorEl.textContent = cleanUiText(requestErrorMessage(error));
      }
    }
    function val(id) { return document.getElementById(id).value; }
    function num(id) { return Number(val(id)); }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function cleanUiText(text) {
      const candidateWord = '\u5019\u9009';
      const bestOralCandidate = 'Best / Nominee / Oral ' + candidateWord;
      const frontierCandidate = 'frontier ' + candidateWord;
      const recentCandidateOpenAlex = 'OpenAlex \u8fd1\u671f' + candidateWord;
      const recentCandidateS2 = 'Semantic Scholar \u8fd1\u671f' + candidateWord;
      const recentCandidate = '\u8fd1\u671f' + candidateWord;
      const remoteCandidatePaper = '\u8054\u7f51' + candidateWord + '\u8bba\u6587';
      const trendRemotePaper = '\u8054\u7f51\u8d8b\u52bf\u8bba\u6587';
      const candidatePaper = candidateWord + '\u8bba\u6587';
      const candidatePapers = candidateWord + '\u8bba\u6587\u4eec';
      const candidateZone = candidateWord + '\u533a';
      const candidateSource = '\u5019\u9009\u6765\u6e90';
      const auxiliaryCandidate = '\u8f85\u52a9' + candidateWord;
      const recentCandidatePapers = '\u8fd1\u671f' + candidateWord + '\u8bba\u6587';
      const recentCandidateFew = '\u8fd1\u671f' + candidateWord + '\u51e0\u4e2a';
      const candidateIdea = candidateWord + ' idea';
      const candidateIdeas = candidateWord + ' ideas';
      const candidateDossiers = 'Candidate' + ' Dossiers';
      const candidateOutside = 'Candidate' + ' ideas outside current Gold context';
      const candidateGeneric = 'Candidate';
      return String(text ?? '')
        .split(bestOralCandidate).join('Best / Nominee / Oral 线索')
        .split(frontierCandidate).join('frontier 线索')
        .split(recentCandidateOpenAlex).join('OpenAlex 趋势')
        .split(recentCandidateS2).join('Semantic Scholar 趋势')
        .split(recentCandidateFew).join('近期趋势线索')
        .split(recentCandidatePapers).join('近期趋势论文')
        .split(recentCandidate).join('趋势')
        .split(remoteCandidatePaper).join(trendRemotePaper)
        .split(candidatePaper).join('趋势论文')
        .split(candidatePapers).join('趋势论文')
        .split(candidateZone).join('趋势区')
        .split(candidateSource).join('趋势来源')
        .split(auxiliaryCandidate).join('辅助趋势')
        .split(candidateIdea).join('idea 草案')
        .split(candidateIdeas).join('idea 草案')
        .split(candidateDossiers).join('Idea dossiers')
        .split(candidateOutside).join('当前 Gold 上下文外的 idea 草案')
        .split(candidateGeneric).join('Trend')
        .split(candidateWord).join('趋势参考');
    }
    function displayText(text) { return cleanUiText(text); }
    function displayJobName(job) {
      const raw = job && job.name ? job.name : (job && job.kind ? job.kind : '任务');
      return displayText(raw);
    }
    function safeUrl(url) { const value=String(url || '').trim(); return /^https?:\/\//i.test(value) ? value : ''; }
    function openSafeUrl(url) { const value = safeUrl(url); if (value) window.open(value, '_blank', 'noopener,noreferrer'); }
    function openUserLibraryPaper(id) {
      const papers = ((window.lastData || {}).user_papers || []);
      const paper = papers.find(item => Number(item.id) === Number(id));
      const ref = paper ? safeUrl(paper.external_ref) : '';
      if (ref) openSafeUrl(ref);
    }
    function empty(text) { return `<div class="empty">${esc(cleanUiText(text))}</div>`; }
    function emptyAction(text, action, title='点击查看') { return `<button class="empty-action" onclick="metricAction('${esc(action)}')" title="${esc(title)}">${esc(cleanUiText(text))}</button>`; }
    function compact(text, n=160) { text = String(text || ''); return text.length > n ? text.slice(0, n) + '…' : text; }
    function sessionScopeForId(id) { return `session:${Number(id)}`; }
    function latestScopePreview(scope, fallback='') {
      const fallbackText = cleanUiText(fallback || '');
      if (fallbackText) return compact(fallbackText, 120);
      const messages = scopeMessages(transientMessagesByScope, scope).concat(scopeMessages(activityMessagesByScope, scope));
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        const item = messages[i] || {};
        if (item.thinking) continue;
        const text = cleanUiText(item.text || '');
        if (text) return compact(text, 120);
      }
      const live = visibleProgressText(scope);
      if (live) return compact(live, 120);
      return '';
    }
    function panelNavigate(event, action) {
      if (event && event.target && event.target.closest('button, a, input, textarea, select, summary, details, label')) return;
      metricAction(action);
    }
    function activatePanel(event, action) {
      if (!event || (event.key !== 'Enter' && event.key !== ' ')) return;
      if (event.target && event.target.closest('button, a, input, textarea, select, summary, details, label')) return;
      event.preventDefault();
      metricAction(action);
    }
    const tooltipPortal = (() => {
      let node = null;
      let activeAnchor = null;
      function ensure() {
        if (!node) {
          node = document.createElement('div');
          node.className = 'tooltip-popover';
          document.body.appendChild(node);
        }
        return node;
      }
      function place(anchor) {
        const el = ensure();
        const rect = anchor.getBoundingClientRect();
        const gap = 10;
        el.style.maxWidth = `${Math.min(292, Math.max(220, window.innerWidth - 24))}px`;
        const tooltipRect = el.getBoundingClientRect();
        let left = rect.left - tooltipRect.width - gap;
        if (left < 12) left = Math.min(window.innerWidth - tooltipRect.width - 12, rect.right + gap);
        let top = rect.top + (rect.height / 2) - (tooltipRect.height / 2);
        top = Math.max(12, Math.min(top, window.innerHeight - tooltipRect.height - 12));
        el.style.left = `${Math.max(12, left)}px`;
        el.style.top = `${top}px`;
      }
      return {
        show(anchor) {
          const text = anchor && anchor.getAttribute('data-tooltip');
          if (!text) return;
          activeAnchor = anchor;
          const el = ensure();
          el.textContent = text;
          place(anchor);
          el.classList.add('visible');
        },
        hide(anchor) {
          if (anchor && activeAnchor && anchor !== activeAnchor) return;
          if (node) node.classList.remove('visible');
          activeAnchor = null;
        },
        reposition() {
          if (activeAnchor && node && node.classList.contains('visible')) place(activeAnchor);
        },
      };
    })();
    document.addEventListener('mouseover', event => {
      const anchor = event.target.closest && event.target.closest('.tooltip-anchor');
      if (anchor) tooltipPortal.show(anchor);
    });
    document.addEventListener('mouseout', event => {
      const anchor = event.target.closest && event.target.closest('.tooltip-anchor');
      if (anchor) tooltipPortal.hide(anchor);
    });
    document.addEventListener('focusin', event => {
      const anchor = event.target.closest && event.target.closest('.tooltip-anchor');
      if (anchor) tooltipPortal.show(anchor);
    });
    document.addEventListener('focusout', event => {
      const anchor = event.target.closest && event.target.closest('.tooltip-anchor');
      if (anchor) tooltipPortal.hide(anchor);
    });
    window.addEventListener('scroll', tooltipPortal.reposition, true);
    window.addEventListener('resize', tooltipPortal.reposition);
    function renderPaperRow(p, mode='excellent') {
      const ref = safeUrl(p.external_ref);
      const source = p.source_label || p.source_kind || 'source';
      const reason = p.quality_reason || (mode === 'frontier' ? '仅作为趋势参考，不参与评分标准' : '已通过质量评分');
      const status = cleanUiText(p.library_status || (mode === 'frontier' ? '趋势参考' : 'Gold'));
      const statusDetail = cleanUiText(p.library_status_detail || reason);
      const statusClass = /待核验|待评分|需补/.test(status) ? 'pending' : (mode === 'frontier' || mode === 'user' || /趋势|辅助|降级|用户库/.test(status) ? 'frontier' : 'gold');
      const stats = mode === 'user'
        ? `${esc(p.venue)} · ${esc(p.year)} · ${esc(source)}`
        : mode === 'frontier'
        ? `${esc(p.venue)} · ${esc(p.year)} · ${esc(source)} · cites=${esc(p.citation_count || 0)} · ${esc(p.award || '待核验')}`
        : `${esc(p.venue)} · ${esc(p.year)} · weight=${esc(p.paper_weight)} · ${esc(p.award || '-')} · ${esc(source)}`;
      const actionHint = ref ? '打开来源' : (mode === 'frontier' ? '查看趋势分区' : (mode === 'user' ? '查看用户库' : '查看依据分区'));
      const body = `<div class="paper-title-line"><div class="paper-title">${esc(cleanUiText(p.title))}</div><span class="paper-status ${esc(statusClass)}" title="${esc(statusDetail)}">${esc(status)}</span></div><div class="paper-meta">${esc(cleanUiText(stats))}</div><div class="paper-reason">${esc(statusDetail)} · ${esc(cleanUiText(reason))} · ${esc(actionHint)}</div>`;
      const userDelete = mode === 'user' ? `<button class="delete-artifact user-paper-delete" onclick="deleteUserLibraryPaper(${Number(p.id)}, event)" title="删除用户库论文">×</button>` : '';
      if (mode === 'user') return `<div class="artifact-row"><button class="paper-row clickable-row" onclick="${ref ? `openUserLibraryPaper(${Number(p.id)})` : "metricAction('library')"}" title="${ref ? '打开论文来源' : '查看用户论文库'}">${body}</button>${userDelete}</div>`;
      if (ref) return `<a class="paper-row clickable-row" href="${esc(ref)}" target="_blank" rel="noreferrer" title="打开论文来源">${body}</a>`;
      const action = mode === 'frontier' ? 'frontier' : (mode === 'user' ? 'library' : 'excellent');
      const title = mode === 'frontier' ? '查看趋势论文分区' : (mode === 'user' ? '查看用户论文库' : '查看核心优秀依据库');
      return `<div class="artifact-row"><button class="paper-row clickable-row" onclick="metricAction('${action}')" title="${title}">${body}</button>${userDelete}</div>`;
    }
    function renderSelect(id, options, selected) { const el=document.getElementById(id); if(!el || el.options.length) return; el.innerHTML=options.map(o=>`<option value="${esc(o.value)}"${o.value===selected?' selected':''}>${esc(o.label)}</option>`).join(''); }
    function renderOptionControls(data) {
      if (appOptions) return;
      appOptions = data.options || {};
      selectedVenues = new Set(appOptions.default_gold_venues || ['ICLR','NeurIPS','ICML','CVPR','ICCV','TPAMI']);
      selectedVenueGroup = ((appOptions.venue_groups || [])[0] || {}).id || 'core';
      document.getElementById('fromYear').value = appOptions.default_from_year || 2024;
      document.getElementById('toYear').value = appOptions.default_to_year || 2026;
      renderSelect('remote', appOptions.remote_options || [], 'openalex');
      renderSelect('providerSelect', appOptions.provider_options || [], data.project.provider || 'ds');
      renderSelect('lang', appOptions.language_options || [], 'zh');
      renderVenuePresets(); renderVenueTabs(); renderVenueChoices(); renderSelectedVenues();
    }
    function renderVenuePresets() { const presets=appOptions.venue_presets || []; document.getElementById('venuePresets').innerHTML=presets.map(p=>`<button class="chip ${selectedVenuePreset===p.id?'active':''}" onclick="selectVenuePreset('${esc(p.id)}')">${esc(p.label)}</button>`).join(''); }
    function selectVenuePreset(id) { const preset=(appOptions.venue_presets || []).find(p=>p.id===id); if(!preset) return; selectedVenuePreset=id; selectedVenues=new Set(preset.venues || []); renderVenuePresets(); renderVenueChoices(); renderSelectedVenues(); }
    function renderVenueTabs() { const groups=appOptions.venue_groups || []; document.getElementById('venueTabs').innerHTML=groups.map(g=>`<button class="tab ${selectedVenueGroup===g.id?'active':''}" onclick="selectVenueGroup('${esc(g.id)}')">${esc(g.label)}</button>`).join(''); }
    function selectVenueGroup(id) { selectedVenueGroup=id; renderVenueTabs(); renderVenueChoices(); }
    function renderVenueChoices() { const groups=appOptions.venue_groups || []; const group=groups.find(g=>g.id===selectedVenueGroup) || groups[0] || {venues:[]}; document.getElementById('venueGroups').innerHTML=`<div class="chips">${(group.venues || []).map(v=>`<button class="chip ${selectedVenues.has(v.value)?'active':''}" onclick="toggleVenue('${esc(v.value)}')">${esc(v.label)}<small>${esc(v.rank)}</small></button>`).join('')}</div>`; }
    function toggleVenue(value) { if(selectedVenues.has(value)) selectedVenues.delete(value); else selectedVenues.add(value); selectedVenuePreset=''; renderVenuePresets(); renderVenueChoices(); renderSelectedVenues(); }
    function renderSelectedVenues() { const values=[...selectedVenues]; document.getElementById('selectedVenueStrip').innerHTML=values.length ? values.map(v=>`<span>${esc(v)}</span>`).join('') : '<span style="border-color:#f6c2bd;background:#fff1f0;color:#b42318">请选择至少一个会议</span>'; }
    function firstLineMatch(text, pattern) {
      const match = String(text || '').match(pattern);
      return match ? String(match[1] || '').trim() : '';
    }
    function parseJsonLine(result, label) {
      const escaped = String(label || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const match = String(result || '').match(new RegExp(escaped + ':\\s*(\\{[^\\n]+\\})'));
      if (!match) return null;
      try { return JSON.parse(match[1]); } catch (error) { return null; }
    }
    function parseGroundingAuditFailure(result) {
      const text = String(result || '');
      const match = text.match(/(Genome|Pattern) grounding audit failed:\s*(\[[^\n]+\])/);
      if (!match) return null;
      try {
        return {kind: match[1].toLowerCase(), failures: JSON.parse(match[2])};
      } catch (error) {
        return {kind: match[1].toLowerCase(), failures: []};
      }
    }
    function groundingAuditJobText(result, displayResult) {
      const audit = parseGroundingAuditFailure(result);
      if (!audit) return '';
      const label = audit.kind === 'genome' ? '论文 Genome 逻辑线' : 'Pattern 聚合逻辑';
      const lines = [`${label}没有通过证据锚点审计，因此没有写入严格证据标准。`];
      lines.push('模型可以做抽象和命名，但每一步必须落在优秀论文标题、摘要或已验证 Genome 的逻辑空间里；系统已拦截这次不可靠输出。');
      const failures = Array.isArray(audit.failures) ? audit.failures.slice(0, 3) : [];
      if (failures.length) {
        const text = failures.map(item => {
          if (!item || typeof item !== 'object') return '';
          const code = item.code || 'grounding_audit_failed';
          const step = item.step ? ` · ${item.step}` : '';
          const detail = item.source_text || item.logic_text || item.text || '';
          return `${code}${step}${detail ? `：${compact(cleanUiText(detail), 140)}` : ''}`;
        }).filter(Boolean).join('\n');
        if (text) lines.push(`未通过项：\n${text}`);
      }
      lines.push('下一步：换用更完整的优秀论文摘要/元数据，或重新运行 Genome/Pattern 构建；用户论文库仍只作为领域知识，不会被当作评分标准。');
      if (!failures.length && displayResult) lines.push(compact(displayResult, 500));
      return lines.join('\n\n');
    }
    function ideaFailureRecoveryLines(failure) {
      const lines = [];
      if (!failure || typeof failure !== 'object') return lines;
      const strategyLabels = {
        retry_with_content_floor: '恢复策略：直接重试，但要求模型补足研究对象、机制假设、首个实验和失败边界。',
        storyline_first_retry: '恢复策略：先重写六步论文故事线，再迁移到当前领域；只迁移逻辑角色，不迁移源论文表面方法。',
        novelty_boundary_retry: '恢复策略：先拉开 prior-art 差异边界，再生成更窄的未解矛盾。',
        frontier_repair_then_retry: '恢复策略：先补当前方向的近期趋势证据，再重新生成。',
        gold_logic_repair_then_retry: '恢复策略：先补 Gold/Genome/Pattern，让五个评分维度都有显式依据。',
        domain_context_only_retry: '恢复策略：保留用户论文库为领域背景，但从 Gold/Pattern/Genome 中选择故事线和评分依据。',
        inspect_rejections_then_retry: '恢复策略：先读取被拒原因，补证据或收窄方向后再生成。'
      };
      if (failure.recovery_strategy && strategyLabels[failure.recovery_strategy]) lines.push(strategyLabels[failure.recovery_strategy]);
      if (Array.isArray(failure.blocked_dimensions) && failure.blocked_dimensions.length) {
        lines.push(`卡住的维度：${failure.blocked_dimensions.slice(0, 4).join('、')}`);
      }
      if (Array.isArray(failure.retry_prompt_constraints) && failure.retry_prompt_constraints.length) {
        const constraints = failure.retry_prompt_constraints.slice(0, 3).map(item => `- ${cleanUiText(item)}`).join('\n');
        if (constraints) lines.push(`下次重试约束：\n${constraints}`);
      }
      return lines;
    }
    function ideateSummaryText(summary) {
      if (!summary || typeof summary !== 'object') return '';
      if (summary.type === 'idea_generation_not_ready' && summary.failure) {
        const failure = summary.failure;
        const reasonLabels = {
          storyline_migration_failed: '生成结果像是在硬搬源论文方法，还没有完成“论文故事线 → 当前领域”的迁移。',
          prior_art_too_close: '生成结果和已有工作过近，需要先拉开差异边界。',
          frontier_evidence_not_ready: '当前方向的近期趋势证据不足，需要先补与问题匹配的 frontier 论文。',
          gold_evidence_not_ready: '核心优秀论文、Genome 或 Pattern 依据不足，五个评分维度还没有显式支撑。',
          user_library_used_as_standard: '模型把用户论文库当成了 Gold/故事线评分依据；用户库只能作为领域知识和路线学习背景。',
          idea_generation_not_ready: '严格证据门没有放行这批 idea。'
        };
        const lines = ['这次没有写入新对话，因为严格生成门没有放行 idea。'];
        const primary = failure.primary_reason || 'idea_generation_not_ready';
        lines.push(reasonLabels[primary] || reasonLabels.idea_generation_not_ready);
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        lines.push(...ideaFailureRecoveryLines(failure));
        return lines.join('\n\n');
      }
      const data = summary.idea_session || {};
      const ideateTypes = new Set(['idea_session_created', 'idea_session_review_ready', 'idea_session_review_incomplete']);
      if (!ideateTypes.has(summary.type) || !data || typeof data !== 'object') return '';
      const lines = ['已生成新的研究 idea，并写入当前对话。'];
      const loop = data.auto_review_loop && typeof data.auto_review_loop === 'object' ? data.auto_review_loop : null;
      if (data.title) lines.push(`标题：${data.title}`);
      if (loop && loop.status === 'completed') lines.push(`自动审稿循环：已完成 ${loop.rounds_completed || 0}/${loop.requested_rounds || 0} 轮，下面是最终 idea。`);
      else if (loop && loop.status) lines.push('自动审稿循环没有完成，下面是生成阶段保留的当前 idea。');
      if (data.current_idea) lines.push(`核心想法：${data.current_idea}`);
      if (data.trend_support) lines.push(`近期依据：${data.trend_support}`);
      if (data.storyline_logic) lines.push(`顶会论文逻辑迁移：${data.storyline_logic}`);
      if (Array.isArray(data.storyline_steps) && data.storyline_steps.length) {
        const steps = data.storyline_steps.slice(0, 6).map(item => {
          const label = item.label || item.step || '故事线';
          const transfer = item.transfer || '';
          return `${label}：${transfer}`;
        }).filter(Boolean).join('\n');
        if (steps) lines.push(`六步故事线迁移：\n${steps}`);
      }
      if (data.current_method_problem) lines.push(`当前方法问题：${data.current_method_problem}`);
      if (data.recent_limitation_signal) lines.push(`近半年局限性/趋势信号：${data.recent_limitation_signal}`);
      if (data.evidence_score !== undefined && data.evidence_score !== '') lines.push(`证据驱动评分：${data.evidence_score}/10`);
      if (data.key_risk) lines.push(`审稿风险：${data.key_risk}`);
      if (data.first_experiment) lines.push(`首个实验：${data.first_experiment}`);
      return lines.join('\n');
    }
    function sessionStepSummaryText(summary) {
      if (!summary || typeof summary !== 'object') return '';
      if (summary.type === 'session_step_failed' && summary.failure) {
        const failure = summary.failure;
        const labels = {
          llm_unavailable: '模型暂时不可用或 API 配置有问题。',
          invalid_model_response: '模型返回格式不稳定，系统没有把这轮写入对话。',
          evidence_not_ready: '多轮修改需要先补足 Genome/Pattern 证据。',
          session_evidence_guard_failed: '模型本轮改写越出了优秀论文逻辑空间，或把用户论文库/趋势论文当成了标准。',
          session_step_failed: '本轮修改没有完成。'
        };
        const lines = ['本轮追问没有写入当前会话。', labels[failure.reason_code] || labels.session_step_failed];
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        if (failure.reason) lines.push(`原因：${compact(cleanUiText(failure.reason), 260)}`);
        return lines.join('\n\n');
      }
      if (summary.type !== 'session_step_completed') return '';
      const lines = ['本轮追问已写入当前会话，并刷新了会话记忆和 dossier。'];
      if (summary.decision) lines.push(`决策：${summary.decision}`);
      if (summary.revised_idea) lines.push(`修改后的 idea：${summary.revised_idea}`);
      if (summary.current_best && summary.current_best !== summary.revised_idea) lines.push(`当前最佳版本：${summary.current_best}`);
      if (summary.trend_alignment) lines.push(`趋势对齐：${summary.trend_alignment}`);
      if (summary.memory) {
        const memory = summary.memory;
        const memoryBits = [];
        if (memory.turn_count !== undefined) memoryBits.push(`轮数 ${memory.turn_count}`);
        if (memory.preferred_language) memoryBits.push(`语言 ${memory.preferred_language}`);
        if (memoryBits.length) lines.push(`记忆状态：${memoryBits.join(' · ')}`);
        if (memory.state_policy) lines.push('记忆策略：压缩摘要保持多轮连续性，原始中间过程不进入回答。');
        if (memory.user_library_policy) lines.push('用户库边界：只作领域背景，不作为评分标准或故事线来源。');
        if (memory.stable_thesis) lines.push(`稳定主线：${memory.stable_thesis}`);
        if (memory.unresolved_risk) lines.push(`未解决风险：${memory.unresolved_risk}`);
        if (memory.next_experiment) lines.push(`下一步实验：${memory.next_experiment}`);
      }
      if (summary.dossier) lines.push(`Dossier：${summary.dossier}`);
      return lines.join('\n');
    }
    function reviewSummaryText(summary) {
      if (!summary || typeof summary !== 'object') return '';
      if (summary.type === 'reviewer_failed' && summary.failure) {
        const failure = summary.failure;
        const labels = {
          llm_unavailable: '模型暂时不可用或 API 配置有问题。',
          invalid_model_response: '模型返回格式不稳定，系统没有保存审稿结果。',
          idea_not_found: '没有找到要审稿的 idea 或会话。',
          evidence_not_ready: '审稿需要先补足核心证据上下文。',
          reviewer_failed: '审稿人视角没有完成。'
        };
        const lines = ['审稿人视角没有写入结果。', labels[failure.reason_code] || labels.reviewer_failed];
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        if (failure.reason) lines.push(`原因：${compact(cleanUiText(failure.reason), 260)}`);
        return lines.join('\n\n');
      }
      if (summary.type !== 'review_completed') return '';
      const lines = ['审稿人视角已完成：已从 novelty、证据链、逻辑和可行性角度攻击当前 idea。'];
      if (summary.decision) lines.push(`审稿结论：${summary.decision}`);
      if (summary.summary) lines.push(`攻击摘要：${summary.summary}`);
      if (summary.review_path) lines.push(`审稿记录：${summary.review_path}`);
      return lines.join('\n');
    }
    function reviewLoopSummaryText(summary) {
      if (!summary || typeof summary !== 'object') return '';
      if (summary.type === 'review_loop_failed') {
        const failure = summary.failure || {};
        const data = summary.review_loop || {};
        const done = Number(data.rounds_completed || 0);
        const requested = Number(data.requested_rounds || 0);
        const lines = done > 0
          ? [`审稿循环没有完整完成：已完成 ${done}/${requested || '?'} 轮，当前保留最新安全版本。`]
          : ['审稿循环没有完成，当前 idea 未被最终覆盖。'];
        if (data.final_idea) lines.push(`当前 idea：${data.final_idea}`);
        const recoveryStrategy = data.recovery_strategy || failure.recovery_strategy || '';
        const safeResume = data.safe_resume_command || failure.safe_resume_command || '';
        const nextAction = data.next_action || failure.next_action || '';
        if (recoveryStrategy) lines.push(`恢复策略：${recoveryStrategy}`);
        if (safeResume) lines.push(`安全继续：${safeResume}`);
        if (nextAction) lines.push(`下一步：${nextAction}`);
        return lines.join('\n\n');
      }
      if (summary.type !== 'review_loop_completed') return '';
      const data = summary.review_loop || {};
      const lines = [`审稿循环已完成：经过 ${data.rounds_completed || 0}/${data.requested_rounds || 0} 轮审稿专家攻击和改写。`];
      if (data.final_idea) lines.push(`最终 idea：${data.final_idea}`);
      lines.push('中间审稿意见已进入会话记忆；对话区只保留最终结果。');
      return lines.join('\n');
    }
    function sessionStepJobText(result, displayResult) {
      const failure = parseJsonLine(result, 'Session-step failure JSON');
      if (failure && failure.status === 'session_step_failed') {
        const labels = {
          llm_unavailable: '模型暂时不可用或 API 配置有问题。',
          invalid_model_response: '模型返回格式不稳定，系统没有把这轮写入对话。',
          evidence_not_ready: '多轮修改需要先补足 Genome/Pattern 证据。',
          session_evidence_guard_failed: '模型本轮改写越出了优秀论文逻辑空间，或把用户论文库/趋势论文当成了标准。',
          session_step_failed: '本轮修改没有完成。'
        };
        const reason = labels[failure.reason_code] || labels.session_step_failed;
        const lines = ['本轮追问没有写入当前会话。', reason];
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        if (failure.reason) lines.push(`原因：${compact(cleanUiText(failure.reason), 260)}`);
        return lines.join('\n\n');
      }
      const memory = parseJsonLine(result, 'Session memory JSON');
      const decision = firstLineMatch(result, /Turn\s+\d+\s+decision:\s*([^\n]+)/i);
      const revised = firstLineMatch(result, /Revised idea:\s*([^\n]+)/i);
      const currentBest = firstLineMatch(result, /Current best:\s*([^\n]+)/i);
      const trend = firstLineMatch(result, /Trend check:\s*([^\n]+)/i);
      const dossier = firstLineMatch(result, /Dossier:\s*([^\n]+)/i);
      const lines = ['本轮追问已写入当前会话，并刷新了会话记忆和 dossier。'];
      if (decision) lines.push(`决策：${decision}`);
      if (revised) lines.push(`修改后的 idea：${revised}`);
      if (currentBest && currentBest !== revised) lines.push(`当前最佳版本：${currentBest}`);
      if (trend) lines.push(`趋势对齐：${trend}`);
      if (memory) {
        const memoryBits = [];
        if (memory.turn_count !== undefined) memoryBits.push(`轮数 ${memory.turn_count}`);
        if (memory.preferred_language) memoryBits.push(`语言 ${memory.preferred_language}`);
        if (memory.recent_limitations_count !== undefined) memoryBits.push(`近半年局限性线索 ${memory.recent_limitations_count}`);
        if (memoryBits.length) lines.push(`记忆状态：${memoryBits.join(' · ')}`);
        if (memory.state_policy) lines.push('记忆策略：压缩摘要保持多轮连续性，原始中间过程不进入回答。');
        if (memory.user_library_policy) lines.push('用户库边界：只作领域背景，不作为评分标准或故事线来源。');
        if (memory.stable_thesis) lines.push(`稳定主线：${memory.stable_thesis}`);
        if (memory.unresolved_risk) lines.push(`未解决风险：${memory.unresolved_risk}`);
        if (memory.next_experiment) lines.push(`下一步实验：${memory.next_experiment}`);
      }
      if (dossier) lines.push(`Dossier：${dossier}`);
      if (!decision && !revised && displayResult) lines.push(compact(displayResult, 900));
      return lines.join('\n');
    }
    function reviewJobText(result, displayResult) {
      const failure = parseJsonLine(result, 'Reviewer failure JSON');
      if (failure && failure.status === 'reviewer_failed') {
        const labels = {
          llm_unavailable: '模型暂时不可用或 API 配置有问题。',
          invalid_model_response: '模型返回格式不稳定，系统没有保存审稿结果。',
          idea_not_found: '没有找到要审稿的 idea 或会话。',
          evidence_not_ready: '审稿需要先补足核心证据上下文。',
          reviewer_failed: '审稿人视角没有完成。'
        };
        const reason = labels[failure.reason_code] || labels.reviewer_failed;
        const lines = ['审稿人视角没有写入结果。', reason];
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        if (failure.reason) lines.push(`原因：${compact(cleanUiText(failure.reason), 260)}`);
        return lines.join('\n\n');
      }
      const decision = firstLineMatch(result, /Reviewer decision:\s*([^\n]+)/i) || firstLineMatch(result, /Decision:\s*([^\n]+)/i);
      const summary = firstLineMatch(result, /Summary:\s*([^\n]+)/i);
      const trend = firstLineMatch(result, /Trend check:\s*([^\n]+)/i);
      const revised = firstLineMatch(result, /Revised idea:\s*([^\n]+)/i);
      const saved = firstLineMatch(result, /JSON:\s*([^\n]+)/i) || firstLineMatch(result, /Saved critique #\d+:\s*([^\n]+)/i);
      const lines = ['审稿人视角已完成：已从 novelty、证据链、逻辑和可行性角度攻击当前 idea。'];
      if (decision) lines.push(`审稿结论：${decision}`);
      if (summary) lines.push(`攻击摘要：${summary}`);
      if (trend) lines.push(`趋势对齐：${trend}`);
      if (revised) lines.push(`建议改写：${revised}`);
      if (saved) lines.push(`审稿记录：${saved}`);
      if (!decision && !summary && displayResult) lines.push(compact(displayResult, 900));
      return lines.join('\n');
    }
    function cleanupJobText(result, displayResult) {
      const legacy = firstLineMatch(result, /Legacy non-core Gold paper records:\s*(\d+)/i);
      const genomes = firstLineMatch(result, /Non-Gold genome cards:\s*(\d+)/i);
      const patterns = firstLineMatch(result, /Non-Gold strict pattern cards:\s*(\d+)/i);
      const candidates = firstLineMatch(result, /Candidate ideas outside current Gold context:\s*(\d+)/i);
      const applied = /Cleanup applied\./i.test(result);
      const lines = [applied ? '证据上下文清理已完成。' : '证据上下文体检已完成。'];
      if (legacy) lines.push(`旧版非核心 Gold 记录：${legacy} 条${applied ? '，已降级到趋势/辅助来源' : '，建议执行清理降级'}`);
      if (genomes) lines.push(`非 Gold Genome 卡：${genomes} 张`);
      if (patterns) lines.push(`非 Gold 严格 Pattern 卡：${patterns} 张`);
      if (candidates) lines.push(`引用已失效 Gold 上下文的 idea：${candidates} 条`);
      if (!legacy && !genomes && !patterns && !candidates && displayResult) lines.push(compact(displayResult, 900));
      return lines.join('\n');
    }
    function reviewLoopJobText(result, displayResult) {
      const summary = parseJsonLine(result, 'Review-loop summary JSON');
      if (summary) {
        const lines = [`审稿循环已完成：经过 ${summary.rounds_completed || 0}/${summary.requested_rounds || 0} 轮审稿专家攻击和改写。`];
        if (summary.final_idea) lines.push(`最终 idea：${summary.final_idea}`);
        lines.push('中间审稿意见已进入会话记忆；对话区只保留最终结果。');
        return lines.join('\n');
      }
      const failure = parseJsonLine(result, 'Review-loop failure JSON');
      if (failure && failure.status === 'failed') {
        const done = Number(failure.rounds_completed || 0);
        const requested = Number(failure.requested_rounds || 0);
        const lines = done > 0
          ? [`审稿循环没有完整完成：已完成 ${done}/${requested || '?'} 轮，当前保留最新安全版本。`]
          : ['审稿循环没有完成，当前 idea 未被覆盖。'];
        if (failure.final_idea) lines.push(`当前 idea：${failure.final_idea}`);
        if (failure.recovery_strategy) lines.push(`恢复策略：${failure.recovery_strategy}`);
        if (failure.safe_resume_command) lines.push(`安全继续：${failure.safe_resume_command}`);
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        return lines.join('\n\n');
      }
      if (/Review loop failed/i.test(result)) return '审稿循环没有完成，当前 idea 未被最终覆盖。';
      return displayResult ? compact(displayResult, 900) : '审稿循环已结束。';
    }
    function ideateJobText(result, displayResult) {
      const summary = parseJsonLine(result, 'Idea-session summary JSON');
      const failure = parseJsonLine(result, 'Idea-generation failure JSON');
      if (failure && failure.status === 'idea_generation_not_ready') {
        const reasonLabels = {
          storyline_migration_failed: '生成结果像是在硬搬源论文方法，还没有完成“论文故事线 → 当前领域”的迁移。',
          prior_art_too_close: '生成结果和已有工作过近，需要先拉开差异边界。',
          frontier_evidence_not_ready: '当前方向的近期趋势证据不足，需要先补与问题匹配的 frontier 论文。',
          gold_evidence_not_ready: '核心优秀论文、Genome 或 Pattern 依据不足，五个评分维度还没有显式支撑。',
          user_library_used_as_standard: '模型把用户论文库当成了 Gold/故事线评分依据；用户库只能作为领域知识和路线学习背景。',
          idea_generation_not_ready: '严格证据门没有放行这批 idea。'
        };
        const primary = failure.primary_reason || 'idea_generation_not_ready';
        const lines = ['这次没有写入新对话，因为严格生成门没有放行 idea。'];
        lines.push(reasonLabels[primary] || reasonLabels.idea_generation_not_ready);
        if (failure.next_action) lines.push(`下一步：${failure.next_action}`);
        lines.push(...ideaFailureRecoveryLines(failure));
        if (Array.isArray(failure.rejected_ideas) && failure.rejected_ideas.length) {
          const rejected = failure.rejected_ideas.slice(0, 3).map(item => `${item.title || 'idea'}：${compact(cleanUiText(item.reason || ''), 180)}`).join('\n');
          lines.push(`被拒摘要：\n${rejected}`);
        }
        return lines.join('\n\n');
      }
      const name = firstLineMatch(result, /Name:\s*([^\n]+)/i);
      const currentIdea = firstLineMatch(result, /Current idea:\s*([^\n]+)/i);
      const trend = firstLineMatch(result, /Trend support:\s*([^\n]+)/i);
      const dossier = firstLineMatch(result, /Dossier:\s*([^\n]+)/i);
      const lines = ['已生成新的研究 idea，并写入当前对话。'];
      const title = summary && summary.title ? summary.title : name;
      const idea = summary && summary.current_idea ? summary.current_idea : currentIdea;
      const trendText = summary && summary.trend_support ? summary.trend_support : trend;
      const loop = summary && summary.auto_review_loop ? summary.auto_review_loop : null;
      if (title) lines.push(`标题：${title}`);
      if (loop && loop.status === 'completed') lines.push(`自动审稿循环：已完成 ${loop.rounds_completed || 0}/${loop.requested_rounds || 0} 轮，下面是最终 idea。`);
      else if (loop && loop.status === 'failed') lines.push('自动审稿循环没有完成，下面是生成阶段保留的当前 idea。');
      if (idea) lines.push(`核心想法：${idea}`);
      if (trendText) lines.push(`近期依据：${trendText}`);
      if (summary && summary.storyline_logic) lines.push(`顶会论文逻辑迁移：${summary.storyline_logic}`);
      if (summary && Array.isArray(summary.storyline_steps) && summary.storyline_steps.length) {
        const steps = summary.storyline_steps.slice(0, 6).map(item => {
          const label = item.label || item.step || '故事线';
          const transfer = item.transfer || '';
          return `${label}：${transfer}`;
        }).filter(Boolean).join('\n');
        if (steps) lines.push(`六步故事线迁移：\n${steps}`);
      }
      if (summary && summary.current_method_problem) lines.push(`当前方法问题：${summary.current_method_problem}`);
      if (summary && summary.recent_limitation_signal) lines.push(`近半年局限性/趋势信号：${summary.recent_limitation_signal}`);
      if (summary && Array.isArray(summary.evidence_basis) && summary.evidence_basis.length) {
        const basis = summary.evidence_basis.slice(0, 3).map(item => {
          const source = [item.source_type, item.source_id].filter(Boolean).join(':') || item.source_title || 'evidence';
          const title = item.source_title ? ` ${item.source_title}` : '';
          const used = item.used_for ? ` -> ${item.used_for}` : '';
          const standard = item.borrowed_standard ? `：${item.borrowed_standard}` : '';
          return `${source}${title}${used}${standard}`;
        }).join('\n');
        lines.push(`参考依据：\n${basis}`);
      }
      if (summary && summary.evidence_score !== undefined && summary.evidence_score !== '') lines.push(`证据驱动评分：${summary.evidence_score}/10`);
      if (summary && Array.isArray(summary.scoring_dimensions) && summary.scoring_dimensions.length) {
        const scoreText = summary.scoring_dimensions.slice(0, 5).map(item => {
          const covered = item.explicit_evidence_covered ? '有显式依据' : '依据不足';
          return `${item.dimension}:${item.score}/10(${covered})`;
        }).join('；');
        lines.push(`评分维度：${scoreText}`);
      }
      if (summary && summary.prior_art_gate_decision) lines.push(`Prior-art gate：${summary.prior_art_gate_decision}${summary.prior_art_gate_summary ? ' · ' + summary.prior_art_gate_summary : ''}`);
      if (summary && summary.key_risk) lines.push(`审稿风险：${summary.key_risk}`);
      if (summary && summary.first_experiment) lines.push(`首个实验：${summary.first_experiment}`);
      if (summary && Array.isArray(summary.why_not_done_yet) && summary.why_not_done_yet.length) lines.push(`为什么现在可做：${summary.why_not_done_yet.slice(0,2).join('；')}`);
      lines.push('回答会继续保留依据链：参考的优秀论文/逻辑资产、当前方法问题、近半年局限性信号、审稿攻击点，以及创新性、逻辑性、可行性、价值、抗审稿攻击性的证据驱动评分。');
      if (dossier) lines.push(`Dossier：${dossier}`);
      if (!name && !currentIdea && displayResult) lines.push(compact(displayResult, 900));
      return lines.join('\n');
    }
    function parseGoldBuildSummary(result) {
      const jsonMatch = String(result || '').match(/Gold-build summary JSON:\s*(\{[^\n]+\})/);
      if (jsonMatch) {
        try { return JSON.parse(jsonMatch[1]); } catch (error) {}
      }
      const summaryMatch = String(result || '').match(/Gold-build summary:\s*fetched=(\d+);\s*candidate_imported=(\d+);\s*excellent_imported=(\d+);\s*poster_skipped=(\d+);\s*openreview_non_excellent_skipped=(\d+);\s*unqualified_skipped=(\d+);\s*failures=(\d+)/);
      const importedMatch = String(result || '').match(/优秀论文库 imported\/updated (\d+) records/);
      const legacyRemoteCandidatePaper = '\u8054\u7f51' + '\u5019\u9009' + '\u8bba\u6587';
      const trendRemotePaper = '\u8054\u7f51\u8d8b\u52bf\u8bba\u6587';
      const candidateMatch = String(result || '').match(new RegExp(`(?:${trendRemotePaper}|${legacyRemoteCandidatePaper}) imported\\/updated (\\d+) records`));
      const scoredMatch = String(result || '').match(/优秀论文:\s*(\d+)/);
      if (!summaryMatch && !importedMatch) return null;
      return {
        fetched: summaryMatch ? Number(summaryMatch[1]) : null,
        candidate_imported: summaryMatch ? Number(summaryMatch[2]) : (candidateMatch ? Number(candidateMatch[1]) : null),
        excellent_imported: summaryMatch ? Number(summaryMatch[3]) : (importedMatch ? Number(importedMatch[1]) : null),
        poster_skipped: summaryMatch ? Number(summaryMatch[4]) : null,
        openreview_non_excellent_skipped: summaryMatch ? Number(summaryMatch[5]) : null,
        unqualified_skipped: summaryMatch ? Number(summaryMatch[6]) : null,
        failures: summaryMatch ? Number(summaryMatch[7]) : null,
        current_excellent_papers: scoredMatch ? Number(scoredMatch[1]) : null,
        metadata_only_remote: /metadata-only source/i.test(result || ''),
        source_failures: [],
        source_reports: [],
        next_actions: []
      };
    }
    function goldBuildDetailLines(summary) {
      if (!summary) return [];
      const lines = [];
      const policy = summary.quality_policy && typeof summary.quality_policy === 'object' ? summary.quality_policy : null;
      if (policy) {
        const venues = Array.isArray(policy.core_venues) ? policy.core_venues.join('、') : '';
        const signals = Array.isArray(policy.accepted_quality_signals) ? policy.accepted_quality_signals.join('、') : '';
        const excluded = Array.isArray(policy.excluded_from_gold) ? policy.excluded_from_gold.join('、') : '';
        const policyBits = [];
        if (venues) policyBits.push(`核心会议：${venues}`);
        if (signals) policyBits.push(`入库信号：${signals}`);
        if (excluded) policyBits.push(`排除：${excluded}`);
        if (policyBits.length) lines.push(`核心 Gold 标准：\n${policyBits.join('\n')}`);
      }
      const sourcePriority = Array.isArray(summary.source_priority) ? summary.source_priority : [];
      if (sourcePriority.length) {
        lines.push(`来源优先级：\n${sourcePriority.slice(0, 4).join('\n')}`);
      }
      const failures = Array.isArray(summary.source_failures) ? summary.source_failures : [];
      if (failures.length) {
        const failureText = failures.slice(0, 3).map(item => {
          if (!item || typeof item !== 'object') return '';
          const where = [item.year, item.venue, item.remote].filter(Boolean).join(' ');
          const action = item.recommended_action ? `；建议：${item.recommended_action}` : '';
          return `${where || '未知来源'}：${item.error || '没有错误详情'}${action}`;
        }).filter(Boolean).join('\n');
        if (failureText) lines.push(`失败源：\n${failureText}`);
      }
      const emptyResults = Array.isArray(summary.source_empty_results) ? summary.source_empty_results : [];
      if (emptyResults.length) {
        const emptyText = emptyResults.slice(0, 4).map(item => {
          if (!item || typeof item !== 'object') return '';
          const where = [item.year, item.venue, item.remote].filter(Boolean).join(' ');
          const action = item.recommended_action ? `；建议：${item.recommended_action}` : '';
          return `${where || '未知来源'}：请求成功但没有返回论文${action}`;
        }).filter(Boolean).join('\n');
        if (emptyText) lines.push(`空结果来源：\n${emptyText}`);
      }
      const reports = Array.isArray(summary.source_reports) ? summary.source_reports : [];
      const meaningful = reports.filter(item => item && typeof item === 'object' && (item.excellent_imported || item.frontier_imported || item.poster_skipped || item.unqualified_skipped || item.openreview_non_excellent_skipped));
      if (meaningful.length) {
        const reportText = meaningful.slice(0, 4).map(item => {
          const mode = item.mode === 'metadata_only' ? '元数据趋势' : (item.mode === 'auxiliary_frontier' ? '辅助趋势' : '核心 Gold');
          return `${item.year || ''} ${item.venue || ''}：${mode}，抓取 ${item.fetched ?? 0}，趋势 ${item.frontier_imported ?? 0}，Gold ${item.excellent_imported ?? 0}，Poster 过滤 ${item.poster_skipped ?? 0}，未达优秀信号 ${((item.unqualified_skipped ?? 0) + (item.openreview_non_excellent_skipped ?? 0))}`;
        }).join('\n');
        lines.push(`来源明细：\n${reportText}`);
      }
      const actions = Array.isArray(summary.next_actions) ? summary.next_actions : [];
      if (actions.length) {
        const actionText = actions.slice(0, 4).map(item => {
          if (!item || typeof item !== 'object') return '';
          const label = item.label || '下一步';
          const reason = item.reason ? `：${item.reason}` : '';
          const command = item.command ? `\n  ${item.command}` : '';
          return `${label}${reason}${command}`;
        }).filter(Boolean).join('\n');
        if (actionText) lines.push(`建议动作：\n${actionText}`);
      }
      return lines.map(cleanUiText);
    }
    function completedGoldBuildGroupForJob(job, jobs) {
      if (!job || job.kind !== 'gold-build' || !job.group_id) return null;
      const groupJobs = (jobs || []).filter(item => item && item.kind === 'gold-build' && item.group_id === job.group_id);
      if (groupJobs.length <= 1) return null;
      if (groupJobs.some(item => item.status === 'running')) return null;
      return groupJobs;
    }
    function summarizeGoldBuildGroup(groupJobs) {
      const summaries = groupJobs.map(item => parseGoldBuildSummary(item.result ? `${item.result.stdout || ''}${item.result.stderr || ''}`.trim() : item.error)).filter(Boolean);
      const total = key => summaries.reduce((sum, item) => sum + Number(item && item[key] !== null && item[key] !== undefined ? item[key] : 0), 0);
      const allReports = summaries.flatMap(item => Array.isArray(item.source_reports) ? item.source_reports : []);
      const allFailures = summaries.flatMap(item => Array.isArray(item.source_failures) ? item.source_failures : []);
      const allEmptyResults = summaries.flatMap(item => Array.isArray(item.source_empty_results) ? item.source_empty_results : []);
      const failedJobs = groupJobs.filter(item => item.status === 'failed').length;
      return {
        fetched: total('fetched'),
        candidate_imported: total('candidate_imported'),
        excellent_imported: total('excellent_imported'),
        poster_skipped: total('poster_skipped'),
        openreview_non_excellent_skipped: total('openreview_non_excellent_skipped'),
        unqualified_skipped: total('unqualified_skipped'),
        failures: total('failures') + failedJobs,
        current_excellent_papers: summaries.reduce((value, item) => Math.max(value, Number(item.current_excellent_papers || 0)), 0) || null,
        metadata_only_remote: summaries.some(item => item.metadata_only_remote),
        source_reports: allReports,
        source_failures: allFailures,
        source_empty_results: allEmptyResults,
        next_actions: summaries.flatMap(item => Array.isArray(item.next_actions) ? item.next_actions : []),
        group_jobs: groupJobs.length
      };
    }
    function goldBuildGroupJobText(groupJobs) {
      const summary = summarizeGoldBuildGroup(groupJobs);
      const lines = [
        `本轮优秀论文库扩充已完成：${summary.group_jobs} 个来源任务。`,
        `聚合结果：抓取 ${summary.fetched} 条；趋势区写入 ${summary.candidate_imported} 条；核心优秀依据库写入 ${summary.excellent_imported} 条；Poster 硬过滤 ${summary.poster_skipped} 条；未达 Best/Nominee/Oral 信号 ${summary.openreview_non_excellent_skipped + summary.unqualified_skipped} 条；失败源 ${summary.failures} 个。`
      ];
      lines.push(`当前核心优秀依据库：${summary.current_excellent_papers ?? '刷新后可见'} 篇。普通联网论文只留在趋势区，不参与 idea 评分标准。`);
      if (summary.metadata_only_remote) lines.push('其中包含元数据趋势来源：需运行质量核验，确认 Best / Nominee / Oral 后才可进入核心优秀依据库。');
      lines.push(...goldBuildDetailLines(summary));
      if (Number(summary.excellent_imported || 0) === 0) {
        lines.push('这不是按钮失效，而是严格过滤后没有发现可进入核心优秀论文库的记录；建议查看趋势论文或继续执行质量核验。');
      }
      return lines.join('\n\n');
    }
    function jobText(job) {
      const result = job.result ? `${job.result.stdout || ''}${job.result.stderr || ''}`.trim() : job.error;
      const displayResult = cleanUiText(result || '');
      const summary = job.result_summary || null;
      const jobName = displayJobName(job);
      if (job.status === 'running') return `${jobName} 正在运行。`;
      if (job.status === 'failed') {
        if (job.kind === 'ideate') {
          const text = ideateSummaryText(summary);
          if (text) return text;
        }
        if (job.kind === 'step') {
          const text = sessionStepSummaryText(summary);
          if (text) return text;
        }
        if (job.kind === 'review') {
          const text = reviewSummaryText(summary);
          if (text) return text;
        }
        if (job.kind === 'review-loop') {
          const text = reviewLoopSummaryText(summary);
          if (text) return text;
        }
        const groundingText = groundingAuditJobText(result, displayResult);
        if (groundingText) return groundingText;
        if (job.kind === 'step') return sessionStepJobText(result, displayResult);
        if (job.kind === 'review') return reviewJobText(result, displayResult);
        if (job.kind === 'review-loop') return reviewLoopSummaryText(summary) || '审稿循环没有完成，当前 idea 未被最终覆盖。';
        if (job.kind === 'gold-build') {
          const summary = parseGoldBuildSummary(result);
          if (summary) {
            const lines = [
              `${jobName} 没有完成写入：抓取 ${summary.fetched ?? 0} 条；趋势区写入 ${summary.candidate_imported ?? 0} 条；核心优秀依据库写入 ${summary.excellent_imported ?? 0} 条；失败源 ${summary.failures ?? 0} 个。`
            ];
            lines.push(...goldBuildDetailLines(summary));
            if (!lines.some(line => /建议动作/.test(line))) lines.push('建议动作：\n重试失败源；如果连续失败，缩小年份范围或改用官方奖项页/OpenReview Oral 来源。');
            return lines.join('\n\n');
          }
        }
        return `${jobName} 失败：${displayResult || '没有返回详情'}`;
      }
      if (job.kind === 'ideate') {
        const text = ideateSummaryText(summary);
        if (text) return text;
        if (job.created_session_id) return '已生成新的研究 idea。';
      }
      if (job.kind === 'step') return sessionStepSummaryText(summary) || sessionStepJobText(result, displayResult);
      if (job.kind === 'review') return reviewSummaryText(summary) || reviewJobText(result, displayResult);
      if (job.kind === 'review-loop') return reviewLoopSummaryText(summary) || reviewLoopJobText(result, displayResult);
      if (job.kind === 'cleanup') return cleanupJobText(result, displayResult);
      const groundingText = groundingAuditJobText(result, displayResult);
      if (groundingText) return groundingText;
      let text = `${jobName} 已完成。${displayResult ? '\n\n' + compact(displayResult, 1200) : ''}`;
      if (job.kind === 'ideate' && !job.created_session_id) {
        const detailMatch = result.match(/Current-hotspot evidence is not ready for this query\.[\s\S]{0,600}/);
        const strictMatch = result.match(/Strict evidence is not ready yet[\s\S]{0,600}/);
        const repairMatch = result.match(/frontier_repair:\s*(completed|no_new_records|failed|skipped)[^\n]*/i) || result.match(/Auto frontier repair:[\s\S]{0,260}/i);
        const repairText = repairMatch ? `\n\n自动补证据状态：${repairMatch[0]}` : '';
        const evidenceText = cleanUiText((detailMatch && detailMatch[0]) || (strictMatch && strictMatch[0]) || result || '请先扩充与该方向匹配的近期优秀论文和趋势论文。');
        text = '这次没有写入新对话，因为严格证据门没有放行 idea 生成。系统会先尝试自动补充与当前方向匹配的趋势论文，并保持这些 frontier 记录不进入 Gold 评分标准；如果补完仍不对齐，就会停在这里等待扩充优秀论文依据。\n\n' + compact(evidenceText, 1100) + repairText;
      }
      if (job.kind === 'gold-build') {
        const summary = parseGoldBuildSummary(result);
        if (summary) {
          text += `\n\n联网扩充结果：抓取 ${summary.fetched ?? '若干'} 条；趋势区写入 ${summary.candidate_imported ?? '若干'} 条；核心优秀依据库写入 ${summary.excellent_imported ?? '若干'} 条；Poster 硬过滤 ${summary.poster_skipped ?? 0} 条；OpenReview 非 Best/Nominee/Oral 跳过 ${summary.openreview_non_excellent_skipped ?? 0} 条；缺少核心优秀论文信号跳过 ${summary.unqualified_skipped ?? 0} 条；失败源 ${summary.failures ?? 0} 个。`;
          text += `\n\n本次联网趋势论文：${summary.candidate_imported ?? '若干'} 篇；进入优秀依据库：${summary.excellent_imported ?? '若干'} 篇；当前优秀依据库：${summary.current_excellent_papers ?? '刷新后可见'} 篇。普通联网论文不会进入评分标准，可在“证据库”的“趋势论文”里查看。`;
          if (summary.metadata_only_remote) {
            text += '\n\n这次来源是元数据收集模式：论文会先进入趋势区，只有经过 quality-enrich 核验出明确 Best/Nominee/Oral 信号后，才会进入核心优秀依据库。';
          }
          const detailLines = goldBuildDetailLines(summary);
          if (detailLines.length) text += '\n\n' + detailLines.join('\n\n');
        }
      }
      if (job.kind === 'gold-build' && result && /imported\/updated 0 records/.test(result)) {
        text += '\n\n这不是按钮失效，而是严格过滤后没有发现可进入核心优秀论文库的记录。当前规则只允许核心来源的最佳论文、提名/Outstanding 或 Oral 进入评分依据；普通顶会论文、Poster、Spotlight 和仅高引用论文都会留在趋势区。';
      }
      if (job.kind === 'quality-enrich' && result) {
        const appliedMatch = result.match(/Applied labels:\s*(\d+)/);
        const skippedMatch = result.match(/Skipped labels:\s*(\d+)/);
        if (appliedMatch) text += `\n\n本次核验后新增/更新优秀标签：${appliedMatch[1]} 篇。`;
        if (skippedMatch) text += `未通过元数据把关、保留为趋势参考：${skippedMatch[1]} 条。`;
        text += '\n\n质量核验只更新论文元数据标签，不会把大模型先验写入 idea 或评分标准；需要二次核验或仅为高引用猜测的记录会留在趋势区。';
      }
      return text;
    }
    function jobActions(job) {
      if (!job || job.status === 'running') return [];
      if (job.kind === 'gold-build') {
        const result = job.result ? `${job.result.stdout || ''}${job.result.stderr || ''}`.trim() : job.error;
        const summary = parseGoldBuildSummary(result);
        const actions = job.status === 'failed'
          ? [{label:'调整检索', action:'settings'}, {label:'查看趋势论文', action:'frontier'}]
          : [{label:'查看核心库', action:'excellent'}, {label:'查看趋势论文', action:'frontier'}];
        if (summary && Number(summary.candidate_imported || 0) > 0 && Number(summary.excellent_imported || 0) === 0) {
          actions.push({label:'核验质量标签', action:'quality-enrich'});
        }
        return actions;
      }
      if (job.kind === 'quality-enrich') return job.status === 'failed'
        ? [{label:'重新核验', action:'quality-enrich'}, {label:'查看趋势论文', action:'frontier'}]
        : [{label:'查看核心库', action:'excellent'}, {label:'查看趋势论文', action:'frontier'}];
      if (job.kind === 'ideate' && job.created_session_id) return [{label:'查看证据库', action:'library'}, {label:'审稿人攻击', action:'review'}];
      if (job.kind === 'cleanup') return job.status === 'failed'
        ? [{label:'调整检索', action:'settings'}, {label:'查看运行记录', action:'runs'}]
        : [{label:'查看核心库', action:'excellent'}, {label:'查看运行记录', action:'runs'}];
      return [];
    }
    function goldBuildGroupActions(groupJobs) {
      const summary = summarizeGoldBuildGroup(groupJobs);
      const actions = Number(summary.failures || 0) > 0
        ? [{label:'调整检索', action:'settings'}, {label:'查看趋势论文', action:'frontier'}]
        : [{label:'查看核心库', action:'excellent'}, {label:'查看趋势论文', action:'frontier'}];
      if (Number(summary.candidate_imported || 0) > 0 && Number(summary.excellent_imported || 0) === 0) actions.push({label:'核验质量标签', action:'quality-enrich'});
      return actions;
    }
    function jobActivity(job) {
      return {role:'assistant', meta:`任务 · ${jobStatusLabel(job.status)}`, text:jobText(job), job_id:job.id, actions:jobActions(job)};
    }
    function jobStatusLabel(status) {
      return ({running:'运行中', completed:'已完成', failed:'失败'}[status] || status || '任务');
    }
    function renderChat(chat) {
      const el=document.getElementById('chatMessages');
      const wasNearBottom = !el || (el.scrollHeight - el.scrollTop - el.clientHeight < 80);
      const active = chat && chat.active && chat.session && !newChatMode;
      chatActive = Boolean(active);
      if (active && !activeSessionId) activeSessionId = Number(chat.session.id);
      const scope = active && chat.session ? `session:${Number(chat.session.id)}` : currentScope();
      if (!isBusy) document.getElementById('sendButton').textContent = active ? '发送' : '生成';
      document.getElementById('instruction').placeholder = active ? '继续追问或修改当前 idea' : '说一个研究领域、问题或粗略方向';
      const serverMessages = active ? ((chat && chat.messages) || []) : (chat && chat.draft ? (chat.messages || []) : [{role:'assistant', kind:'draft', meta:'new workspace', text:'已准备好新对话。直接输入你想研究的方向，我会从论文逻辑库和近期证据里生成新 idea。'}]);
      const transientMessages = scopeMessages(transientMessagesByScope, scope);
      const adoptedUserMessages = active ? transientMessages.filter(m => m.adoptedFromDraft && !m.thinking) : [];
      const liveTransientMessages = active ? transientMessages.filter(m => !m.adoptedFromDraft) : transientMessages;
      let messages = adoptedUserMessages.concat(serverMessages).concat(scopeMessages(activityMessagesByScope, scope)).concat(liveTransientMessages);
      const jobs = ((window.lastData || {}).jobs || []).slice().reverse();
      let hasRunning = false;
      let hasBlockingRunning = false;
      for (const job of jobs) {
        const jobScope = job.ui_scope || (job.session_id ? `session:${job.session_id}` : '');
        const belongsToRenderedScope = jobScope === scope || isCreatedIdeateForScope(job, scope);
        if (!belongsToRenderedScope) continue;
        if (job.status === 'running') {
          hasRunning = true;
          if (isConversationBlockingJob(job)) hasBlockingRunning = true;
        }
        if (job.status === 'running' && lastJobStatuses.get(job.id) !== 'running') {
          lastJobStatuses.set(job.id, 'running');
        }
        if (job.status !== 'running' && lastJobStatuses.get(job.id) !== job.status) {
          let finalizedInThisPass = false;
          const goldGroupJobs = completedGoldBuildGroupForJob(job, jobs);
          if (goldGroupJobs) {
            const groupKey = `gold-group:${job.group_id}`;
            if (lastJobStatuses.get(groupKey)) {
              lastJobStatuses.set(job.id, job.status);
              continue;
            }
            const activity = {role:'assistant', meta:`任务 · ${goldGroupJobs.some(item => item.status === 'failed') ? '部分完成' : '已完成'}`, text:goldBuildGroupJobText(goldGroupJobs), job_id:groupKey, actions:goldBuildGroupActions(goldGroupJobs)};
            setScopeMessages(activityMessagesByScope, jobScope, scopeMessages(activityMessagesByScope, jobScope).filter(m => m.job_id !== groupKey && !goldGroupJobs.some(item => m.job_id === item.id)).concat([activity]).slice(-8));
            messages = messages.concat([activity]);
            lastJobStatuses.set(groupKey, 'completed');
            goldGroupJobs.forEach(item => lastJobStatuses.set(item.id, item.status));
            clearProgress(jobScope);
            messages = messages.filter(m => !m.thinking);
            continue;
          }
          if (job.kind === 'ideate') {
            if (!isPendingIdeateJob(job) && !isCreatedIdeateForScope(job, scope)) continue;
            finalizedInThisPass = finalizeCompletedIdeateJob(job, scope, {activate:Boolean(job.created_session_id && job.ui_scope === scope)});
          }
          clearProgress(jobScope);
          messages = messages.filter(m => !m.thinking);
          if (finalizedInThisPass) {
            if (isCreatedIdeateForScope(job, scope)) messages = messages.concat(scopeMessages(activityMessagesByScope, scope).filter(m => m.job_id === job.id));
            continue;
          }
          if (job.kind === 'ideate' && job.created_session_id) {
            lastJobStatuses.set(job.id, job.status);
            continue;
          }
          const activity = jobActivity(job);
          const targetScope = jobScope;
          setScopeMessages(activityMessagesByScope, targetScope, scopeMessages(activityMessagesByScope, targetScope).filter(m => m.job_id !== job.id).concat([activity]).slice(-8));
          messages = messages.concat([activity]);
          lastJobStatuses.set(job.id, job.status);
          setScopeMessages(transientMessagesByScope, jobScope, []);
        }
      }
      const awaitingCurrentScope = scopeAwaitingJobStart(scope);
      if (!awaitingCurrentScope && !hasRunning && scopeMessages(transientMessagesByScope, scope).some(m => m.thinking)) setScopeMessages(transientMessagesByScope, scope, []);
      setBusy(hasBlockingRunning || awaitingCurrentScope);
      renderLiveProgress(scope);
      const isFreshDraft = !active && !liveTransientMessages.length && !scopeMessages(activityMessagesByScope, scope).length;
      const scopeProgressText = visibleProgressText(scope);
      const nextKey = JSON.stringify([isFreshDraft, scopeProgressText].concat(messages.map(m => [m.role, m.meta, m.text, Boolean(m.thinking), m.actions || []])));
      if (nextKey !== chatRenderKey) {
        const openingCard = isFreshDraft
          ? `<div class="query-card"><div class="query-card-title">Start a deep research search</div><div class="query-card-text">输入领域、问题或目标会议，先从一个清楚方向开始。</div><div class="quick-prompts"><button class="quick-prompt" onclick="fillPrompt('科研智能体可靠评测，重点考虑近半年优秀论文中的局限性')">科研智能体可靠评测</button><button class="quick-prompt" onclick="fillPrompt('多智能体协作中的可验证失败发现，要求输出审稿人攻击点')">多智能体失败发现</button><button class="quick-prompt" onclick="fillPrompt('长上下文智能体的记忆压缩和可追溯评测，面向 ICLR / NeurIPS')">长上下文记忆</button></div></div>`
          : '';
        const visibleMessages = isFreshDraft ? [] : messages;
        el.innerHTML = openingCard + visibleMessages.map(m => {
        const role = m.role === 'user' ? 'user' : 'assistant';
        const avatar = role === 'user' ? '你' : 'RA';
        const note = visibleProgressText(scope) || '正在组织证据和回答';
        const actions = !m.thinking && Array.isArray(m.actions) && m.actions.length
          ? `<div class="bubble-actions">${m.actions.map(action => `<button class="bubble-action" onclick="messageAction('${esc(action.action)}')" title="${esc(action.label)}">${esc(action.label)}</button>`).join('')}</div>`
          : '';
        const messageText = displayText(m.text || '');
        const messageMeta = displayText(m.meta || (role==='user'?'user':'assistant'));
        const thinking = m.thinking ? `<div class="thinking-row"><div class="thinking-pulse"><span class="bubble-spinner"></span><div class="typing-dots"><span></span><span></span><span></span></div></div><div class="thinking-note">${esc(note)}</div></div>` : `<div class="bubble-text">${esc(messageText)}</div>${actions}`;
        return `<div class="msg ${role} ${m.thinking?'typing':''}"><div class="avatar">${avatar}</div><div class="bubble"><div class="bubble-meta">${esc(messageMeta)}</div>${thinking}</div></div>`;
        }).join('');
        chatRenderKey = nextKey;
        if (pendingAutoScroll || wasNearBottom) el.scrollTop=el.scrollHeight;
      }
      pendingAutoScroll = false;
    }
    function fillPrompt(text) {
      const instruction = document.getElementById('instruction');
      instruction.value = text;
      saveInstructionDraft(currentScope());
      autosizeInstruction();
      instruction.focus();
    }
    function renderSessionHistory(data) {
      const sessions = data.sessions || [];
      const pendingEntries = Object.entries(pendingSearchesByScope).filter(([scope]) => !String(scope).startsWith('session:'));
      document.getElementById('sessionCount').textContent = sessions.length + pendingEntries.length;
      const pendingRows = pendingEntries.map(([scope, item]) => `<div class="session-row"><button class="session-item pending ${scope===currentScope()?'active':''}" onclick="selectPendingSearch('${esc(scope)}')"><div class="session-title">${esc(cleanUiText(item.title || 'New search'))}</div><div class="session-preview">等待生成正式对话</div><div class="session-status"><span class="mini-dot"></span>${esc(sidebarProgressText(scope) || '任务排队中')}</div></button><button class="delete-session" onclick="cancelPendingSearch('${esc(scope)}', event)" title="隐藏临时记录">×</button></div>`).join('');
      const sessionRows = sessions.map(s => {
        const scope = sessionScopeForId(s.id);
        const progress = sidebarProgressText(scope);
        const preview = latestScopePreview(scope, s.current_idea);
        const active = !newChatMode && Number(s.id)===Number(activeSessionId || (data.chat && data.chat.session && data.chat.session.id));
        const status = progress ? `<div class="session-status"><span class="mini-dot"></span>${esc(cleanUiText(progress))}</div>` : '';
        return `<div class="session-row"><button class="session-item ${active?'active':''}" onclick="selectSession(${Number(s.id)})"><div class="session-title">${esc(cleanUiText(s.name))}</div><div class="session-preview">${esc(cleanUiText(preview))}</div>${status}</button><button class="delete-session" onclick="deleteSession(${Number(s.id)}, event)" title="删除对话">×</button></div>`;
      }).join('');
      document.getElementById('sessionHistory').innerHTML = pendingRows + sessionRows || emptyAction('还没有历史对话。', 'chat', '回到对话输入研究方向');
    }
    function selectPendingSearch(scope) {
      if (!scope || !pendingSearchesByScope[scope]) return;
      showChat();
      bumpViewToken();
      saveInstructionDraft(currentScope());
      activeSessionId = null;
      newChatMode = true;
      draftScopeId = scope;
      chatRenderKey = '';
      restoreInstructionDraft(scope);
      renderChat({active:false, session:null, draft:true, messages:[]});
      renderSessionHistory(window.lastData || {});
    }
    function cancelPendingSearch(scope, event) {
      event.stopPropagation();
      removePendingSearch(scope);
      clearProgress(scope);
      clearPendingIdeate(scope);
      clearAwaitingUiState(scope);
      setScopeMessages(transientMessagesByScope, scope, scopeMessages(transientMessagesByScope, scope).filter(m => !m.thinking));
      renderSessionHistory(window.lastData || {});
      renderChat((window.lastData || {}).chat || {active:chatActive, messages:[]});
    }
    function renderLibrary(data) {
      document.getElementById('libraryCount').textContent = (data.top_papers || []).length;
      const excellent = data.top_papers || [];
      const frontier = data.frontier_papers || [];
      const userPapers = data.user_papers || [];
      const panel = (slot, id, label, count, body) => {
        const collapsed = Boolean(libraryCollapsed[id]);
        const icon = collapsed ? '+' : '−';
        const node = document.getElementById(slot);
        node.className = `library-panel ${collapsed ? 'collapsed' : ''}`;
        node.innerHTML = `<button id="${esc(id)}" class="library-toggle" onclick="toggleLibraryPanel('${esc(id)}')" aria-expanded="${collapsed ? 'false' : 'true'}"><h2>${esc(label)}</h2><span class="section-right"><span class="pill">${esc(count)}</span><span class="library-toggle-icon">${esc(icon)}</span></span></button><div class="library-body">${body}</div>`;
      };
      renderUserLibraryPanel(userPapers);
      renderUserLibraryMainPanel(userPapers);
      const papersBody = (excellent.length ? `<div class="paper-list">${excellent.map(p=>renderPaperRow(p, 'excellent')).join('')}</div>` : emptyAction('还没有可进入生成/评分标准的核心优秀论文。点击去扩充核心优秀论文库。', 'settings', '打开检索设置')) + `<button id="frontierPanel" class="section-link" onclick="metricAction('frontier')" title="查看趋势论文分区"><h2>趋势论文</h2><span class="section-right"><span class="pill">${esc(frontier.length)}</span></span></button>` + (frontier.length ? `<div class="paper-list">${frontier.map(p=>renderPaperRow(p, 'frontier')).join('')}</div>` : emptyAction('联网检索到但未通过核心优秀论文标准的论文会显示在这里。点击回到对话生成 idea 或补充证据。', 'chat', '回到对话'));
      panel('papers', 'excellentPanel', '核心优秀依据库', excellent.length, papersBody);
      panel('ideas', 'ideasPanel', '想法记录', (data.ideas||[]).length, ((data.ideas||[]).length ? data.ideas.map(i=>`<div class="artifact-row"><button class="artifact-card clickable-row" onclick="showChat()" title="回到对话查看 idea"><div class="title">${esc(cleanUiText(i.title))}</div><div class="meta">#${esc(i.id)} · ${esc(cleanUiText(i.query || ''))}<br/>score=${esc(i.evidence_score || '-')}</div></button><button class="delete-artifact" onclick="deleteIdea(${Number(i.id)}, event)" title="删除想法记录">×</button></div>`).join('') : emptyAction('还没有想法记录。点击回到对话输入研究方向。', 'chat', '回到对话')));
      panel('patterns', 'patternsPanel', '逻辑卡', (data.patterns||[]).length, ((data.patterns||[]).length ? data.patterns.map(p=>`<button class="artifact-card clickable-row" onclick="metricAction('excellent')" title="查看逻辑卡来源论文"><div class="title">${esc(cleanUiText(p.pattern_key))}</div><div class="meta">#${esc(p.id)} · ${esc(cleanUiText(p.created_at))}</div></button>`).join('') : emptyAction('还没有逻辑卡。点击查看核心优秀论文库并补充证据。', 'excellent', '查看核心优秀论文库')));
      panel('runs', 'runsPanel', '运行记录', (data.runs||[]).length, ((data.runs||[]).length ? data.runs.map(r=>`<button class="artifact-card clickable-row" onclick="metricAction('runs')" title="查看运行记录"><div class="title">${esc(cleanUiText(r.query))}</div><div class="meta">#${esc(r.id)} · ${esc(cleanUiText(r.status))}<br/>${esc(cleanUiText(r.created_at || ''))}</div></button>`).join('') : emptyAction('还没有运行记录。点击回到对话启动一次搜索。', 'chat', '回到对话')));
    }
    function renderUserLibraryMainPanel(userPapers=[]) {
      const node = document.getElementById('userLibraryPanel');
      if (!node) return;
      const body = `<form class="user-library-form" onsubmit="addUserLibraryPaper(event)"><div><label>论文链接</label><input id="userLibraryUrl" placeholder="arXiv / DOI / OpenReview" /></div><div class="row"><div><label>标题</label><input id="userLibraryTitle" placeholder="可选" /></div><div><label>备注</label><input id="userLibraryNote" placeholder="路线线索，可选" /></div></div><button id="addUserLibraryButton" type="submit">添加</button><div id="userLibraryError" class="user-library-error"></div></form><div class="user-library-note">只作领域背景和路线学习，不进入 Gold、评分标准或故事线来源。</div>${userPapers.length ? `<div class="paper-list">${userPapers.map(p=>renderPaperRow(p, 'user')).join('')}</div>` : empty('还没有用户论文。')}`;
      node.className = 'library-panel user-library-main';
      node.innerHTML = `<button id="userLibraryPanelToggle" class="library-toggle" onclick="toggleLibraryPanel('userLibraryPanel')" aria-expanded="${libraryCollapsed.userLibraryPanel ? 'false' : 'true'}"><h2>用户论文库</h2><span class="section-right"><span class="pill">${esc(userPapers.length)}</span><span class="library-toggle-icon">${libraryCollapsed.userLibraryPanel ? '+' : '−'}</span></span></button><div class="library-body">${body}</div>`;
      if (libraryCollapsed.userLibraryPanel) node.classList.add('collapsed');
    }
    function renderUserLibraryPanel(userPapers=[]) {
      const node = document.getElementById('sideUserLibrary');
      if (!node) return;
      node.innerHTML = `<span class="user-library-tab-label">用户论文库 <span class="pill">${esc(userPapers.length)}</span></span><span class="user-library-tab-arrow">›</span>`;
    }
    function toggleLibraryPanel(id) {
      libraryCollapsed[id] = !libraryCollapsed[id];
      renderLibrary(window.lastData || {});
    }
    function evidenceFailureHint(job) {
      const summary = job && job.failure_summary && typeof job.failure_summary === 'object' ? job.failure_summary : null;
      if (!summary) return displayText(job && job.error ? job.error : '证据任务失败；可重试或缩小检索范围。');
      const parts = [];
      if (summary.hint) parts.push(displayText(summary.hint));
      if (summary.next_step) parts.push(`下一步：${displayText(summary.next_step)}`);
      if (summary.candidate_imported !== undefined || summary.excellent_imported !== undefined || summary.failures !== undefined) {
        parts.push(`趋势 ${summary.candidate_imported ?? 0} · Gold ${summary.excellent_imported ?? 0} · 失败源 ${summary.failures ?? 0}`);
      }
      return parts.filter(Boolean).join(' · ');
    }
    function evidenceStatusText(status, ready) {
      const labels = {
        missing_provider_key: '需要模型',
        needs_evidence: '需要证据',
        building: '证据构建中',
        evidence_failed: '证据任务失败',
        ready: '证据已就绪'
      };
      return labels[status] || (ready ? labels.ready : labels.needs_evidence);
    }
    function renderJobs(data) {
      const jobs=data.jobs || [];
      document.getElementById('jobCount').textContent = jobs.length;
      const runningJobs = jobs.filter(j => j.status === 'running');
      const running = runningJobs[0];
      const health = data.health || {};
      const failedEvidence = health.latest_failed_evidence_job || jobs.find(j => j.status === 'failed' && ['gold-build', 'quality-enrich', 'cleanup'].includes(j.kind));
      const el = document.getElementById('jobFeed');
      el.className = `status-mini status-link ${running ? 'running' : ''}`;
      el.onclick = () => metricAction('chat');
      el.textContent = running ? `正在运行 ${runningJobs.length} 个任务` : '空闲';
      if (!running && failedEvidence) {
        const action = failedEvidence.kind === 'quality-enrich' ? 'quality-enrich' : 'settings';
        el.onclick = () => metricAction(action);
        const hint = evidenceFailureHint(failedEvidence);
        el.textContent = failedEvidence.kind === 'quality-enrich' ? `最近失败：${displayJobName(failedEvidence)}。点击重新核验。` : `最近失败：${displayJobName(failedEvidence)}。点击调整检索。`;
        el.title = hint;
      } else {
        el.title = running ? '任务结果只会出现在中间对话' : '任务结果只会出现在中间对话';
      }
    }
    function metricAction(action) {
      if (action === 'library') showLibrary();
      if (action === 'user-library') { libraryCollapsed.userLibraryPanel = false; showLibrary(); renderLibrary(window.lastData || {}); scrollLibraryTo('userLibraryPanel'); }
      if (action === 'excellent') { libraryCollapsed.excellentPanel = false; showLibrary(); renderLibrary(window.lastData || {}); scrollLibraryTo('excellentPanel'); }
      if (action === 'frontier') { libraryCollapsed.excellentPanel = false; showLibrary(); renderLibrary(window.lastData || {}); scrollLibraryTo('frontierPanel'); }
      if (action === 'patterns') { libraryCollapsed.patternsPanel = false; showLibrary(); renderLibrary(window.lastData || {}); scrollLibraryTo('patternsPanel'); }
      if (action === 'runs') { libraryCollapsed.runsPanel = false; showLibrary(); renderLibrary(window.lastData || {}); scrollLibraryTo('runsPanel'); }
      if (action === 'chat') showChat();
      if (action === 'quality-enrich') qualityEnrich();
      if (action === 'review') review();
      if (action === 'settings') { showChat(); const target=document.getElementById('searchSettings'); if (target) { target.open=true; target.scrollIntoView({block:'start', behavior:'smooth'}); } }
    }
    function messageAction(action) { metricAction(action); }
    function scrollLibraryTo(id) {
      window.setTimeout(() => {
        const target = document.getElementById(id);
        if (target) {
          target.scrollIntoView({block:'start', behavior:'smooth'});
          target.classList.add('flash');
          window.setTimeout(() => target.classList.remove('flash'), 900);
        }
      }, 40);
    }
    function renderMetricCard(label, value, action, title, destination, hint) {
      const target = destination || '证据分区';
      const helper = hint || '打开对应位置';
      return `<button class="metric metric-button" onclick="metricAction('${esc(action)}')" title="${esc(title)} · ${esc(target)}" aria-label="${esc(title)}，打开${esc(target)}" data-action-target="${esc(target)}" data-tooltip="${esc(helper)} · ${esc(target)}"><span class="metric-top"><span><strong>${esc(value)}</strong><span class="metric-label">${esc(label)}</span></span></span><span class="metric-open" aria-hidden="true">›</span></button>`;
    }
    function renderMetrics(metricCards) {
      const nextKey = JSON.stringify(metricCards || []);
      if (nextKey === metricsRenderKey) return;
      document.getElementById('metrics').innerHTML = '<div class="metric-grid">' + metricCards.map(x=>renderMetricCard(x[0], x[1], x[2], x[3], x[4], x[5])).join('') + '</div>';
      metricsRenderKey = nextKey;
    }
    function scheduleRefresh(delay=2500) {
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(refresh, delay);
    }
    async function refresh() {
      const seq = ++refreshSeq;
      const token = currentViewToken();
      let activeScope = currentScope();
      const params = activeSessionId && !newChatMode ? `?session_id=${encodeURIComponent(activeSessionId)}` : (newChatMode || pendingIdeateForScope(activeScope) ? '?draft=1' : '');
      let data = {};
      try {
        data = await fetchStateJson('/api/state' + params);
        if (lastStateError) clearRefreshFailureMessages();
        lastStateError = '';
      } catch (error) {
        const message = cleanUiText(requestErrorMessage(error));
        if (lastStateError !== message) {
          lastStateError = message;
          pushActivityMessage(`状态刷新失败：${message}\n\n当前对话没有被切换；请刷新或从左侧历史重新选择。`, '请求失败', activeScope);
          renderChat((window.lastData || {}).chat || {active:chatActive, messages:[]});
        }
        setBusy(false);
        renderLiveProgress(activeScope);
        scheduleRefresh(2500);
        return;
      }
      if (seq !== refreshSeq) return;
      if (!viewStillCurrent(token, activeScope)) return;
      if (handleMissingActiveSession(data, activeScope)) activeScope = currentScope();
      const jobs = data.jobs || [];
      const hasRunningJob = jobs.some(j => j.status === 'running') || awaitingJobStart;
      const activePending = pendingIdeateForScope(activeScope);
      const pendingJob = activePending ? jobs.find(j => j.kind === 'ideate' && j.ui_scope === activeScope && (!activePending.jobId || Number(j.id) === Number(activePending.jobId))) : null;
      if (activePending && pendingJob && pendingJob.status !== 'running') {
        if (pendingJob.created_session_id) {
          finalizeCompletedIdeateJob(pendingJob, activeScope, {activate:true});
          data = await fetchStateJson('/api/state?session_id=' + encodeURIComponent(activeSessionId));
          if (seq !== refreshSeq) return;
        } else {
          finalizeCompletedIdeateJob(pendingJob, activeScope, {activate:false});
        }
      } else if (activePending && !pendingJob && !runningJobsForScope(jobs, activeScope).length) {
        if (!scopeAwaitingJobStart(activeScope)) clearPendingIdeate(activeScope);
      }
      finalizeCompletedIdeateJobs(jobs, activeScope);
      reconcilePendingIdeates(jobs, activeScope);
      window.lastData = data;
      renderOptionControls(data);
      if (!activeSessionId && data.chat && data.chat.session && !newChatMode) activeSessionId = Number(data.chat.session.id);
      renderChat(data.chat || {});
      renderSessionHistory(data);
      renderLibrary(data);
      renderJobs(data);
      document.getElementById('provider').textContent = `${data.project.provider} · ${data.project.api_key_configured ? '已连接' : '未配置'}`;
      document.getElementById('provider').className = `status-pill ${data.project.api_key_configured ? 'ok' : 'bad'}`;
      const health = data.health || {};
      const ready = health.evidence_ready !== undefined ? Boolean(health.evidence_ready) : (data.readiness.high_weight_papers >= 3 && data.readiness.genome_cards >= 2 && data.readiness.pattern_cards >= 1);
      const failedEvidence = health.latest_failed_evidence_job || null;
      const statusText = evidenceStatusText(health.evidence_status, ready);
      const missing = Array.isArray(health.missing_requirements) ? health.missing_requirements.map(item => `${item.key}:${item.current}/${item.required}`).join(' · ') : '';
      document.getElementById('strictStatus').textContent = statusText;
      document.getElementById('strictStatus').title = failedEvidence ? `最近失败：${displayJobName(failedEvidence)} · ${evidenceFailureHint(failedEvidence)}` : (missing ? `证据缺口：${missing}` : '查看核心优秀论文库');
      document.getElementById('strictStatus').onclick = () => metricAction(failedEvidence ? 'settings' : 'excellent');
      document.getElementById('strictStatus').className = `status-pill ${ready ? 'ok' : 'bad'}`;
      const healthDot = document.getElementById('healthDot');
      if (healthDot) healthDot.className = `health-dot ${health.evidence_status === 'building' ? 'building' : (ready && data.project.api_key_configured ? '' : 'bad')}`;
      const metricCards = [
        ['核心论文', data.readiness.high_weight_papers, 'excellent', '查看核心优秀论文库', '核心库', '只含 Best / Nominee / Oral'],
        ['逻辑卡', data.readiness.genome_cards, 'patterns', '查看严格论文逻辑卡', '逻辑库', '严格可用故事线'],
        ['模式卡', data.readiness.pattern_cards, 'patterns', '查看严格模式卡', '模式库', '严格可迁移模式'],
        ['对话记录', data.counts.idea_sessions || 0, 'chat', '回到对话工作区', '对话区', '回到当前研究对话'],
      ];
      renderMetrics(metricCards);
      if (activeView === 'library') showLibrary();
      renderLiveProgress(currentScope());
      scheduleRefresh(hasRunningJob ? 700 : 2500);
    }
    document.getElementById('instruction').addEventListener('input', () => {
      saveInstructionDraft(currentScope());
      autosizeInstruction();
    });
    document.getElementById('instruction').addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChat();
      }
    });
    autosizeInstruction();
    refresh();
  </script>
</body>
</html>"""
