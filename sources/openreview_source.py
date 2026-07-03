import argparse
import hashlib
import json

from core.config import LLMConfig, CommonConfig
from email_utils.arxiv_template import get_paper_block_html
from email_utils.base_template import get_stars
from fetchers.openreview_fetcher import fetch_openreview_submissions
from sources.base import BaseSource


class OpenReviewSource(BaseSource):
    name = "openreview"
    default_title = "Daily OpenReview"

    def __init__(self, source_args: dict, llm_config: LLMConfig, common_config: CommonConfig):
        super().__init__(source_args, llm_config, common_config)
        self.venues = source_args.get("venues", [])
        self.invitations = source_args.get("invitations", [])
        self.queries = source_args.get("queries", [])
        self.max_results = source_args.get("max_results", 50)
        self.max_papers = source_args.get("max_papers", 20)
        self.days = source_args.get("days", 0)
        self.api_versions = source_args.get("api_versions", ["2", "1"])

        config_text = "|".join(self.venues + self.invitations + self.queries + self.api_versions)
        config_hash = hashlib.sha1(config_text.encode("utf-8")).hexdigest()[:12]
        cache_key = "_".join(["openreview", str(self.max_results), str(self.days), config_hash])
        cached = self._load_fetch_cache(cache_key)
        if cached is not None:
            self.items = cached
        else:
            self.items = fetch_openreview_submissions(
                venues=self.venues,
                invitations=self.invitations,
                queries=self.queries,
                max_results=self.max_results,
                days=self.days,
                api_versions=self.api_versions,
            )
            if self.items:
                self._save_fetch_cache(cache_key, self.items)
        print(f"[{self.name}] {len(self.items)} submissions fetched")

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--openreview_venues", nargs="+", default=[],
            help="[OpenReview] Venue IDs, e.g. ICLR.cc/2026/Conference",
        )
        parser.add_argument(
            "--openreview_invitations", nargs="+", default=[],
            help="[OpenReview] Explicit invitation IDs, e.g. ICLR.cc/2026/Conference/-/Submission",
        )
        parser.add_argument(
            "--openreview_queries", nargs="+", default=[],
            help="[OpenReview] Local keyword filters after fetching",
        )
        parser.add_argument(
            "--openreview_max_results", type=int, default=50,
            help="[OpenReview] Max public submissions to fetch before scoring",
        )
        parser.add_argument(
            "--openreview_max_papers", type=int, default=20,
            help="[OpenReview] Max papers to recommend after scoring",
        )
        parser.add_argument(
            "--openreview_days", type=int, default=0,
            help="[OpenReview] Only keep notes created within N days; 0 disables date filtering",
        )
        parser.add_argument(
            "--openreview_api_versions", nargs="+", default=["2", "1"],
            help="[OpenReview] API versions to try: 2 1",
        )

    @staticmethod
    def extract_args(args) -> dict:
        return {
            "venues": args.openreview_venues,
            "invitations": args.openreview_invitations,
            "queries": args.openreview_queries,
            "max_results": args.openreview_max_results,
            "max_papers": args.openreview_max_papers,
            "days": args.openreview_days,
            "api_versions": args.openreview_api_versions,
        }

    def get_max_items(self) -> int:
        return self.max_papers

    def fetch_items(self) -> list[dict]:
        return self.items

    def get_item_cache_id(self, item: dict) -> str:
        identifier = item.get("forum") or item.get("id") or item.get("title", "unknown")
        return "openreview_" + str(identifier).replace("/", "_").replace(".", "_")

    def build_eval_prompt(self, item: dict) -> str:
        authors = ", ".join(item.get("authors", [])[:8])
        keywords = ", ".join(item.get("keywords", [])[:10])
        return f"""
You are screening one OpenReview submission for a daily research digest. Be selective.

Reader profile:
{self.description}

Candidate paper:
Title: {item.get("title", "")}
Abstract: {item.get("abstract", "")}
Authors: {authors or "未说明"}
Venue/status: {item.get("venue_or_status", "OpenReview submission")}
Year: {item.get("year", "")}
Keywords: {keywords or "未说明"}
OpenReview URL: {item.get("url", "")}

Evaluate the paper against the reader profile. The reader mainly cares about:
1. spatiotemporal data mining/modeling, forecasting, mobility analytics, traffic systems, urban computing, human dynamics, and spatiotemporal representation learning;
2. data condensation, dataset distillation, synthetic data generation, gradient/distribution/trajectory matching, matching training trajectories, and diffusion-based data synthesis;
3. efficient vector search, vector databases, approximate nearest neighbor search, indexing, retrieval optimization, and RAG retrieval systems;
4. agent memory, long-term retrieval, personalized memory, knowledge organization, and memory-augmented LLM agents.

Use a strict relevance rubric:
- 9-10: Directly matches one core area, has a clear methodological contribution, and includes strong experiments, benchmarks, code, or practical system implications.
- 7-8: Directly relevant to one core area, but the contribution or validation is less complete.
- 5-6: Adjacent or potentially useful, but not central to the reader's work.
- 0-4: Out of scope, weakly supported, purely application-driven, or only applies existing models without meaningful technical novelty.

Apply caps:
- If it is only a domain-specific application without reusable methodological value, cap relevance at 5.
- If it mainly applies an existing LLM/model/tool without technical innovation, cap relevance at 4.
- If it is about agents but not memory, retrieval, personalization, or knowledge organization, cap relevance at 6.
- If it is about RAG but not retrieval efficiency, indexing, vector search, memory, or data organization, cap relevance at 6.
- If the abstract gives weak or unclear empirical evidence, cap relevance at 7.

Write a concise summary and a structured paper_card in Simplified Chinese.
Important reliability rule: only use the title, abstract, and provided OpenReview metadata. If venue, ablation, code/data, efficiency, or training objective is not stated, write "未说明" or "不适用". Do not guess.

Return strict JSON only. No markdown fence. No extra text.
{{
  "summary": "问题：... 方法：... 实验/证据：... 相关性：... 局限：...",
  "relevance": 0.0,
  "paper_card": {{
    "title": "Paper title copied from candidate",
    "source": "openreview",
    "year": "year copied from metadata or 未说明",
    "venue_or_status": "OpenReview venue/status copied from metadata",
    "area": "Spatiotemporal / Data Condensation / Vector Search / Agent Memory / Adjacent / Out of Scope",
    "score": 0.0,
    "one_sentence": "这篇论文解决……问题，提出……方法，在……上证明……",
    "problem": "现有方法的主要不足是……",
    "method": {{
      "input": "...",
      "core_modules": "...",
      "training_objective": "训练目标/优化目标；如果不适用或摘要未说明则写不适用/未说明",
      "output": "..."
    }},
    "innovations": ["...", "..."],
    "experiments": {{
      "main_results": "...",
      "ablations": "摘要未说明",
      "efficiency_or_cost": "摘要未说明"
    }},
    "limitations": "...",
    "value_to_me": {{
      "uses": ["相关工作", "baseline", "方法借鉴", "项目改进", "benchmark设计"],
      "reason": "..."
    }},
    "evidence_strength": "强 / 中 / 弱 / 摘要未说明",
    "reusable_assets": "代码 / 数据集 / benchmark / 未说明",
    "reading_priority": "必读 / 可读 / 跟踪 / 跳过",
    "url": "candidate URL"
  }}
}}
"""

    @staticmethod
    def _listify(value) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    def _normalize_paper_card(self, item: dict, data: dict, score: float) -> dict:
        raw = data.get("paper_card") if isinstance(data.get("paper_card"), dict) else {}
        method = raw.get("method") if isinstance(raw.get("method"), dict) else {}
        experiments = raw.get("experiments") if isinstance(raw.get("experiments"), dict) else {}
        value = raw.get("value_to_me") if isinstance(raw.get("value_to_me"), dict) else {}
        return {
            "title": str(raw.get("title") or item.get("title", "")).strip(),
            "source": "openreview",
            "year": str(raw.get("year") or item.get("year", "")).strip(),
            "venue_or_status": str(raw.get("venue_or_status") or item.get("venue_or_status", "OpenReview submission")).strip(),
            "area": str(raw.get("area", "")).strip() or "Adjacent",
            "score": round(score, 2),
            "one_sentence": str(raw.get("one_sentence", "")).strip(),
            "problem": str(raw.get("problem", "")).strip(),
            "method": {
                "input": str(method.get("input", "")).strip(),
                "core_modules": str(method.get("core_modules", "")).strip(),
                "training_objective": str(method.get("training_objective", "")).strip(),
                "output": str(method.get("output", "")).strip(),
            },
            "innovations": self._listify(raw.get("innovations"))[:4],
            "experiments": {
                "main_results": str(experiments.get("main_results", "")).strip(),
                "ablations": str(experiments.get("ablations", "")).strip(),
                "efficiency_or_cost": str(experiments.get("efficiency_or_cost", "")).strip(),
            },
            "limitations": str(raw.get("limitations", "")).strip(),
            "value_to_me": {
                "uses": self._listify(value.get("uses"))[:5],
                "reason": str(value.get("reason", "")).strip(),
            },
            "evidence_strength": str(raw.get("evidence_strength", "")).strip(),
            "reusable_assets": str(raw.get("reusable_assets", "")).strip(),
            "reading_priority": str(raw.get("reading_priority", "")).strip(),
            "url": str(raw.get("url") or item.get("url", "")).strip(),
        }

    def parse_eval_response(self, item: dict, response: str) -> dict:
        response = str(response or "").strip("```").strip("json")
        data = json.loads(response)
        score = float(data.get("relevance", 0))
        return {
            "title": item.get("title", ""),
            "id": item.get("id", ""),
            "forum": item.get("forum", ""),
            "abstract": item.get("abstract", ""),
            "authors": item.get("authors", []),
            "keywords": item.get("keywords", []),
            "year": item.get("year", ""),
            "venue_or_status": item.get("venue_or_status", "OpenReview submission"),
            "summary": self._ensure_str(data.get("summary", "")),
            "score": score,
            "paper_card": self._normalize_paper_card(item, data, score),
            "url": item.get("url", ""),
        }

    def render_item_html(self, item: dict) -> str:
        rate = get_stars(item.get("score", 0))
        return get_paper_block_html(
            item.get("title", "Untitled"),
            rate,
            item.get("forum", item.get("id", "")),
            item.get("summary", ""),
            item.get("url", ""),
        )

    def get_theme_color(self) -> str:
        return "64,81,181"

    def get_section_header(self) -> str:
        return '<div class="section-title" style="border-bottom-color: #4051b5;">OpenReview Papers</div>'

    def build_summary_overview(self, recommendations: list[dict]) -> str:
        lines = []
        for i, r in enumerate(recommendations):
            lines.append(
                f"{i + 1}. {r.get('title', '')} ({r.get('venue_or_status', '')}) "
                f"- Score: {r.get('score', 0)} - {r.get('summary', '')}"
            )
        return "\n".join(lines)

    def get_summary_prompt_template(self) -> str:
        return """
            请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
            <div class="summary-wrapper">
              <div class="summary-section">
                <h2>今日 OpenReview 论文观察</h2>
                <p>概括这些 OpenReview submission 与我的研究兴趣的关系。</p>
              </div>
              <div class="summary-section">
                <h2>重点推荐</h2>
                <ol class="summary-list">
                  <li class="summary-item">
                    <div class="summary-item__header"><span class="summary-item__title">论文标题</span><span class="summary-pill">相关性</span></div>
                    <p><strong>推荐理由：</strong>...</p>
                    <p><strong>关键贡献：</strong>...</p>
                  </li>
                </ol>
              </div>
            </div>
            用中文撰写内容，重点推荐部分建议返回 3-5 篇论文。
        """
