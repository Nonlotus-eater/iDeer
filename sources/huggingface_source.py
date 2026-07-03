import argparse
import json
import time

from sources.base import BaseSource
from core.config import LLMConfig, CommonConfig
from fetchers.huggingface_fetcher import get_daily_papers, get_trending_models_api
from email_utils.base_template import get_stars, framework, get_empty_html
from email_utils.huggingface_template import get_paper_block_html, get_model_block_html
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class HuggingFaceSource(BaseSource):
    name = "huggingface"
    default_title = "Daily HuggingFace"

    def __init__(self, source_args: dict, llm_config: LLMConfig, common_config: CommonConfig):
        super().__init__(source_args, llm_config, common_config)
        self.content_types = [ct.lower() for ct in source_args.get("content_type", ["papers", "models"])]
        self.max_papers = source_args.get("max_papers", 30)
        self.max_models = source_args.get("max_models", 15)

        self.papers = []
        self.models = []
        if "papers" in self.content_types:
            cached = self._load_fetch_cache("daily_papers")
            if cached is not None:
                self.papers = cached
            else:
                self.papers = get_daily_papers(self.max_papers * 2)
                if self.papers:
                    self._save_fetch_cache("daily_papers", self.papers)
            print(f"[{self.name}] {len(self.papers)} daily papers")
        if "models" in self.content_types:
            cached = self._load_fetch_cache("trending_models")
            if cached is not None:
                self.models = cached
            else:
                self.models = get_trending_models_api(self.max_models * 2)
                if self.models:
                    self._save_fetch_cache("trending_models", self.models)
            print(f"[{self.name}] {len(self.models)} trending models")

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--hf_content_type", nargs="+", choices=["papers", "models"],
            default=["papers", "models"],
            help="[HuggingFace] Content types to fetch",
        )
        parser.add_argument(
            "--hf_max_papers", type=int, default=30,
            help="[HuggingFace] Max papers to recommend",
        )
        parser.add_argument(
            "--hf_max_models", type=int, default=15,
            help="[HuggingFace] Max models to recommend",
        )

    @staticmethod
    def extract_args(args) -> dict:
        return {
            "content_type": args.hf_content_type,
            "max_papers": args.hf_max_papers,
            "max_models": args.hf_max_models,
        }

    def fetch_items(self) -> list[dict]:
        items = []
        for p in self.papers:
            p["_hf_type"] = "paper"
            arxiv_id = str(p.get("arxiv_id") or p.get("id") or "")
            p.setdefault("venue_or_status", "HuggingFace Daily / arXiv preprint")
            if not p.get("year") and len(arxiv_id) >= 5 and arxiv_id[:4].isdigit() and arxiv_id[4] == ".":
                yy = int(arxiv_id[:2])
                p["year"] = str(1900 + yy if yy >= 91 else 2000 + yy)
            items.append(p)
        for m in self.models:
            m["_hf_type"] = "model"
            items.append(m)
        return items

    def get_item_cache_id(self, item: dict) -> str:
        if item.get("_hf_type") == "paper":
            return "paper_" + item.get("id", "unknown")
        else:
            return "model_" + item.get("model_id", "unknown").replace("/", "_")

    def build_eval_prompt(self, item: dict) -> str:
        if item.get("_hf_type") == "paper":
            return self._build_paper_prompt(item)
        else:
            return self._build_model_prompt(item)

    def _build_paper_prompt(self, item: dict) -> str:
        return f"""
You are screening one HuggingFace Daily Paper for a daily research digest. Be selective; community popularity is only a weak signal.

Reader profile:
{self.description}

Candidate paper:
Title: {item["title"]}
Abstract: {item["abstract"]}
arXiv ID: {item.get("arxiv_id", item.get("id", ""))}
Year: {item.get("year", "")}
Status: {item.get("venue_or_status", "HuggingFace Daily / arXiv preprint")}
Community upvotes: {item.get("upvotes", 0)}

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

Apply these caps:
- If it is popular but only broadly about LLMs, agents, multimodal models, or benchmarks without matching the reader profile, cap relevance at 5.
- If it mainly applies an existing LLM/model/tool without technical innovation, cap relevance at 4.
- If it is about agents but not memory, retrieval, personalization, or knowledge organization, cap relevance at 6.
- If it is about RAG but not retrieval efficiency, indexing, vector search, memory, or data organization, cap relevance at 6.
- If the abstract gives weak or unclear empirical evidence, cap relevance at 7.

Write a concise summary and a structured paper_card in Simplified Chinese.
Important reliability rule: only use the title, abstract, and provided metadata. If venue, ablation, code/data, efficiency, or training objective is not stated, write "未说明" or "不适用". Do not guess.

Return strict JSON only. No markdown fence. No extra text.
{{
  "summary": "问题：... 方法：... 实验/证据：... 相关性：... 局限：...",
  "relevance": 0.0,
  "paper_card": {{
    "title": "Paper title copied from candidate",
    "source": "huggingface",
    "year": "year copied from metadata or 未说明",
    "venue_or_status": "HuggingFace Daily / arXiv preprint",
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
            "source": "huggingface",
            "year": str(raw.get("year") or item.get("year", "")).strip(),
            "venue_or_status": str(raw.get("venue_or_status") or item.get("venue_or_status", "HuggingFace Daily / arXiv preprint")).strip(),
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
            "url": str(raw.get("url") or item.get("paper_url", "")).strip(),
        }

    def _build_model_prompt(self, item: dict) -> str:
        tags = item.get("tags", [])
        return f"""
You are screening one HuggingFace model for a research-focused daily digest. Be selective; general popularity is not enough.

Reader profile:
{self.description}

Candidate model:
Model ID: {item["model_id"]}
Description: {item.get("description", "") or "No description"}
Downloads: {item.get("downloads", 0)}
Likes: {item.get("likes", 0)}
Tags: {", ".join(tags) if tags else "No tags"}

Evaluate usefulness for the reader's research and system-building interests:
- spatiotemporal modeling, forecasting, mobility/traffic/urban data, or spatiotemporal representation learning;
- data condensation, dataset distillation, synthetic data generation, or diffusion-based data synthesis;
- efficient vector search, indexing, retrieval optimization, vector databases, or RAG systems;
- agent memory, long-term retrieval, personalized memory, knowledge organization, or memory-augmented agents.

Use a strict usefulness rubric:
- 9-10: Directly useful for one core area, with clear practical or methodological value.
- 7-8: Relevant and likely worth checking, but the connection is narrower or less validated.
- 5-6: Adjacent utility only.
- 0-4: Generic model release, broad LLM/multimodal model, demo, or weakly related artifact.

Apply caps:
- If it is a generic LLM, vision-language model, chatbot, or benchmark model without direct relevance to the core areas, cap usefulness at 5.
- If the description is sparse and no clear research/system value is visible, cap usefulness at 6.
- If it is only an application demo, cap usefulness at 5.

Write the summary in Simplified Chinese as one plain-text paragraph. Include:
功能: what the model does; 适用场景: where it could be useful; 相关性: why it matters to the reader; 谨慎点: why it may be less important.

Return strict JSON only. No markdown fence. No extra text.
{{
  "summary": "功能：... 适用场景：... 相关性：... 谨慎点：...",
  "usefulness": 0.0
}}
"""

    def parse_eval_response(self, item: dict, response: str) -> dict:
        response = str(response or "").strip("```").strip("json")
        data = json.loads(response)

        if item.get("_hf_type") == "paper":
            score = float(data.get("relevance", 0))
            return {
                "_hf_type": "paper",
                "title": item["title"],
                "id": item.get("id", ""),
                "arxiv_id": item.get("arxiv_id", item.get("id", "")),
                "abstract": item.get("abstract", ""),
                "year": item.get("year", ""),
                "venue_or_status": item.get("venue_or_status", "HuggingFace Daily / arXiv preprint"),
                "summary": self._ensure_str(data.get("summary", "")),
                "score": score,
                "paper_card": self._normalize_paper_card(item, data, score),
                "upvotes": item.get("upvotes", 0),
                "url": item["paper_url"],
            }
        else:
            return {
                "_hf_type": "model",
                "title": item["model_id"],
                "id": item.get("model_id", ""),
                "description": item.get("description", ""),
                "summary": self._ensure_str(data["summary"]),
                "score": float(data["usefulness"]),
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "tags": item.get("tags", []),
                "url": item["model_url"],
            }

    def render_item_html(self, item: dict) -> str:
        rate = get_stars(item.get("score", 0))
        if item.get("_hf_type") == "paper":
            return get_paper_block_html(
                item["title"], rate, item["id"], item["summary"],
                item["url"], item.get("upvotes", 0),
            )
        else:
            return get_model_block_html(
                item["title"], rate, item["id"], item["summary"],
                item["url"], item.get("likes", 0), item.get("downloads", 0),
            )

    def get_theme_color(self) -> str:
        return "255,111,0"

    def get_section_header(self) -> str:
        return '<div class="section-title" style="border-bottom-color: #ff6f00;">🤗 HuggingFace Daily</div>'

    def get_max_items(self) -> int:
        return self.max_papers + self.max_models

    def get_recommendations(self) -> list[dict]:
        """Override: process papers and models separately with independent limits."""
        all_items = self.fetch_items()
        if not all_items:
            print(f"[{self.name}] No items fetched.")
            return []

        papers = [i for i in all_items if i.get("_hf_type") == "paper"]
        models = [i for i in all_items if i.get("_hf_type") == "model"]

        paper_recs = self._process_batch(papers, "papers") if papers else []
        model_recs = self._process_batch(models, "models") if models else []

        paper_recs = sorted(paper_recs, key=lambda x: x.get("score", 0), reverse=True)[:self.max_papers]
        model_recs = sorted(model_recs, key=lambda x: x.get("score", 0), reverse=True)[:self.max_models]

        combined = sorted(paper_recs + model_recs, key=lambda x: x.get("score", 0), reverse=True)[:self.MAX_RECOMMEND]

        if self.save_dir:
            self._save_markdown(combined)

        return combined

    def _process_batch(self, items: list[dict], label: str) -> list[dict]:
        results = []
        print(f"[{self.name}] Processing {len(items)} {label}...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = [executor.submit(self.process_item, item) for item in items]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"[{self.name}] {label}", unit="item"):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def render_email(self, recommendations: list[dict]) -> str:
        """Override: render papers and models in separate sections."""
        papers = [r for r in recommendations if r.get("_hf_type") == "paper"]
        models = [r for r in recommendations if r.get("_hf_type") == "model"]

        if not papers and not models:
            return framework.replace("__CONTENT__", get_empty_html())

        parts = []

        if papers:
            parts.append('<div class="section-title" style="border-bottom-color: #ff6f00;">📄 Daily Papers</div>')
            for i, p in enumerate(tqdm(papers, desc=f"[{self.name}] Rendering papers")):
                parts.append(self.render_item_html(p))

        if models:
            parts.append('<div class="section-title" style="border-bottom-color: #1976d2;">🤖 Trending Models</div>')
            for i, m in enumerate(tqdm(models, desc=f"[{self.name}] Rendering models")):
                parts.append(self.render_item_html(m))

        summary = self.summarize(recommendations)
        content = summary + "<br>" + "</br><br>".join(parts) + "</br>"
        email_html = framework.replace("__CONTENT__", content)

        # Save to history as snapshot (not used as cache)
        if self.save_dir:
            email_path = os.path.join(self.save_dir, f"{self.name}_email.html")
            os.makedirs(os.path.dirname(email_path), exist_ok=True)
            with open(email_path, "w", encoding="utf-8") as f:
                f.write(email_html)

        return email_html

    def build_summary_overview(self, recommendations: list[dict]) -> str:
        papers = [r for r in recommendations if r.get("_hf_type") == "paper"]
        models = [r for r in recommendations if r.get("_hf_type") == "model"]

        overview = ""
        if papers:
            overview += "=== Papers ===\n"
            for i, p in enumerate(papers):
                overview += f"{i + 1}. {p['title']} - {p['summary']}\n"
        if models:
            overview += "\n=== Models ===\n"
            for i, m in enumerate(models):
                overview += f"{i + 1}. {m['title']} - {m['summary']}\n"
        return overview

    def get_summary_prompt_template(self) -> str:
        return """
            请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
            <div class="summary-wrapper">
              <div class="summary-section">
                <h2>今日趋势</h2>
                <p>...</p>
              </div>
              <div class="summary-section">
                <h2>重点推荐</h2>
                <ol class="summary-list">
                  <li class="summary-item">
                    <div class="summary-item__header"><span class="summary-item__title">标题</span><span class="summary-pill">类型</span></div>
                    <p><strong>推荐理由：</strong>...</p>
                    <p><strong>关键亮点：</strong>...</p>
                  </li>
                </ol>
              </div>
              <div class="summary-section">
                <h2>补充观察</h2>
                <p>暂无或其他补充。</p>
              </div>
            </div>

            用中文撰写内容，重点推荐部分建议返回 3-5 项内容。
        """
