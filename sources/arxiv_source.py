import argparse
import json

from sources.base import BaseSource
from core.config import LLMConfig, CommonConfig
from fetchers.arxiv_fetcher import fetch_papers_for_categories
from email_utils.base_template import get_stars
from email_utils.arxiv_template import get_paper_block_html


class ArxivSource(BaseSource):
    name = "arxiv"
    default_title = "Daily arXiv"

    def __init__(self, source_args: dict, llm_config: LLMConfig, common_config: CommonConfig):
        super().__init__(source_args, llm_config, common_config)
        self.categories = source_args.get("categories", ["cs.AI"])
        self.max_entries = source_args.get("max_entries", 100)
        self.max_papers = source_args.get("max_papers", 60)

        cache_key = f"papers_{'_'.join(sorted(self.categories))}_{self.max_entries}"
        cached = self._load_fetch_cache(cache_key)
        if cached is not None:
            self.papers_by_category = cached
        else:
            self.papers_by_category = fetch_papers_for_categories(
                self.categories,
                max_entries=self.max_entries,
            )
            if self.papers_by_category:
                self._save_fetch_cache(cache_key, self.papers_by_category)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--arxiv_categories", nargs="+", default=["cs.AI"],
            help="[arXiv] Categories to fetch (e.g. cs.AI cs.CL cs.CV)",
        )
        parser.add_argument(
            "--arxiv_max_entries", type=int, default=100,
            help="[arXiv] Max entries to fetch per category from arXiv listing",
        )
        parser.add_argument(
            "--arxiv_max_papers", type=int, default=60,
            help="[arXiv] Max papers to recommend after scoring",
        )

    @staticmethod
    def extract_args(args) -> dict:
        return {
            "categories": args.arxiv_categories,
            "max_entries": args.arxiv_max_entries,
            "max_papers": args.arxiv_max_papers,
        }

    def get_max_items(self) -> int:
        return self.max_papers

    def fetch_items(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for cat, papers in self.papers_by_category.items():
            for paper in papers:
                aid = paper.get("arxiv_id", "")
                if aid and aid not in seen:
                    paper.setdefault("category", cat)
                    paper.setdefault("venue_or_status", "arXiv preprint")
                    if not paper.get("year") and len(aid) >= 5 and aid[:4].isdigit() and aid[4] == ".":
                        yy = int(aid[:2])
                        paper["year"] = str(1900 + yy if yy >= 91 else 2000 + yy)
                    seen[aid] = paper
        print(f"[{self.name}] {len(seen)} unique papers after dedup across categories")
        return list(seen.values())

    def get_item_cache_id(self, item: dict) -> str:
        return "paper_" + item.get("arxiv_id", "unknown").replace("/", "_").replace(".", "_")

    def build_eval_prompt(self, item: dict) -> str:
        return f"""
You are screening one arXiv paper for a daily research digest. Be selective.

Reader profile:
{self.description}

Candidate paper:
Title: {item["title"]}
Abstract: {item["abstract"]}
arXiv ID: {item.get("arxiv_id", "")}
Category: {item.get("category", "")}
Year: {item.get("year", "")}
Status: {item.get("venue_or_status", "arXiv preprint")}

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
- If it is only a domain-specific application without reusable methodological value, cap relevance at 5.
- If it mainly applies an existing LLM/model/tool without technical innovation, cap relevance at 4.
- If it is about agents but not memory, retrieval, personalization, or knowledge organization, cap relevance at 6.
- If it is about RAG but not retrieval efficiency, indexing, vector search, memory, or data organization, cap relevance at 6.
- If the abstract gives weak or unclear empirical evidence, cap relevance at 7.

Write a concise summary and a structured paper_card in Simplified Chinese.
Important reliability rule: only use the title and abstract. If venue, ablation, code/data, efficiency, or training objective is not stated, write "未说明" or "不适用". Do not guess.

Return strict JSON only. No markdown fence. No extra text.
{{
  "summary": "问题：... 方法：... 实验/证据：... 相关性：... 局限：...",
  "relevance": 0.0,
  "paper_card": {{
    "title": "Paper title copied from candidate",
    "source": "arxiv",
    "year": "year copied from metadata or 未说明",
    "venue_or_status": "arXiv preprint",
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
        url = item.get("abstract_url", "") or item.get("pdf_url", "")
        return {
            "title": str(raw.get("title") or item.get("title", "")).strip(),
            "source": "arxiv",
            "year": str(raw.get("year") or item.get("year", "")).strip(),
            "venue_or_status": str(raw.get("venue_or_status") or item.get("venue_or_status", "arXiv preprint")).strip(),
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
            "url": str(raw.get("url") or url).strip(),
        }

    def parse_eval_response(self, item: dict, response: str) -> dict:
        response = str(response or "").strip("```").strip("json")
        data = json.loads(response)
        score = float(data.get("relevance", 0))
        return {
            "title": item["title"],
            "arxiv_id": item.get("arxiv_id", ""),
            "abstract": item.get("abstract", ""),
            "category": item.get("category", ""),
            "year": item.get("year", ""),
            "venue_or_status": item.get("venue_or_status", "arXiv preprint"),
            "summary": self._ensure_str(data.get("summary", "")),
            "score": score,
            "paper_card": self._normalize_paper_card(item, data, score),
            "pdf_url": item.get("pdf_url", ""),
            "url": item.get("abstract_url", "") or item.get("pdf_url", ""),
        }

    def render_item_html(self, item: dict) -> str:
        rate = get_stars(item.get("score", 0))
        return get_paper_block_html(
            item["title"],
            rate,
            item.get("arxiv_id", ""),
            item["summary"],
            item.get("pdf_url", ""),
        )

    def get_theme_color(self) -> str:
        return "179,27,27"

    def get_section_header(self) -> str:
        cats = ", ".join(self.categories)
        return f'<div class="section-title" style="border-bottom-color: #b31b1b;">📄 arXiv Papers ({cats})</div>'

    def build_summary_overview(self, recommendations: list[dict]) -> str:
        lines = []
        for i, r in enumerate(recommendations):
            lines.append(
                f"{i + 1}. {r['title']} (arXiv: {r.get('arxiv_id', '')}) "
                f"- Score: {r.get('score', 0)} - {r['summary']}"
            )
        return "\n".join(lines)

    def get_summary_prompt_template(self) -> str:
        return """
            请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
            <div class="summary-wrapper">
              <div class="summary-section">
                <h2>今日arXiv研究趋势</h2>
                <p>分析今天论文体现的研究趋势，解释其与我研究兴趣的联系...</p>
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
              <div class="summary-section">
                <h2>补充观察</h2>
                <p>值得持续关注的方向或潜在研究机会...</p>
              </div>
            </div>

            用中文撰写内容，重点推荐部分建议返回 3-5 篇论文。
        """
