"""Generate a cross-source personalized narrative report from daily recommendations."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from sources.base import BaseSource
from core.config import CommonConfig, EmailConfig, LLMConfig, PROJECT_ROOT
from email_utils.report_template import render_report_email
from llm.GPT import GPT
from llm.Ollama import Ollama

REPORT_EMAIL_TITLE = "Daily Personal Briefing"


class ReportGenerator:
    def __init__(
        self,
        all_recs: dict[str, list[dict]],
        profile_text: str,
        llm_config: LLMConfig,
        common_config: CommonConfig,
        report_title: str = REPORT_EMAIL_TITLE,
        min_score: float = 4.0,
        max_items: int = 18,
        theme_count: int = 4,
        prediction_count: int = 4,
        idea_count: int = 4,
    ):
        self.all_recs = all_recs
        self.profile_text = profile_text.strip()
        self.llm_config = llm_config
        self.common_config = common_config
        self.report_title = report_title
        self.min_score = min_score
        self.max_items = max_items
        self.theme_count = theme_count
        self.prediction_count = prediction_count
        self.idea_count = idea_count

        self.run_datetime = datetime.now(timezone.utc)
        self.run_date = self.run_datetime.strftime("%Y-%m-%d")

        self.model = self._build_model(llm_config)

        base_dir = str(PROJECT_ROOT)
        self.save_dir = os.path.join(base_dir, common_config.save_dir, "reports", self.run_date)
        self.email_cache_path = os.path.join(self.save_dir, "report.html")

        if common_config.save:
            os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _build_model(llm_config: LLMConfig):
        provider = llm_config.provider.lower()
        if provider == "ollama":
            return Ollama(llm_config.model)
        if provider in ("openai", "siliconflow"):
            return GPT(llm_config.model, llm_config.base_url, llm_config.api_key)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = str(text or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _format_time(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        try:
            return datetime.fromisoformat(raw).astimezone().strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return raw

    @staticmethod
    def _listify(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    def _normalize_paper_card(self, raw: Any, item: dict | None = None) -> dict[str, Any]:
        item = item or {}
        card = raw if isinstance(raw, dict) else {}
        method = card.get("method") if isinstance(card.get("method"), dict) else {}
        experiments = card.get("experiments") if isinstance(card.get("experiments"), dict) else {}
        value = card.get("value_to_me") if isinstance(card.get("value_to_me"), dict) else {}
        score = self._safe_float(card.get("score", item.get("score", 0)))
        source = str(card.get("source") or item.get("source", "")).strip()
        url = str(card.get("url") or item.get("url", "")).strip()
        summary = str(item.get("summary", "")).strip()

        return {
            "title": str(card.get("title") or item.get("title", "Untitled")).strip(),
            "source": source,
            "year": str(card.get("year") or item.get("year", "")).strip(),
            "venue_or_status": str(
                card.get("venue_or_status")
                or item.get("venue_or_status", "")
                or ("arXiv preprint" if source == "arxiv" else "")
            ).strip(),
            "area": str(card.get("area", "")).strip() or str(item.get("category", "")).strip() or "Adjacent",
            "score": round(score, 2),
            "one_sentence": str(card.get("one_sentence", "")).strip() or summary,
            "problem": str(card.get("problem", "")).strip(),
            "method": {
                "input": str(method.get("input", "")).strip(),
                "core_modules": str(method.get("core_modules", "")).strip(),
                "training_objective": str(method.get("training_objective", "")).strip(),
                "output": str(method.get("output", "")).strip(),
            },
            "innovations": self._listify(card.get("innovations"))[:4],
            "experiments": {
                "main_results": str(experiments.get("main_results", "")).strip(),
                "ablations": str(experiments.get("ablations", "")).strip(),
                "efficiency_or_cost": str(experiments.get("efficiency_or_cost", "")).strip(),
            },
            "limitations": str(card.get("limitations", "")).strip(),
            "value_to_me": {
                "uses": self._listify(value.get("uses"))[:5],
                "reason": str(value.get("reason", "")).strip(),
            },
            "evidence_strength": str(card.get("evidence_strength", "")).strip(),
            "reusable_assets": str(card.get("reusable_assets", "")).strip(),
            "reading_priority": str(card.get("reading_priority", "")).strip(),
            "url": url,
        }

    def _normalize_item(self, source_name: str, rec: dict) -> dict:
        source_label = {
            "github": "GitHub",
            "huggingface": "HuggingFace",
            "twitter": "X/Twitter",
        }.get(source_name, source_name)
        score = self._safe_float(rec.get("score", 0))
        summary = str(rec.get("summary", "")).strip()
        url = str(rec.get("url", "")).strip()

        normalized = {
            "source": source_name,
            "source_label": source_label,
            "score": round(score, 2),
            "title": str(rec.get("title", "Untitled")).strip(),
            "summary": summary,
            "url": url,
            "category": str(rec.get("category", "")).strip(),
            "year": str(rec.get("year", "")).strip(),
            "venue_or_status": str(rec.get("venue_or_status", "")).strip(),
            "time": "",
            "metrics": "",
            "entity": "",
            "detail": "",
        }

        if source_name == "arxiv":
            normalized["entity"] = str(rec.get("arxiv_id", rec.get("title", ""))).strip()
            normalized["detail"] = self._truncate(rec.get("abstract", ""), 320)
            normalized["metrics"] = " / ".join(
                part
                for part in [
                    f"arxiv_id={rec.get('arxiv_id', '')}" if rec.get("arxiv_id") else "",
                    f"category={rec.get('category', '')}" if rec.get("category") else "",
                ]
                if part
            )
            normalized["venue_or_status"] = normalized["venue_or_status"] or "arXiv preprint"
        elif source_name == "github":
            normalized["entity"] = str(rec.get("repo_name", rec.get("title", ""))).strip()
            normalized["detail"] = " / ".join(
                part
                for part in [
                    str(rec.get("language", "")).strip(),
                    self._truncate(rec.get("description", ""), 220),
                    "；".join(str(x).strip() for x in rec.get("highlights", [])[:3] if str(x).strip()),
                ]
                if part
            )
            normalized["metrics"] = (
                f"stars={int(rec.get('stars', 0) or 0)}, "
                f"stars_today={int(rec.get('stars_today', 0) or 0)}, "
                f"forks={int(rec.get('forks', 0) or 0)}"
            )
        elif source_name == "huggingface":
            hf_type = str(rec.get("_hf_type", "")).strip() or "item"
            normalized["category"] = hf_type
            normalized["entity"] = str(rec.get("id", rec.get("title", ""))).strip()
            if hf_type == "paper":
                normalized["detail"] = self._truncate(rec.get("abstract", ""), 260)
                normalized["metrics"] = f"upvotes={int(rec.get('upvotes', 0) or 0)}"
            else:
                tags = ", ".join(str(x).strip() for x in rec.get("tags", [])[:6] if str(x).strip())
                normalized["detail"] = " / ".join(
                    part for part in [self._truncate(rec.get("description", ""), 200), tags] if part
                )
                normalized["metrics"] = (
                    f"likes={int(rec.get('likes', 0) or 0)}, "
                    f"downloads={int(rec.get('downloads', 0) or 0)}"
                )
        elif source_name == "twitter":
            author = str(rec.get("author_name", rec.get("author_username", ""))).strip()
            handle = str(rec.get("author_username", "")).strip()
            normalized["entity"] = f"{author} (@{handle})" if author and handle else author or handle
            normalized["time"] = self._format_time(rec.get("created_at", ""))
            normalized["detail"] = " / ".join(
                part
                for part in [
                    self._truncate(rec.get("text", ""), 280),
                    "；".join(str(x).strip() for x in rec.get("key_points", [])[:3] if str(x).strip()),
                ]
                if part
            )
            normalized["metrics"] = (
                f"likes={int(rec.get('likes', 0) or 0)}, "
                f"retweets={int(rec.get('retweets', 0) or 0)}, "
                f"replies={int(rec.get('replies', 0) or 0)}"
            )

        is_paper_item = (
            isinstance(rec.get("paper_card"), dict)
            or source_name in ("arxiv", "semanticscholar", "pubmed")
            or (source_name == "huggingface" and normalized.get("category") == "paper")
        )
        if is_paper_item:
            normalized["paper_card"] = self._normalize_paper_card(rec.get("paper_card"), normalized)

        return normalized

    def _filter_items(self) -> list[dict]:
        normalized: list[dict] = []
        for source_name, recs in self.all_recs.items():
            source_items = [self._normalize_item(source_name, rec) for rec in recs]
            source_items.sort(key=lambda item: item.get("score", 0), reverse=True)
            qualified = [item for item in source_items if item.get("score", 0) >= self.min_score]
            if not qualified:
                qualified = source_items[: min(3, len(source_items))]
            normalized.extend(qualified)

        if not normalized:
            return []

        by_source: dict[str, list[dict]] = {}
        for item in normalized:
            by_source.setdefault(item["source"], []).append(item)

        selected: list[dict] = []
        source_names = sorted(
            by_source.keys(),
            key=lambda name: by_source[name][0].get("score", 0),
            reverse=True,
        )
        index_by_source = {name: 0 for name in source_names}

        while len(selected) < self.max_items:
            added_any = False
            for source_name in source_names:
                source_items = by_source[source_name]
                source_index = index_by_source[source_name]
                if source_index >= len(source_items):
                    continue
                selected.append(source_items[source_index])
                index_by_source[source_name] += 1
                added_any = True
                if len(selected) >= self.max_items:
                    break
            if not added_any:
                break

        return selected

    def _format_item_for_prompt(self, item: dict, index: int) -> str:
        lines = [
            f"[{index}] source={item.get('source')} / score={item.get('score', 0)}",
            f"title={item.get('title', '')}",
        ]
        if item.get("entity"):
            lines.append(f"entity={item.get('entity')}")
        if item.get("category"):
            lines.append(f"category={item.get('category')}")
        if item.get("year"):
            lines.append(f"year={item.get('year')}")
        if item.get("venue_or_status"):
            lines.append(f"venue_or_status={item.get('venue_or_status')}")
        if item.get("time"):
            lines.append(f"time={item.get('time')}")
        if item.get("metrics"):
            lines.append(f"metrics={item.get('metrics')}")
        if item.get("summary"):
            lines.append(f"summary={item.get('summary')}")
        if item.get("detail"):
            lines.append(f"detail={item.get('detail')}")
        if item.get("paper_card"):
            lines.append(
                "paper_card="
                + json.dumps(item.get("paper_card"), ensure_ascii=False, separators=(",", ":"))
            )
        if item.get("url"):
            lines.append(f"url={item.get('url')}")
        return "\n".join(lines)

    def _build_prompt(self, filtered_items: list[dict]) -> str:
        items_text = "\n\n".join(
            self._format_item_for_prompt(item, index)
            for index, item in enumerate(filtered_items, 1)
        )
        profile_excerpt = self._truncate(self.profile_text, 6000)

        return f"""You are writing a paper-card daily research digest for a single reader.

The output language must be Simplified Chinese, but keep the reasoning prompt in English and do not return any explanatory preamble.

Reader profile:
{profile_excerpt}

Curated source material for today:
{items_text}

Your job is to select the most relevant papers/items and return structured paper cards. This is not a broad analyst essay.

Reader's core interests:
1. spatiotemporal data mining/modeling, forecasting, mobility analytics, traffic systems, urban computing, human dynamics, and spatiotemporal representation learning;
2. data condensation, dataset distillation, synthetic data generation, gradient/distribution/trajectory matching, matching training trajectories, and diffusion-based data synthesis;
3. efficient vector search, vector databases, approximate nearest neighbor search, indexing structures, retrieval optimization, and RAG retrieval systems;
4. agent memory, long-term retrieval, personalized memory, knowledge organization, and memory-augmented LLM agents.

Card-writing priorities:
1. Prefer papers with an existing paper_card in the input. Preserve their title, source, URL, score, and factual fields unless a field is empty.
2. For each top paper, fill the card fields using only the provided material. If a field is not stated, write "未说明" or "不适用".
3. Do not invent conference names, code availability, datasets, ablations, efficiency numbers, or benchmarks.
4. Rank by direct relevance to the reader, methodological value, experimental support, reusable assets, and practical implications.
5. Use watchlist for relevant but lower-priority items, including models or adjacent papers.
6. Keep the opening and area_summary short. The email should mainly be a list of paper cards.

Output strict JSON only. No markdown fence. No extra text.

Schema:
{{
  "report_title": "A concise Chinese title for today's paper digest",
  "subtitle": "One-sentence Chinese subtitle focused on the top papers",
  "opening": "One short Chinese paragraph naming the strongest 2-3 papers and why they matter",
  "top_papers": [
    {{
      "title": "Paper title",
      "source": "arxiv/huggingface/semanticscholar/pubmed/rss",
      "year": "2026 or 未说明",
      "venue_or_status": "arXiv preprint / HuggingFace Daily / conference / 未说明",
      "area": "Spatiotemporal / Data Condensation / Vector Search / Agent Memory / Adjacent",
      "score": 0.0,
      "one_sentence": "这篇论文解决……问题，提出……方法，在……上证明……",
      "problem": "现有方法的主要不足是……",
      "method": {{
        "input": "...",
        "core_modules": "...",
        "training_objective": "训练目标/优化目标；若不适用或未说明则写不适用/未说明",
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
      "url": "https://..."
    }}
  ],
  "area_summary": [
    {{
      "area": "Vector Search / Agent Memory / Spatiotemporal / Data Condensation",
      "summary": "One concise Chinese sentence about today's papers in this area"
    }}
  ],
  "watchlist": [
    {{
      "title": "Item title",
      "source": "source",
      "reason": "Why to monitor it or why it is lower priority",
      "url": "https://..."
    }}
  ]
}}

Requirements:
- Return 5-10 top_papers if enough relevant papers exist; fewer is fine when evidence is weak.
- Every top_papers entry must be one concrete paper/item from the provided source material.
- Prefer top_papers from arxiv, huggingface papers, semanticscholar, pubmed, or RSS paper sources. Put models/repos/social items in watchlist unless they are essential.
- Do not output predictions or research ideas.
- Use concise Chinese. Each card field should be informative but not essay-length.
"""

    @staticmethod
    def _clean_llm_json(raw: str) -> str:
        cleaned = str(raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        return cleaned.strip()

    @staticmethod
    def _safe_slug(text: str, limit: int = 32) -> str:
        compact = re.sub(r"\s+", " ", str(text or "").strip())
        return compact[:limit].rstrip(" /-_")

    def _parse_json_object(self, raw: str) -> dict[str, Any] | None:
        cleaned = self._clean_llm_json(raw)
        if not cleaned:
            return None

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return data if isinstance(data, dict) else None

    def _build_repair_prompt(self, invalid_json: str) -> str:
        return f"""You are fixing invalid JSON produced by another model.

Return valid JSON only. No markdown fence. No commentary.
Keep the original meaning, preserve Chinese text, and keep the same schema.
If a field is missing, keep it as an empty string, empty array, or empty object as appropriate.
Do not add any fields outside the schema.

Invalid JSON:
{invalid_json}
"""

    def _repair_report_json(self, invalid_json: str) -> dict[str, Any] | None:
        print("[ReportGenerator] Attempting JSON repair pass.")
        repair_prompt = self._build_repair_prompt(invalid_json)
        repaired_raw = self.model.inference(repair_prompt, temperature=0.0)
        repaired = self._parse_json_object(repaired_raw)
        if repaired is None:
            print("[ReportGenerator] JSON repair pass failed.")
        return repaired

    @staticmethod
    def _normalize_signal(signal: dict[str, Any]) -> dict[str, str]:
        return {
            "source": str(signal.get("source", "")).strip(),
            "title": str(signal.get("title", "")).strip(),
            "why_it_matters": str(signal.get("why_it_matters", "")).strip(),
            "url": str(signal.get("url", "")).strip(),
        }

    def _fallback_signals(self, filtered_items: list[dict], limit: int = 3) -> list[dict[str, str]]:
        signals = []
        for item in filtered_items[:limit]:
            signals.append(
                {
                    "source": str(item.get("source", "")),
                    "title": str(item.get("title", "")),
                    "why_it_matters": str(item.get("summary", "")),
                    "url": str(item.get("url", "")),
                }
            )
        return signals

    def _build_fallback_report(self, filtered_items: list[dict], reason: str) -> dict[str, Any]:
        top_items = filtered_items[: max(1, min(self.max_items, len(filtered_items)))]
        source_counts: dict[str, int] = {}
        for item in top_items:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1

        top_papers = [
            self._normalize_paper_card(item.get("paper_card"), item)
            for item in top_items
            if item.get("paper_card")
        ]
        if not top_papers:
            top_papers = [
                self._normalize_paper_card({}, item)
                for item in top_items
                if item.get("source") in ("arxiv", "huggingface", "semanticscholar", "pubmed", "rss")
            ]
        top_titles = "、".join(self._safe_slug(item.get("title", "")) for item in top_papers[:3])
        opening = (
            f"今天最值得优先阅读的论文包括 {top_titles or '高分论文'}。"
            "这份版本基于已完成的抓取与评分结果自动生成，优先保留单篇论文的问题、方法、实验和阅读价值。"
        )

        watchlist = []
        for item in top_items[:5]:
            reason_text = str(item.get("summary", "")).strip() or str(item.get("detail", "")).strip()
            watchlist.append(
                {
                    "title": self._safe_slug(item.get("title", "Untitled"), limit=52) or "重点条目",
                    "source": str(item.get("source", "")),
                    "reason": reason_text or "今天的多源筛选结果中优先级较高。",
                    "url": str(item.get("url", "")),
                }
            )

        return {
            "report_title": self.report_title or "今日论文卡片速览",
            "subtitle": "基于已完成抓取结果生成的稳定版论文卡片",
            "opening": opening,
            "top_papers": top_papers,
            "area_summary": [],
            "watchlist": watchlist,
            "metadata": {
                "date": self.run_date,
                "generated_at": self.run_datetime.isoformat(),
                "source_counts": source_counts,
                "input_item_count": len(filtered_items),
                "generation_mode": "fallback",
                "fallback_reason": reason,
            },
        }

    def _normalize_report(self, data: dict[str, Any], filtered_items: list[dict]) -> dict[str, Any]:
        title = str(data.get("report_title", "")).strip() or self.report_title
        subtitle = str(data.get("subtitle", "")).strip()
        opening = str(data.get("opening", "")).strip()
        item_by_url = {
            str(item.get("url", "")).strip(): item
            for item in filtered_items
            if str(item.get("url", "")).strip()
        }
        item_by_title = {
            str(item.get("title", "")).strip(): item
            for item in filtered_items
            if str(item.get("title", "")).strip()
        }

        top_papers = []
        seen_titles: set[str] = set()
        for raw_card in data.get("top_papers") or []:
            if not isinstance(raw_card, dict):
                continue
            matched_item = None
            card_url = str(raw_card.get("url", "")).strip()
            card_title = str(raw_card.get("title", "")).strip()
            if card_url:
                matched_item = item_by_url.get(card_url)
            if not matched_item and card_title:
                matched_item = item_by_title.get(card_title)

            if matched_item:
                base_card = matched_item.get("paper_card")
                if not isinstance(base_card, dict):
                    base_card = {}
                merged_card = {**base_card, **raw_card}
                merged_card["title"] = matched_item.get("title", merged_card.get("title", ""))
                merged_card["source"] = matched_item.get("source", merged_card.get("source", ""))
                merged_card["score"] = matched_item.get("score", merged_card.get("score", 0))
                merged_card["url"] = matched_item.get("url", merged_card.get("url", ""))
                normalized_card = self._normalize_paper_card(merged_card, matched_item)
            else:
                normalized_card = self._normalize_paper_card(raw_card, {})

            title_key = normalized_card.get("title", "").lower()
            if normalized_card.get("title") and title_key not in seen_titles:
                seen_titles.add(title_key)
                top_papers.append(normalized_card)

        if not top_papers:
            for item in filtered_items:
                if not item.get("paper_card"):
                    continue
                normalized_card = self._normalize_paper_card(item.get("paper_card"), item)
                title_key = normalized_card.get("title", "").lower()
                if normalized_card.get("title") and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    top_papers.append(normalized_card)
                if len(top_papers) >= self.max_items:
                    break

        top_papers = top_papers[: self.max_items]

        area_summary = []
        for raw_area in data.get("area_summary") or []:
            if not isinstance(raw_area, dict):
                continue
            area = str(raw_area.get("area", "")).strip()
            summary = str(raw_area.get("summary", "")).strip()
            if area and summary:
                area_summary.append({"area": area, "summary": summary})

        watchlist = []
        for watch in data.get("watchlist") or []:
            if not isinstance(watch, dict):
                continue
            item = str(watch.get("title") or watch.get("item", "")).strip()
            reason = str(watch.get("reason", "")).strip()
            if item:
                watchlist.append(
                    {
                        "title": item,
                        "source": str(watch.get("source", "")).strip(),
                        "reason": reason,
                        "url": str(watch.get("url", "")).strip(),
                    }
                )

        return {
            "report_title": title,
            "subtitle": subtitle,
            "opening": opening,
            "top_papers": top_papers,
            "area_summary": area_summary,
            "watchlist": watchlist,
            "metadata": {
                "date": self.run_date,
                "generated_at": self.run_datetime.isoformat(),
                "source_counts": {
                    source_name: len(recs) for source_name, recs in self.all_recs.items()
                },
                "input_item_count": len(filtered_items),
            },
        }

    def generate(self) -> dict[str, Any] | None:
        filtered = self._filter_items()
        if not filtered:
            print("[ReportGenerator] No recommendation items available for report generation.")
            return None

        print(
            f"[ReportGenerator] Building report from {len(filtered)} curated items "
            f"(min_score={self.min_score})."
        )
        prompt = self._build_prompt(filtered)
        raw = self.model.inference(prompt, temperature=self.llm_config.temperature)
        cleaned = self._clean_llm_json(raw)
        data = self._parse_json_object(cleaned)

        if data is None:
            try:
                json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"[ReportGenerator] Failed to parse report JSON: {e}")
            else:
                print("[ReportGenerator] LLM response is not a JSON object.")
            print(f"[ReportGenerator] Raw response (first 600 chars): {cleaned[:600]}")
            data = self._repair_report_json(cleaned)

        if data is None:
            print("[ReportGenerator] Falling back to deterministic report rendering.")
            report = self._build_fallback_report(
                filtered,
                reason="llm_report_json_invalid",
            )
            report["input_items"] = filtered
            return report

        report = self._normalize_report(data, filtered)
        report["input_items"] = filtered
        return report

    def render_email(self, report: dict[str, Any]) -> str:
        html = render_report_email(report)
        if self.common_config.save:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(self.email_cache_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def save(self, report: dict[str, Any]) -> None:
        if not self.common_config.save:
            print("[ReportGenerator] Save disabled, skipping.")
            return

        os.makedirs(self.save_dir, exist_ok=True)

        json_path = os.path.join(self.save_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[ReportGenerator] JSON saved to {json_path}")

        md_path = os.path.join(self.save_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {report.get('report_title', self.report_title)}\n")
            subtitle = str(report.get("subtitle", "")).strip()
            if subtitle:
                f.write(f"> {subtitle}\n\n")
            f.write(f"- 日期：{self.run_date}\n")
            f.write(
                f"- 覆盖来源："
                f"{', '.join(f'{name}({count})' for name, count in report.get('metadata', {}).get('source_counts', {}).items())}\n\n"
            )

            opening = str(report.get("opening", "")).strip()
            if opening:
                f.write("## 今日重点\n\n")
                f.write(opening + "\n\n")

            top_papers = report.get("top_papers") or []
            if top_papers:
                f.write("## 今日重点论文卡片\n\n")
                for index, card in enumerate(top_papers, 1):
                    title = card.get("title", "Untitled")
                    url = card.get("url", "")
                    heading = f"### {index}. [{title}]({url})" if url else f"### {index}. {title}"
                    f.write(heading + "\n\n")
                    f.write(
                        f"- **论文**：{title} / {card.get('year', '未说明')} / "
                        f"{card.get('venue_or_status', '未说明')} / {card.get('area', '未说明')} / "
                        f"Score {card.get('score', '')}\n"
                    )
                    f.write(f"- **一句话**：{card.get('one_sentence', '未说明')}\n")
                    f.write(f"- **问题**：{card.get('problem', '未说明')}\n")
                    method = card.get("method") or {}
                    f.write(
                        "- **方法**："
                        f"输入：{method.get('input', '未说明')} → "
                        f"核心模块：{method.get('core_modules', '未说明')} → "
                        f"训练/优化目标：{method.get('training_objective', '未说明')} → "
                        f"输出：{method.get('output', '未说明')}\n"
                    )
                    innovations = card.get("innovations") or []
                    f.write(
                        "- **创新**："
                        + ("；".join(str(x) for x in innovations) if innovations else "未说明")
                        + "\n"
                    )
                    experiments = card.get("experiments") or {}
                    f.write(
                        "- **实验**："
                        f"主要提升：{experiments.get('main_results', '未说明')}；"
                        f"关键消融：{experiments.get('ablations', '未说明')}；"
                        f"效率/成本：{experiments.get('efficiency_or_cost', '未说明')}\n"
                    )
                    f.write(f"- **局限**：{card.get('limitations', '未说明')}\n")
                    value = card.get("value_to_me") or {}
                    uses = value.get("uses") or []
                    uses_text = " / ".join(str(x) for x in uses) if uses else "未说明"
                    reason = value.get("reason", "")
                    f.write(f"- **对我价值**：适合用于：{uses_text}；原因：{reason or '未说明'}\n")
                    f.write(f"- **证据强度**：{card.get('evidence_strength', '未说明')}\n")
                    f.write(f"- **代码/数据**：{card.get('reusable_assets', '未说明')}\n")
                    f.write(f"- **阅读优先级**：{card.get('reading_priority', '未说明')}\n\n")

            area_summary = report.get("area_summary") or []
            if area_summary:
                f.write("## 按方向速览\n\n")
                for item in area_summary:
                    f.write(f"- **{item.get('area', '')}**：{item.get('summary', '')}\n")
                f.write("\n")

            watchlist = report.get("watchlist") or []
            if watchlist:
                f.write("## 继续跟踪\n\n")
                for watch in watchlist:
                    title = watch.get("title") or watch.get("item", "")
                    url = watch.get("url", "")
                    label = f"[{title}]({url})" if url else title
                    source = watch.get("source", "")
                    prefix = f"[{source}] " if source else ""
                    f.write(f"- {prefix}**{label}**：{watch.get('reason', '')}\n")
                f.write("\n")
        print(f"[ReportGenerator] Markdown saved to {md_path}")

    def send_email(self, report: dict[str, Any], email_config: EmailConfig):
        html = self.render_email(report)
        BaseSource._send_email_html(html, email_config, self.report_title, self.run_datetime)
