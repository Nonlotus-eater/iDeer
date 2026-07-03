import html

from email_utils.base_template import framework


def _escape(text) -> str:
    return html.escape(str(text or ""), quote=True)


def _source_badge(source: str) -> tuple[str, str]:
    mapping = {
        "github": ("GitHub", "#24292e"),
        "huggingface": ("HuggingFace", "#ff6f00"),
        "twitter": ("X/Twitter", "#1d9bf0"),
        "rss": ("RSS", "#0e7490"),
        "arxiv": ("arXiv", "#b31b1b"),
        "openreview": ("OpenReview", "#4051b5"),
        "semanticscholar": ("Semantic Scholar", "#1857b6"),
        "pubmed": ("PubMed", "#2e7d32"),
    }
    return mapping.get(str(source or "").lower(), (str(source or "source"), "#6b7280"))


def _pill(label: str, color: str = "#475569") -> str:
    if not str(label or "").strip():
        return ""
    return (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:999px;'
        f'background:{color}14;color:{color};font-size:12px;font-weight:700;'
        f'margin:0 6px 6px 0;">{_escape(label)}</span>'
    )


def _field(label: str, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        value = "未说明"
    return f"""
    <div style="margin-top:10px;">
      <span style="font-weight:800;color:#0f172a;">【{_escape(label)}】</span>
      <span style="color:#334155;">{_escape(value)}</span>
    </div>
    """


def _link_field(label: str, url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return _field(label, "未说明")
    escaped_url = _escape(url)
    return f"""
    <div style="margin-top:10px;">
      <span style="font-weight:800;color:#0f172a;">【{_escape(label)}】</span>
      <a href="{escaped_url}" style="color:#2563eb;text-decoration:none;">{escaped_url}</a>
    </div>
    """


def _method_block(method: dict) -> str:
    method = method or {}
    parts = [
        ("输入", method.get("input", "")),
        ("核心模块", method.get("core_modules", "")),
        ("训练/优化目标", method.get("training_objective", "")),
        ("输出", method.get("output", "")),
    ]
    text = " → ".join(
        f"{label}: {str(value or '未说明').strip() or '未说明'}" for label, value in parts
    )
    return _field("方法", text)


def _list_block(label: str, values: list) -> str:
    items = [str(value).strip() for value in (values or []) if str(value).strip()]
    if not items:
        return _field(label, "未说明")
    lis = "".join(
        f'<li style="margin:4px 0;color:#334155;line-height:1.65;">{_escape(value)}</li>'
        for value in items
    )
    return f"""
    <div style="margin-top:10px;">
      <span style="font-weight:800;color:#0f172a;">【{_escape(label)}】</span>
      <ol style="margin:6px 0 0 20px;padding:0;">{lis}</ol>
    </div>
    """


def _experiments_block(experiments: dict) -> str:
    experiments = experiments or {}
    value = (
        f"主要提升：{str(experiments.get('main_results') or '未说明').strip() or '未说明'}；"
        f"关键消融：{str(experiments.get('ablations') or '未说明').strip() or '未说明'}；"
        f"效率/成本：{str(experiments.get('efficiency_or_cost') or '未说明').strip() or '未说明'}"
    )
    return _field("实验", value)


def _value_block(value_to_me: dict) -> str:
    value_to_me = value_to_me or {}
    uses = [str(x).strip() for x in (value_to_me.get("uses") or []) if str(x).strip()]
    uses_text = " / ".join(uses) if uses else "未说明"
    reason = str(value_to_me.get("reason") or "").strip()
    text = f"适合用于：{uses_text}"
    if reason:
        text += f"；原因：{reason}"
    return _field("对我价值", text)


def _render_paper_card(card: dict, index: int) -> str:
    label, color = _source_badge(card.get("source", ""))
    title = _escape(card.get("title", "Untitled"))
    url = _escape(card.get("url", ""))
    link_open = f'<a href="{url}" style="color:#0f172a;text-decoration:none;">' if url else ""
    link_close = "</a>" if url else ""
    score = card.get("score", "")
    score_text = f"评分 {score}" if score != "" else ""
    meta = " / ".join(
        part
        for part in [
            str(card.get("year", "")).strip(),
            str(card.get("venue_or_status", "")).strip(),
            str(card.get("area", "")).strip(),
        ]
        if part
    )

    return f"""
    <div style="padding:20px 22px;border-radius:8px;background:#ffffff;border:1px solid #e2e8f0;
                box-shadow:0 8px 22px rgba(15,23,42,0.05);margin-bottom:18px;">
      <div style="font-size:13px;font-weight:800;color:#64748b;margin-bottom:8px;">
        论文 {index}
      </div>
      <div style="font-size:21px;font-weight:800;color:#0f172a;line-height:1.35;">
        {link_open}{title}{link_close}
      </div>
      <div style="margin-top:10px;">
        {_pill(label, color)}
        {_pill(score_text, "#2563eb")}
        {_pill(card.get("reading_priority", ""), "#0f766e")}
        {_pill(card.get("evidence_strength", ""), "#7c3aed")}
        {_pill(card.get("reusable_assets", ""), "#b45309")}
      </div>
      <div style="font-size:13px;color:#64748b;line-height:1.65;margin-top:2px;">
        {_escape(meta or "年份 / 会议或状态 / 领域未说明")}
      </div>
      <div style="font-size:14px;line-height:1.75;color:#334155;margin-top:14px;">
        {_link_field("原文", card.get("url", ""))}
        {_field("一句话", card.get("one_sentence", ""))}
        {_field("问题", card.get("problem", ""))}
        {_method_block(card.get("method") or {})}
        {_list_block("创新", card.get("innovations") or [])}
        {_experiments_block(card.get("experiments") or {})}
        {_field("局限", card.get("limitations", ""))}
        {_value_block(card.get("value_to_me") or {})}
      </div>
    </div>
    """


def _render_area_summary(item: dict) -> str:
    return f"""
    <li style="margin:8px 0;line-height:1.7;color:#334155;">
      <strong style="color:#0f172a;">{_escape(item.get("area", ""))}</strong>：
      {_escape(item.get("summary", ""))}
    </li>
    """


def _render_watch(watch: dict) -> str:
    label, color = _source_badge(watch.get("source", ""))
    title = _escape(watch.get("title") or watch.get("item", ""))
    reason = _escape(watch.get("reason", ""))
    url = _escape(watch.get("url", ""))
    link_open = f'<a href="{url}" style="color:#0f172a;text-decoration:none;">' if url else ""
    link_close = "</a>" if url else ""
    return f"""
    <li style="margin:10px 0;line-height:1.7;color:#475569;">
      {_pill(label, color)}
      {link_open}<strong style="color:#0f172a;">{title}</strong>{link_close}：{reason}
    </li>
    """


def render_report_email(report: dict) -> str:
    title = _escape(report.get("report_title", "Daily Paper Digest"))
    subtitle = _escape(report.get("subtitle", ""))
    opening = _escape(report.get("opening", "")).replace("\n", "<br><br>")
    metadata = report.get("metadata", {}) or {}
    source_counts = metadata.get("source_counts", {}) or {}
    source_line = " · ".join(f"{_escape(name)} {count}" for name, count in source_counts.items())
    top_papers = report.get("top_papers") or []
    area_summary = report.get("area_summary") or []
    watchlist = report.get("watchlist") or []

    header = f"""
    <div style="font-family:'Helvetica Neue',Arial,sans-serif;margin-bottom:24px;">
      <div style="font-size:28px;font-weight:800;color:#0f172a;line-height:1.25;">{title}</div>
      <div style="margin-top:10px;font-size:15px;line-height:1.7;color:#475569;">{subtitle}</div>
      <div style="margin-top:14px;font-size:12px;color:#64748b;">
        生成日期：{_escape(metadata.get('date', ''))} · 来源覆盖：{source_line}
      </div>
    </div>
    """

    overview = ""
    if opening:
        overview = f"""
        <div style="padding:18px 20px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;
                    margin-bottom:24px;">
          <div style="font-size:13px;font-weight:800;color:#334155;">今日重点</div>
          <div style="margin-top:10px;font-size:15px;line-height:1.8;color:#1e293b;">{opening}</div>
        </div>
        """

    paper_cards_html = "".join(
        _render_paper_card(card, index) for index, card in enumerate(top_papers, 1)
    )
    if not paper_cards_html:
        paper_cards_html = """
        <div style="padding:18px 20px;border-radius:8px;background:#ffffff;border:1px solid #e2e8f0;
                    color:#64748b;line-height:1.7;">
          今天没有足够高相关的论文卡片。可以检查数据源、最低分阈值或 LLM 输出日志。
        </div>
        """

    area_html = ""
    if area_summary:
        area_html = (
            '<div class="section-title" style="border-bottom-color:#0f766e;">按方向速览</div>'
            + f'<ul style="padding-left:20px;margin:0 0 22px 0;">'
            + "".join(_render_area_summary(item) for item in area_summary)
            + "</ul>"
        )

    watch_html = ""
    if watchlist:
        watch_html = (
            '<div class="section-title" style="border-bottom-color:#64748b;">继续跟踪</div>'
            + f'<ul style="padding-left:20px;margin:0;">'
            + "".join(_render_watch(item) for item in watchlist)
            + "</ul>"
        )

    content = (
        header
        + overview
        + '<div class="section-title" style="border-bottom-color:#0f172a;">今日重点论文卡片</div>'
        + paper_cards_html
        + area_html
        + watch_html
    )
    return framework.replace("__CONTENT__", content)
