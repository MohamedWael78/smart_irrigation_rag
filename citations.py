"""
Citation parsing + rendering.

The knowledge-base tool formats retrieved chunks as:
    [1] (fao56_ch6.pdf, p.42): <chunk text...>
    [2] (drip_manual.pdf, p.7): <chunk text...>

so the LLM can cite them by number in its answer. This module extracts
that structure back out of the tool's raw text output so the UI can render
real citation cards (source + page + snippet) instead of a flat filename
list, and builds the HTML for the custom citation-card component.

Keeping this as string parsing (rather than depending on LangChain's
`response_format="content_and_artifact"` tool-artifact channel) makes it
robust across LangChain versions, since AgentExecutor implementations
don't uniformly propagate tool artifacts.
"""
import html
import re

_CITATION_RE = re.compile(
    r"\[(\d+)\]\s*\(([^,]+),\s*p\.([^\)]+)\):\s*(.+?)(?=\n\n\[\d+\]\s*\(|\Z)",
    re.DOTALL,
)


def parse_citations(tool_output_text: str) -> list[dict]:
    if not tool_output_text:
        return []
    citations = []
    for m in _CITATION_RE.finditer(tool_output_text):
        idx, source, page, snippet = m.groups()
        snippet = " ".join(snippet.split())  # collapse whitespace/newlines
        if len(snippet) > 240:
            snippet = snippet[:240].rsplit(" ", 1)[0] + "\u2026"
        citations.append({
            "index": int(idx),
            "source": source.strip(),
            "page": page.strip(),
            "snippet": snippet,
        })
    return citations


def render_citation_cards_html(citations: list[dict]) -> str:
    """Builds a small self-contained HTML/CSS grid of citation cards for
    st.components.v1.html. Kept dependency-free (no external CSS/JS)."""
    if not citations:
        return ""

    cards = []
    for c in citations:
        source = html.escape(c["source"])
        page = html.escape(str(c["page"]))
        snippet = html.escape(c["snippet"])
        cards.append(f"""
        <div class="cite-card">
          <div class="cite-badge">{c['index']}</div>
          <div class="cite-body">
            <div class="cite-source">{source}</div>
            <div class="cite-page">Page {page}</div>
            <div class="cite-snippet">{snippet}</div>
          </div>
        </div>
        """)

    return f"""
    <div class="cite-wrap">
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
        .cite-wrap {{
          font-family: 'Plus Jakarta Sans', sans-serif;
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 10px;
          padding: 6px 2px 14px 2px;
        }}
        .cite-card {{
          display: flex;
          gap: 10px;
          background: #ffffff;
          border: 1px solid #dbe6e2;
          border-radius: 10px;
          padding: 12px;
          box-shadow: 0 2px 6px rgba(14, 27, 26, 0.05);
          transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        .cite-card:hover {{
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(14, 165, 183, 0.15);
          border-color: #0ea5b7;
        }}
        .cite-badge {{
          flex: 0 0 auto;
          width: 22px; height: 22px;
          border-radius: 6px;
          background: linear-gradient(135deg, #0ea5b7 0%, #67a63c 100%);
          color: white;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 700;
          display: flex; align-items: center; justify-content: center;
        }}
        .cite-source {{
          font-size: 0.8rem;
          font-weight: 700;
          color: #0e1b1a;
          word-break: break-word;
        }}
        .cite-page {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.66rem;
          color: #0b8494;
          margin: 2px 0 6px 0;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }}
        .cite-snippet {{
          font-size: 0.76rem;
          line-height: 1.4;
          color: #5b6b67;
        }}
      </style>
      {''.join(cards)}
    </div>
    """
