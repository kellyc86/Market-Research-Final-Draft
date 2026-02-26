"""
Market Research Assistant
=========================
A Streamlit RAG application that generates structured industry reports
from Wikipedia sources in three stages:

    1. Validate the user's industry input via LLM
    2. Retrieve and filter Wikipedia pages via multi-query search
    3. Generate a grounded, cited report under 500 words

Retrieval follows a broad-then-filter design: the LLM generates five
Wikipedia-style queries covering distinct aspects of the industry, the
retriever casts a wide net across all of them in parallel, and a second
LLM pass reranks and selects the five most relevant pages. This reduces
the risk of missing important sub-topics that a single query would overlook
and mirrors ensemble reasoning -- diverse weak signals outperform one strong one.

Grounding is enforced through explicit prompt constraints and low temperature
(0.2), which reduces stochastic variation and keeps outputs closer to the
retrieved source material. Temperature alone does not eliminate hallucination;
it only shifts the probability distribution toward more conservative completions.
A programmatic word-limit check runs after generation as a hard backstop,
since LLMs cannot reliably track their own output length.
"""

import io
import re
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF


# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
APP_TITLE = "Market Research Assistant"
APP_ICON = ":material/query_stats:"
LLM_OPTIONS = [
    "Gemini 2.5 Flash",
    "Gemini 2.5 Pro",
]
LLM_MODEL_MAP = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}
LLM_DESCRIPTIONS = {
    "Gemini 2.5 Flash": "Fast & free -- great for quick reports",
    "Gemini 2.5 Pro": "Most capable Gemini -- best report quality",
}
# Pipeline tuning parameters -- balance retrieval breadth against API cost.
# Values were settled through repeated testing across a range of industry queries.
DEFAULT_TEMPERATURE = 0.2          # Lower = more deterministic, source-faithful outputs
MAX_WIKI_RESULTS = 10              # Pages fetched per search query
FINAL_SOURCE_COUNT = 5             # Sources used in the final report
MAX_REPORT_WORDS = 500             # Hard prose ceiling aligned with assignment requirement
REPORT_WORD_TARGET = 380           # Soft prose target: kept low so LLM lands ~400-430,
                                   # well below the 500 ceiling before any trimming runs
HARD_WORD_LIMIT = 500              # Enforced programmatically on prose only -- the Key Data
                                   # table is excluded from the count and never truncated
WIKI_CONTENT_CHARS = 8000          # Characters extracted per Wikipedia page
NUM_SEARCH_QUERIES = 5             # Number of search queries the LLM generates


# --------------------------------------------------------------
# HELPER FUNCTIONS -- modular pipeline stages
# --------------------------------------------------------------

def handle_api_error(e: Exception, context: str = "Operation") -> None:
    """Surface a user-friendly message for common Google Gemini API errors.

    Raw API error strings are noisy and unhelpful for non-technical users.
    Pattern-matching on the error message maps known failure modes -- bad
    key, quota exhaustion, missing model -- to actionable guidance.
    """
    error_msg = str(e).lower()
    if "api key" in error_msg or "api_key" in error_msg or "authentication" in error_msg:
        st.error(
            "**Invalid API key.** Please check the Google AI API key "
            "in the sidebar and try again."
        )
    elif "resource_exhausted" in error_msg or "429" in error_msg or "quota" in error_msg:
        st.warning(
            "**Rate limit reached.** The Gemini API has per-minute quotas. "
            "Please wait 1-2 minutes and try again, or switch to the other model."
        )
    elif "404" in error_msg or "not_found" in error_msg:
        st.error(
            "**Model not found.** The selected model may be temporarily "
            "unavailable. Please try the other Gemini model."
        )
    else:
        st.error(f"{context} failed: {e}")


def initialise_llm(model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini model with a fixed low temperature.

    Temperature 0.2 keeps outputs closer to the training distribution's
    high-probability completions, which tend to be more factually conservative.
    The tradeoff is reduced creativity, which is acceptable here -- we want
    the model to report what the sources say, not speculate beyond them.
    """
    model_id = LLM_MODEL_MAP[model_name]
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_validate_industry(model_name: str, api_key: str, user_input: str) -> dict:
    """Cached wrapper so repeated validation calls don't burn API quota.

    Streamlit reruns the whole script on every interaction, so without caching
    the validation LLM call would fire again on unrelated UI events. Keying the
    cache on (model, api_key, input) means genuinely different queries always
    run fresh while re-submissions hit the cache in under a millisecond.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_MAP[model_name],
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )
    return validate_industry(llm, user_input)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_generate_search_queries(model_name: str, api_key: str, industry: str) -> list[str]:
    """Cached wrapper for search query generation.

    Query generation is cheap but adds ~1 second of latency. Since the same
    industry name produces consistent queries at low temperature, caching avoids
    redundant calls when the user re-runs or navigates back through the pipeline.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_MAP[model_name],
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )
    return generate_search_queries(llm, industry)


def validate_industry(llm, user_input: str) -> dict:
    """LLM-based industry validation with normalisation.

    Rule-based matching (regex, word lists) can't handle the range of informal
    inputs users actually type -- 'pharma', 'EVs', 'oil'. An LLM handles
    synonyms and abbreviations naturally while still rejecting company names
    and nonsense. The structured three-line output format makes parsing
    deterministic and avoids having to interpret free-form prose responses.

    Returns a dict with keys: is_valid (bool), normalised (str), reason (str).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a validation assistant. Your ONLY job is to decide "
         "whether the user's input refers to a legitimate industry or "
         "economic sector.\n\n"
         "Respond in EXACTLY this format (three lines, no extras):\n"
         "VALID: yes or no\n"
         "NORMALISED: <the cleaned, standard industry name if valid, "
         "otherwise empty>\n"
         "REASON: <one-sentence explanation>\n\n"
         "VALID examples: semiconductor manufacturing, renewable energy, "
         "pharmaceutical, fast fashion, fintech, automotive, agriculture, "
         "telecommunications, e-commerce, cybersecurity.\n\n"
         "VALID even if informal -- normalise to the standard name:\n"
         "  'cars' -> Automotive Industry\n"
         "  'AI' -> Artificial Intelligence Industry\n"
         "  'tech' -> Technology Industry\n"
         "  'pharma' -> Pharmaceutical Industry\n"
         "  'oil' -> Oil and Gas Industry\n\n"
         "INVALID examples (must reject these):\n"
         "  - Company names: 'Apple', 'Tesla', 'Google'\n"
         "  - Product names: 'iPhone', 'Model 3'\n"
         "  - People's names: 'Elon Musk', 'Jeff Bezos'\n"
         "  - Random words: 'hello', 'asdfg', 'pizza'\n"
         "  - Questions or sentences: 'what is AI?'\n"
         "  - Empty or whitespace-only input\n\n"
         "If the input is a loose synonym, abbreviation, or informal "
         "name for a real industry, mark it VALID and normalise it."),
        ("human", "Is this a valid industry? Input: '{input}'"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": user_input})

    # Normalise the raw response before parsing -- LLMs occasionally use
    # en-dashes, smart quotes, or other Unicode in their output even when
    # following a strict format. ASCII-safe substitutions prevent crashes
    # in the startswith() comparisons below.
    response = (
        response
        .replace("\u2013", "-")   # en-dash
        .replace("\u2014", "-")   # em-dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u00a0", " ")   # non-breaking space
    )

    # Parse the structured three-line response.
    # If the LLM deviates from the format, we default to invalid rather
    # than silently accepting input that may have no Wikipedia coverage.
    lines = response.strip().split("\n")
    result = {"is_valid": False, "reason": "", "normalised": ""}
    parsed_valid_line = False

    for line in lines:
        lower = line.lower().strip()
        if lower.startswith("valid:"):
            result["is_valid"] = "yes" in lower
            parsed_valid_line = True
        elif lower.startswith("normalised:") or lower.startswith("normalized:"):
            result["normalised"] = line.split(":", 1)[1].strip()
        elif lower.startswith("reason:"):
            result["reason"] = line.split(":", 1)[1].strip()

    # Default to invalid if no VALID: line was found at all
    if not parsed_valid_line:
        result["is_valid"] = False
        result["reason"] = (
            "Could not determine validity. Please try rephrasing "
            "your input as a standard industry name."
        )

    return result


def generate_search_queries(llm, industry: str) -> list[str]:
    """Generate five aspect-targeted Wikipedia queries for broader source coverage.

    A single query reliably surfaces the main overview article but misses
    adjacent pages on regulation, key firms, and technology -- all of which
    matter for a useful market report. Splitting retrieval across five targeted
    queries gets a more representative candidate pool before the reranking step
    selects the best five. The Wikipedia article-title format is important:
    conversational queries ('what is the semiconductor market size?') return
    poor results from the Wikipedia API compared to encyclopaedic titles.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research strategist. Given an industry, generate exactly "
         "5 distinct Wikipedia search queries that together would provide "
         "comprehensive coverage for a market research report.\n\n"
         "The queries should target different aspects:\n"
         "1. The industry itself (overview, definition)\n"
         "2. The market size or economics of the industry\n"
         "3. Key technology or innovation in the industry\n"
         "4. Regulation, policy, or risks in the industry\n"
         "5. Major companies or competitive landscape\n\n"
         "IMPORTANT: Format each query as a Wikipedia article title -- "
         "use proper capitalisation and standard encyclopaedic naming.\n"
         "GOOD: 'Semiconductor industry', 'Automotive safety'\n"
         "BAD: 'semiconductor market size trends 2024'\n\n"
         "Respond with ONLY the 5 queries, one per line. No numbering, "
         "no explanation."),
        ("human", "Industry: {industry}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"industry": industry})

    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    # Always include the industry name itself as a baseline query
    if industry not in queries:
        queries.insert(0, industry)
    return queries[:NUM_SEARCH_QUERIES + 1]


def _fetch_single_query(query: str) -> list[dict]:
    """Fetch Wikipedia pages for one query -- designed to run inside a thread.

    Returning an empty list on failure rather than raising means one bad query
    (e.g. a term with no Wikipedia article) doesn't abort the whole retrieval
    batch. The caller aggregates results across threads so partial failures
    degrade gracefully rather than crashing the pipeline.
    """
    retriever = WikipediaRetriever(
        top_k_results=MAX_WIKI_RESULTS,
        doc_content_chars_max=WIKI_CONTENT_CHARS,
    )
    try:
        docs = retriever.invoke(query)
    except Exception:
        return []

    results = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "") or (
            "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        )
        results.append({
            "title": title,
            "url": source,
            "content": doc.page_content,
        })
    return results


def retrieve_wikipedia_pages(industry: str, queries: list[str]) -> list[dict]:
    """Run all Wikipedia queries concurrently and return deduplicated results.

    Sequential retrieval was the biggest bottleneck in early versions -- five
    queries at ~2 seconds each meant 10 seconds of waiting before any ranking
    could begin. Parallelising with ThreadPoolExecutor brings this closer to
    the slowest single request (~2-3s total). Threads rather than asyncio
    because WikipediaRetriever is synchronous and has no async interface.
    Deduplication by title prevents the same article appearing multiple times
    when different queries happen to surface it.
    """
    pages = []
    seen_titles = set()

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = {executor.submit(_fetch_single_query, q): q for q in queries}
        for future in as_completed(futures):
            for page in future.result():
                if page["title"] not in seen_titles:
                    seen_titles.add(page["title"])
                    pages.append(page)

    return pages


def select_top_pages(
    llm,
    industry: str,
    pages: list[dict],
) -> list[dict]:
    """LLM-as-reranker: select the five most relevant pages from the candidate pool.

    Broad retrieval reliably brings back tangential pages -- founder biographies,
    geographic overviews, narrowly scoped sub-articles -- that would dilute the
    final report. Having the LLM evaluate title and opening snippet for relevance
    to a market research brief filters these out cheaply. The snippet-only input
    keeps token cost low while giving enough signal for relevance judgement.
    Fallback to the first five pages if the LLM response is unparseable ensures
    the pipeline never stalls on a malformed reranker output.
    """
    if len(pages) <= FINAL_SOURCE_COUNT:
        return pages

    # Build a numbered list of candidates -- title + snippet -- for the LLM to evaluate
    candidate_descriptions = ""
    for i, page in enumerate(pages):
        snippet = page["content"][:600]
        candidate_descriptions += (
            f"[{i}] Title: {page['title']}\n"
            f"    Snippet: {snippet}...\n\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research librarian selecting the most relevant "
         "Wikipedia sources for a market research report.\n\n"
         "TASK: From the numbered candidate pages below, select EXACTLY "
         "5 that are MOST relevant to writing an industry report about "
         "'{industry}'. Prioritise pages that cover:\n"
         "- The industry overview and structure\n"
         "- Market size, trends, and growth drivers\n"
         "- Key companies and competitive landscape\n"
         "- Regulation, risks, and challenges\n"
         "- Technology and innovation in the sector\n\n"
         "AVOID selecting pages about:\n"
         "- Individual people or biographies\n"
         "- Unrelated tangential topics\n"
         "- Overly narrow sub-topics that don't inform the big picture\n\n"
         "Respond with ONLY the 5 index numbers, one per line, "
         "in order of relevance (most relevant first).\n"
         "Example response:\n3\n0\n7\n1\n5"),
        ("human",
         "Industry: {industry}\n\nCandidate pages:\n{candidates}"),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "industry": industry,
        "candidates": candidate_descriptions,
    })

    # Extract integers from the response, robust to varied output formats.
    # Handles one-per-line, comma-separated, bracketed, or annotated responses.
    selected = []
    seen_indices = set()
    for match in re.findall(r"\d+", response):
        idx = int(match)
        if 0 <= idx < len(pages) and idx not in seen_indices:
            seen_indices.add(idx)
            selected.append(pages[idx].copy())
        if len(selected) == FINAL_SOURCE_COUNT:
            break

    # If parsing fails entirely, fall back to the first five pages
    if len(selected) < FINAL_SOURCE_COUNT:
        for i, page in enumerate(pages):
            if i not in seen_indices:
                selected.append(page)
                seen_indices.add(i)
            if len(selected) == FINAL_SOURCE_COUNT:
                break

    return selected[:FINAL_SOURCE_COUNT]


def filter_low_quality_pages(pages: list[dict]) -> list[dict]:
    """Remove stubs and disambiguation pages before passing candidates to the reranker.

    Without this step, the LLM reranker wastes ranking capacity on near-empty
    articles. Two patterns reliably identify low-value pages: disambiguation pages
    open with 'may refer to', and stub articles are identifiable by length alone.
    The 1,500-character threshold is intentionally conservative -- a genuine
    industry overview article is almost always longer than this. Setting it higher
    risked discarding useful but concise pages on niche sub-sectors.
    """
    MIN_CONTENT_LENGTH = 1500
    filtered = []
    for page in pages:
        content = page["content"]
        # Reject disambiguation pages
        if "may refer to" in content[:300].lower():
            continue
        # Reject stubs -- too short to contain useful market data
        if len(content) < MIN_CONTENT_LENGTH:
            continue
        filtered.append(page)

    # If filtering removes everything, fall back to the full list
    # to avoid returning zero candidates to the ranking step
    return filtered if filtered else pages


def check_source_diversity(pages: list[dict]) -> dict:
    """Flag low source diversity using pairwise Jaccard similarity.

    Jaccard similarity (|intersection| / |union| on word sets) measures how much
    vocabulary two pages share. High overlap between multiple sources suggests
    they cover the same narrow sub-topic rather than complementary angles, which
    tends to produce shallow, repetitive reports. The 0.4 threshold came from
    manually checking outputs: same-industry Wikipedia pages typically share
    15-25% vocabulary through shared terminology; above 40% usually means
    near-duplicate coverage. Requiring two or more high-overlap pairs to trigger
    the warning avoids false positives from one coincidentally similar pair.
    """
    if len(pages) < 2:
        return {"is_diverse": True, "avg_overlap": 0.0, "warning": ""}

    # Word sets built from the first 2000 characters of each page for speed
    word_sets = []
    for page in pages:
        words = set(page["content"][:2000].lower().split())
        word_sets.append(words)

    # Pairwise Jaccard similarity across all page combinations
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if union:
                overlaps.append(len(intersection) / len(union))

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    # Require two or more high-overlap pairs to trigger the warning --
    # a single coincidental match should not penalise otherwise diverse sources
    high_overlap_count = sum(1 for o in overlaps if o > 0.4)

    if high_overlap_count >= 2:
        return {
            "is_diverse": False,
            "avg_overlap": avg_overlap,
            "warning": (
                f"Some sources have high content overlap "
                f"(avg similarity: {avg_overlap:.0%}). "
                f"Consider trying a more specific industry name "
                f"for more diverse results."
            ),
        }
    return {"is_diverse": True, "avg_overlap": avg_overlap, "warning": ""}


def generate_related_industries(llm, industry: str) -> list[str]:
    """Suggest adjacent industries to encourage broader contextual research.

    Analysing an industry in isolation misses upstream suppliers, downstream
    customers, and substitutes that often explain the most important market
    dynamics. Surfacing these connections at the end of the report prompts
    follow-on searches and makes the tool more useful for iterative research
    rather than one-off lookups.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a market research analyst. Given an industry, return exactly "
         "6 related or adjacent industries that a researcher would benefit from "
         "exploring next for deeper context.\n\n"
         "Focus on industries that are:\n"
         "- Upstream suppliers or downstream customers\n"
         "- Competitive substitutes or adjacent markets\n"
         "- Sectors that heavily influence or are influenced by this one\n\n"
         "Respond with ONLY the 6 industry names, one per line. "
         "Use standard industry names (2-5 words). No numbering, no explanation."),
        ("human", "Industry: {industry}"),
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"industry": industry})
    industries = [r.strip() for r in response.strip().split("\n") if r.strip()]
    return industries[:6]


def enforce_word_limit(text: str, limit: int = HARD_WORD_LIMIT) -> str:
    """Programmatic backstop ensuring prose never exceeds the 500-word limit.

    The prompt targets 380 words, so this function is rarely needed -- but
    LLMs cannot reliably self-count, so relying on the prompt alone is not
    sufficient. The algorithm separates table pipe rows first (they are
    excluded from the count and always reattached in full), then trims
    sentences from the longest prose sections iteratively until count_words()
    confirms compliance. Protected sections (Key Metrics, Key Data, Final
    Takeaway) are never trimmed to preserve report integrity.
    """
    # Step 1: pull out ALL table rows so they are never touched
    prose_lines = []
    table_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
        else:
            prose_lines.append(line)

    prose = "\n".join(prose_lines)

    # Already fine -- reattach table and return
    if count_words(prose) <= limit:
        return (prose.strip() + ("\n\n" + "\n".join(table_lines) if table_lines else "")).strip()

    # Step 2: split prose into sections, trim one sentence at a time from the
    # end of the longest section until we are under the limit.
    # Find section boundaries by looking for ## headings.
    section_pattern = re.compile(r'(^#{1,3}\s+.+$)', re.MULTILINE)
    parts = section_pattern.split(prose)
    # parts alternates: [pre-heading text, heading, body, heading, body, ...]

    # Build list of (heading, body) blocks preserving order
    blocks = []  # list of [heading_or_none, body_text]
    i = 0
    if parts and not section_pattern.match(parts[0].strip()):
        blocks.append(["", parts[0]])
        i = 1
    while i < len(parts):
        heading = parts[i] if i < len(parts) else ""
        body = parts[i + 1] if i + 1 < len(parts) else ""
        blocks.append([heading, body])
        i += 2

    # Trim sentences one at a time from the body with the most words
    # until we are under the limit -- never touch headings or Key Data
    max_iterations = 200
    iteration = 0
    while count_words("\n".join(h + b for h, b in blocks)) > limit and iteration < max_iterations:
        iteration += 1
        # Find the longest body that isn't Key Data or Final Takeaway
        longest_idx = -1
        longest_wc = 0
        for idx, (heading, body) in enumerate(blocks):
            if any(k in heading for k in ("Key Data", "Key Metrics", "Final Takeaway")):
                continue
            wc = count_words(body)
            if wc > longest_wc:
                longest_wc = wc
                longest_idx = idx

        if longest_idx == -1:
            break  # Nothing left to trim safely

        body = blocks[longest_idx][1]
        # Remove the last sentence from this body
        last_period = body.rstrip().rfind(".")
        if last_period > 0:
            blocks[longest_idx][1] = body[:last_period + 1].rstrip()
        else:
            # No sentence boundary -- trim last word
            words = body.split()
            blocks[longest_idx][1] = " ".join(words[:-1])

    prose = "\n".join(h + b for h, b in blocks)

    # Absolute final guarantee: if smart trimming still left us over (e.g.
    # all remaining sections were protected), do a hard word-level cut on
    # the prose only so count_words() can never return > limit.
    if count_words(prose) > limit:
        # Strip markdown symbols the same way count_words does, cut to limit,
        # then return the cut prose (structure may be imperfect but word
        # count is guaranteed -- this path should almost never be reached).
        clean_words = re.sub(r"[#*|]", "", re.sub(r"-{3,}", "", prose)).split()
        prose = " ".join(clean_words[:limit])

    return (prose.strip() + ("\n\n" + "\n".join(table_lines) if table_lines else "")).strip()


def count_words(text: str) -> int:
    """Count prose words only, excluding table rows and markdown formatting.

    Table pipe rows are structured data sitting outside the prose limit, so
    they are stripped before counting. Heading markers, bold/italic asterisks,
    and separator rows are also removed so the count reflects readable words
    rather than formatting artefacts. This function is used both to display
    the word count and inside enforce_word_limit -- using the same logic in
    both places guarantees the displayed figure and the truncation threshold
    can never disagree.
    """
    # Remove table rows before counting
    lines = [
        line for line in text.split("\n")
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]
    clean = "\n".join(lines)
    clean = re.sub(r"[#*|]", "", clean)
    clean = re.sub(r"-{3,}", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return len(clean.split()) if clean else 0


def sanitise_for_streamlit(text: str) -> str:
    """Strip characters that Streamlit's markdown renderer misinterprets as LaTeX.

    Streamlit renders $...$ and $$...$$ as math, which breaks whenever Wikipedia
    content or LLM output contains currency values or LaTeX-style sequences.
    Rather than escaping them (which adds backslashes visible to users), the
    simpler fix is to remove dollar signs entirely -- currency is written as
    'USD X billion' throughout the prompt, so nothing meaningful is lost.
    """
    text = text.replace("$", "")
    text = text.replace("\\(", "(")
    text = text.replace("\\)", ")")
    text = text.replace("\\[", "[")
    text = text.replace("\\]", "]")
    return text


HEADING_LABELS = [
    "Executive Summary",
    "Key Metrics",
    "Industry Overview",
    "Market Structure & Competitive Dynamics",
    "Growth Drivers",
    "Risks & Constraints",
    "Key Data",
    "Strategic Interpretation",
    "Final Takeaway",
]


def split_report_into_sections(report: str) -> list[tuple[str, str]]:
    """Parse the report into (heading, body) pairs, format-agnostically.

    The same model can output '## Executive Summary', '**Executive Summary**',
    or plain 'Executive Summary' across different runs even with identical prompts.
    Searching for the known heading label strings by position rather than by
    markdown format makes the parser resilient to this variation. Body content
    is then the text between consecutive heading positions, so partial or
    reordered outputs still parse correctly.
    """
    # Matches any heading label regardless of ## markers, ** bold wrappers,
    # or trailing colons -- all of which LLMs produce inconsistently
    heading_positions = []

    for label in HEADING_LABELS:
        pattern = (
            r"(?:\#{1,3}\s*)?"        # Optional ## markers
            r"(?:\*\*\s*)?"           # Optional opening **
            + re.escape(label)        # The heading label itself
            + r"(?:\s*\*\*)?"         # Optional closing **
            r"\s*:?\s*"              # Optional colon and whitespace
        )
        match = re.search(pattern, report)
        if match:
            heading_positions.append((match.start(), match.end(), label))

    heading_positions.sort(key=lambda x: x[0])

    if not heading_positions:
        return [("Report", report.strip())]

    sections = []
    for i, (start, end, label) in enumerate(heading_positions):
        if i + 1 < len(heading_positions):
            body = report[end:heading_positions[i + 1][0]]
        else:
            body = report[end:]
        sections.append((label, body.strip()))

    return sections


def generate_report(
    llm,
    industry: str,
    pages: list[dict],
) -> str:
    """Generate a structured, source-grounded industry report via the LLM.

    The prompt uses few-shot examples to demonstrate grounded vs hallucinated
    output -- instruction alone proved insufficient during testing. Dollar signs
    are explicitly banned in the prompt because Streamlit renders $...$ as LaTeX;
    sanitise_for_streamlit() catches any that slip through as a second defence.
    The word limit appears in the prompt as a soft target (380 words) and is
    enforced programmatically afterwards as a hard guarantee, since LLMs
    consistently overshoot word counts regardless of how the instruction is phrased.
    """
    source_titles = [page["title"] for page in pages]
    titles_str = ", ".join(f'"{t}"' for t in source_titles)

    # Combine all retrieved source text with clear attribution headers
    source_material = ""
    for i, page in enumerate(pages, 1):
        source_material += (
            f"--- Source {i}: {page['title']} ---\n"
            f"URL: {page['url']}\n"
            f"{page['content']}\n\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior market intelligence analyst preparing a "
         "professional industry report for a corporate strategy team.\n\n"
         "Generate a concise, executive-ready industry report that "
         "strictly follows the rules below.\n\n"
         "================================ INPUT\n"
         "You will receive the industry name and extracted text from "
         "Wikipedia pages. Available sources: {source_titles}.\n"
         "You MUST base your report ONLY on those sources.\n"
         "Do NOT use outside knowledge. Do NOT invent facts. Do NOT "
         "add statistics not present in the provided material.\n"
         "If information is missing, state: 'Data not available in "
         "retrieved sources.'\n\n"
         "================================ GROUNDING RULES\n"
         "Every factual claim, statistic, and figure MUST be directly "
         "traceable to one of the provided sources. Cite the source "
         "title in parentheses after each claim.\n\n"
         "GROUNDED (correct):\n"
         "'The semiconductor market was valued at US580 billion "
         "(Semiconductor industry).'\n\n"
         "HALLUCINATED (wrong):\n"
         "'The market is projected to reach US1 trillion by 2030.'\n"
         "^ This would be wrong if the figure does not appear in any "
         "of the provided source texts.\n\n"
         "If you cannot find a specific figure in the sources, write "
         "'Data not available in retrieved sources' rather than "
         "inventing a number. It is far better to admit missing data "
         "than to fabricate a statistic.\n\n"
         "================================ WORD LIMIT\n"
         "STRICT MAXIMUM: {max_words} prose words. Target: {max_words_target} words.\n"
         "Count every word in every prose section EXCEPT the Key Data table.\n"
         "Write CONCISELY. Each section should be 2-4 sentences maximum.\n"
         "Do NOT pad. Do NOT repeat points made in earlier sections.\n"
         "The Key Data table does NOT count -- include it in full.\n\n"
         "================================ MANDATORY STRUCTURE\n"
         "Use markdown headings (##) followed by a NEWLINE, then the "
         "section body on the NEXT line. Never put heading and body "
         "text on the same line.\n\n"
         "CORRECT format example:\n"
         "## Executive Summary\n"
         "The industry is growing rapidly...\n\n"
         "WRONG format (do NOT do this):\n"
         "## Executive Summary The industry is growing rapidly...\n\n"
         "Use this exact section order:\n\n"
         "## Executive Summary\n"
         "Key insight, strategic implication, and recommendation in "
         "2-3 sentences.\n\n"
         "## Key Metrics\n"
         "Output EXACTLY 3 lines. Each line is ONE metric. "
         "Each line MUST end with a newline character before the next metric starts. "
         "Do NOT put multiple metrics on the same line.\n\n"
         "Line format: Label: Value\n\n"
         "CORRECT (3 separate lines, each on its own line):\n"
         "Global Market Size: USD 1.5 trillion\n"
         "Annual Growth Rate: 8.2% CAGR\n"
         "Market Concentration: Top 5 firms hold 45% share\n\n"
         "WRONG (all on one line -- do NOT do this):\n"
         "Global Market Size: USD 1.5 trillion Annual Growth Rate: 8.2% CAGR\n\n"
         "WRONG (with citations -- do NOT do this):\n"
         "Global Market Size: USD 1.5 trillion (Semiconductor industry)\n\n"
         "Rules: exactly 3 lines, no bullets, no numbering, no parentheses.\n\n"
         "## Industry Overview\n"
         "Definition, scope, and scale indicators.\n\n"
         "## Market Structure & Competitive Dynamics\n"
         "Key segments, major players, level of competition, "
         "differentiation factors, barriers to entry.\n\n"
         "## Growth Drivers\n"
         "Economic, technological, and behavioural factors.\n\n"
         "## Risks & Constraints\n"
         "Regulatory, structural, and operational risks.\n\n"
         "## Key Data\n"
         "Include a markdown table with AT LEAST 4 data rows (not counting "
         "the header) summarising the most decision-relevant quantitative "
         "figures found in the sources.\n"
         "STRICT FORMAT -- every row on its own line, pipe character at "
         "start and end of every line:\n"
         "| Metric | Value | Source |\n"
         "| --- | --- | --- |\n"
         "| Global market size | USD 580 billion | Semiconductor industry |\n"
         "| Annual growth rate | 8.2% CAGR | Semiconductor industry |\n"
         "| Leading market | United States (47% share) | Semiconductor industry |\n"
         "| Key players | TSMC, Samsung, Intel | Semiconductor industry |\n\n"
         "Rules: minimum 4 rows, each row on its own line, "
         "never put the whole table on one line.\n"
         "If fewer than 4 quantitative figures exist in the sources, "
         "use qualitative descriptors in the Value column instead.\n\n"
         "## Strategic Interpretation\n"
         "Explain what the findings mean for decision-makers. Do not "
         "repeat numbers. Interpret them.\n\n"
         "## Final Takeaway\n"
         "One strong concluding insight in 1-2 sentences.\n\n"
         "================================ ANALYTICAL STANDARDS\n"
         "The report must:\n"
         "- Synthesise information across sources\n"
         "- Compare information where possible\n"
         "- Highlight contradictions if present\n"
         "- Prioritise insight over description\n"
         "- Clearly distinguish facts vs interpretation\n"
         "- Cite sources inline using their titles\n\n"
         "================================ WRITING STYLE\n"
         "Tone: concise, analytical, objective, professional, confident.\n"
         "Avoid: fluff, generic phrases, marketing language, repetition.\n"
         "Each sentence must add value.\n"
         "CRITICAL: Never use dollar signs ($). Write currency as "
         "'US1.5 billion' or 'USD 1.5 billion', never '$1.5 billion'. "
         "Dollar signs cause rendering errors.\n\n"
         "================================ QUALITY CONTROL\n"
         "Before output, perform this self-check:\n"
         "1. Word count: under {max_words} words?\n"
         "2. Grounding: every number and statistic appears in the "
         "source material? Remove any that do not.\n"
         "3. Citations: every factual claim cites a source title?\n"
         "4. Structure: all mandatory sections present?\n"
         "5. No repetition: same point not made twice?\n"
         "6. Executive readability: a senior executive could scan "
         "this in 2 minutes and extract key insights?\n\n"
         "Do NOT include a word count line at the end.\n"
         "Output ONLY the final report."),
        ("human",
         "Industry: **{industry}**\n\nSources:\n{sources}"),
    ])

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "industry": industry,
        "sources": source_material,
        "max_words": MAX_REPORT_WORDS,
        "max_words_target": REPORT_WORD_TARGET,
        "source_titles": titles_str,
    })

    # Strip any "Word count: X" line the LLM appends despite instructions
    lines = report.strip().split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith("word count")
        and not line.strip().lower().startswith("*word count")
    ]
    report = "\n".join(cleaned_lines).strip()

    report = sanitise_for_streamlit(report)

    # Programmatic safety net -- runs every time regardless of word count.
    # The prompt targets 380 words so the LLM should land well under 500;
    # enforce_word_limit handles the rare cases where it overshoots.
    report = enforce_word_limit(report, HARD_WORD_LIMIT)

    return report


# --------------------------------------------------------------
# SESSION STATE -- manages the multi-step flow
# --------------------------------------------------------------

def init_session_state():
    """Initialise session state variables on the first run.

    Streamlit reruns the entire script on every user interaction. Session
    state persists values across reruns so the pipeline remembers which
    step it is on, what it has retrieved, and what report it has generated.
    """
    defaults = {
        "current_step": 1,
        "industry_input": "",
        "validated_industry": "",
        "wiki_pages": [],
        "report": "",
        "search_queries": [],
        "report_model": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline():
    """Clear all pipeline state to start a fresh query.

    Called when the user submits a new industry name. Clearing stale
    results ensures the UI never shows a mismatch between the current
    input and previously displayed sources or report.
    """
    st.session_state.current_step = 1
    st.session_state.validated_industry = ""
    st.session_state.wiki_pages = []
    st.session_state.report = ""
    st.session_state.search_queries = []


# --------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------

def inject_custom_css():
    """Apply custom CSS to create a professional, scannable report layout.

    Streamlit's default styling gives no visual hierarchy between sections.
    Distinct treatments for the executive summary (callout box), key metrics
    (highlighted data box), data table (navy header), and conclusion (dark
    banner) let a reader scan and extract the most important points quickly --
    which matters when the target audience is time-pressured decision-makers.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

    /* -- Global Typography -- */
    .report-section, .insight-callout, .takeaway-box,
    .kpi-card, .source-card, .report-header {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* -- Report Section Cards -- */
    .report-section {
        background: #FFFFFF;
        border-left: 4px solid #003A70;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        line-height: 1.6;
        color: #333333;
        font-size: 15px;
    }
    .report-section h3 {
        color: #003A70;
        margin-top: 0;
        font-size: 20px;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* -- Executive Summary -- Insight Callout -- */
    .insight-callout {
        background: linear-gradient(135deg, #F0F6FC 0%, #E8F0FE 100%);
        border-left: 5px solid #003A70;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 6px rgba(0,58,112,0.08);
        line-height: 1.7;
        color: #1A1A2E;
        font-size: 15px;
    }
    .insight-callout .callout-label {
        font-weight: 700;
        color: #003A70;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        display: block;
    }
    .insight-callout h3 {
        color: #003A70;
        margin-top: 0;
        font-size: 22px;
        font-weight: 700;
    }
    .insight-callout p {
        margin: 0.5rem 0 0 0;
        font-size: 15px;
    }

    /* -- Final Takeaway -- Conclusion Box -- */
    .takeaway-box {
        background: #003A70;
        color: #FFFFFF;
        padding: 1.4rem 1.6rem;
        margin: 1.5rem 0 1.2rem 0;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,58,112,0.2);
        line-height: 1.7;
        font-size: 15px;
    }
    .takeaway-box .callout-label {
        font-weight: 700;
        color: #80B8E3;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        display: block;
    }
    .takeaway-box h3 {
        color: #FFFFFF !important;
        margin-top: 0;
        font-size: 20px;
        font-weight: 700;
    }
    .takeaway-box p {
        color: rgba(255,255,255,0.92);
        margin: 0.5rem 0 0 0;
    }

    /* -- KPI Metrics Box -- */
    .kpi-box {
        background: #F0F6FC;
        border: 1px solid #C8DDEF;
        border-left: 5px solid #0085CA;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0 1.2rem 0;
    }
    .kpi-box .kpi-row {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }
    .kpi-box .kpi-item {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        font-size: 15px;
        color: #1A1A2E;
    }
    .kpi-box .kpi-label {
        font-weight: 600;
        color: #003A70;
        min-width: 0;
        white-space: nowrap;
    }
    .kpi-box .kpi-sep {
        color: #888;
        flex-shrink: 0;
    }
    .kpi-box .kpi-value {
        font-weight: 500;
        color: #222;
    }

    /* -- Styled Data Table -- */
    .mckinsey-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 14px;
    }
    .mckinsey-table thead th {
        background: #003A70;
        color: #FFFFFF;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.3px;
    }
    .mckinsey-table tbody td {
        padding: 9px 14px;
        border-bottom: 1px solid #E8ECF0;
        color: #333333;
    }
    .mckinsey-table tbody tr:nth-child(even) {
        background: #F8F9FB;
    }
    .mckinsey-table tbody tr:hover {
        background: #EEF3F8;
    }
    .table-source {
        font-size: 11px;
        color: #999999;
        font-style: italic;
        margin-top: 4px;
    }

    /* -- Source Cards -- */
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E0E4E8;
        border-radius: 8px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        font-size: 14px;
    }

    /* -- Report Header -- */
    .report-header {
        background: linear-gradient(135deg, #003A70 0%, #00578A 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .report-header h2 {
        color: white !important;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .report-header .subtitle {
        color: rgba(255,255,255,0.75);
        margin: 0.4rem 0 0 0;
        font-size: 14px;
        font-weight: 400;
    }
    .report-header .model-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.9);
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 12px;
        margin-top: 8px;
    }

    /* -- Source Citations Footer -- */
    .sources-footer {
        font-size: 12px;
        color: #888888;
        font-style: italic;
        line-height: 1.6;
        padding-top: 0.5rem;
    }
    .sources-footer a {
        color: #0085CA;
    }

    /* -- Related Industries -- */
    .related-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 2px solid #E8ECF0;
    }
    .related-section h4 {
        font-family: 'Playfair Display', 'Georgia', serif;
        color: #003A70;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .related-section .related-subtitle {
        color: #888;
        font-size: 13px;
        margin-bottom: 1rem;
    }
    .related-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    .related-chip {
        background: #EEF3F8;
        border: none;
        color: #444;
        border-radius: 4px;
        padding: 0.3rem 0.8rem;
        font-size: 13px;
        font-weight: 400;
        cursor: default;
        white-space: nowrap;
    }

    /* -- Upgraded Typography -- */
    .report-header h2 {
        font-family: 'Playfair Display', 'Georgia', serif !important;
        letter-spacing: -0.01em;
    }
    .report-section h3,
    .insight-callout h3,
    .takeaway-box h3 {
        font-family: 'Playfair Display', 'Georgia', serif !important;
    }
    .report-section,
    .insight-callout p,
    .takeaway-box p,
    .kpi-card,
    .source-card {
        font-family: 'DM Sans', 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> tuple[str, str]:
    """Render sidebar controls and return the selected model name and API key.

    The API key uses Streamlit's password input type so it is masked on screen
    and held only in session memory -- it is never written to disk or included
    in any log. It is passed directly to the Gemini API on each call and goes
    no further.
    """
    with st.sidebar:
        st.markdown("### Configuration")

        selected_model = st.selectbox(
            "Select Model",
            options=LLM_OPTIONS,
            index=0,
            help="Choose the Gemini model for validation and report generation.",
        )

        desc = LLM_DESCRIPTIONS.get(selected_model, "")
        if desc:
            st.caption(desc)

        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            placeholder="Enter your Google AI API key",
            help="Your key is not stored and is only used for this session.",
            key="api_key",
        )

        st.divider()
        st.caption(
            "Get a free API key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)"
        )

        st.divider()
        st.markdown("### Pipeline Status")
        step = st.session_state.get("current_step", 1)

        progress_map = {1: 0.0, 2: 0.33, 3: 0.66, 4: 1.0}
        st.progress(progress_map.get(step, 0.0))

        steps_info = [
            ("Industry Input", step >= 2),
            ("Source Retrieval", step >= 3),
            ("Report Generation", step >= 4),
        ]
        for i, (label, done) in enumerate(steps_info, 1):
            if done:
                st.markdown(f"~~Step {i}: {label}~~ :green[Done]")
            elif step == i:
                st.markdown(f"**Step {i}: {label}** :orange[In progress]")
            else:
                st.markdown(f"Step {i}: {label}")

    return selected_model, api_key


INDUSTRY_SUGGESTIONS = [
    "Renewable Energy", "Semiconductor Manufacturing", "Pharmaceutical Industry",
    "Fintech", "Automotive Industry", "Artificial Intelligence Industry",
    "E-commerce", "Aerospace and Defence", "Telecommunications", "Cybersecurity",
]


def render_step_1(llm, model_name: str = "", api_key: str = ""):
    """Step 1: Industry input and validation.

    Validation is LLM-powered so it handles informal names and abbreviations
    gracefully. Invalid inputs show a helpful recovery UI -- the reason the
    LLM rejected the input plus a clickable list of example industries -- so
    the user never hits a dead end.
    """
    st.header("Step 1: Enter an Industry")
    st.markdown(
        "Enter the name of an industry or economic sector to research. "
        "The assistant accepts standard names, abbreviations, and informal "
        "terms -- for example, *'pharma'*, *'AI'*, or *'renewables'*."
    )

    industry = st.text_input(
        "Industry name",
        placeholder="e.g. Renewable Energy, Semiconductor Manufacturing, Fintech",
        key="industry_text_input",
    )

    if st.button("Validate Industry", type="primary", disabled=not industry):
        reset_pipeline()
        st.session_state.industry_input = industry

        with st.spinner("Validating your input..."):
            try:
                # Use cached validation when model name and key are available
                if model_name and api_key:
                    result = _cached_validate_industry(model_name, api_key, industry)
                else:
                    result = validate_industry(llm, industry)
            except Exception as e:
                handle_api_error(e, "Validation")
                return

        if result["is_valid"]:
            normalised = result["normalised"] or industry
            st.session_state.validated_industry = normalised
            st.session_state.current_step = 2
            st.success(f"Recognised industry: **{normalised}**")
            if result["reason"]:
                st.caption(result["reason"])
            st.rerun()
        else:
            # Show a clear rejection message with the LLM's specific reason,
            # then offer clickable example industries so the user has an
            # immediate path forward rather than guessing what to type.
            st.warning(
                f"**'{industry}'** wasn't recognised as an industry or economic sector."
            )
            reason = result.get("reason", "")
            if reason:
                st.caption(f"Reason: {reason}")

            st.markdown(
                "Please enter a **specific industry name** -- for example an "
                "economic sector, market, or technology vertical. "
                "Company names, product names, and general words are not accepted."
            )

            st.markdown(
                "**Examples you could try:** "
                + ", ".join(f"*{s}*" for s in INDUSTRY_SUGGESTIONS)
            )

            if st.button("<- Try again", type="secondary"):
                st.rerun()

    elif not industry:
        st.info("Enter an industry name above to get started.")


def render_step_2(llm, model_name: str = "", api_key: str = ""):
    """Step 2: Retrieve and display 5 most relevant Wikipedia sources."""
    industry = st.session_state.validated_industry

    st.header("Step 2: Relevant Wikipedia Sources")

    if not st.session_state.wiki_pages:
        with st.spinner("Generating search queries and retrieving sources..."):
            try:
                # Use cached query generation when possible, then retrieve in parallel
                if model_name and api_key:
                    queries = _cached_generate_search_queries(model_name, api_key, industry)
                else:
                    queries = generate_search_queries(llm, industry)
                st.session_state.search_queries = queries

                raw_pages = retrieve_wikipedia_pages(industry, queries)

                # If parallel retrieval returns nothing, fall back to a direct
                # search on just the industry name -- handles cases where LLM-
                # generated queries are too specific and miss the main article.
                if not raw_pages:
                    fallback_pages = retrieve_wikipedia_pages(industry, [industry])
                    if fallback_pages:
                        raw_pages = fallback_pages
                    else:
                        st.error(
                            "No Wikipedia pages found for this industry. "
                            "Try a broader or more common industry name."
                        )
                        return

                # Remove stubs and disambiguation pages before ranking
                raw_pages = filter_low_quality_pages(raw_pages)

                if len(raw_pages) < FINAL_SOURCE_COUNT:
                    st.info(
                        f"Found {len(raw_pages)} relevant page(s) "
                        f"(fewer than the ideal {FINAL_SOURCE_COUNT}). "
                        f"The report may be less comprehensive."
                    )

                # LLM reranking: filter the quality-checked pool to the best five
                top_pages = select_top_pages(llm, industry, raw_pages)
                st.session_state.wiki_pages = top_pages

            except Exception as e:
                handle_api_error(e, "Retrieval")
                return

    if st.session_state.search_queries:
        with st.expander("Search strategy", expanded=False):
            st.markdown("Queries used to find sources:")
            for q in st.session_state.search_queries:
                st.markdown(f"- *{q}*")
            st.caption(
                f"Retrieved {len(st.session_state.wiki_pages)} most relevant "
                f"pages from initial candidate pool."
            )

    pages = st.session_state.wiki_pages
    st.markdown(f"**Top {len(pages)} sources selected by relevance:**")

    for i, page in enumerate(pages, 1):
        snippet = page["content"][:120].replace("\n", " ")
        st.markdown(
            f'<div class="source-card">'
            f'<strong>{i}. <a href="{page["url"]}" target="_blank">'
            f'{page["title"]}</a></strong><br>'
            f'<small style="color:#666">{snippet}...</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    diversity = check_source_diversity(pages)
    if not diversity["is_diverse"]:
        st.warning(diversity["warning"])
    else:
        st.caption(
            f"Source diversity: good (avg overlap: "
            f"{diversity['avg_overlap']:.0%})"
        )

    st.divider()

    if st.button("Generate Industry Report", type="primary"):
        st.session_state.current_step = 3
        st.rerun()


@st.cache_data(show_spinner=False, ttl=600)
def fetch_industry_image(industry: str, page_titles: tuple | None = None) -> str | None:
    """Fetch a relevant thumbnail image URL from Wikipedia for the report header.

    Tries the industry name first, then falls back through the retrieved page
    titles until an image is found. SVG files are skipped as they are typically
    logos or icons that render poorly as header images. Cached for 10 minutes
    to avoid repeated network calls on Streamlit reruns. page_titles is a tuple
    rather than a list because st.cache_data requires hashable arguments.
    """
    import urllib.parse
    import requests

    titles_to_try = [industry]
    if page_titles:
        titles_to_try.extend(page_titles)

    for title in titles_to_try:
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php?"
                "action=query&format=json&prop=pageimages&piprop=original"
                f"&titles={urllib.parse.quote(title)}"
                "&redirects=1"
            )
            resp = requests.get(search_url, timeout=3)
            data = resp.json()
            api_pages = data.get("query", {}).get("pages", {})
            for page in api_pages.values():
                img = page.get("original", {}).get("source")
                if img:
                    if not img.lower().endswith(".svg"):  # SVGs render poorly as headers
                        return img
        except Exception:
            continue
    return None


def parse_markdown_table(text: str) -> list[list[str]] | None:
    """Extract a markdown table from text and return it as a list of rows.

    Handles two formatting edge cases that LLMs produce:
    1. Normal: each row on its own line starting and ending with '|'
    2. Single-line: the entire table concatenated onto one line -- detected
       by a high pipe count combined with '---' separator markers.

    Returns None if no valid table (at least header + one data row) is found.
    """
    if "|" in text:
        fixed_lines = []
        for line in text.strip().split("\n"):
            stripped = line.strip()
            if stripped.count("|") >= 6 and "---" in stripped:
                parts = re.split(r'\|\s*\|', stripped)
                if len(parts) > 2:
                    rows_text = []
                    for part in parts:
                        clean_part = part.strip().strip("|").strip()
                        if clean_part:
                            rows_text.append(f"| {clean_part} |")
                    fixed_lines.extend(rows_text)
                else:
                    fixed_lines.append(stripped)
            else:
                fixed_lines.append(stripped)
        text = "\n".join(fixed_lines)

    lines = text.strip().split("\n")
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            break

    if len(table_lines) < 2:
        return None

    rows = []
    for i, line in enumerate(table_lines):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c.strip()) <= set("-: ") for c in cells):  # skip separator rows
            continue
        rows.append(cells)

    return rows if len(rows) >= 2 else None


def extract_table_from_body(body: str) -> tuple[str, list[list[str]] | None, str]:
    """Split section body text into pre-table text, table data, and post-table text.

    Returns (pre_table_text, table_rows_or_None, post_table_text).
    If no table is found, returns (body, None, "").

    Handles both normal pipe-row tables and inline table fragments where
    pipes appear mid-paragraph rather than at line boundaries.
    """
    table_data = parse_markdown_table(body)
    if not table_data:
        return body, None, ""

    lines = body.split("\n")
    pre_lines = []
    post_lines = []
    in_table = False
    table_ended = False

    for line in lines:
        stripped = line.strip()
        is_pipe_row = stripped.startswith("|") and stripped.endswith("|")
        is_inline_table = (
            not is_pipe_row
            and stripped.count("|") >= 4
            and "---" in stripped
        )
        if (is_pipe_row or is_inline_table) and not table_ended:
            in_table = True
            continue
        if in_table and not is_pipe_row and not is_inline_table:
            table_ended = True
            in_table = False
        if not in_table and not table_ended:
            pre_lines.append(line)
        elif table_ended:
            post_lines.append(line)

    return (
        "\n".join(pre_lines).strip(),
        table_data,
        "\n".join(post_lines).strip(),
    )


def render_kpi_cards(body: str):
    """Render Key Metrics section as a single styled box with labelled rows.

    Simpler and more robust than individual cards: all metrics go into one
    box, so partial parses still render something useful. Handles any format
    the LLM produces -- one metric per line, run-on single line, bulleted,
    or numbered -- by normalising the body into candidate lines first.
    """
    # Normalise body: collapse to single string then split on metric boundaries.
    # Handles both run-on single lines and normal newline-separated metrics.
    # Strategy: find all "Label: Value" pairs using a broad pattern, regardless
    # of whether they are on separate lines or jammed together.
    normalised = re.sub(r"\s+", " ", body.replace("\n", " ")).strip()

    SKIP_PREFIXES = (
        "extract", "strict format", "correct", "wrong", "rules",
        "note", "source", "http", "label", "value", "follow",
        "output", "line format", "do not",
    )

    # Extract all Label: Value pairs from the normalised string.
    # Pattern: a label (1-8 words, no colon) followed by ': ' and a value
    # (everything up to the next label pattern or end of string).
    # This works whether metrics are on separate lines or run together.
    pair_pattern = re.compile(
        r'([A-Z][^:]{2,50}?)'          # Label: starts with capital, 3-50 chars
        r':\s*'                          # colon separator
        r'([^:]{3,120}?)'               # Value: 3-120 chars (non-greedy)
        r'(?=\s+[A-Z][^:]{2,50}:|$)',   # lookahead: next label or end
    )

    metrics = []
    for match in pair_pattern.finditer(normalised):
        label = match.group(1).strip().strip("*").strip("#")
        value = match.group(2).strip().strip("*")
        # Strip trailing parenthetical citations
        value = re.sub(r"\s*\([^)]{0,80}\)\s*$", "", value).strip()
        # Skip instruction-echo lines
        if any(label.lower().startswith(p) for p in SKIP_PREFIXES):
            continue
        if label and value and len(label.split()) <= 8:
            metrics.append((label, value))
        if len(metrics) == 5:
            break

    # If still empty, fall back to plain markdown
    if not metrics:
        st.markdown(sanitise_for_streamlit(body))
        return

    # Render as a single styled box -- one row per metric
    rows_html = ""
    for label, value in metrics[:5]:  # cap at 5 in case of over-parsing
        safe_label = sanitise_for_streamlit(label)
        safe_value = sanitise_for_streamlit(value)
        rows_html += (
            f'<div class="kpi-item">'
            f'<span class="kpi-label">{safe_label}</span>'
            f'<span class="kpi-sep">--</span>'
            f'<span class="kpi-value">{safe_value}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="kpi-box"><div class="kpi-row">{rows_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_styled_table(table_data: list[list[str]]):
    """Render a parsed table as styled HTML with a navy header and alternating rows.

    Rendered as HTML rather than via st.dataframe() because Streamlit's
    markdown table support is inconsistent -- pipe tables sometimes render
    correctly and sometimes do not, depending on surrounding content.
    """
    if not table_data or len(table_data) < 2:
        return

    headers = table_data[0]
    num_cols = len(headers)

    html = '<table class="mckinsey-table"><thead><tr>'
    for h in headers:
        html += f'<th>{sanitise_for_streamlit(h)}</th>'
    html += '</tr></thead><tbody>'

    for row in table_data[1:]:
        while len(row) < num_cols:
            row.append("")
        row = row[:num_cols]
        html += '<tr>'
        for cell in row:
            html += f'<td>{sanitise_for_streamlit(cell)}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    html += '<div class="table-source">Source: Wikipedia (retrieved data)</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_report_section(heading: str, body: str):
    """Render a single report section with visual treatment matched to its role.

    Different sections carry different cognitive weight for the reader:
    - Executive Summary: first thing a reader scans -- gets a highlighted callout
    - Key Metrics: quantitative anchor points -- gets large KPI number cards
    - Final Takeaway: the conclusion -- gets a bold navy box for emphasis
    - Key Data: tabular data -- rendered as styled HTML table
    - All others: standard card with left border for visual grouping
    """
    clean_heading = heading.strip().strip("#").strip("*").strip()

    with st.container():

        if clean_heading == "Executive Summary":
            safe_body = sanitise_for_streamlit(body) if body else ""
            st.markdown(
                f'<div class="insight-callout">'
                f'<span class="callout-label">Key Insight</span>'
                f'<h3>{clean_heading}</h3>'
                f'<p>{safe_body}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

        if clean_heading == "Key Metrics":
            st.markdown(
                f'<div class="report-section"><h3>{clean_heading}</h3></div>',
                unsafe_allow_html=True,
            )
            if body:
                render_kpi_cards(body)
            return

        if clean_heading == "Final Takeaway":
            safe_body = sanitise_for_streamlit(body) if body else ""
            st.markdown(
                f'<div class="takeaway-box">'
                f'<span class="callout-label">Bottom Line</span>'
                f'<h3>{clean_heading}</h3>'
                f'<p>{safe_body}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return

        # For Key Data: check table exists BEFORE rendering the heading.
        # If the table is missing (truncated or not generated), skip the
        # entire section silently rather than showing an empty header.
        if clean_heading == "Key Data":
            if not body:
                return
            pre_text, table_data, post_text = extract_table_from_body(body)
            if not table_data:
                return  # No table -- hide the section entirely
            st.markdown(
                f'<div class="report-section"><h3>{clean_heading}</h3></div>',
                unsafe_allow_html=True,
            )
            if pre_text:
                st.markdown(sanitise_for_streamlit(pre_text))
            render_styled_table(table_data)
            if post_text:
                st.markdown(sanitise_for_streamlit(post_text))
            return

        st.markdown(
            f'<div class="report-section">'
            f'<h3>{clean_heading}</h3></div>',
            unsafe_allow_html=True,
        )

        if not body:
            return

        pre_text, table_data, post_text = extract_table_from_body(body)

        if table_data:
            if pre_text:
                st.markdown(sanitise_for_streamlit(pre_text))
            render_styled_table(table_data)
            if post_text:
                st.markdown(sanitise_for_streamlit(post_text))
        else:
            st.markdown(sanitise_for_streamlit(body))


def sanitise_for_pdf(text: str) -> str:
    """Replace Unicode characters that fall outside the latin-1 character set.

    fpdf2's built-in Helvetica font only covers latin-1. Wikipedia content
    frequently contains em-dashes, smart quotes, and non-breaking spaces that
    trigger UnicodeEncodeError during PDF rendering. Common offenders are mapped
    to safe equivalents; anything remaining outside latin-1 is replaced with a
    fallback character.
    """
    replacements = {
        "\u2013": "-",    # en-dash
        "\u2014": " - ",  # em-dash
        "\u2015": " - ",  # horizontal bar
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote (apostrophe)
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
        "\u2010": "-",    # hyphen
        "\u2011": "-",    # non-breaking hyphen
        "\u2012": "-",    # figure dash
        "\u00b7": " ",    # middle dot
        "\u2022": "-",    # bullet
        "\u2023": "-",    # triangular bullet
        "\u25cf": "-",    # black circle
        "\u00d7": "x",    # multiplication sign
        "\u2264": "<=",   # less than or equal
        "\u2265": ">=",   # greater than or equal
        "\u00b1": "+/-",  # plus-minus sign
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    text = text.replace("**", "")
    return text


def generate_pdf(industry: str, report: str, pages: list[dict]) -> bytes:
    """Generate a formatted PDF of the industry report for download.

    Sections are parsed and rendered individually so headings get bold navy
    styling and tables are drawn with borders. All text passes through
    sanitise_for_pdf() before rendering to prevent encoding errors from
    non-latin-1 characters in Wikipedia content. fpdf2 was chosen over
    reportlab for its simpler API and smaller dependency footprint.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    from datetime import date
    today_str = date.today().strftime("%B %d, %Y")

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 58, 112)
    pdf.cell(
        0, 14, txt=sanitise_for_pdf(industry),
        new_x="LMARGIN", new_y="NEXT", align="C",
    )

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        txt=f"Market Intelligence Report  |  {today_str}",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )

    pdf.ln(4)
    pdf.set_draw_color(0, 58, 112)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    sections = split_report_into_sections(report)
    for heading, body in sections:

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(0, 58, 112)
        pdf.cell(
            0, 10, txt=sanitise_for_pdf(heading),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(1)

        if not body:
            continue

        pre_text, table_data, post_text = extract_table_from_body(body)

        if table_data:
            if pre_text:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 6, txt=sanitise_for_pdf(pre_text))
                pdf.ln(2)

            headers = table_data[0]
            data_rows = table_data[1:]
            num_cols = len(headers)
            usable_width = 170
            col_width = usable_width / num_cols

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(0, 58, 112)
            pdf.set_text_color(255, 255, 255)
            for header in headers:
                pdf.cell(
                    col_width, 7,
                    txt=sanitise_for_pdf(header)[:30],
                    border=1, fill=True, align="C",
                )
            pdf.ln()

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for row_idx, row in enumerate(data_rows):
                if row_idx % 2 == 0:
                    pdf.set_fill_color(245, 245, 245)
                else:
                    pdf.set_fill_color(255, 255, 255)
                for j, cell in enumerate(row):
                    cell_text = sanitise_for_pdf(cell)[:30] if j < num_cols else ""
                    pdf.cell(
                        col_width, 7, txt=cell_text,
                        border=1, fill=True, align="C",
                    )
                pdf.ln()

            pdf.ln(3)

            if post_text:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 6, txt=sanitise_for_pdf(post_text))
                pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 6, txt=sanitise_for_pdf(body))
            pdf.ln(3)

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 58, 112)
    pdf.cell(0, 10, txt="References", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for i, page in enumerate(pages, 1):
        ref_text = sanitise_for_pdf(f"[{i}] {page['title']} - {page['url']}")
        pdf.multi_cell(0, 5, txt=ref_text)
        pdf.ln(1)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(
        0, 5,
        txt="Generated by Market Research Assistant | Data sourced from Wikipedia",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )

    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)


def render_related_industries(llm, industry: str):
    """Render a row of clickable chip buttons for related industries.

    Placing related industries at the end of the report encourages the user
    to explore adjacent sectors rather than stopping at a single data point.
    Each chip is a Streamlit button -- clicking it pre-fills the industry
    input and resets the pipeline, turning the suggestion into a one-click
    follow-on search. Related industries are generated once and stored in
    session state so re-renders do not trigger additional LLM calls.
    """
    if "related_industries" not in st.session_state or \
       st.session_state.get("related_for") != industry:
        with st.spinner("Finding related industries..."):
            try:
                related = generate_related_industries(llm, industry)
                st.session_state.related_industries = related
                st.session_state.related_for = industry
            except Exception:
                st.session_state.related_industries = []
                st.session_state.related_for = industry

    related = st.session_state.get("related_industries", [])
    if not related:
        return

    st.markdown(
        '<div class="related-section">'
        '<h4>Related Industries</h4>'
        '<p class="related-subtitle">Type one of these into the search box above to generate a new report</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Display related industries as plain styled text chips -- no buttons,
    # which avoids Streamlit rerun conflicts. User reads them and types
    # their chosen industry into the search box above.
    chips_html = '<div class="related-grid">'
    for item in related:
        chips_html += f'<span class="related-chip">{item}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_step_3(llm, model_name: str = ""):
    """Step 3: Generate the industry report and render it section by section.

    Each section is rendered individually rather than as a single markdown block.
    This gives precise control over the visual treatment of each section type
    and avoids relying on Streamlit's markdown renderer to handle pipe tables
    and nested formatting consistently.
    """
    industry = st.session_state.validated_industry
    pages = st.session_state.wiki_pages

    st.header("Step 3: Industry Report")

    if not st.session_state.report:
        with st.spinner("Generating your industry report..."):
            try:
                report = generate_report(llm, industry, pages)
                st.session_state.report = report
                st.session_state.report_model = model_name
                st.session_state.current_step = 4
            except Exception as e:
                handle_api_error(e, "Report generation")
                return

    report = st.session_state.report
    used_model = st.session_state.get("report_model", model_name)

    from datetime import date
    today = date.today().strftime("%B %d, %Y")

    page_titles = tuple(p["title"] for p in pages)
    img_url = fetch_industry_image(industry, page_titles)

    model_badge = (
        f'<span class="model-badge">Generated with {used_model}</span>'
        if used_model else ""
    )
    st.markdown(
        f'<div class="report-header">'
        f'<h2>{industry}</h2>'
        f'<p class="subtitle">Market Intelligence Report &nbsp;|&nbsp; {today}</p>'
        f'{model_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if img_url:
        st.image(img_url, use_container_width=True)

    sections = split_report_into_sections(report)
    for heading, body in sections:
        render_report_section(heading, body)

    wc = count_words(report)
    if wc <= HARD_WORD_LIMIT:
        st.success(f"Prose word count: {wc} / {HARD_WORD_LIMIT} (Key Data table excluded)")
    else:
        st.error(f"Prose word count: {wc} / {HARD_WORD_LIMIT} -- over limit")

    st.markdown("---")
    sources_html = '<div class="sources-footer"><strong>References</strong><br>'
    for i, page in enumerate(st.session_state.wiki_pages, 1):
        title = page.get("title", "Unknown")
        url = page.get("url", "")
        sources_html += (
            f'[{i}] {title} -- '
            f'<a href="{url}" target="_blank">Wikipedia</a><br>'
        )
    sources_html += (
        f'<br><em>Source: Wikipedia | Analysis generated by Market Research '
        f'Assistant | {today}</em></div>'
    )
    st.markdown(sources_html, unsafe_allow_html=True)

    render_related_industries(llm, industry)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Research a different industry"):
            reset_pipeline()
            st.rerun()
    with col2:
        try:
            pdf_bytes = generate_pdf(industry, report, pages)
            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name=f"{industry.lower().replace(' ', '_')}_report.pdf",
                mime="application/pdf",
            )
        except Exception:
            # Fall back to plain text if PDF generation fails
            st.download_button(
                label="Download report (text)",
                data=report,
                file_name=f"{industry.lower().replace(' ', '_')}_report.txt",
                mime="text/plain",
            )
            st.caption("PDF generation failed -- text version provided.")


# --------------------------------------------------------------
# MAIN -- application entry point
# --------------------------------------------------------------

def main():
    """Entry point. Initialises state and routes the user through the pipeline.

    Streamlit reruns this entire script on every user interaction. Session
    state preserves the current step, validated input, retrieved pages, and
    generated report across reruns. Each pipeline step checks session state
    before making any API calls, so completed steps are not re-executed.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
    )

    init_session_state()
    inject_custom_css()

    st.title(APP_TITLE)
    st.caption(
        "AI-powered industry analysis from Wikipedia sources  |  "
        "Built with LangChain & Google Gemini"
    )

    selected_model, api_key = render_sidebar()

    if not api_key:
        st.info(
            "Select a model and enter your Google AI API key in the sidebar "
            "to begin."
        )
        return

    llm = initialise_llm(selected_model, api_key)

    # Steps remain visible once reached so the user can review earlier stages
    step = st.session_state.current_step

    render_step_1(llm, model_name=selected_model, api_key=api_key)
    if step >= 2:
        st.divider()
        render_step_2(llm, model_name=selected_model, api_key=api_key)
    if step >= 3:
        st.divider()
        render_step_3(llm, selected_model)


if __name__ == "__main__":
    main()
