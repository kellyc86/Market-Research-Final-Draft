"""
Market Research Assistant
=========================
A Streamlit RAG application that generates structured industry reports
from Wikipedia sources in three stages:

    1. Validate the user's industry input via LLM
    2. Retrieve and filter Wikipedia pages via multi-query search
    3. Generate a grounded, cited report under 500 words

Retrieval uses a broad-then-filter approach: the LLM generates several
Wikipedia-style queries covering different aspects of the industry, the
retriever casts a wide net across all of them, and a second LLM pass
reranks and selects the five most relevant pages. This reduces the risk
of missing important sub-topics that a single query would overlook.

Grounding is enforced through prompt constraints and low temperature (0.2),
which pushes the model toward deterministic, source-faithful completions.
Temperature alone does not eliminate hallucination -- it only shifts the
probability distribution toward more conservative outputs. A programmatic
word-limit check runs after generation as a hard backstop, since LLMs are
unreliable at self-counting tokens.
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
MAX_REPORT_WORDS = 500             # Hard ceiling aligned with assignment requirement
REPORT_WORD_TARGET = 430           # Soft target: leaves buffer below limit for safety
HARD_WORD_LIMIT = 500              # Enforced programmatically after generation
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
    """Instantiate the LangChain Gemini model for a given selection and key.

    Temperature 0.2 was chosen deliberately: lower values produce more
    consistent, factual completions, which matters here because the report
    must stay grounded in retrieved sources. A higher temperature risks
    the model generating plausible-sounding but unsupported statistics.
    """
    model_id = LLM_MODEL_MAP[model_name]
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_validate_industry(model_name: str, api_key: str, user_input: str) -> dict:
    """Cached wrapper around industry validation.

    Validation makes an LLM call for every submission, even if the user
    types the same industry twice in a session. Caching by (model, key, input)
    means repeat submissions -- common when users tweak capitalisation or
    re-run -- hit the cache instantly instead of spending 1-2 seconds on
    an API round-trip. TTL of one hour prevents stale results across
    very long sessions.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_MAP[model_name],
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )
    return validate_industry(llm, user_input)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_generate_search_queries(model_name: str, api_key: str, industry: str) -> list[str]:
    """Cached wrapper around query generation.

    The same industry name always produces the same set of five queries --
    the output is deterministic at temperature 0.2. Caching this call
    saves another 1-2 seconds on re-runs and on the common case where
    the user goes back and re-generates after reading the sources.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_MAP[model_name],
        google_api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
    )
    return generate_search_queries(llm, industry)


def validate_industry(llm, user_input: str) -> dict:
    """Use the LLM to determine whether the user's input names a real industry.

    Simple string matching would miss abbreviations, informal names, and
    misspellings. LLM-based validation handles these naturally -- 'pharma'
    normalises to 'Pharmaceutical Industry', 'AI' to 'Artificial Intelligence
    Industry' -- while still rejecting company names, person names, and
    freeform text.

    Returns a dict: is_valid (bool), normalised name (str), reason (str).
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
    """Generate multiple Wikipedia search queries to broaden source coverage.

    A single query (e.g. 'renewable energy') tends to surface the same
    high-level overview page repeatedly. Prompting the LLM to produce
    queries targeting distinct aspects -- market size, regulation, key
    companies, technology -- retrieves a more diverse candidate pool.
    This mirrors the intuition behind ensemble methods: combining varied
    weak signals produces a stronger result than any single signal alone.
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
    """Fetch Wikipedia pages for a single query. Runs inside a thread.

    Isolated into its own function so ThreadPoolExecutor can call it
    independently per query. Returns an empty list on any failure so
    a single bad query does not interrupt the other concurrent fetches.
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
    """Retrieve Wikipedia pages for all queries in parallel, deduplicated by title.

    Sequential retrieval (one query at a time) is the main bottleneck in the
    pipeline -- each Wikipedia API call takes 1-3 seconds and they have no
    dependency on each other. Running them concurrently with a thread pool
    cuts retrieval time by roughly 4-5x for five queries, since the wall-clock
    time is determined by the slowest single request rather than the sum of all.

    Threads are used rather than async because WikipediaRetriever is a
    synchronous blocking call and does not expose an async interface.
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
    """Use the LLM to rank retrieved pages and select the five most relevant.

    Without this filtering step, broad retrieval often returns tangentially
    related pages -- a founder biography, a geographic region article -- that
    would dilute the report. The LLM sees each page's title and a short snippet,
    then returns the indices of the five pages most useful for a market research
    report. This is an LLM-as-reranker pattern: cheap to run, but meaningfully
    improves the signal-to-noise ratio of what reaches the generation step.
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
    """Remove Wikipedia stub pages and disambiguation pages before LLM ranking.

    Two failure modes degrade report quality without this filter:
    1. Stub pages -- Wikipedia articles under ~300 words that contain
       almost no substantive content. Passing these to the LLM ranker
       wastes ranking capacity on pages that would be useless as sources.
    2. Disambiguation pages -- pages that only list alternative meanings
       of a term. These contain no industry content at all and are
       identifiable by the presence of 'may refer to' in the opening text.

    The minimum length threshold (1500 characters) is deliberately
    conservative. A page this short is almost certainly a stub or a
    very narrow sub-article that adds little to a broad industry report.
    Raising the threshold would risk discarding legitimately concise but
    useful overview articles.
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
    """Measure content overlap between retrieved pages using Jaccard similarity.

    Jaccard similarity between two sets A and B is |A intersection B| / |A union B|.
    Applied to word sets, it flags when pages share too much vocabulary --
    a sign that multiple sources cover the same narrow sub-topic rather than
    providing complementary perspectives. High overlap tends to produce
    shallow, repetitive reports. The 0.4 threshold was chosen empirically:
    Wikipedia pages on the same industry typically share 15-25% vocabulary
    through common terminology; above 40% suggests near-duplicate coverage.
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
    """Generate a list of related industries for deeper research suggestions.

    A good analyst does not stop at a single industry in isolation -- adjacent
    sectors often explain demand dynamics, supply chain dependencies, or
    competitive threats that shape the focal industry. For example, researching
    Electric Vehicles naturally connects to Battery Manufacturing, Semiconductor
    Industry, and Renewable Energy. Surfacing these connections helps the user
    build a more complete picture with follow-on searches.
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
    """Truncate generated text to the hard word limit at the nearest sentence boundary.

    LLMs routinely overshoot stated word limits -- the model predicts the next
    token without tracking an accurate running count. Truncating to the nearest
    full stop before the limit guarantees compliance without cutting mid-sentence.
    The sentence-boundary check only applies when that boundary falls past the
    halfway point, preventing excessive content loss from an early full stop.
    """
    words = text.split()
    if len(words) <= limit:
        return text

    truncated = " ".join(words[:limit])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.5:
        truncated = truncated[:last_period + 1]

    return truncated


def count_words(text: str) -> int:
    """Count substantive words in a report, stripping markdown formatting symbols.

    Heading markers (##), bold/italic markers (* **), pipe characters (|),
    and separator rows (---) are removed so the count reflects actual prose
    content rather than formatting artefacts.
    """
    clean = re.sub(r"[#*|]", "", text)
    clean = re.sub(r"-{3,}", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return len(clean.split()) if clean else 0


def sanitise_for_streamlit(text: str) -> str:
    """Remove characters that Streamlit's markdown renderer interprets as LaTeX.

    Streamlit treats $...$ and $$...$$ as inline and block math delimiters.
    Wikipedia content and LLM outputs occasionally contain dollar signs and
    LaTeX-style backslash sequences that trigger this, producing garbled
    mixed-font output. Stripping them keeps the rendered report clean without
    changing the meaning of the text.
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
    """Split a report string into (heading, body) tuples.

    LLMs are inconsistent with heading formatting -- the same model may output
    '## Executive Summary', '**Executive Summary**', or plain 'Executive Summary'
    across different runs. Rather than relying on a specific format, this function
    searches for the known heading label strings and uses their positions in the
    text to extract body content. This makes the parser format-agnostic and
    robust to prompt variation. If no recognised headings are found, the full
    report is returned as a single section.
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
    """Generate a structured industry report grounded in retrieved Wikipedia sources.

    The prompt balances two competing constraints: enforcing a fixed section
    structure while keeping every claim grounded in the retrieved sources.
    Good/bad examples in the prompt demonstrate what grounded output looks like
    rather than just describing the rule -- a few-shot approach that is more
    reliable than instruction alone.

    The word limit appears twice: as a soft instruction in the prompt and as a
    hard programmatic check after generation. The programmatic check is the
    reliable one. Dollar signs are prohibited in the prompt because Streamlit
    renders $...$ as LaTeX math -- this is also caught by sanitise_for_streamlit()
    as a fallback.
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
         "Maximum: {max_words} words. Target range: {max_words_target}-{max_words} words.\n"
         "If output exceeds {max_words} words, rewrite more concisely.\n\n"
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
         "Extract exactly 3 key quantitative metrics from the sources.\n"
         "STRICT FORMAT -- each metric on its own line, nothing else on that line:\n"
         "LABEL: value\n\n"
         "CORRECT example (follow this exactly):\n"
         "Global Market Size: USD 1.5 trillion\n"
         "Annual Growth Rate: 8.2% CAGR\n"
         "Market Concentration: Top 5 firms hold 45% share\n\n"
         "WRONG (do NOT add source citations or extra text on these lines):\n"
         "Global Market Size: USD 1.5 trillion (Semiconductor industry)\n\n"
         "Rules: exactly 3 lines, each line is ONLY 'Label: Value', "
         "no bullets, no numbering, no parentheses, no extra words.\n\n"
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

    # Hard word limit enforced after generation -- the prompt alone is not sufficient
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
    """Inject custom CSS to style the report output professionally.

    Streamlit's default styling is functional but generic. Custom CSS creates
    visual hierarchy that makes the report easier to scan -- distinct treatments
    for the executive summary, KPI cards, data table, and conclusion help a
    reader quickly locate what matters.
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

    /* -- KPI Metric Cards -- */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .kpi-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E0E4E8;
        border-top: 4px solid #0085CA;
        border-radius: 8px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .kpi-card .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #003A70;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .kpi-card .kpi-label {
        font-size: 12px;
        font-weight: 500;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
        background: #F0F6FC;
        border: 1.5px solid #003A70;
        color: #003A70;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        white-space: nowrap;
    }
    .related-chip:hover {
        background: #003A70;
        color: #FFFFFF;
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
    """Render the sidebar with model selection and API key entry.

    The API key is collected via a password-type field (masked input) and
    held only in Streamlit's in-memory session state. It is never written
    to disk or logged, and is only transmitted to the Google Gemini API
    when making LLM calls.
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
    """Render Key Metrics section as large KPI cards in a horizontal row.

    Parses 'LABEL: value' lines from the body. Handles bullet points,
    bold markers, numbered prefixes, and inline source citations that
    LLMs occasionally add despite prompt instructions.
    Always renders exactly 3 cards -- pads with 'N/A' if fewer than 3
    metrics are found so the layout stays consistent.
    Falls back to plain markdown only if no colon-separated lines exist at all.
    """
    # Skip instruction-style lines the LLM sometimes echoes back
    SKIP_PREFIXES = (
        "extract", "strict format", "correct example", "wrong", "rules",
        "note", "source", "http", "label:", "value:", "follow this",
    )

    metrics = []
    for line in body.strip().split("\n"):
        # Strip leading bullets, dashes, asterisks, and numbered prefixes
        line = re.sub(r"^\s*[-*\d]+[.)]*\s*", "", line)
        # Strip bold markers, hash markers, and surrounding whitespace
        line = line.strip().strip("*").strip("#").strip()
        if not line or ":" not in line:
            continue
        # Skip lines that look like prompt instructions echoed back
        if any(line.lower().startswith(p) for p in SKIP_PREFIXES):
            continue
        parts = line.split(":", 1)
        label = parts[0].strip().strip("*").strip()
        value = parts[1].strip().strip("*").strip()
        # Strip trailing parenthetical source citations e.g. "(Semiconductor industry)"
        value = re.sub(r"\s*\([^)]{0,80}\)\s*$", "", value).strip()
        # Label should be short (a metric name, not a sentence or URL)
        if (label
                and value
                and len(label) < 80
                and len(label.split()) <= 8
                and not label.lower().startswith("http")):
            metrics.append((label, value))

    if not metrics:
        st.markdown(sanitise_for_streamlit(body))
        return

    # Always show exactly 3 cards -- pad if LLM returned fewer
    metrics = metrics[:3]
    while len(metrics) < 3:
        metrics.append(("Data not available", "N/A"))

    cards_html = '<div class="kpi-row">'
    for label, value in metrics:
        safe_value = sanitise_for_streamlit(value)
        safe_label = sanitise_for_streamlit(label)
        cards_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{safe_value}</div>'
            f'<div class="kpi-label">{safe_label}</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


def render_styled_table(table_data: list[list[str]]):
    """Render a parsed table as styled HTML with a navy header and alternating rows.

    Rendered as HTML rather than via st.dataframe() because Streamlit's
    markdown table support is inconsistent -- pipe tables sometimes render
    correctly and sometimes do not, depending on surrounding content.
    """
    if not table_data or len(table_data) < 2:
        st.info("Table data not available.")
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
        '<h4>Explore Related Industries</h4>'
        '<p class="related-subtitle">Search one of these in the box above to generate a new report</p>'
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
        st.success(f"Word count: {wc} / {HARD_WORD_LIMIT}")
    else:
        st.error(f"Word count: {wc} / {HARD_WORD_LIMIT} -- over limit")

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
