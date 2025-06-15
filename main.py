from __future__ import annotations

# Standard Library
import os
from typing import List, Optional, Any, Dict

# Third-Party Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

# Automatically load variables from a local .env file (if present) so that
# developers don't need to export them manually during local development.
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment-driven configuration to avoid hard-coding sensitive values.
AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY: str | None = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
DEFAULT_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "256"))

# Guard clauses to surface configuration errors early.
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT]):
    missing = [
        name for name, value in (
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_KEY", AZURE_OPENAI_KEY),
            ("AZURE_OPENAI_DEPLOYMENT", AZURE_OPENAI_DEPLOYMENT),
        )
        if value is None
    ]
    raise RuntimeError(
        "Missing required Azure OpenAI environment variables: " + ", ".join(missing)
    )

# Initialise the AzureOpenAI client (new SDK interface).
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class SEOItem(BaseModel):
    """Represents a single page\'s SEO-relevant data."""

    url: str = Field(..., description="Canonical URL of the page")
    title: Optional[str] = Field(None, description="SEO title tag text")
    meta_description: Optional[str] = Field(
        None, description="Meta description content for the page"
    )
    content: str = Field(..., description="Primary textual content of the page")


class SummaryRequest(BaseModel):
    """Request body for the summarisation endpoint."""

    items: List[SEOItem] = Field(..., description="List of SEO items to summarise")
    language: Optional[str] = Field(
        "English", description="Target language for generated summaries"
    )


class SummaryResponse(BaseModel):
    """Response payload containing generated summaries."""

    summaries: List[str]


# ---------------------------------------------------------------------------
# Models specific to Project Overview Summaries
# ---------------------------------------------------------------------------


class KeywordStat(BaseModel):
    """Represents statistics for a single keyword within the project overview."""

    id: int
    keyword: str
    position: int
    position_change: int
    volume: int
    cpc: float
    kd: float
    intent: str


class ProjectOverview(BaseModel):
    """Subset of Rank-Tracker project overview fields used for summarisation."""

    url: str
    domain_authority: int = Field(..., alias="domain_authority")
    domain_authority_change: int
    no_of_kw: int
    avg_pos_change: float
    avg_pos_desktop: float
    avg_pos_mobile: float
    increased_keywords: int
    decreased_keywords: int
    range_distribution: Dict[str, int]
    top_keywords: List[KeywordStat]

    # Allow additional keys without validation errors (history arrays, etc.).
    class Config:
        extra = "allow"


class OverviewSummaryRequest(BaseModel):
    """Request body for overview summarisation endpoint."""

    projects: List[ProjectOverview]
    language: Optional[str] = Field("English", description="Target output language")


# ---------------------------------------------------------------------------
# Models specific to Keyword Summaries
# ---------------------------------------------------------------------------

class Keyword(BaseModel):
    """Represents a single keyword with tracking data."""
    
    id: int
    keyword: str
    volume: int
    cpc: float
    kd: float
    intent: str
    cur_desk: int
    prev_desk: int
    cur_mobile: int
    prev_mobile: int
    project_id: int
    owner_id: int
    track_freq: str
    location: str
    last_updated: str
    no_of_tracks: int
    advanced: bool
    ranking_page_for_kw_desktop: Optional[str]
    ranking_page_for_kw_mobile: Optional[str]


class KeywordSummaryRequest(BaseModel):
    """Request body for keyword summarisation endpoint."""
    
    keywords: List[Keyword]
    language: Optional[str] = Field("English", description="Target output language")


# ---------------------------------------------------------------------------
# Models specific to Historical Rank Summaries
# ---------------------------------------------------------------------------

class RankingEntry(BaseModel):
    """Represents a single ranking entry for a keyword."""
    
    url: str
    Rank: int
    Type: str
    Domain: str
    Keyword: str


class RankHistory(BaseModel):
    """Represents ranking history for mobile and desktop."""
    
    mobile: List[RankingEntry]
    desktop: Optional[List[RankingEntry]] = None


class HistoricalKeyword(BaseModel):
    """Represents a keyword with its historical ranking data."""
    
    id: int
    keyword: str
    intents: str
    volume: int
    position_desktop_change: int
    position_mobile_change: int
    rank_history: List[RankHistory]


class HistoricalRankRequest(BaseModel):
    """Request body for historical rank summarisation endpoint."""
    
    project_id: int
    page: int
    limit: int
    total_keywords: int
    keywords: List[HistoricalKeyword]
    language: Optional[str] = Field("English", description="Target output language")


# ---------------------------------------------------------------------------
# Models specific to Page Rankings
# ---------------------------------------------------------------------------

class PageSummary(BaseModel):
    """Represents a page with its ranking performance data."""
    
    page_url: str
    keyword_count: int
    avg_current_position: float
    avg_previous_position: float
    total_volume: int
    top_keyword_id: int
    top_keyword: str
    top_keyword_desktop_position: int
    top_keyword_desktop_prev_position: int
    top_keyword_mobile_position: int
    top_keyword_mobile_prev_position: int
    top_keyword_country: str


class PageRankingsRequest(BaseModel):
    """Request body for page rankings summarisation endpoint."""
    
    device_type: str
    project_id: int
    page: int
    limit: int
    total_pages: int
    total_results: int
    pages_summary: List[PageSummary]
    language: Optional[str] = Field("English", description="Target output language")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="SEO Summarisation API", version="1.0.0")


# Utility -------------------------------------------------------------------

def _build_prompt(item: SEOItem, language: str) -> List[dict[str, str]]:
    """Create a ChatGPT prompt for the given SEO item."""

    system_prompt: str = (
        "You are an expert SEO analyst. Given a web page's SEO-related data, "
        "produce a concise, actionable summary focusing on key optimisations. "
        f"Respond in {language}."
    )

    user_content: str = (
        f"URL: {item.url}\n"
        f"Title: {item.title or 'N/A'}\n"
        f"Meta Description: {item.meta_description or 'N/A'}\n"
        "Content:\n" + item.content
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_overview_prompt(project: ProjectOverview, language: str) -> List[dict[str, str]]:
    """Create a Chat prompt for a Rank-Tracker project overview."""

    system_prompt: str = (
        "You are an SEO strategist. Given ranking-tracker overview metrics for a website, "
        "produce a concise summary highlighting performance trends, opportunities, and warnings. "
        f"Respond in {language}."
    )

    # Summarise key numeric metrics and top 5 keywords (already limited to 10 in sample).
    metrics_section = (
        f"Domain Authority: {project.domain_authority} (Δ {project.domain_authority_change})\n"
        f"Keywords Tracked: {project.no_of_kw}\n"
        f"Avg Desktop Pos: {project.avg_pos_desktop}\n"
        f"Avg Mobile Pos: {project.avg_pos_mobile}\n"
        f"Avg Position Change: {project.avg_pos_change}\n"
        f"Increased Keywords: {project.increased_keywords}\n"
        f"Decreased Keywords: {project.decreased_keywords}\n"
        f"Range Distribution: {project.range_distribution}\n"
    )

    top_kw_lines = [
        f"- {kw.keyword}: pos {kw.position} (Δ {kw.position_change}), vol {kw.volume}, intent {kw.intent}"
        for kw in project.top_keywords[:5]
    ]

    user_content = (
        f"Website URL: {project.url}\n" + metrics_section + "Top Keywords:\n" + "\n".join(top_kw_lines)
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_keyword_prompt(keyword: Keyword, language: str) -> List[dict[str, str]]:
    """Create a Chat prompt for a keyword summary."""
    
    system_prompt: str = (
        "You are an SEO keyword analyst. Given tracking data for a specific keyword, "
        "produce a concise summary of its performance, trends, and optimization opportunities. "
        f"Respond in {language}."
    )
    
    # Calculate position changes
    desktop_change = keyword.prev_desk - keyword.cur_desk
    mobile_change = keyword.prev_mobile - keyword.cur_mobile
    
    user_content: str = (
        f"Keyword: {keyword.keyword}\n"
        f"Volume: {keyword.volume}\n"
        f"CPC: {keyword.cpc}\n"
        f"Keyword Difficulty: {keyword.kd}\n"
        f"Intent: {keyword.intent}\n"
        f"Location: {keyword.location}\n"
        f"Desktop Position: {keyword.cur_desk} (Change: {desktop_change})\n"
        f"Mobile Position: {keyword.cur_mobile} (Change: {mobile_change})\n"
        f"Ranking Page Desktop: {keyword.ranking_page_for_kw_desktop or 'None'}\n"
        f"Ranking Page Mobile: {keyword.ranking_page_for_kw_mobile or 'None'}\n"
        f"Last Updated: {keyword.last_updated}\n"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_historical_rank_prompt(keyword: HistoricalKeyword, language: str) -> List[dict[str, str]]:
    """Create a Chat prompt for a historical rank summary."""
    
    system_prompt: str = (
        "You are an SEO ranking analyst. Given historical ranking data for a keyword, "
        "produce a concise summary of ranking patterns, competitive insights, and optimization opportunities. "
        f"Respond in {language}."
    )
    
    # Create a summary of the top 5 rankings for mobile and desktop
    mobile_rankings = []
    desktop_rankings = []
    
    if keyword.rank_history and len(keyword.rank_history) > 0:
        # Get the most recent history entry
        latest_history = keyword.rank_history[0]
        
        # Process mobile rankings
        if latest_history.mobile:
            for i, entry in enumerate(latest_history.mobile[:5]):  # Limit to top 5
                mobile_rankings.append(
                    f"Rank {entry.Rank}: {entry.url} ({entry.Domain}, {entry.Type})"
                )
        
        # Process desktop rankings if available
        if latest_history.desktop:
            for i, entry in enumerate(latest_history.desktop[:5]):  # Limit to top 5
                desktop_rankings.append(
                    f"Rank {entry.Rank}: {entry.url} ({entry.Domain}, {entry.Type})"
                )
    
    user_content: str = (
        f"Keyword: {keyword.keyword}\n"
        f"Intent: {keyword.intents}\n"
        f"Volume: {keyword.volume}\n"
        f"Desktop Position Change: {keyword.position_desktop_change}\n"
        f"Mobile Position Change: {keyword.position_mobile_change}\n\n"
        f"Top Mobile Rankings:\n" + "\n".join(mobile_rankings or ["No mobile ranking data available"]) + "\n\n"
        f"Top Desktop Rankings:\n" + "\n".join(desktop_rankings or ["No desktop ranking data available"])
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_page_rankings_prompt(page: PageSummary, device_type: str, language: str) -> List[dict[str, str]]:
    """Create a Chat prompt for a page rankings summary."""
    
    system_prompt: str = (
        "You are an SEO page analyst. Given ranking data for a specific page, "
        "produce a concise summary of its performance, keyword rankings, and optimization opportunities. "
        f"Focus on {device_type} performance. Respond in {language}."
    )
    
    # Calculate position changes
    avg_position_change = page.avg_previous_position - page.avg_current_position
    top_keyword_desktop_change = page.top_keyword_desktop_prev_position - page.top_keyword_desktop_position
    top_keyword_mobile_change = page.top_keyword_mobile_prev_position - page.top_keyword_mobile_position
    
    user_content: str = (
        f"Page URL: {page.page_url}\n"
        f"Number of Keywords Ranking: {page.keyword_count}\n"
        f"Average Current Position: {page.avg_current_position}\n"
        f"Average Previous Position: {page.avg_previous_position}\n"
        f"Position Change: {avg_position_change}\n"
        f"Total Search Volume: {page.total_volume}\n\n"
        f"Top Keyword: {page.top_keyword}\n"
        f"Top Keyword Desktop Position: {page.top_keyword_desktop_position} (Change: {top_keyword_desktop_change})\n"
        f"Top Keyword Mobile Position: {page.top_keyword_mobile_position} (Change: {top_keyword_mobile_change})\n"
        f"Top Keyword Country: {page.top_keyword_country}\n"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _summarise_item(item: SEOItem, language: str) -> str:
    """Generate a summary for a single SEO item using Azure OpenAI."""

    messages = _build_prompt(item, language)

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response.choices[0].message.content.strip()


def _summarise_overview(project: ProjectOverview, language: str) -> str:
    """Generate a summary for a project overview via Azure OpenAI."""

    messages = _build_overview_prompt(project, language)

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response.choices[0].message.content.strip()


def _summarise_keyword(keyword: Keyword, language: str) -> str:
    """Generate a summary for a keyword via Azure OpenAI."""
    
    messages = _build_keyword_prompt(keyword, language)
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return response.choices[0].message.content.strip()


def _summarise_historical_rank(keyword: HistoricalKeyword, language: str) -> str:
    """Generate a summary for historical rank data via Azure OpenAI."""
    
    messages = _build_historical_rank_prompt(keyword, language)
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return response.choices[0].message.content.strip()


def _summarise_page_rankings(page: PageSummary, device_type: str, language: str) -> str:
    """Generate a summary for page rankings data via Azure OpenAI."""
    
    messages = _build_page_rankings_prompt(page, device_type, language)
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return response.choices[0].message.content.strip()


# Routes --------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """Lightweight endpoint to confirm the service is running."""

    return {"status": "ok"}


@app.post("/summaries", response_model=SummaryResponse, summary="Generate summaries")
async def generate_summaries(req: SummaryRequest) -> SummaryResponse:  # noqa: D401
    """Generate SEO-focused summaries for the provided items."""

    summaries: List[str] = [
        _summarise_item(item, req.language or "English") for item in req.items
    ]

    return SummaryResponse(summaries=summaries)


@app.post("/overview-summaries", response_model=SummaryResponse, summary="Generate overview summaries")
async def generate_overview_summaries(req: OverviewSummaryRequest) -> SummaryResponse:  # noqa: D401
    """Generate summaries for Rank-Tracker project overviews."""

    summaries: List[str] = [
        _summarise_overview(project, req.language or "English") for project in req.projects
    ]

    return SummaryResponse(summaries=summaries)


@app.post("/keyword-summaries", response_model=SummaryResponse, summary="Generate keyword summaries")
async def generate_keyword_summaries(req: KeywordSummaryRequest) -> SummaryResponse:  # noqa: D401
    """Generate summaries for tracked keywords with performance analysis."""
    
    summaries: List[str] = [
        _summarise_keyword(keyword, req.language or "English") for keyword in req.keywords
    ]
    
    return SummaryResponse(summaries=summaries)


@app.post("/historical-rank-summaries", response_model=SummaryResponse, summary="Generate historical rank summaries")
async def generate_historical_rank_summaries(req: HistoricalRankRequest) -> SummaryResponse:  # noqa: D401
    """Generate summaries for keywords with historical ranking data."""
    
    summaries: List[str] = [
        _summarise_historical_rank(keyword, req.language or "English") for keyword in req.keywords
    ]
    
    return SummaryResponse(summaries=summaries)


@app.post("/page-rankings-summaries", response_model=SummaryResponse, summary="Generate page rankings summaries")
async def generate_page_rankings_summaries(req: PageRankingsRequest) -> SummaryResponse:  # noqa: D401
    """Generate summaries for pages with their keyword ranking performance."""
    
    summaries: List[str] = [
        _summarise_page_rankings(page, req.device_type, req.language or "English") for page in req.pages_summary
    ]
    
    return SummaryResponse(summaries=summaries)
