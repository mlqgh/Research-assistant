# schemas.py
from typing import List, Literal
from pydantic import BaseModel, Field

class QueryRoute(BaseModel):
    """Schema for routing decisions made by the Router Agent."""
    datasource: Literal["vectorstore", "tavily_search", "direct_answer"] = Field(
        description="The target data source based on user query intent."
    )
    reasoning: str = Field(description="Explanation for selecting this route.")

class QueryRefinement(BaseModel):
    """Schema for rephrasing user queries into optimal search queries."""
    optimized_query: str = Field(
        description="A high-density search query stripped of conversational filler."
    )

class SourceCitation(BaseModel):
    """Schema representing an individual source citation."""
    source_id: str = Field(description="File path, document title, or web URL.")
    snippet: str = Field(description="Key text snippet from the document supporting the answer.")


class GroundingEvaluation(BaseModel):
    citations: List[SourceCitation ] = Field(default_factory=list, description="Citations supporting the response")
    hallucinations_detected: List[str] = Field(default_factory=list, description="Ungrounded claims made in response")
    is_grounded: bool = Field(description="True if response is fully supported by context")