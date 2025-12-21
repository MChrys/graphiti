from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


# Enums matching the core Graphiti search configuration
class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='
    is_null = 'IS NULL'
    is_not_null = 'IS NOT NULL'


class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'


class EdgeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'


class Reranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'


# Filter classes matching the core Graphiti search filters
class DateFilter(BaseModel):
    date: datetime | None = Field(description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )


class PropertyFilter(BaseModel):
    property_name: str = Field(description='Property name')
    property_value: str | int | float | None = Field(
        description='Value you want to match on for the property'
    )
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for the property'
    )


class SearchFilters(BaseModel):
    node_labels: list[str] | None = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: list[str] | None = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: list[list[DateFilter]] | None = Field(default=None)
    invalid_at: list[list[DateFilter]] | None = Field(default=None)
    created_at: list[list[DateFilter]] | None = Field(default=None)
    expired_at: list[list[DateFilter]] | None = Field(default=None)
    edge_uuids: list[str] | None = Field(default=None)
    property_filters: list[PropertyFilter] | None = Field(default=None)


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')

    # Date range filters (simplified interface for common use cases)
    created_at_start: datetime | None = Field(
        None, description='Filter for facts created at or after this datetime'
    )
    created_at_end: datetime | None = Field(
        None, description='Filter for facts created at or before this datetime'
    )
    valid_at_start: datetime | None = Field(
        None, description='Filter for facts valid at or after this datetime'
    )
    valid_at_end: datetime | None = Field(
        None, description='Filter for facts valid at or before this datetime'
    )
    invalid_at_start: datetime | None = Field(
        None, description='Filter for facts invalid at or after this datetime'
    )
    invalid_at_end: datetime | None = Field(
        None, description='Filter for facts invalid at or before this datetime'
    )
    expired_at_start: datetime | None = Field(
        None, description='Filter for facts expired at or after this datetime'
    )
    expired_at_end: datetime | None = Field(
        None, description='Filter for facts expired at or before this datetime'
    )

    # Advanced filtering (for complex filtering needs)
    filters: SearchFilters | None = Field(
        None, description='Advanced filtering options including complex date filters, entity filters, and property filters'
    )

    # Search method configuration
    node_search_methods: list[NodeSearchMethod] | None = Field(
        None, description='Search methods to use for node search (defaults to cosine_similarity and bm25)'
    )
    edge_search_methods: list[EdgeSearchMethod] | None = Field(
        None, description='Search methods to use for edge search (defaults to cosine_similarity and bm25)'
    )

    # Ranking and scoring options
    reranker: Reranker | None = Field(
        None, description='Reranking method to use for search results (defaults to reciprocal_rank_fusion)'
    )
    min_score: float | None = Field(
        None, description='Minimum similarity score threshold for results (defaults to core system default)'
    )
    mmr_lambda: float | None = Field(
        None, description='Lambda parameter for MMR reranking (0-1, defaults to core system default)'
    )

    # Search configuration
    include_nodes: bool = Field(default=True, description='Whether to include nodes in search results')
    include_edges: bool = Field(default=True, description='Whether to include edges in search results')
    include_episodes: bool = Field(default=False, description='Whether to include episodes in search results')
    include_communities: bool = Field(default=False, description='Whether to include communities in search results')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]

    # Ranking and scoring information
    fact_scores: list[float] | None = Field(
        default=None, description='Relevance scores for each fact (higher is more relevant)'
    )

    # Search metadata
    total_results: int | None = Field(
        default=None, description='Total number of results available (may be more than returned facts)'
    )
    search_time_ms: int | None = Field(
        default=None, description='Time taken for the search operation in milliseconds'
    )

    # Advanced result types (for future use when exposing full search results)
    nodes: list[dict[str, Any]] | None = Field(
        default=None, description='Node results from the search (when include_nodes is true)'
    )
    node_scores: list[float] | None = Field(
        default=None, description='Relevance scores for each node'
    )

    edges: list[dict[str, Any]] | None = Field(
        default=None, description='Edge results from the search (when include_edges is true)'
    )
    edge_scores: list[float] | None = Field(
        default=None, description='Relevance scores for each edge'
    )

    episodes: list[dict[str, Any]] | None = Field(
        default=None, description='Episode results from the search (when include_episodes is true)'
    )
    episode_scores: list[float] | None = Field(
        default=None, description='Relevance scores for each episode'
    )

    communities: list[dict[str, Any]] | None = Field(
        default=None, description='Community results from the search (when include_communities is true)'
    )
    community_scores: list[float] | None = Field(
        default=None, description='Relevance scores for each community'
    )


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
