from datetime import datetime, timezone
from time import time

from fastapi import APIRouter, status

from graph_service.dto import (
    ComparisonOperator,
    DateFilter,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    NodeSearchMethod,
    PropertyFilter,
    Reranker,
    SearchFilters,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter(tags=["search"])


def _convert_date_filter_to_core(dto_date_filter: DateFilter):
    """Convert API DateFilter to core DateFilter"""
    from graphiti_core.search.search_filters import DateFilter as CoreDateFilter
    from graphiti_core.search.search_filters import ComparisonOperator as CoreComparisonOperator

    return CoreDateFilter(
        date=dto_date_filter.date,
        comparison_operator=CoreComparisonOperator(dto_date_filter.comparison_operator.value)
    )


def _convert_property_filter_to_core(dto_property_filter: PropertyFilter):
    """Convert API PropertyFilter to core PropertyFilter"""
    from graphiti_core.search.search_filters import PropertyFilter as CorePropertyFilter
    from graphiti_core.search.search_filters import ComparisonOperator as CoreComparisonOperator

    return CorePropertyFilter(
        property_name=dto_property_filter.property_name,
        property_value=dto_property_filter.property_value,
        comparison_operator=CoreComparisonOperator(dto_property_filter.comparison_operator.value)
    )


def _convert_search_filters_to_core(dto_filters: SearchFilters | None):
    """Convert API SearchFilters to core SearchFilters"""
    from graphiti_core.search.search_filters import SearchFilters as CoreSearchFilters

    if dto_filters is None:
        return None

    core_created_at = None
    if dto_filters.created_at is not None:
        core_created_at = [
            [_convert_date_filter_to_core(date_filter) for date_filter in or_list]
            for or_list in dto_filters.created_at
        ]

    core_valid_at = None
    if dto_filters.valid_at is not None:
        core_valid_at = [
            [_convert_date_filter_to_core(date_filter) for date_filter in or_list]
            for or_list in dto_filters.valid_at
        ]

    core_invalid_at = None
    if dto_filters.invalid_at is not None:
        core_invalid_at = [
            [_convert_date_filter_to_core(date_filter) for date_filter in or_list]
            for or_list in dto_filters.invalid_at
        ]

    core_expired_at = None
    if dto_filters.expired_at is not None:
        core_expired_at = [
            [_convert_date_filter_to_core(date_filter) for date_filter in or_list]
            for or_list in dto_filters.expired_at
        ]

    core_property_filters = None
    if dto_filters.property_filters is not None:
        core_property_filters = [
            _convert_property_filter_to_core(prop_filter)
            for prop_filter in dto_filters.property_filters
        ]

    return CoreSearchFilters(
        node_labels=dto_filters.node_labels,
        edge_types=dto_filters.edge_types,
        created_at=core_created_at,
        valid_at=core_valid_at,
        invalid_at=core_invalid_at,
        expired_at=core_expired_at,
        edge_uuids=dto_filters.edge_uuids,
        property_filters=core_property_filters,
    )


def _build_search_filters_from_query(query: SearchQuery):
    """Build SearchFilters from SearchQuery, combining simplified date/entity filters with advanced filters"""
    from graphiti_core.search.search_filters import (
        SearchFilters as CoreSearchFilters,
        DateFilter as CoreDateFilter,
        ComparisonOperator as CoreComparisonOperator,
    )

    # Start with advanced filters if provided
    core_filters = _convert_search_filters_to_core(query.filters)

    # If no advanced filters, create a new one
    if core_filters is None:
        core_filters = CoreSearchFilters()

    # Add simplified date filters if provided
    if query.created_at_start is not None or query.created_at_end is not None:
        if core_filters.created_at is None:
            core_filters.created_at = []

        date_filters = []
        if query.created_at_start is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.created_at_start,
                    comparison_operator=CoreComparisonOperator.greater_than_equal
                )
            )
        if query.created_at_end is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.created_at_end,
                    comparison_operator=CoreComparisonOperator.less_than_equal
                )
            )
        if date_filters:  # Only add if we have filters
            core_filters.created_at.append(date_filters)

    # Similar logic for other date ranges
    if query.valid_at_start is not None or query.valid_at_end is not None:
        if core_filters.valid_at is None:
            core_filters.valid_at = []

        date_filters = []
        if query.valid_at_start is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.valid_at_start,
                    comparison_operator=CoreComparisonOperator.greater_than_equal
                )
            )
        if query.valid_at_end is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.valid_at_end,
                    comparison_operator=CoreComparisonOperator.less_than_equal
                )
            )
        if date_filters:  # Only add if we have filters
            core_filters.valid_at.append(date_filters)

    # Handle invalid_at date ranges
    if query.invalid_at_start is not None or query.invalid_at_end is not None:
        if core_filters.invalid_at is None:
            core_filters.invalid_at = []

        date_filters = []
        if query.invalid_at_start is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.invalid_at_start,
                    comparison_operator=CoreComparisonOperator.greater_than_equal
                )
            )
        if query.invalid_at_end is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.invalid_at_end,
                    comparison_operator=CoreComparisonOperator.less_than_equal
                )
            )
        if date_filters:  # Only add if we have filters
            core_filters.invalid_at.append(date_filters)

    # Handle expired_at date ranges
    if query.expired_at_start is not None or query.expired_at_end is not None:
        if core_filters.expired_at is None:
            core_filters.expired_at = []

        date_filters = []
        if query.expired_at_start is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.expired_at_start,
                    comparison_operator=CoreComparisonOperator.greater_than_equal
                )
            )
        if query.expired_at_end is not None:
            date_filters.append(
                CoreDateFilter(
                    date=query.expired_at_end,
                    comparison_operator=CoreComparisonOperator.less_than_equal
                )
            )
        if date_filters:  # Only add if we have filters
            core_filters.expired_at.append(date_filters)

    # Add simplified entity filters if provided
    if query.node_labels is not None:
        core_filters.node_labels = query.node_labels
    if query.edge_types is not None:
        core_filters.edge_types = query.edge_types

    return core_filters


def _convert_search_config_from_query(query: SearchQuery):
    """Convert SearchQuery to SearchConfig for advanced search"""
    from graphiti_core.search.search_config import (
        SearchConfig,
        EdgeSearchConfig,
        NodeSearchConfig,
        EdgeSearchMethod as CoreEdgeSearchMethod,
        NodeSearchMethod as CoreNodeSearchMethod,
        EdgeReranker as CoreEdgeReranker,
        NodeReranker as CoreNodeReranker,
    )
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_RRF,
        COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    )

    # Use advanced search if any advanced parameters are provided
    use_advanced_search = (
        query.node_search_methods is not None or
        query.edge_search_methods is not None or
        query.reranker is not None or
        query.min_score is not None or
        query.mmr_lambda is not None or
        query.reranker_min_score is not None or
        query.bfs_max_depth is not None or
        not query.include_edges or  # Default search only returns edges
        query.include_nodes or
        query.include_episodes or
        query.include_communities
    )

    if use_advanced_search:
        # Determine search methods
        edge_methods = [CoreEdgeSearchMethod.bm25, CoreEdgeSearchMethod.cosine_similarity]
        if query.edge_search_methods is not None:
            edge_methods = [CoreEdgeSearchMethod(method.value) for method in query.edge_search_methods]

        node_methods = [CoreNodeSearchMethod.bm25, CoreNodeSearchMethod.cosine_similarity]
        if query.node_search_methods is not None:
            node_methods = [CoreNodeSearchMethod(method.value) for method in query.node_search_methods]

        # Determine rerankers (EdgeReranker for edges, NodeReranker for nodes)
        edge_reranker = CoreEdgeReranker.rrf
        node_reranker = CoreNodeReranker.rrf
        if query.reranker is not None:
            edge_reranker = CoreEdgeReranker(query.reranker.value)
            node_reranker = CoreNodeReranker(query.reranker.value)

        # Create configs
        edge_config = EdgeSearchConfig(
            search_methods=edge_methods,
            reranker=edge_reranker,
            sim_min_score=query.min_score or 0.0,
            mmr_lambda=query.mmr_lambda or 0.5,
            reranker_min_score=query.reranker_min_score or 0.0,
            bfs_max_depth=query.bfs_max_depth or 3,
        )

        node_config = None
        if query.include_nodes:
            node_config = NodeSearchConfig(
                search_methods=node_methods,
                reranker=node_reranker,
                sim_min_score=query.min_score or 0.0,
                mmr_lambda=query.mmr_lambda or 0.5,
                reranker_min_score=query.reranker_min_score or 0.0,
                bfs_max_depth=query.bfs_max_depth or 3,
            )

        return SearchConfig(
            edge_config=edge_config,
            node_config=node_config,
            limit=query.max_facts,
        )
    else:
        # Use default search config for backward compatibility
        config = EDGE_HYBRID_SEARCH_RRF
        config.limit = query.max_facts
        return config


@router.post(
    '/search',
    status_code=status.HTTP_200_OK,
    summary="Search knowledge graph with advanced filtering and ranking",
    description="""
    Search the knowledge graph with comprehensive filtering, ranking, and result configuration options.

    This endpoint automatically detects when to use advanced search features while maintaining backward compatibility.
    Simple queries use optimized basic search, while queries with advanced parameters trigger enhanced search capabilities.

    **Features:**
    - Date range filtering (created_at, valid_at, invalid_at, expired_at)
    - Entity type filtering (node_labels, edge_types)
    - Advanced property-based filtering with comparison operators
    - Multiple search methods (cosine similarity, BM25, BFS)
    - Advanced ranking options (RRF, MMR, cross-encoder, node distance, episode mentions)
    - Configurable result types (nodes, edges, episodes, communities)
    - Search metadata and performance timing

    **Backward Compatibility:**
    Existing clients continue to work without any changes. Simple queries are automatically optimized for performance.
    """,
    responses={
        200: {
            "description": "Search results with facts and optional metadata",
            "content": {
                "application/json": {
                    "example": {
                        "facts": [
                            {
                                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                                "name": "John works at TechCorp",
                                "fact": "John Doe is employed as a Senior Engineer at TechCorp since 2023",
                                "valid_at": "2023-01-01T00:00:00Z",
                                "invalid_at": None,
                                "created_at": "2023-01-01T00:00:00Z",
                                "expired_at": None
                            }
                        ],
                        "fact_scores": [0.95, 0.87],
                        "total_results": 25,
                        "search_time_ms": 150,
                        "search_config_used": {
                            "edge_search_methods": ["cosine_similarity", "bm25"],
                            "node_search_methods": None,
                            "edge_reranker": "reciprocal_rank_fusion",
                            "limit": 10
                        },
                        "ranking_method": "rrf",
                        "score_normalization": "none",
                        "max_score_possible": None,
                        "min_score_threshold": 0.6
                    }
                }
            }
        }
    }
)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    start_time = time()

    # Determine if we need to use advanced search
    use_advanced_search = (
        query.node_search_methods is not None or
        query.edge_search_methods is not None or
        query.reranker is not None or
        query.min_score is not None or
        query.mmr_lambda is not None or
        query.reranker_min_score is not None or
        query.bfs_max_depth is not None or
        not query.include_edges or  # Default search only returns edges
        query.include_nodes or
        query.include_episodes or
        query.include_communities or
        query.filters is not None or
        query.created_at_start is not None or
        query.created_at_end is not None or
        query.valid_at_start is not None or
        query.valid_at_end is not None or
        query.invalid_at_start is not None or
        query.invalid_at_end is not None or
        query.expired_at_start is not None or
        query.expired_at_end is not None or
        query.node_labels is not None or
        query.edge_types is not None
    )

    if use_advanced_search:
        # Use advanced search with all parameters
        search_config = _convert_search_config_from_query(query)
        search_filters = _build_search_filters_from_query(query)

        search_results = await graphiti.search(
            query=query.query,
            config=search_config,
            group_ids=query.group_ids,
            search_filter=search_filters,
        )

        # Convert edges to facts
        facts = [get_fact_result_from_edge(edge) for edge in search_results.edges]

        # Calculate search time
        search_time_ms = int((time() - start_time) * 1000)

        # Build search configuration metadata
        search_config_metadata = {
            "edge_search_methods": [method.value for method in search_config.edge_config.search_methods],
            "node_search_methods": [method.value for method in search_config.node_config.search_methods] if search_config.node_config else None,
            "edge_reranker": search_config.edge_config.reranker.value,
            "node_reranker": search_config.node_config.reranker.value if search_config.node_config else None,
            "limit": search_config.limit,
        }

        # Determine ranking method and thresholds
        ranking_method = search_config.edge_config.reranker.value
        min_score_threshold = search_config.edge_config.sim_min_score

        # Add MMR-specific metadata
        if ranking_method == "mmr":
            search_config_metadata["mmr_lambda"] = search_config.edge_config.mmr_lambda
            max_score_possible = 1.0  # MMR scores are typically normalized
        else:
            max_score_possible = None  # Varies by method

        # Build response with all result types and scores
        response = SearchResults(
            facts=facts,
            fact_scores=search_results.edge_reranker_scores if search_results.edge_reranker_scores else None,
            total_results=len(facts),
            search_time_ms=search_time_ms,
            search_config_used=search_config_metadata,
            ranking_method=ranking_method,
            score_normalization="none",  # Core search doesn't normalize by default
            max_score_possible=max_score_possible,
            min_score_threshold=min_score_threshold,
        )

        # Add advanced result types if requested
        if query.include_nodes and search_results.nodes:
            response.nodes = [node.to_dict() for node in search_results.nodes]
            response.node_scores = search_results.node_reranker_scores

        if query.include_edges and search_results.edges:
            response.edges = [edge.to_dict() for edge in search_results.edges]
            response.edge_scores = search_results.edge_reranker_scores

        if query.include_episodes and search_results.episodes:
            response.episodes = [episode.to_dict() for episode in search_results.episodes]
            response.episode_scores = search_results.episode_reranker_scores

        if query.include_communities and search_results.communities:
            response.communities = [community.to_dict() for community in search_results.communities]
            response.community_scores = search_results.community_reranker_scores

        return response
    else:
        # Use basic search for backward compatibility
        relevant_edges = await graphiti.search(
            group_ids=query.group_ids,
            query=query.query,
            num_results=query.max_facts,
        )
        facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]

        # Calculate search time
        search_time_ms = int((time() - start_time) * 1000)

        return SearchResults(
            facts=facts,
            total_results=len(facts),
            search_time_ms=search_time_ms,
            search_config_used={"method": "hybrid_search_rrf", "description": "Default hybrid search with reciprocal rank fusion"},
            ranking_method="rrf",
            score_normalization="none",
            max_score_possible=None,
            min_score_threshold=None,
        )


@router.post(
    '/search-advanced',
    status_code=status.HTTP_200_OK,
    summary="Advanced search with full configuration access",
    description="""
    Advanced search endpoint that always uses the enhanced search system with complete access to all configurations.

    Unlike the standard /search endpoint, this endpoint never falls back to basic search, ensuring consistent
    advanced behavior regardless of query complexity. This provides predictable performance and feature set.

    **Key Differences from /search:**
    - Always uses advanced search (no automatic fallback)
    - Consistent behavior for all query types
    - Includes all result types when available
    - Full access to search configuration options

    **Use Cases:**
    - Applications requiring consistent advanced features
    - Complex filtering and ranking scenarios
    - When you need all result types (nodes, edges, episodes, communities)
    - Debugging advanced search configurations
    """,
    responses={
        200: {
            "description": "Advanced search results with comprehensive metadata and all result types",
            "content": {
                "application/json": {
                    "example": {
                        "facts": [
                            {
                                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                                "name": "AI Research Breakthrough",
                                "fact": "Research team published breakthrough paper on transformer architectures",
                                "valid_at": "2024-03-15T00:00:00Z",
                                "invalid_at": None,
                                "created_at": "2024-03-15T00:00:00Z",
                                "expired_at": None
                            }
                        ],
                        "fact_scores": [0.98, 0.92],
                        "nodes": [
                            {
                                "uuid": "node-123",
                                "name": "Transformer Architecture",
                                "labels": ["Concept", "Technology"]
                            }
                        ],
                        "node_scores": [0.95],
                        "edges": [
                            {
                                "uuid": "edge-456",
                                "source_node": "node-123",
                                "target_node": "node-789",
                                "type": "ENABLES"
                            }
                        ],
                        "edge_scores": [0.97],
                        "total_results": 50,
                        "search_time_ms": 220
                    }
                }
            }
        }
    }
)
async def search_advanced(query: SearchQuery, graphiti: ZepGraphitiDep):
    start_time = time()

    # Always use advanced search with all parameters
    search_config = _convert_search_config_from_query(query)
    search_filters = _build_search_filters_from_query(query)

    search_results = await graphiti.search(
        query=query.query,
        config=search_config,
        group_ids=query.group_ids,
        search_filter=search_filters,
    )

    # Convert edges to facts
    facts = [get_fact_result_from_edge(edge) for edge in search_results.edges]

    # Calculate search time
    search_time_ms = int((time() - start_time) * 1000)

    # Build response with all result types and scores
    response = SearchResults(
        facts=facts,
        fact_scores=search_results.edge_reranker_scores if search_results.edge_reranker_scores else None,
        total_results=len(facts),
        search_time_ms=search_time_ms,
    )

    # Always include advanced result types when available
    if search_results.nodes:
        response.nodes = [node.to_dict() for node in search_results.nodes]
        response.node_scores = search_results.node_reranker_scores

    if search_results.edges:
        response.edges = [edge.to_dict() for edge in search_results.edges]
        response.edge_scores = search_results.edge_reranker_scores

    if search_results.episodes:
        response.episodes = [episode.to_dict() for episode in search_results.episodes]
        response.episode_scores = search_results.episode_reranker_scores

    if search_results.communities:
        response.communities = [community.to_dict() for community in search_results.communities]
        response.community_scores = search_results.community_reranker_scores

    return response


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post(
    '/get-memory',
    status_code=status.HTTP_200_OK,
    summary="Retrieve contextual memory with advanced search capabilities",
    description="""
    Retrieve memory facts based on message context with comprehensive search and filtering options.

    This endpoint builds a search query from the provided messages and applies the same advanced
    filtering and ranking capabilities available in the search endpoints. It's designed for
    conversational AI applications that need to retrieve relevant context from the knowledge graph.

    **Key Features:**
    - Automatic query composition from message history
    - Advanced filtering (date ranges, entity types, properties)
    - Multiple ranking methods (RRF, MMR, cross-encoder, etc.)
    - Configurable result types and scoring
    - Backward compatible with existing get-memory implementations

    **Use Cases:**
    - Conversational AI context retrieval
    - Chatbot memory access
    - Context-aware question answering
    - Session-based memory management
    """,
    responses={
        200: {
            "description": "Memory retrieval results with contextual facts and metadata",
            "content": {
                "application/json": {
                    "example": {
                        "facts": [
                            {
                                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                                "name": "User Preferences",
                                "fact": "User prefers TypeScript over JavaScript for frontend development",
                                "valid_at": "2024-01-01T00:00:00Z",
                                "invalid_at": None,
                                "created_at": "2024-01-01T00:00:00Z",
                                "expired_at": None
                            }
                        ],
                        "fact_scores": [0.94, 0.88],
                        "total_results": 15,
                        "search_time_ms": 180,
                        "search_config_used": {
                            "edge_search_methods": ["cosine_similarity", "bm25"],
                            "edge_reranker": "mmr",
                            "mmr_lambda": 0.5,
                            "limit": 10
                        },
                        "ranking_method": "mmr",
                        "score_normalization": "none",
                        "max_score_possible": 1.0,
                        "min_score_threshold": 0.7
                    }
                }
            }
        }
    }
)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    start_time = time()

    combined_query = compose_query_from_messages(request.messages)

    # Determine if we need to use advanced search
    use_advanced_search = (
        request.node_search_methods is not None or
        request.edge_search_methods is not None or
        request.reranker is not None or
        request.min_score is not None or
        request.mmr_lambda is not None or
        request.reranker_min_score is not None or
        request.bfs_max_depth is not None or
        not request.include_edges or  # Default search only returns edges
        request.include_nodes or
        request.include_episodes or
        request.include_communities or
        request.filters is not None or
        request.created_at_start is not None or
        request.created_at_end is not None or
        request.valid_at_start is not None or
        request.valid_at_end is not None or
        request.invalid_at_start is not None or
        request.invalid_at_end is not None or
        request.expired_at_start is not None or
        request.expired_at_end is not None or
        request.node_labels is not None or
        request.edge_types is not None
    )

    if use_advanced_search:
        # Create a SearchQuery from the GetMemoryRequest to reuse helper functions
        search_query = SearchQuery(
            group_ids=[request.group_id],
            query=combined_query,
            max_facts=request.max_facts,
            created_at_start=request.created_at_start,
            created_at_end=request.created_at_end,
            valid_at_start=request.valid_at_start,
            valid_at_end=request.valid_at_end,
            invalid_at_start=request.invalid_at_start,
            invalid_at_end=request.invalid_at_end,
            expired_at_start=request.expired_at_start,
            expired_at_end=request.expired_at_end,
            node_labels=request.node_labels,
            edge_types=request.edge_types,
            filters=request.filters,
            node_search_methods=request.node_search_methods,
            edge_search_methods=request.edge_search_methods,
            reranker=request.reranker,
            min_score=request.min_score,
            mmr_lambda=request.mmr_lambda,
            reranker_min_score=request.reranker_min_score,
            bfs_max_depth=request.bfs_max_depth,
            include_nodes=request.include_nodes,
            include_edges=request.include_edges,
            include_episodes=request.include_episodes,
            include_communities=request.include_communities,
        )

        # Use advanced search with all parameters
        search_config = _convert_search_config_from_query(search_query)
        search_filters = _build_search_filters_from_query(search_query)

        search_results = await graphiti.search(
            query=search_query.query,
            config=search_config,
            group_ids=search_query.group_ids,
            search_filter=search_filters,
        )

        # Convert edges to facts
        facts = [get_fact_result_from_edge(edge) for edge in search_results.edges]

        # Calculate search time
        search_time_ms = int((time() - start_time) * 1000)

        # Build search configuration metadata
        search_config_metadata = {
            "edge_search_methods": [method.value for method in search_config.edge_config.search_methods],
            "node_search_methods": [method.value for method in search_config.node_config.search_methods] if search_config.node_config else None,
            "edge_reranker": search_config.edge_config.reranker.value,
            "node_reranker": search_config.node_config.reranker.value if search_config.node_config else None,
            "limit": search_config.limit,
        }

        # Determine ranking method and thresholds
        ranking_method = search_config.edge_config.reranker.value
        min_score_threshold = search_config.edge_config.sim_min_score

        # Add MMR-specific metadata
        if ranking_method == "mmr":
            search_config_metadata["mmr_lambda"] = search_config.edge_config.mmr_lambda
            max_score_possible = 1.0  # MMR scores are typically normalized
        else:
            max_score_possible = None  # Varies by method

        # Build response with all result types and scores
        response = GetMemoryResponse(
            facts=facts,
            fact_scores=search_results.edge_reranker_scores if search_results.edge_reranker_scores else None,
            total_results=len(facts),
            search_time_ms=search_time_ms,
            search_config_used=search_config_metadata,
            ranking_method=ranking_method,
            score_normalization="none",  # Core search doesn't normalize by default
            max_score_possible=max_score_possible,
            min_score_threshold=min_score_threshold,
        )

        # Add advanced result types if requested
        if request.include_nodes and search_results.nodes:
            response.nodes = [node.to_dict() for node in search_results.nodes]
            response.node_scores = search_results.node_reranker_scores

        if request.include_edges and search_results.edges:
            response.edges = [edge.to_dict() for edge in search_results.edges]
            response.edge_scores = search_results.edge_reranker_scores

        if request.include_episodes and search_results.episodes:
            response.episodes = [episode.to_dict() for episode in search_results.episodes]
            response.episode_scores = search_results.episode_reranker_scores

        if request.include_communities and search_results.communities:
            response.communities = [community.to_dict() for community in search_results.communities]
            response.community_scores = search_results.community_reranker_scores

        return response
    else:
        # Use basic search for backward compatibility
        result = await graphiti.search(
            group_ids=[request.group_id],
            query=combined_query,
            num_results=request.max_facts,
        )
        facts = [get_fact_result_from_edge(edge) for edge in result]

        # Calculate search time
        search_time_ms = int((time() - start_time) * 1000)

        return GetMemoryResponse(
            facts=facts,
            total_results=len(facts),
            search_time_ms=search_time_ms,
            search_config_used={"method": "hybrid_search_rrf", "description": "Default hybrid search with reciprocal rank fusion"},
            ranking_method="rrf",
            score_normalization="none",
            max_score_possible=None,
            min_score_threshold=None,
        )


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
