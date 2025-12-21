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

router = APIRouter()


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

        # Determine reranker
        reranker = CoreEdgeReranker.rrf
        if query.reranker is not None:
            reranker = CoreEdgeReranker(query.reranker.value)

        # Create configs
        edge_config = EdgeSearchConfig(
            search_methods=edge_methods,
            reranker=reranker,
            sim_min_score=query.min_score or 0.0,
            mmr_lambda=query.mmr_lambda or 0.5,
        )

        node_config = None
        if query.include_nodes:
            node_config = NodeSearchConfig(
                search_methods=node_methods,
                reranker=reranker,
                sim_min_score=query.min_score or 0.0,
                mmr_lambda=query.mmr_lambda or 0.5,
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


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    """Search the knowledge graph with advanced filtering and ranking options."""
    start_time = time()

    # Determine if we need to use advanced search
    use_advanced_search = (
        query.node_search_methods is not None or
        query.edge_search_methods is not None or
        query.reranker is not None or
        query.min_score is not None or
        query.mmr_lambda is not None or
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

        search_results = await graphiti.search_(
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


@router.post('/search-advanced', status_code=status.HTTP_200_OK)
async def search_advanced(query: SearchQuery, graphiti: ZepGraphitiDep):
    """
    Advanced search endpoint with full access to all search configurations.

    This endpoint always uses the advanced search system and provides complete
    access to all filtering, ranking, and result configuration options.
    Unlike the standard /search endpoint, it doesn't fall back to basic search
    for simple queries, ensuring consistent advanced behavior.
    """
    start_time = time()

    # Always use advanced search with all parameters
    search_config = _convert_search_config_from_query(query)
    search_filters = _build_search_filters_from_query(query)

    search_results = await graphiti.search_(
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


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
