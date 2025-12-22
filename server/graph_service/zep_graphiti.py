import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None, embedder=None):
        super().__init__(uri, user, password, llm_client, embedder=embedder)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        num_results: int = 10,
        search_filter=None,
        config=None,
        driver=None,
        **kwargs
    ):
        """
        Enhanced search method that accepts and properly handles new filter parameters.

        This method extends the base Graphiti.search method to provide better integration
        with the API layer and support for advanced filtering options.

        Parameters
        ----------
        query : str
            The search query string.
        group_ids : list[str] | None, optional
            The graph partitions to return data from.
        center_node_uuid : str | None, optional
            Facts will be reranked based on proximity to this node.
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.
        search_filter : SearchFilters | None, optional
            Advanced search filters for date ranges, entity types, and property-based filtering.
        config : SearchConfig | None, optional
            Advanced search configuration for ranking methods and search behavior.
        driver : GraphDriver | None, optional
            The database driver to use for the search.
        **kwargs
            Additional keyword arguments for future extensibility.

        Returns
        -------
        list[EntityEdge] | SearchResults
            Either a list of EntityEdge objects (basic search) or SearchResults object (advanced search)
            depending on whether advanced parameters are provided.

        Raises
        ------
        ValueError
            If search configuration is invalid or parameters are malformed.
        """
        try:
            # Validate input parameters
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")

            if config is not None:
                # Validate the search configuration before using it
                self.validate_search_config(config)

                # Use the advanced search_ method
                search_results = await super().search_(
                    query=query,
                    config=config,
                    center_node_uuid=center_node_uuid,
                    group_ids=group_ids,
                    search_filter=search_filter,
                    driver=driver
                )
                return search_results
            else:
                # Use the basic parent class search implementation for backward compatibility
                return await super().search(
                    query=query,
                    center_node_uuid=center_node_uuid,
                    group_ids=group_ids,
                    num_results=num_results,
                    search_filter=search_filter,
                    driver=driver
                )
        except Exception as e:
            # Log the error and re-raise with more context
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            # Re-raise ValueError as-is, wrap other exceptions
            if isinstance(e, ValueError):
                raise
            else:
                raise RuntimeError(f"Search operation failed: {e}") from e

    def create_search_filters(
        self,
        created_at_start=None,
        created_at_end=None,
        valid_at_start=None,
        valid_at_end=None,
        invalid_at_start=None,
        invalid_at_end=None,
        expired_at_start=None,
        expired_at_end=None,
        node_labels=None,
        edge_types=None,
        property_filters=None
    ):
        """
        Create SearchFilters object from simplified parameters.

        This helper method provides a convenient way to create SearchFilters
        from individual date range and entity filter parameters.

        Parameters
        ----------
        created_at_start, created_at_end : datetime, optional
            Date range for filtering by creation time.
        valid_at_start, valid_at_end : datetime, optional
            Date range for filtering by validity time.
        invalid_at_start, invalid_at_end : datetime, optional
            Date range for filtering by invalidation time.
        expired_at_start, expired_at_end : datetime, optional
            Date range for filtering by expiration time.
        node_labels : list[str], optional
            Filter results by node labels.
        edge_types : list[str], optional
            Filter results by edge types.
        property_filters : list, optional
            Advanced property-based filters.

        Returns
        -------
        SearchFilters
            A SearchFilters object configured with the provided criteria.
        """
        from graphiti_core.search.search_filters import (
            SearchFilters as CoreSearchFilters,
            DateFilter as CoreDateFilter,
            ComparisonOperator as CoreComparisonOperator,
        )

        filters = CoreSearchFilters()

        # Add date range filters
        if created_at_start is not None or created_at_end is not None:
            if filters.created_at is None:
                filters.created_at = []

            date_filters = []
            if created_at_start is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=created_at_start,
                        comparison_operator=CoreComparisonOperator.greater_than_equal
                    )
                )
            if created_at_end is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=created_at_end,
                        comparison_operator=CoreComparisonOperator.less_than_equal
                    )
                )
            if date_filters:
                filters.created_at.append(date_filters)

        # Similar logic for other date ranges
        if valid_at_start is not None or valid_at_end is not None:
            if filters.valid_at is None:
                filters.valid_at = []

            date_filters = []
            if valid_at_start is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=valid_at_start,
                        comparison_operator=CoreComparisonOperator.greater_than_equal
                    )
                )
            if valid_at_end is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=valid_at_end,
                        comparison_operator=CoreComparisonOperator.less_than_equal
                    )
                )
            if date_filters:
                filters.valid_at.append(date_filters)

        if invalid_at_start is not None or invalid_at_end is not None:
            if filters.invalid_at is None:
                filters.invalid_at = []

            date_filters = []
            if invalid_at_start is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=invalid_at_start,
                        comparison_operator=CoreComparisonOperator.greater_than_equal
                    )
                )
            if invalid_at_end is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=invalid_at_end,
                        comparison_operator=CoreComparisonOperator.less_than_equal
                    )
                )
            if date_filters:
                filters.invalid_at.append(date_filters)

        if expired_at_start is not None or expired_at_end is not None:
            if filters.expired_at is None:
                filters.expired_at = []

            date_filters = []
            if expired_at_start is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=expired_at_start,
                        comparison_operator=CoreComparisonOperator.greater_than_equal
                    )
                )
            if expired_at_end is not None:
                date_filters.append(
                    CoreDateFilter(
                        date=expired_at_end,
                        comparison_operator=CoreComparisonOperator.less_than_equal
                    )
                )
            if date_filters:
                filters.expired_at.append(date_filters)

        # Add entity filters
        if node_labels is not None:
            filters.node_labels = node_labels
        if edge_types is not None:
            filters.edge_types = edge_types
        if property_filters is not None:
            filters.property_filters = property_filters

        return filters

    def create_search_config(
        self,
        limit: int = 10,
        edge_search_methods=None,
        edge_reranker=None,
        node_search_methods=None,
        node_reranker=None,
        min_score: float = None,
        mmr_lambda: float = None,
        reranker_min_score: float = None,
        bfs_max_depth: int = None
    ):
        """
        Create SearchConfig object from simplified parameters.

        This helper method provides a convenient way to create SearchConfig
        objects for common search scenarios while supporting full customization.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results to return. Defaults to 10.
        edge_search_methods : list, optional
            List of edge search methods. Defaults to [bm25, cosine_similarity].
        edge_reranker : str, optional
            Edge reranker method. Defaults to 'rrf'.
        node_search_methods : list, optional
            List of node search methods. Defaults to [bm25, cosine_similarity].
        node_reranker : str, optional
            Node reranker method. Defaults to 'rrf'.
        min_score : float, optional
            Minimum similarity score threshold. Defaults to system default.
        mmr_lambda : float, optional
            MMR lambda parameter for diversity vs relevance balance.
        reranker_min_score : float, optional
            Minimum reranker score threshold. Defaults to 0.
        bfs_max_depth : int, optional
            Maximum BFS depth for breadth-first search. Defaults to system default.

        Returns
        -------
        SearchConfig
            A SearchConfig object configured with the provided criteria.
        """
        from graphiti_core.search.search_config import (
            SearchConfig,
            EdgeSearchConfig,
            NodeSearchConfig,
            EdgeSearchMethod as CoreEdgeSearchMethod,
            NodeSearchMethod as CoreNodeSearchMethod,
            EdgeReranker as CoreEdgeReranker,
            NodeReranker as CoreNodeReranker,
        )
        from graphiti_core.search.search_utils import (
            DEFAULT_MIN_SCORE,
            DEFAULT_MMR_LAMBDA,
            MAX_SEARCH_DEPTH,
        )

        # Convert string values to core enums
        if edge_search_methods is None:
            edge_search_methods = [CoreEdgeSearchMethod.bm25, CoreEdgeSearchMethod.cosine_similarity]
        else:
            edge_search_methods = [CoreEdgeSearchMethod(method) if isinstance(method, str) else method for method in edge_search_methods]

        if edge_reranker is None:
            edge_reranker = CoreEdgeReranker.rrf
        else:
            edge_reranker = CoreEdgeReranker(edge_reranker) if isinstance(edge_reranker, str) else edge_reranker

        if node_search_methods is None:
            node_search_methods = [CoreNodeSearchMethod.bm25, CoreNodeSearchMethod.cosine_similarity]
        else:
            node_search_methods = [CoreNodeSearchMethod(method) if isinstance(method, str) else method for method in node_search_methods]

        if node_reranker is None:
            node_reranker = CoreNodeReranker.rrf
        else:
            node_reranker = CoreNodeReranker(node_reranker) if isinstance(node_reranker, str) else node_reranker

        # Create edge configuration
        edge_config = EdgeSearchConfig(
            search_methods=edge_search_methods,
            reranker=edge_reranker,
            sim_min_score=min_score or DEFAULT_MIN_SCORE,
            mmr_lambda=mmr_lambda or DEFAULT_MMR_LAMBDA,
            bfs_max_depth=bfs_max_depth or MAX_SEARCH_DEPTH,
        )

        # Create node configuration
        node_config = NodeSearchConfig(
            search_methods=node_search_methods,
            reranker=node_reranker,
            sim_min_score=min_score or DEFAULT_MIN_SCORE,
            mmr_lambda=mmr_lambda or DEFAULT_MMR_LAMBDA,
            bfs_max_depth=bfs_max_depth or MAX_SEARCH_DEPTH,
        )

        return SearchConfig(
            edge_config=edge_config,
            node_config=node_config,
            limit=limit,
            reranker_min_score=reranker_min_score or 0,
        )

    def create_edge_search_config(
        self,
        limit: int = 10,
        search_methods=None,
        reranker: str = 'rrf',
        min_score: float = None,
        mmr_lambda: float = None,
        reranker_min_score: float = None,
        bfs_max_depth: int = None
    ):
        """
        Create edge-only SearchConfig object.

        This helper method creates a SearchConfig that only searches edges,
        which is useful for focused entity relationship searches.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results to return. Defaults to 10.
        search_methods : list, optional
            List of edge search methods. Defaults to [bm25, cosine_similarity].
        reranker : str, optional
            Edge reranker method. Defaults to 'rrf'.
        min_score : float, optional
            Minimum similarity score threshold. Defaults to system default.
        mmr_lambda : float, optional
            MMR lambda parameter for diversity vs relevance balance.
        reranker_min_score : float, optional
            Minimum reranker score threshold. Defaults to 0.
        bfs_max_depth : int, optional
            Maximum BFS depth for breadth-first search. Defaults to system default.

        Returns
        -------
        SearchConfig
            A SearchConfig object configured for edge-only search.
        """
        from graphiti_core.search.search_config import (
            SearchConfig,
            EdgeSearchConfig,
            EdgeSearchMethod as CoreEdgeSearchMethod,
            EdgeReranker as CoreEdgeReranker,
        )
        from graphiti_core.search.search_utils import (
            DEFAULT_MIN_SCORE,
            DEFAULT_MMR_LAMBDA,
            MAX_SEARCH_DEPTH,
        )

        # Convert string values to core enums
        if search_methods is None:
            search_methods = [CoreEdgeSearchMethod.bm25, CoreEdgeSearchMethod.cosine_similarity]
        else:
            search_methods = [CoreEdgeSearchMethod(method) if isinstance(method, str) else method for method in search_methods]

        core_reranker = CoreEdgeReranker(reranker) if isinstance(reranker, str) else reranker

        edge_config = EdgeSearchConfig(
            search_methods=search_methods,
            reranker=core_reranker,
            sim_min_score=min_score or DEFAULT_MIN_SCORE,
            mmr_lambda=mmr_lambda or DEFAULT_MMR_LAMBDA,
            bfs_max_depth=bfs_max_depth or MAX_SEARCH_DEPTH,
        )

        return SearchConfig(
            edge_config=edge_config,
            limit=limit,
            reranker_min_score=reranker_min_score or 0,
        )

    def create_node_search_config(
        self,
        limit: int = 10,
        search_methods=None,
        reranker: str = 'rrf',
        min_score: float = None,
        mmr_lambda: float = None,
        reranker_min_score: float = None,
        bfs_max_depth: int = None
    ):
        """
        Create node-only SearchConfig object.

        This helper method creates a SearchConfig that only searches nodes,
        which is useful for focused entity discovery.

        Parameters
        ----------
        limit : int, optional
            Maximum number of results to return. Defaults to 10.
        search_methods : list, optional
            List of node search methods. Defaults to [bm25, cosine_similarity].
        reranker : str, optional
            Node reranker method. Defaults to 'rrf'.
        min_score : float, optional
            Minimum similarity score threshold. Defaults to system default.
        mmr_lambda : float, optional
            MMR lambda parameter for diversity vs relevance balance.
        reranker_min_score : float, optional
            Minimum reranker score threshold. Defaults to 0.
        bfs_max_depth : int, optional
            Maximum BFS depth for breadth-first search. Defaults to system default.

        Returns
        -------
        SearchConfig
            A SearchConfig object configured for node-only search.
        """
        from graphiti_core.search.search_config import (
            SearchConfig,
            NodeSearchConfig,
            NodeSearchMethod as CoreNodeSearchMethod,
            NodeReranker as CoreNodeReranker,
        )
        from graphiti_core.search.search_utils import (
            DEFAULT_MIN_SCORE,
            DEFAULT_MMR_LAMBDA,
            MAX_SEARCH_DEPTH,
        )

        # Convert string values to core enums
        if search_methods is None:
            search_methods = [CoreNodeSearchMethod.bm25, CoreNodeSearchMethod.cosine_similarity]
        else:
            search_methods = [CoreNodeSearchMethod(method) if isinstance(method, str) else method for method in search_methods]

        core_reranker = CoreNodeReranker(reranker) if isinstance(reranker, str) else reranker

        node_config = NodeSearchConfig(
            search_methods=search_methods,
            reranker=core_reranker,
            sim_min_score=min_score or DEFAULT_MIN_SCORE,
            mmr_lambda=mmr_lambda or DEFAULT_MMR_LAMBDA,
            bfs_max_depth=bfs_max_depth or MAX_SEARCH_DEPTH,
        )

        return SearchConfig(
            node_config=node_config,
            limit=limit,
            reranker_min_score=reranker_min_score or 0,
        )

    def create_search_config_from_recipe(
        self,
        recipe_name: str,
        limit: int = None,
        **override_params
    ):
        """
        Create SearchConfig from a predefined recipe with optional parameter overrides.

        This helper method provides easy access to predefined search configurations
        while allowing customization of specific parameters.

        Parameters
        ----------
        recipe_name : str
            Name of the predefined recipe. Available options:
            - 'edge_hybrid_rrf': Hybrid edge search with reciprocal rank fusion
            - 'edge_hybrid_mmr': Hybrid edge search with MMR reranking
            - 'edge_hybrid_cross_encoder': Hybrid edge search with cross-encoder reranking
            - 'edge_hybrid_node_distance': Hybrid edge search with node distance reranking
            - 'edge_hybrid_episode_mentions': Hybrid edge search with episode mentions reranking
            - 'node_hybrid_rrf': Hybrid node search with reciprocal rank fusion
            - 'node_hybrid_mmr': Hybrid node search with MMR reranking
            - 'node_hybrid_cross_encoder': Hybrid node search with cross-encoder reranking
            - 'node_hybrid_node_distance': Hybrid node search with node distance reranking
            - 'node_hybrid_episode_mentions': Hybrid node search with episode mentions reranking
            - 'combined_hybrid_rrf': Hybrid search over edges, nodes, and communities with RRF
            - 'combined_hybrid_mmr': Hybrid search over edges, nodes, and communities with MMR
            - 'combined_hybrid_cross_encoder': Hybrid search with cross-encoder reranking
            - 'community_hybrid_rrf': Hybrid community search with reciprocal rank fusion
            - 'community_hybrid_mmr': Hybrid community search with MMR reranking
            - 'community_hybrid_cross_encoder': Hybrid community search with cross-encoder reranking
        limit : int, optional
            Override the default limit from the recipe.
        **override_params
            Additional parameters to override in the search configuration.

        Returns
        -------
        SearchConfig
            A SearchConfig object based on the specified recipe with overrides applied.

        Raises
        ------
        ValueError
            If the specified recipe name is not found.
        """
        from graphiti_core.search.search_config_recipes import (
            EDGE_HYBRID_SEARCH_RRF,
            EDGE_HYBRID_SEARCH_MMR,
            EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            EDGE_HYBRID_SEARCH_NODE_DISTANCE,
            EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
            NODE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_MMR,
            NODE_HYBRID_SEARCH_CROSS_ENCODER,
            NODE_HYBRID_SEARCH_NODE_DISTANCE,
            NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
            COMBINED_HYBRID_SEARCH_RRF,
            COMBINED_HYBRID_SEARCH_MMR,
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            COMMUNITY_HYBRID_SEARCH_RRF,
            COMMUNITY_HYBRID_SEARCH_MMR,
            COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER,
        )

        # Map recipe names to configuration objects
        recipe_map = {
            'edge_hybrid_rrf': EDGE_HYBRID_SEARCH_RRF,
            'edge_hybrid_mmr': EDGE_HYBRID_SEARCH_MMR,
            'edge_hybrid_cross_encoder': EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            'edge_hybrid_node_distance': EDGE_HYBRID_SEARCH_NODE_DISTANCE,
            'edge_hybrid_episode_mentions': EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
            'node_hybrid_rrf': NODE_HYBRID_SEARCH_RRF,
            'node_hybrid_mmr': NODE_HYBRID_SEARCH_MMR,
            'node_hybrid_cross_encoder': NODE_HYBRID_SEARCH_CROSS_ENCODER,
            'node_hybrid_node_distance': NODE_HYBRID_SEARCH_NODE_DISTANCE,
            'node_hybrid_episode_mentions': NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
            'combined_hybrid_rrf': COMBINED_HYBRID_SEARCH_RRF,
            'combined_hybrid_mmr': COMBINED_HYBRID_SEARCH_MMR,
            'combined_hybrid_cross_encoder': COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            'community_hybrid_rrf': COMMUNITY_HYBRID_SEARCH_RRF,
            'community_hybrid_mmr': COMMUNITY_HYBRID_SEARCH_MMR,
            'community_hybrid_cross_encoder': COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER,
        }

        if recipe_name not in recipe_map:
            available_recipes = list(recipe_map.keys())
            raise ValueError(f"Unknown recipe '{recipe_name}'. Available recipes: {available_recipes}")

        # Get the base configuration
        config = recipe_map[recipe_name]

        # Create a copy to avoid modifying the original
        config_dict = config.model_dump()

        # Apply limit override if provided
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"Limit must be a positive integer, got {limit}")
            config_dict['limit'] = limit

        # Apply additional parameter overrides with validation
        for param_name, param_value in override_params.items():
            try:
                if param_name in config_dict:
                    # Validate top-level parameters
                    if param_name == 'limit' and param_value is not None:
                        if param_value <= 0:
                            raise ValueError(f"Limit must be a positive integer, got {param_value}")
                    elif param_name == 'reranker_min_score' and param_value is not None:
                        if not isinstance(param_value, (int, float)) or param_value < 0 or param_value > 1:
                            raise ValueError(f"reranker_min_score must be a number between 0 and 1, got {param_value}")

                    config_dict[param_name] = param_value
                elif '.' in param_name:
                    # Handle nested parameters like 'edge_config.reranker'
                    parts = param_name.split('.')
                    if len(parts) == 2 and parts[0] in config_dict:
                        nested_config_name = parts[0]
                        nested_param_name = parts[1]

                        # Initialize nested config if it's None
                        if config_dict[nested_config_name] is None:
                            from graphiti_core.search.search_config import (
                                EdgeSearchConfig, NodeSearchConfig, EpisodeSearchConfig, CommunitySearchConfig
                            )
                            if nested_config_name == 'edge_config':
                                config_dict[nested_config_name] = EdgeSearchConfig().model_dump()
                            elif nested_config_name == 'node_config':
                                config_dict[nested_config_name] = NodeSearchConfig().model_dump()
                            elif nested_config_name == 'episode_config':
                                config_dict[nested_config_name] = EpisodeSearchConfig().model_dump()
                            elif nested_config_name == 'community_config':
                                config_dict[nested_config_name] = CommunitySearchConfig().model_dump()
                            else:
                                continue  # Skip unknown nested config

                        # Validate common nested parameters
                        if nested_param_name in ['sim_min_score', 'mmr_lambda'] and param_value is not None:
                            if not isinstance(param_value, (int, float)) or param_value < 0 or param_value > 1:
                                raise ValueError(f"{param_name} must be a number between 0 and 1, got {param_value}")
                        elif nested_param_name == 'bfs_max_depth' and param_value is not None:
                            if not isinstance(param_value, int) or param_value <= 0:
                                raise ValueError(f"{param_name} must be a positive integer, got {param_value}")

                        config_dict[nested_config_name][nested_param_name] = param_value
            except (ValueError, TypeError, AttributeError) as e:
                raise ValueError(f"Invalid parameter override for '{param_name}': {e}") from e

        # Reconstruct the SearchConfig from the modified dictionary with validation
        try:
            from graphiti_core.search.search_config import SearchConfig
            return SearchConfig.model_validate(config_dict)
        except Exception as e:
            raise ValueError(f"Failed to create valid SearchConfig from recipe '{recipe_name}': {e}") from e

    def get_available_recipes(self):
        """
        Get a list of all available search recipe names.

        Returns
        -------
        list[str]
            List of available recipe names that can be used with create_search_config_from_recipe().
        """
        from graphiti_core.search.search_config_recipes import (
            EDGE_HYBRID_SEARCH_RRF,
            EDGE_HYBRID_SEARCH_MMR,
            EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            EDGE_HYBRID_SEARCH_NODE_DISTANCE,
            EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
            NODE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_MMR,
            NODE_HYBRID_SEARCH_CROSS_ENCODER,
            NODE_HYBRID_SEARCH_NODE_DISTANCE,
            NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
            COMBINED_HYBRID_SEARCH_RRF,
            COMBINED_HYBRID_SEARCH_MMR,
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            COMMUNITY_HYBRID_SEARCH_RRF,
            COMMUNITY_HYBRID_SEARCH_MMR,
            COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER,
        )

        return [
            'edge_hybrid_rrf',
            'edge_hybrid_mmr',
            'edge_hybrid_cross_encoder',
            'edge_hybrid_node_distance',
            'edge_hybrid_episode_mentions',
            'node_hybrid_rrf',
            'node_hybrid_mmr',
            'node_hybrid_cross_encoder',
            'node_hybrid_node_distance',
            'node_hybrid_episode_mentions',
            'combined_hybrid_rrf',
            'combined_hybrid_mmr',
            'combined_hybrid_cross_encoder',
            'community_hybrid_rrf',
            'community_hybrid_mmr',
            'community_hybrid_cross_encoder',
        ]

    def validate_search_config(self, config):
        """
        Validate a SearchConfig object for common configuration issues.

        This method performs validation checks on search configurations to ensure
        they are properly configured and will work correctly with the search system.

        Parameters
        ----------
        config : SearchConfig
            The SearchConfig object to validate.

        Returns
        -------
        bool
            True if the configuration is valid.

        Raises
        ------
        ValueError
            If the configuration has invalid parameters.
        """
        if config is None:
            raise ValueError("SearchConfig cannot be None")

        # Validate limit
        if hasattr(config, 'limit') and config.limit is not None:
            if not isinstance(config.limit, int) or config.limit <= 0:
                raise ValueError(f"SearchConfig.limit must be a positive integer, got {config.limit}")

        # Validate reranker_min_score
        if hasattr(config, 'reranker_min_score') and config.reranker_min_score is not None:
            if not isinstance(config.reranker_min_score, (int, float)) or config.reranker_min_score < 0 or config.reranker_min_score > 1:
                raise ValueError(f"SearchConfig.reranker_min_score must be between 0 and 1, got {config.reranker_min_score}")

        # Validate edge_config if present
        if hasattr(config, 'edge_config') and config.edge_config is not None:
            edge_config = config.edge_config
            if hasattr(edge_config, 'sim_min_score') and edge_config.sim_min_score is not None:
                if not isinstance(edge_config.sim_min_score, (int, float)) or edge_config.sim_min_score < 0 or edge_config.sim_min_score > 1:
                    raise ValueError(f"edge_config.sim_min_score must be between 0 and 1, got {edge_config.sim_min_score}")
            if hasattr(edge_config, 'mmr_lambda') and edge_config.mmr_lambda is not None:
                if not isinstance(edge_config.mmr_lambda, (int, float)) or edge_config.mmr_lambda < 0 or edge_config.mmr_lambda > 1:
                    raise ValueError(f"edge_config.mmr_lambda must be between 0 and 1, got {edge_config.mmr_lambda}")
            if hasattr(edge_config, 'bfs_max_depth') and edge_config.bfs_max_depth is not None:
                if not isinstance(edge_config.bfs_max_depth, int) or edge_config.bfs_max_depth <= 0:
                    raise ValueError(f"edge_config.bfs_max_depth must be a positive integer, got {edge_config.bfs_max_depth}")

        # Validate node_config if present
        if hasattr(config, 'node_config') and config.node_config is not None:
            node_config = config.node_config
            if hasattr(node_config, 'sim_min_score') and node_config.sim_min_score is not None:
                if not isinstance(node_config.sim_min_score, (int, float)) or node_config.sim_min_score < 0 or node_config.sim_min_score > 1:
                    raise ValueError(f"node_config.sim_min_score must be between 0 and 1, got {node_config.sim_min_score}")
            if hasattr(node_config, 'mmr_lambda') and node_config.mmr_lambda is not None:
                if not isinstance(node_config.mmr_lambda, (int, float)) or node_config.mmr_lambda < 0 or node_config.mmr_lambda > 1:
                    raise ValueError(f"node_config.mmr_lambda must be between 0 and 1, got {node_config.mmr_lambda}")
            if hasattr(node_config, 'bfs_max_depth') and node_config.bfs_max_depth is not None:
                if not isinstance(node_config.bfs_max_depth, int) or node_config.bfs_max_depth <= 0:
                    raise ValueError(f"node_config.bfs_max_depth must be a positive integer, got {node_config.bfs_max_depth}")

        return True


async def get_graphiti(settings: ZepEnvDep):
    print(f"[DEBUG] Creating Graphiti client - embedding_model_name: {settings.embedding_model_name}", flush=True)
    logger.info(f"Creating Graphiti client - embedding_model_name: {settings.embedding_model_name}")

    # Create embedder first if custom model is specified
    embedder = None
    if settings.embedding_model_name is not None:
        print(f"[DEBUG] Creating custom embedder with model: {settings.embedding_model_name}", flush=True)
        embedder_config = OpenAIEmbedderConfig(
            embedding_model=settings.embedding_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        embedder = OpenAIEmbedder(embedder_config)
        print(f"[DEBUG] Embedder created - model: {embedder.config.embedding_model}", flush=True)
    else:
        print("[DEBUG] No custom embedding_model_name set, using default embedder", flush=True)
        logger.warning("No custom embedding_model_name set, using default embedder")

    # Create Graphiti client with custom embedder
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        embedder=embedder,
    )
    print(f"[DEBUG] Client created - embedder model: {client.embedder.config.embedding_model if hasattr(client.embedder, 'config') else 'unknown'}", flush=True)

    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
