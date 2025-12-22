"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from typing import List

import pytest

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, CommunityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_similarity_search,
    node_similarity_search,
    community_similarity_search,
    get_embeddings_for_nodes,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
)


class TestVectorSearch:
    """Tests for vector search functionality."""

    @pytest.mark.asyncio
    async def test_edge_similarity_search(self):
        """Test edge similarity search function."""
        # Mock driver
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock search results
        mock_edges = [
            EntityEdge(
                uuid='edge1',
                source_node_uuid='node1',
                target_node_uuid='node2',
                fact='test fact 1',
                created_at=datetime.now(),
                group_id='group1'
            ),
            EntityEdge(
                uuid='edge2',
                source_node_uuid='node3',
                target_node_uuid='node4',
                fact='test fact 2',
                created_at=datetime.now(),
                group_id='group2'
            )
        ]
        mock_driver.search_interface.edge_similarity_search.return_value = mock_edges

        # Test parameters
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        search_filter = SearchFilters()
        group_ids = ['group1', 'group2']
        limit = 10
        min_score = 0.7

        # Call the function
        result = await edge_similarity_search(
            driver=mock_driver,
            query_vector=query_vector,
            source_node_uuids=None,
            target_node_uuids=None,
            search_filter=search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        # Verify the driver interface was called correctly
        mock_driver.search_interface.edge_similarity_search.assert_called_once_with(
            query_vector=query_vector,
            source_node_uuids=None,
            target_node_uuids=None,
            search_filter=search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        # Verify results
        assert len(result) == 2
        assert result[0].uuid == 'edge1'
        assert result[1].uuid == 'edge2'

    @pytest.mark.asyncio
    async def test_edge_similarity_search_with_node_filters(self):
        """Test edge similarity search with source and target node filters."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()
        mock_driver.search_interface.edge_similarity_search.return_value = []

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        source_node_uuids = ['node1', 'node2']
        target_node_uuids = ['node3', 'node4']
        search_filter = SearchFilters()

        result = await edge_similarity_search(
            driver=mock_driver,
            query_vector=query_vector,
            source_node_uuids=source_node_uuids,
            target_node_uuids=target_node_uuids,
            search_filter=search_filter,
            group_ids=None,
            limit=10,
            min_score=0.7
        )

        mock_driver.search_interface.edge_similarity_search.assert_called_once_with(
            query_vector=query_vector,
            source_node_uuids=source_node_uuids,
            target_node_uuids=target_node_uuids,
            search_filter=search_filter,
            group_ids=None,
            limit=10,
            min_score=0.7
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_node_similarity_search(self):
        """Test node similarity search function."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock search results
        mock_nodes = [
            EntityNode(
                uuid='node1',
                name='Test Node 1',
                summary='Test summary 1',
                created_at=datetime.now(),
                group_id='group1'
            ),
            EntityNode(
                uuid='node2',
                name='Test Node 2',
                summary='Test summary 2',
                created_at=datetime.now(),
                group_id='group2'
            )
        ]
        mock_driver.search_interface.node_similarity_search.return_value = mock_nodes

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        search_filter = SearchFilters()
        group_ids = ['group1', 'group2']
        limit = 10
        min_score = 0.7

        result = await node_similarity_search(
            driver=mock_driver,
            query_vector=query_vector,
            search_filter=search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        mock_driver.search_interface.node_similarity_search.assert_called_once_with(
            query_vector=query_vector,
            search_filter=search_filter,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        assert len(result) == 2
        assert result[0].uuid == 'node1'
        assert result[1].uuid == 'node2'

    @pytest.mark.asyncio
    async def test_node_similarity_search_empty_results(self):
        """Test node similarity search with no results."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()
        mock_driver.search_interface.node_similarity_search.return_value = []

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        search_filter = SearchFilters()

        result = await node_similarity_search(
            driver=mock_driver,
            query_vector=query_vector,
            search_filter=search_filter,
            group_ids=None,
            limit=10,
            min_score=0.7
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_community_similarity_search(self):
        """Test community similarity search function."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock search results
        mock_communities = [
            CommunityNode(
                uuid='comm1',
                name='Test Community 1',
                summary='Test community summary 1',
                created_at=datetime.now()
            ),
            CommunityNode(
                uuid='comm2',
                name='Test Community 2',
                summary='Test community summary 2',
                created_at=datetime.now()
            )
        ]
        mock_driver.search_interface.community_similarity_search.return_value = mock_communities

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        group_ids = ['group1', 'group2']
        limit = 10
        min_score = 0.7

        result = await community_similarity_search(
            driver=mock_driver,
            query_vector=query_vector,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        mock_driver.search_interface.community_similarity_search.assert_called_once_with(
            query_vector=query_vector,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score
        )

        assert len(result) == 2
        assert result[0].uuid == 'comm1'
        assert result[1].uuid == 'comm2'


class TestGetEmbeddings:
    """Tests for embedding retrieval functions."""

    @pytest.mark.asyncio
    async def test_get_embeddings_for_nodes(self):
        """Test get_embeddings_for_nodes function."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock nodes
        nodes = [
            EntityNode(
                uuid='node1',
                name='Test Node 1',
                summary='Test summary 1',
                created_at=datetime.now(),
                group_id='group1'
            ),
            EntityNode(
                uuid='node2',
                name='Test Node 2',
                summary='Test summary 2',
                created_at=datetime.now(),
                group_id='group2'
            )
        ]

        # Mock embedding results
        expected_embeddings = [
            ('node1', [0.1, 0.2, 0.3]),
            ('node2', [0.4, 0.5, 0.6])
        ]
        mock_driver.search_interface.get_embeddings_for_nodes.return_value = expected_embeddings

        result = await get_embeddings_for_nodes(mock_driver, nodes)

        mock_driver.search_interface.get_embeddings_for_nodes.assert_called_once_with(nodes)
        assert result == expected_embeddings

    @pytest.mark.asyncio
    async def test_get_embeddings_for_empty_nodes(self):
        """Test get_embeddings_for_nodes with empty node list."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()
        mock_driver.search_interface.get_embeddings_for_nodes.return_value = []

        result = await get_embeddings_for_nodes(mock_driver, [])

        mock_driver.search_interface.get_embeddings_for_nodes.assert_called_once_with([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_embeddings_for_communities(self):
        """Test get_embeddings_for_communities function."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock communities
        communities = [
            CommunityNode(
                uuid='comm1',
                name='Test Community 1',
                summary='Test community summary 1',
                created_at=datetime.now()
            ),
            CommunityNode(
                uuid='comm2',
                name='Test Community 2',
                summary='Test community summary 2',
                created_at=datetime.now()
            )
        ]

        # Mock embedding results
        expected_embeddings = [
            ('comm1', [0.1, 0.2, 0.3]),
            ('comm2', [0.4, 0.5, 0.6])
        ]
        mock_driver.search_interface.get_embeddings_for_communities.return_value = expected_embeddings

        result = await get_embeddings_for_communities(mock_driver, communities)

        mock_driver.search_interface.get_embeddings_for_communities.assert_called_once_with(communities)
        assert result == expected_embeddings

    @pytest.mark.asyncio
    async def test_get_embeddings_for_edges(self):
        """Test get_embeddings_for_edges function."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Mock edges
        edges = [
            EntityEdge(
                uuid='edge1',
                source_node_uuid='node1',
                target_node_uuid='node2',
                fact='test fact 1',
                created_at=datetime.now(),
                group_id='group1'
            ),
            EntityEdge(
                uuid='edge2',
                source_node_uuid='node3',
                target_node_uuid='node4',
                fact='test fact 2',
                created_at=datetime.now(),
                group_id='group2'
            )
        ]

        # Mock embedding results
        expected_embeddings = [
            ('edge1', [0.1, 0.2, 0.3]),
            ('edge2', [0.4, 0.5, 0.6])
        ]
        mock_driver.search_interface.get_embeddings_for_edges.return_value = expected_embeddings

        result = await get_embeddings_for_edges(mock_driver, edges)

        mock_driver.search_interface.get_embeddings_for_edges.assert_called_once_with(edges)
        assert result == expected_embeddings

    @pytest.mark.asyncio
    async def test_get_embeddings_for_empty_edges(self):
        """Test get_embeddings_for_edges with empty edge list."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()
        mock_driver.search_interface.get_embeddings_for_edges.return_value = []

        result = await get_embeddings_for_edges(mock_driver, [])

        mock_driver.search_interface.get_embeddings_for_edges.assert_called_once_with([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_embeddings_integration(self):
        """Test integration between different embedding functions."""
        mock_driver = MagicMock()
        mock_driver.search_interface = AsyncMock()

        # Setup different types of objects
        nodes = [
            EntityNode(
                uuid='node1',
                name='Test Node',
                summary='Test summary',
                created_at=datetime.now(),
                group_id='group1'
            )
        ]

        communities = [
            CommunityNode(
                uuid='comm1',
                name='Test Community',
                summary='Test community summary',
                created_at=datetime.now()
            )
        ]

        edges = [
            EntityEdge(
                uuid='edge1',
                source_node_uuid='node1',
                target_node_uuid='node2',
                fact='test fact',
                created_at=datetime.now(),
                group_id='group1'
            )
        ]

        # Mock different embedding results
        mock_driver.search_interface.get_embeddings_for_nodes.return_value = [
            ('node1', [0.1, 0.2, 0.3])
        ]
        mock_driver.search_interface.get_embeddings_for_communities.return_value = [
            ('comm1', [0.4, 0.5, 0.6])
        ]
        mock_driver.search_interface.get_embeddings_for_edges.return_value = [
            ('edge1', [0.7, 0.8, 0.9])
        ]

        # Call all functions
        node_embeddings = await get_embeddings_for_nodes(mock_driver, nodes)
        community_embeddings = await get_embeddings_for_communities(mock_driver, communities)
        edge_embeddings = await get_embeddings_for_edges(mock_driver, edges)

        # Verify all were called correctly
        mock_driver.search_interface.get_embeddings_for_nodes.assert_called_once_with(nodes)
        mock_driver.search_interface.get_embeddings_for_communities.assert_called_once_with(communities)
        mock_driver.search_interface.get_embeddings_for_edges.assert_called_once_with(edges)

        # Verify results are distinct
        assert node_embeddings == [('node1', [0.1, 0.2, 0.3])]
        assert community_embeddings == [('comm1', [0.4, 0.5, 0.6])]
        assert edge_embeddings == [('edge1', [0.7, 0.8, 0.9])]

        # Verify embeddings are different
        assert node_embeddings[0][1] != community_embeddings[0][1]
        assert community_embeddings[0][1] != edge_embeddings[0][1]
        assert node_embeddings[0][1] != edge_embeddings[0][1]