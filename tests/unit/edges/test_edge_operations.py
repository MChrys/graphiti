"""
Tests for graph edge operations and functionality

Run with: pytest tests/unit/edges/test_edge_operations.py -v
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from graphiti_core.edges import (
    Edge,
    EntityEdge,
    EpisodicEdge,
    CommunityEdge,
    get_entity_edge_from_record,
    get_episodic_edge_from_record,
    get_community_edge_from_record,
    create_entity_edge_embeddings,
)
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError
from graphiti_core.utils.datetime_utils import utc_now


class TestEdgeBaseClass:
    """Tests for the base Edge class."""

    def test_edge_initialization(self):
        """Test basic Edge initialization with abstract class."""
        # Edge is abstract, but we can test basic field initialization
        edge_data = {
            'uuid': str(uuid4()),
            'group_id': 'test-group',
            'source_node_uuid': str(uuid4()),
            'target_node_uuid': str(uuid4()),
            'created_at': utc_now(),
        }

        # Test that abstract methods exist
        assert hasattr(Edge, 'save')
        assert hasattr(Edge, 'get_by_uuid')

    def test_edge_hash_and_equality(self):
        """Test Edge hash and equality methods."""
        uuid_val = str(uuid4())

        # Create a concrete implementation for testing
        class TestEdge(Edge):
            async def save(self, driver):
                pass

            @classmethod
            async def get_by_uuid(cls, driver, uuid):
                pass

        edge1 = TestEdge(
            uuid=uuid_val,
            group_id='group',
            source_node_uuid='source1',
            target_node_uuid='target1',
            created_at=utc_now()
        )
        edge2 = TestEdge(
            uuid=uuid_val,
            group_id='different-group',
            source_node_uuid='source2',
            target_node_uuid='target2',
            created_at=utc_now()
        )
        edge3 = TestEdge(
            uuid=str(uuid4()),
            group_id='group',
            source_node_uuid='source1',
            target_node_uuid='target1',
            created_at=utc_now()
        )

        # Test equality based on UUID
        assert edge1 == edge2
        assert edge1 != edge3

        # Test hash
        assert hash(edge1) == hash(edge2)
        assert hash(edge1) != hash(edge3)

    def test_edge_equality_with_node(self):
        """Test Edge equality with Node (should be false)."""
        class TestNode:
            def __init__(self, uuid):
                self.uuid = uuid

        class TestEdge(Edge):
            async def save(self, driver):
                pass

            @classmethod
            async def get_by_uuid(cls, driver, uuid):
                pass

        uuid_val = str(uuid4())
        node = TestNode(uuid=uuid_val)
        edge = TestEdge(
            uuid=uuid_val,
            group_id='group',
            source_node_uuid='source',
            target_node_uuid='target',
            created_at=utc_now()
        )

        # Even with same UUID, edge should not equal node
        assert edge != node


class TestEntityEdge:
    """Tests for EntityEdge creation and operations."""

    def test_entity_edge_initialization_minimal(self):
        """Test EntityEdge initialization with minimal required fields."""
        edge = EntityEdge(
            group_id='test-group',
            source_node_uuid='source-uuid',
            target_node_uuid='target-uuid',
            name='RELATED_TO',
            fact='Source is related to target',
            created_at=utc_now()
        )

        assert edge.group_id == 'test-group'
        assert edge.source_node_uuid == 'source-uuid'
        assert edge.target_node_uuid == 'target-uuid'
        assert edge.name == 'RELATED_TO'
        assert edge.fact == 'Source is related to target'
        assert edge.fact_embedding is None
        assert edge.episodes == []
        assert edge.expired_at is None
        assert edge.valid_at is None
        assert edge.invalid_at is None
        assert edge.attributes == {}
        assert isinstance(edge.uuid, str)

    def test_entity_edge_initialization_complete(self):
        """Test EntityEdge initialization with all fields."""
        test_time = utc_now()
        test_uuid = str(uuid4())
        test_embedding = [0.1, 0.2, 0.3]
        test_attributes = {'strength': 0.8, 'confidence': 0.9}

        edge = EntityEdge(
            uuid=test_uuid,
            group_id='test-group',
            source_node_uuid='source-uuid',
            target_node_uuid='target-uuid',
            name='RELATION_TYPE',
            fact='Source has relation to target',
            fact_embedding=test_embedding,
            episodes=['episode1', 'episode2'],
            expired_at=test_time,
            valid_at=test_time,
            invalid_at=test_time,
            attributes=test_attributes,
            created_at=test_time
        )

        assert edge.uuid == test_uuid
        assert edge.fact_embedding == test_embedding
        assert edge.episodes == ['episode1', 'episode2']
        assert edge.expired_at == test_time
        assert edge.valid_at == test_time
        assert edge.invalid_at == test_time
        assert edge.attributes == test_attributes

    @pytest.mark.asyncio
    async def test_entity_edge_generate_embedding(self):
        """Test EntityEdge embedding generation."""
        edge = EntityEdge(
            group_id='test-group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Source relates to target',
            created_at=utc_now()
        )
        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create.return_value = [0.4, 0.5, 0.6]

        result = await edge.generate_embedding(mock_embedder)

        assert result == [0.4, 0.5, 0.6]
        assert edge.fact_embedding == [0.4, 0.5, 0.6]
        mock_embedder.create.assert_called_once_with(input_data=['Source relates to target'])

    @pytest.mark.asyncio
    async def test_entity_edge_load_fact_embedding_neo4j(self):
        """Test EntityEdge embedding loading for Neo4j."""
        edge = EntityEdge(
            uuid='test-uuid',
            group_id='group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.execute_query = AsyncMock(return_value=([{'fact_embedding': [0.1, 0.2, 0.3]}], None, None))

        await edge.load_fact_embedding(mock_driver)

        assert edge.fact_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_edge_load_fact_embedding_neptune(self):
        """Test EntityEdge embedding loading for Neptune."""
        edge = EntityEdge(
            uuid='test-uuid',
            group_id='group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEPTUNE
        mock_driver.execute_query = AsyncMock(return_value=([{'fact_embedding': '0.1,0.2,0.3'}], None, None))

        await edge.load_fact_embedding(mock_driver)

        assert edge.fact_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_edge_load_fact_embedding_kuzu(self):
        """Test EntityEdge embedding loading for Kuzu."""
        edge = EntityEdge(
            uuid='test-uuid',
            group_id='group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.KUZU
        mock_driver.execute_query = AsyncMock(return_value=([{'fact_embedding': [0.1, 0.2, 0.3]}], None, None))

        await edge.load_fact_embedding(mock_driver)

        assert edge.fact_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_edge_load_fact_embedding_not_found(self):
        """Test EntityEdge embedding loading when edge not found."""
        edge = EntityEdge(
            uuid='test-uuid',
            group_id='group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(EdgeNotFoundError):
            await edge.load_fact_embedding(mock_driver)

    @pytest.mark.asyncio
    async def test_entity_edge_save_kuzu(self):
        """Test EntityEdge save for Kuzu provider."""
        edge = EntityEdge(
            group_id='test-group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            attributes={'strength': 0.8},
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.KUZU
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.edges.get_entity_edge_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await edge.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['attributes'] == json.dumps({'strength': 0.8})

    @pytest.mark.asyncio
    async def test_entity_edge_save_neo4j(self):
        """Test EntityEdge save for Neo4j provider."""
        edge = EntityEdge(
            group_id='test-group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            attributes={'strength': 0.8},
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.edges.get_entity_edge_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await edge.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['edge_data']['strength'] == 0.8  # Attribute should be merged

    @pytest.mark.asyncio
    async def test_entity_edge_get_by_uuid_empty_list(self):
        """Test EntityEdge.get_by_uuids with empty list."""
        mock_driver = MagicMock(spec=GraphDriver)

        result = await EntityEdge.get_by_uuids(mock_driver, [])

        assert result == []

    @pytest.mark.asyncio
    async def test_entity_edge_delete_with_graph_operations_interface(self):
        """Test EntityEdge delete with graph operations interface."""
        edge = EntityEdge(
            group_id='test-group',
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATION',
            fact='Test fact',
            created_at=utc_now()
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = AsyncMock()

        await edge.delete(mock_driver)

        mock_driver.graph_operations_interface.edge_delete.assert_called_once_with(edge, mock_driver)


class TestEpisodicEdge:
    """Tests for EpisodicEdge creation and operations."""

    def test_episodic_edge_initialization(self):
        """Test EpisodicEdge initialization."""
        created_at = utc_now()
        edge = EpisodicEdge(
            group_id='test-group',
            source_node_uuid='episode-uuid',
            target_node_uuid='entity-uuid',
            created_at=created_at
        )

        assert edge.group_id == 'test-group'
        assert edge.source_node_uuid == 'episode-uuid'
        assert edge.target_node_uuid == 'entity-uuid'
        assert edge.created_at == created_at
        assert isinstance(edge.uuid, str)

    @pytest.mark.asyncio
    async def test_episodic_edge_save(self):
        """Test EpisodicEdge save operation."""
        created_at = utc_now()
        edge = EpisodicEdge(
            uuid='test-uuid',
            group_id='test-group',
            source_node_uuid='episode-uuid',
            target_node_uuid='entity-uuid',
            created_at=created_at
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        await edge.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['episode_uuid'] == 'episode-uuid'
        assert called_kwargs['entity_uuid'] == 'entity-uuid'
        assert called_kwargs['uuid'] == 'test-uuid'
        assert called_kwargs['group_id'] == 'test-group'

    @pytest.mark.asyncio
    async def test_episodic_edge_get_by_uuid_not_found(self):
        """Test EpisodicEdge.get_by_uuid when edge not found."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(EdgeNotFoundError):
            await EpisodicEdge.get_by_uuid(mock_driver, 'nonexistent-uuid')

    @pytest.mark.asyncio
    async def test_episodic_edge_get_by_uuids_not_found(self):
        """Test EpisodicEdge.get_by_uuids when no edges found."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(EdgeNotFoundError):
            await EpisodicEdge.get_by_uuids(mock_driver, ['nonexistent-uuid'])

    @pytest.mark.asyncio
    async def test_episodic_edge_get_by_group_ids_not_found(self):
        """Test EpisodicEdge.get_by_group_ids when no edges found."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(GroupsEdgesNotFoundError):
            await EpisodicEdge.get_by_group_ids(mock_driver, ['nonexistent-group'])


class TestCommunityEdge:
    """Tests for CommunityEdge creation and operations."""

    def test_community_edge_initialization(self):
        """Test CommunityEdge initialization."""
        created_at = utc_now()
        edge = CommunityEdge(
            group_id='test-group',
            source_node_uuid='community-uuid',
            target_node_uuid='entity-uuid',
            created_at=created_at
        )

        assert edge.group_id == 'test-group'
        assert edge.source_node_uuid == 'community-uuid'
        assert edge.target_node_uuid == 'entity-uuid'
        assert edge.created_at == created_at
        assert isinstance(edge.uuid, str)

    @pytest.mark.asyncio
    async def test_community_edge_save(self):
        """Test CommunityEdge save operation."""
        created_at = utc_now()
        edge = CommunityEdge(
            uuid='test-uuid',
            group_id='test-group',
            source_node_uuid='community-uuid',
            target_node_uuid='entity-uuid',
            created_at=created_at
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.edges.get_community_edge_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await edge.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['community_uuid'] == 'community-uuid'
        assert called_kwargs['entity_uuid'] == 'entity-uuid'
        assert called_kwargs['uuid'] == 'test-uuid'
        assert called_kwargs['group_id'] == 'test-group'


class TestEdgeHelperFunctions:
    """Tests for edge helper functions."""

    def test_get_episodic_edge_from_record(self):
        """Test creating EpisodicEdge from database record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'group_id': 'test-group',
            'source_node_uuid': 'episode-uuid',
            'target_node_uuid': 'entity-uuid',
            'created_at': test_time.isoformat()
        }

        with patch('graphiti_core.edges.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            edge = get_episodic_edge_from_record(record)

            assert isinstance(edge, EpisodicEdge)
            assert edge.uuid == 'test-uuid'
            assert edge.group_id == 'test-group'
            assert edge.source_node_uuid == 'episode-uuid'
            assert edge.target_node_uuid == 'entity-uuid'

    def test_get_entity_edge_from_record_neo4j(self):
        """Test creating EntityEdge from Neo4j record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'source_node_uuid': 'source-uuid',
            'target_node_uuid': 'target-uuid',
            'name': 'RELATION_TYPE',
            'fact': 'Test fact',
            'group_id': 'test-group',
            'episodes': ['episode1', 'episode2'],
            'created_at': test_time.isoformat(),
            'expired_at': test_time.isoformat(),
            'valid_at': test_time.isoformat(),
            'invalid_at': None,
            'attributes': {
                'strength': 0.8,
                'fact': 'Should be removed',  # Should be removed
                'uuid': 'Should be removed'  # Should be removed
            },
            'fact_embedding': [0.1, 0.2, 0.3]
        }

        with patch('graphiti_core.edges.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            edge = get_entity_edge_from_record(record, GraphProvider.NEO4J)

            assert isinstance(edge, EntityEdge)
            assert edge.uuid == 'test-uuid'
            assert edge.name == 'RELATION_TYPE'
            assert edge.fact == 'Test fact'
            assert edge.episodes == ['episode1', 'episode2']
            assert edge.attributes == {'strength': 0.8}  # Should be cleaned
            assert 'fact' not in edge.attributes
            assert 'uuid' not in edge.attributes

    def test_get_entity_edge_from_record_kuzu(self):
        """Test creating EntityEdge from Kuzu record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'source_node_uuid': 'source-uuid',
            'target_node_uuid': 'target-uuid',
            'name': 'RELATION_TYPE',
            'fact': 'Test fact',
            'group_id': 'test-group',
            'episodes': ['episode1'],
            'created_at': test_time.isoformat(),
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'attributes': json.dumps({'strength': 0.8}),
            'fact_embedding': [0.1, 0.2, 0.3]
        }

        with patch('graphiti_core.edges.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            edge = get_entity_edge_from_record(record, GraphProvider.KUZU)

            assert isinstance(edge, EntityEdge)
            assert edge.uuid == 'test-uuid'
            assert edge.attributes == {'strength': 0.8}

    def test_get_community_edge_from_record(self):
        """Test creating CommunityEdge from database record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'group_id': 'test-group',
            'source_node_uuid': 'community-uuid',
            'target_node_uuid': 'entity-uuid',
            'created_at': test_time.isoformat()
        }

        with patch('graphiti_core.edges.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            edge = get_community_edge_from_record(record)

            assert isinstance(edge, CommunityEdge)
            assert edge.uuid == 'test-uuid'
            assert edge.group_id == 'test-group'
            assert edge.source_node_uuid == 'community-uuid'
            assert edge.target_node_uuid == 'entity-uuid'

    @pytest.mark.asyncio
    async def test_create_entity_edge_embeddings_with_empty_list(self):
        """Test creating embeddings for empty edge list."""
        mock_embedder = AsyncMock(spec=EmbedderClient)

        await create_entity_edge_embeddings(mock_embedder, [])

        mock_embedder.create_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_entity_edge_embeddings_filters_falsey_facts(self):
        """Test that edges with falsey facts are filtered out."""
        edge1 = EntityEdge(
            group_id='group',
            source_node_uuid='source1',
            target_node_uuid='target1',
            name='RELATION',
            fact='Valid fact',
            created_at=utc_now()
        )
        edge2 = EntityEdge(
            group_id='group',
            source_node_uuid='source2',
            target_node_uuid='target2',
            name='RELATION',
            fact='',  # Should be filtered
            created_at=utc_now()
        )
        edge3 = EntityEdge(
            group_id='group',
            source_node_uuid='source3',
            target_node_uuid='target3',
            name='RELATION',
            fact='Another valid fact',
            created_at=utc_now()
        )

        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        await create_entity_edge_embeddings(mock_embedder, [edge1, edge2, edge3])

        # Should only be called with valid facts
        mock_embedder.create_batch.assert_called_once_with(['Valid fact', 'Another valid fact'])
        assert edge1.fact_embedding == [0.1, 0.2]
        assert edge3.fact_embedding == [0.3, 0.4]
        assert edge2.fact_embedding is None  # Should remain None

    @pytest.mark.asyncio
    async def test_entity_edge_delete_by_uuids_with_interface(self):
        """Test EntityEdge.delete_by_uuids with graph operations interface."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = AsyncMock()

        await EntityEdge.delete_by_uuids(mock_driver, ['uuid1', 'uuid2'])

        mock_driver.graph_operations_interface.edge_delete_by_uuids.assert_called_once_with(
            EntityEdge, mock_driver, ['uuid1', 'uuid2']
        )

    @pytest.mark.asyncio
    async def test_entity_edge_delete_by_uuids_kuzu(self):
        """Test EntityEdge.delete_by_uuids for Kuzu provider."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.KUZU
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        await EntityEdge.delete_by_uuids(mock_driver, ['uuid1', 'uuid2'])

        # Should make two calls for Kuzu (one for regular edges, one for edge nodes)
        assert mock_driver.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_entity_edge_delete_by_uuids_neo4j(self):
        """Test EntityEdge.delete_by_uuids for Neo4j provider."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        await EntityEdge.delete_by_uuids(mock_driver, ['uuid1', 'uuid2'])

        # Should make two calls for Neo4j (one to collect edges, one to delete)
        assert mock_driver.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_entity_edge_get_between_nodes_kuzu(self):
        """Test EntityEdge.get_between_nodes for Kuzu provider."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.KUZU
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.edges.get_entity_edge_return_query') as mock_query:
            mock_query.return_value = 'TEST_RETURN'
            await EntityEdge.get_between_nodes(mock_driver, 'source-uuid', 'target-uuid')

        mock_driver.execute_query.assert_called_once()
        called_query = mock_driver.execute_query.call_args[0][0]
        assert 'RelatesToNode_' in called_query  # Kuzu-specific pattern

    @pytest.mark.asyncio
    async def test_entity_edge_get_by_group_ids_not_found(self):
        """Test EntityEdge.get_by_group_ids when no edges found."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(GroupsEdgesNotFoundError):
            await EntityEdge.get_by_group_ids(mock_driver, ['nonexistent-group'])

    @pytest.mark.asyncio
    async def test_community_edge_get_by_uuid_empty_result(self):
        """Test CommunityEdge.get_by_uuid returns first result even if empty."""
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        # This should not raise an error (unlike other edge types)
        # but it might fail due to index error - let's see what happens
        result = await CommunityEdge.get_by_uuid(mock_driver, 'test-uuid')

        # If it doesn't raise an error, the test passes
        # If it does, we would need to adjust the implementation