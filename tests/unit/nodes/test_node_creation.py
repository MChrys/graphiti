"""
Tests for graph node creation and operations

Run with: pytest tests/unit/nodes/test_node_creation.py -v
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from graphiti_core.nodes import (
    Node,
    EntityNode,
    EpisodicNode,
    CommunityNode,
    EpisodeType,
    get_entity_node_from_record,
    get_episodic_node_from_record,
    get_community_node_from_record,
    create_entity_node_embeddings,
)
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.utils.datetime_utils import utc_now


class TestEpisodeType:
    """Tests for EpisodeType enum functionality."""

    def test_episode_type_values(self):
        """Test EpisodeType enum has correct values."""
        assert EpisodeType.message.value == 'message'
        assert EpisodeType.json.value == 'json'
        assert EpisodeType.text.value == 'text'

    def test_from_str_message(self):
        """Test from_str with message type."""
        result = EpisodeType.from_str('message')
        assert result == EpisodeType.message

    def test_from_str_json(self):
        """Test from_str with json type."""
        result = EpisodeType.from_str('json')
        assert result == EpisodeType.json

    def test_from_str_text(self):
        """Test from_str with text type."""
        result = EpisodeType.from_str('text')
        assert result == EpisodeType.text

    def test_from_str_unimplemented(self):
        """Test from_str raises error for unimplemented type."""
        with pytest.raises(NotImplementedError):
            EpisodeType.from_str('unknown')


class TestNodeBaseClass:
    """Tests for the base Node class."""

    def test_node_initialization(self):
        """Test basic Node initialization with abstract class."""
        # Node is abstract, but we can test basic field initialization
        node_data = {
            'uuid': str(uuid4()),
            'name': 'Test Node',
            'group_id': 'test-group',
            'labels': ['TestLabel'],
            'created_at': utc_now(),
        }

        # Test that abstract methods exist
        assert hasattr(Node, 'save')
        assert hasattr(Node, 'get_by_uuid')
        assert hasattr(Node, 'get_by_uuids')

    def test_node_hash_and_equality(self):
        """Test Node hash and equality methods."""
        uuid_val = str(uuid4())

        # Create a concrete implementation for testing
        class TestNode(Node):
            async def save(self, driver):
                pass

            @classmethod
            async def get_by_uuid(cls, driver, uuid):
                pass

            @classmethod
            async def get_by_uuids(cls, driver, uuids):
                pass

        node1 = TestNode(uuid=uuid_val, name='Test', group_id='group')
        node2 = TestNode(uuid=uuid_val, name='Different', group_id='group')
        node3 = TestNode(uuid=str(uuid4()), name='Test', group_id='group')

        # Test equality based on UUID
        assert node1 == node2
        assert node1 != node3

        # Test hash
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)


class TestEntityNode:
    """Tests for EntityNode creation and operations."""

    def test_entity_node_initialization(self):
        """Test EntityNode initialization with minimal fields."""
        node = EntityNode(
            name='Test Entity',
            group_id='test-group',
            labels=['Person', 'Test']
        )

        assert node.name == 'Test Entity'
        assert node.group_id == 'test-group'
        assert node.labels == ['Person', 'Test']
        assert node.name_embedding is None
        assert node.summary == ''
        assert node.attributes == {}
        assert isinstance(node.uuid, str)
        assert isinstance(node.created_at, datetime)

    def test_entity_node_with_all_fields(self):
        """Test EntityNode initialization with all fields."""
        test_uuid = str(uuid4())
        test_time = utc_now()
        test_embedding = [0.1, 0.2, 0.3]
        test_attributes = {'age': 25, 'city': 'New York'}

        node = EntityNode(
            uuid=test_uuid,
            name='Complete Entity',
            group_id='test-group',
            labels=['Person'],
            summary='A complete test entity',
            name_embedding=test_embedding,
            attributes=test_attributes,
            created_at=test_time
        )

        assert node.uuid == test_uuid
        assert node.name == 'Complete Entity'
        assert node.group_id == 'test-group'
        assert node.labels == ['Person']
        assert node.summary == 'A complete test entity'
        assert node.name_embedding == test_embedding
        assert node.attributes == test_attributes
        assert node.created_at == test_time

    @pytest.mark.asyncio
    async def test_entity_node_generate_name_embedding(self):
        """Test EntityNode name embedding generation."""
        node = EntityNode(name='Test Entity', group_id='test-group')
        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create.return_value = [0.1, 0.2, 0.3]

        result = await node.generate_name_embedding(mock_embedder)

        assert result == [0.1, 0.2, 0.3]
        assert node.name_embedding == [0.1, 0.2, 0.3]
        mock_embedder.create.assert_called_once_with(input_data=['Test Entity'])

    @pytest.mark.asyncio
    async def test_entity_node_load_name_embedding_neo4j(self):
        """Test EntityNode embedding loading for Neo4j."""
        node = EntityNode(uuid='test-uuid', name='Test', group_id='group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.execute_query = AsyncMock(return_value=([{'name_embedding': [0.1, 0.2, 0.3]}], None, None))

        await node.load_name_embedding(mock_driver)

        assert node.name_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_node_load_name_embedding_neptune(self):
        """Test EntityNode embedding loading for Neptune."""
        node = EntityNode(uuid='test-uuid', name='Test', group_id='group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEPTUNE
        mock_driver.execute_query = AsyncMock(return_value=([{'name_embedding': '0.1,0.2,0.3'}], None, None))

        await node.load_name_embedding(mock_driver)

        assert node.name_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_node_load_name_embedding_not_found(self):
        """Test EntityNode embedding loading when node not found."""
        node = EntityNode(uuid='test-uuid', name='Test', group_id='group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with pytest.raises(NodeNotFoundError):
            await node.load_name_embedding(mock_driver)

    @pytest.mark.asyncio
    async def test_entity_node_save_with_graph_operations_interface(self):
        """Test EntityNode save with graph operations interface."""
        node = EntityNode(name='Test Entity', group_id='test-group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = AsyncMock()

        await node.save(mock_driver)

        mock_driver.graph_operations_interface.node_save.assert_called_once_with(node, mock_driver)

    @pytest.mark.asyncio
    async def test_entity_node_save_neo4j(self):
        """Test EntityNode save for Neo4j provider."""
        node = EntityNode(
            name='Test Entity',
            group_id='test-group',
            labels=['Person'],
            name_embedding=[0.1, 0.2, 0.3],
            summary='Test summary',
            attributes={'age': 25}
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.nodes.get_entity_node_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await node.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['entity_data']['name'] == 'Test Entity'
        assert called_kwargs['entity_data']['age'] == 25  # Attribute should be merged

    @pytest.mark.asyncio
    async def test_entity_node_save_kuzu(self):
        """Test EntityNode save for Kuzu provider."""
        node = EntityNode(
            name='Test Entity',
            group_id='test-group',
            labels=['Person'],
            attributes={'age': 25}
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.KUZU
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.nodes.get_entity_node_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await node.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['attributes'] == json.dumps({'age': 25})
        assert 'Person' in called_kwargs['labels']

    @pytest.mark.asyncio
    async def test_entity_node_delete_with_graph_operations_interface(self):
        """Test EntityNode delete with graph operations interface."""
        node = EntityNode(name='Test Entity', group_id='test-group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = AsyncMock()

        await node.delete(mock_driver)

        mock_driver.graph_operations_interface.node_delete.assert_called_once_with(node, mock_driver)

    @pytest.mark.asyncio
    async def test_entity_node_delete_neo4j(self):
        """Test EntityNode delete for Neo4j provider."""
        node = EntityNode(name='Test Entity', group_id='test-group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEO4J
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        await node.delete(mock_driver)

        mock_driver.execute_query.assert_called_once()


class TestEpisodicNode:
    """Tests for EpisodicNode creation and operations."""

    def test_episodic_node_initialization(self):
        """Test EpisodicNode initialization with all required fields."""
        valid_at = utc_now()
        node = EpisodicNode(
            name='Test Episode',
            group_id='test-group',
            source=EpisodeType.message,
            source_description='Test message episode',
            content='user: Hello world',
            valid_at=valid_at,
            entity_edges=['edge1', 'edge2']
        )

        assert node.name == 'Test Episode'
        assert node.group_id == 'test-group'
        assert node.source == EpisodeType.message
        assert node.source_description == 'Test message episode'
        assert node.content == 'user: Hello world'
        assert node.valid_at == valid_at
        assert node.entity_edges == ['edge1', 'edge2']
        assert isinstance(node.uuid, str)
        assert isinstance(node.created_at, datetime)

    @pytest.mark.asyncio
    async def test_episodic_node_save_with_graph_operations_interface(self):
        """Test EpisodicNode save with graph operations interface."""
        valid_at = utc_now()
        node = EpisodicNode(
            name='Test Episode',
            group_id='test-group',
            source=EpisodeType.text,
            source_description='Test episode',
            content='Test content',
            valid_at=valid_at
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = AsyncMock()

        await node.save(mock_driver)

        mock_driver.graph_operations_interface.episodic_node_save.assert_called_once_with(node, mock_driver)

    @pytest.mark.asyncio
    async def test_episodic_node_save_without_interface(self):
        """Test EpisodicNode save without graph operations interface."""
        valid_at = utc_now()
        node = EpisodicNode(
            name='Test Episode',
            group_id='test-group',
            source=EpisodeType.json,
            source_description='Test JSON episode',
            content='{"test": "data"}',
            valid_at=valid_at,
            entity_edges=['edge1']
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.graph_operations_interface = None
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.nodes.get_episode_node_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await node.save(mock_driver)

        mock_driver.execute_query.assert_called_once()
        called_kwargs = mock_driver.execute_query.call_args[1]
        assert called_kwargs['name'] == 'Test Episode'
        assert called_kwargs['source'] == 'json'
        assert called_kwargs['content'] == '{"test": "data"}'
        assert called_kwargs['entity_edges'] == ['edge1']


class TestCommunityNode:
    """Tests for CommunityNode creation and operations."""

    def test_community_node_initialization(self):
        """Test CommunityNode initialization."""
        node = CommunityNode(
            name='Test Community',
            group_id='test-group',
            summary='A test community'
        )

        assert node.name == 'Test Community'
        assert node.group_id == 'test-group'
        assert node.summary == 'A test community'
        assert node.name_embedding is None
        assert isinstance(node.uuid, str)
        assert isinstance(node.created_at, datetime)

    @pytest.mark.asyncio
    async def test_community_node_generate_name_embedding(self):
        """Test CommunityNode name embedding generation."""
        node = CommunityNode(name='Test Community', group_id='test-group')
        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create.return_value = [0.4, 0.5, 0.6]

        result = await node.generate_name_embedding(mock_embedder)

        assert result == [0.4, 0.5, 0.6]
        assert node.name_embedding == [0.4, 0.5, 0.6]
        mock_embedder.create.assert_called_once_with(input_data=['Test Community'])

    @pytest.mark.asyncio
    async def test_community_node_save_neptune(self):
        """Test CommunityNode save for Neptune provider."""
        node = CommunityNode(
            name='Test Community',
            group_id='test-group',
            summary='Test summary',
            name_embedding=[0.1, 0.2, 0.3]
        )
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEPTUNE
        mock_driver.save_to_aoss = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=([], None, None))

        with patch('graphiti_core.nodes.get_community_node_save_query') as mock_query:
            mock_query.return_value = 'TEST_QUERY'
            await node.save(mock_driver)

        mock_driver.save_to_aoss.assert_called_once_with(
            'communities',
            [{'name': 'Test Community', 'uuid': node.uuid, 'group_id': 'test-group'}]
        )
        mock_driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_community_node_load_name_embedding_neptune(self):
        """Test CommunityNode embedding loading for Neptune."""
        node = CommunityNode(uuid='test-uuid', name='Test Community', group_id='group')
        mock_driver = MagicMock(spec=GraphDriver)
        mock_driver.provider = GraphProvider.NEPTUNE
        mock_driver.execute_query = AsyncMock(return_value=([{'name_embedding': '0.1,0.2,0.3'}], None, None))

        await node.load_name_embedding(mock_driver)

        assert node.name_embedding == [0.1, 0.2, 0.3]
        mock_driver.execute_query.assert_called_once()


class TestNodeHelperFunctions:
    """Tests for node helper functions."""

    def test_get_episodic_node_from_record(self):
        """Test creating EpisodicNode from database record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Episode',
            'group_id': 'test-group',
            'source': 'message',
            'source_description': 'Test message',
            'content': 'user: Hello world',
            'valid_at': test_time.isoformat(),
            'created_at': test_time.isoformat(),
            'entity_edges': ['edge1', 'edge2']
        }

        with patch('graphiti_core.nodes.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            node = get_episodic_node_from_record(record)

            assert isinstance(node, EpisodicNode)
            assert node.uuid == 'test-uuid'
            assert node.name == 'Test Episode'
            assert node.source == EpisodeType.message
            assert node.content == 'user: Hello world'

    def test_get_entity_node_from_record_neo4j(self):
        """Test creating EntityNode from Neo4j record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Entity',
            'group_id': 'test-group',
            'summary': 'Test summary',
            'created_at': test_time.isoformat(),
            'labels': ['Person', 'Entity'],
            'attributes': {'age': 25, 'name': 'Should be removed'},
            'name_embedding': [0.1, 0.2, 0.3]
        }

        with patch('graphiti_core.nodes.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            node = get_entity_node_from_record(record, GraphProvider.NEO4J)

            assert isinstance(node, EntityNode)
            assert node.uuid == 'test-uuid'
            assert node.name == 'Test Entity'
            assert node.attributes == {'age': 25}  # name should be removed
            assert 'name' not in node.attributes

    def test_get_entity_node_from_record_kuzu(self):
        """Test creating EntityNode from Kuzu record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Entity',
            'group_id': 'test-group',
            'summary': 'Test summary',
            'created_at': test_time.isoformat(),
            'labels': ['Person', 'Entity', 'Entity_testgroup'],
            'attributes': json.dumps({'age': 25})
        }

        with patch('graphiti_core.nodes.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            node = get_entity_node_from_record(record, GraphProvider.KUZU)

            assert isinstance(node, EntityNode)
            assert node.uuid == 'test-uuid'
            assert node.name == 'Test Entity'
            assert node.attributes == {'age': 25}
            assert 'Entity_testgroup' not in node.labels

    def test_get_community_node_from_record(self):
        """Test creating CommunityNode from database record."""
        test_time = utc_now()
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Community',
            'group_id': 'test-group',
            'summary': 'Test summary',
            'created_at': test_time.isoformat(),
            'name_embedding': [0.1, 0.2, 0.3]
        }

        with patch('graphiti_core.nodes.parse_db_date') as mock_parse:
            mock_parse.return_value = test_time

            node = get_community_node_from_record(record)

            assert isinstance(node, CommunityNode)
            assert node.uuid == 'test-uuid'
            assert node.name == 'Test Community'
            assert node.summary == 'Test summary'

    @pytest.mark.asyncio
    async def test_create_entity_node_embeddings_with_empty_list(self):
        """Test creating embeddings for empty node list."""
        mock_embedder = AsyncMock(spec=EmbedderClient)

        await create_entity_node_embeddings(mock_embedder, [])

        mock_embedder.create_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_entity_node_embeddings_filters_falsey_names(self):
        """Test that nodes with falsey names are filtered out."""
        node1 = EntityNode(name='Valid Node', group_id='group')
        node2 = EntityNode(name='', group_id='group')  # Should be filtered
        node3 = EntityNode(name='Another Valid Node', group_id='group')
        node4 = EntityNode(name=None, group_id='group')  # Should be filtered if possible

        mock_embedder = AsyncMock(spec=EmbedderClient)
        mock_embedder.create_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        await create_entity_node_embeddings(mock_embedder, [node1, node2, node3, node4])

        # Should only be called with valid names
        mock_embedder.create_batch.assert_called_once_with(['Valid Node', 'Another Valid Node'])
        assert node1.name_embedding == [0.1, 0.2]
        assert node3.name_embedding == [0.3, 0.4]

    def test_episodic_node_from_record_missing_dates(self):
        """Test error handling for missing dates in episodic node record."""
        record = {
            'uuid': 'test-uuid',
            'name': 'Test Episode',
            'group_id': 'test-group',
            'source': 'message',
            'source_description': 'Test message',
            'content': 'Test content',
            'valid_at': None,  # Missing date
            'created_at': None,  # Missing date
            'entity_edges': []
        }

        with pytest.raises(ValueError, match='created_at cannot be None'):
            get_episodic_node_from_record(record)