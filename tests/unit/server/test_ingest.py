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

import sys
import os

# Add server directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'server'))

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
from functools import partial

from fastapi import status
from fastapi.testclient import TestClient

from graph_service.routers.ingest import (
    router, async_worker, AsyncWorker,
    add_messages, add_entity_node, delete_entity_edge,
    delete_group, delete_episode, clear
)
from graph_service.dto import AddMessagesRequest, AddEntityNodeRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep


class TestAsyncWorker:
    """Test AsyncWorker functionality"""

    @pytest.mark.asyncio
    async def test_async_worker_initialization(self):
        """Test AsyncWorker initializes correctly"""
        worker = AsyncWorker()
        assert worker.queue is not None
        assert isinstance(worker.queue, asyncio.Queue)
        assert worker.task is None

    @pytest.mark.asyncio
    async def test_async_worker_start(self):
        """Test AsyncWorker starts correctly"""
        worker = AsyncWorker()
        await worker.start()
        assert worker.task is not None
        assert not worker.task.done()

        # Cleanup
        await worker.stop()

    @pytest.mark.asyncio
    async def test_async_worker_stop(self):
        """Test AsyncWorker stops correctly"""
        worker = AsyncWorker()
        await worker.start()

        # Add a task to the queue
        mock_job = AsyncMock()
        await worker.queue.put(mock_job)

        await worker.stop()

        assert worker.task is None
        assert worker.queue.empty()

    @pytest.mark.asyncio
    async def test_async_worker_processes_jobs(self):
        """Test AsyncWorker processes jobs correctly"""
        worker = AsyncWorker()
        await worker.start()

        # Create a mock job
        mock_job = AsyncMock()
        await worker.queue.put(mock_job)

        # Wait a bit for processing
        await asyncio.sleep(0.1)

        # Verify job was called
        mock_job.assert_called_once()

        await worker.stop()

    @pytest.mark.asyncio
    async def test_async_worker_handles_cancellation(self):
        """Test AsyncWorker handles cancellation gracefully"""
        worker = AsyncWorker()
        await worker.start()

        # Cancel the task
        worker.task.cancel()

        # Should not raise exception
        await worker.stop()

    @pytest.mark.asyncio
    async def test_async_worker_clears_queue_on_stop(self):
        """Test AsyncWorker clears queue when stopped"""
        worker = AsyncWorker()
        await worker.start()

        # Add multiple jobs
        for i in range(5):
            mock_job = AsyncMock()
            await worker.queue.put(mock_job)

        assert worker.queue.qsize() == 5

        await worker.stop()

        # Queue should be empty
        assert worker.queue.empty()


class TestAddMessagesEndpoint:
    """Test add messages endpoint"""

    @pytest.mark.asyncio
    async def test_add_messages_success(self):
        """Test successful message addition"""
        # Create test request
        request = AddMessagesRequest(
            group_id="test-group-123",
            messages=[
                Message(
                    content="Hello world",
                    role_type="user",
                    uuid="msg-123",
                    name="Test Message",
                    role="TestUser",
                    timestamp=datetime.now(timezone.utc),
                    source_description="Test source"
                )
            ]
        )

        # Mock dependencies
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        with patch('graph_service.routers.ingest.async_worker') as mock_worker:
            mock_worker.queue = AsyncMock()

            # Call the endpoint
            response = await add_messages(request, mock_graphiti)

            # Verify response
            assert isinstance(response, Result)
            assert response.success is True
            assert "Messages added to processing queue" in response.message

            # Verify job was queued
            mock_worker.queue.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_messages_with_multiple_messages(self):
        """Test adding multiple messages"""
        messages = [
            Message(
                content=f"Message {i}",
                role_type="user" if i % 2 == 0 else "assistant",
                uuid=f"msg-{i}"
            )
            for i in range(3)
        ]

        request = AddMessagesRequest(
            group_id="test-group",
            messages=messages
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        with patch('graph_service.routers.ingest.async_worker') as mock_worker:
            mock_worker.queue = AsyncMock()

            response = await add_messages(request, mock_graphiti)

            assert response.success is True
            # Should queue one job per message
            assert mock_worker.queue.put.call_count == 3

    @pytest.mark.asyncio
    async def test_add_messages_with_optional_fields(self):
        """Test adding messages with optional fields"""
        request = AddMessagesRequest(
            group_id="test-group",
            messages=[
                Message(
                    content="Test message",
                    role_type="user",
                    # uuid, name, role, timestamp, source_description use defaults
                )
            ]
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        with patch('graph_service.routers.ingest.async_worker') as mock_worker:
            mock_worker.queue = AsyncMock()

            response = await add_messages(request, mock_graphiti)

            assert response.success is True
            mock_worker.queue.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_messages_job_execution(self):
        """Test that queued jobs execute correctly"""
        # Set up real async worker for this test
        worker = AsyncWorker()
        await worker.start()

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        # Create test message
        message = Message(
            content="Test message",
            role_type="user",
            uuid="test-uuid",
            name="Test Message"
        )

        request = AddMessagesRequest(
            group_id="test-group",
            messages=[message]
        )

        # Call add_messages directly to queue the job
        with patch('graph_service.routers.ingest.async_worker', worker):
            response = await add_messages(request, mock_graphiti)

        assert response.success is True

        # Wait for job to process
        await asyncio.sleep(0.1)

        # Verify graphiti.add_episode was called
        mock_graphiti.add_episode.assert_called_once()
        call_args = mock_graphiti.add_episode.call_args
        assert call_args.kwargs['uuid'] == "test-uuid"
        assert call_args.kwargs['group_id'] == "test-group"
        assert call_args.kwargs['name'] == "Test Message"

        await worker.stop()


class TestAddEntityNodeEndpoint:
    """Test add entity node endpoint"""

    @pytest.mark.asyncio
    async def test_add_entity_node_success(self):
        """Test successful entity node creation"""
        request = AddEntityNodeRequest(
            uuid="node-123",
            group_id="group-123",
            name="Test Entity",
            summary="Test entity summary"
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_node = MagicMock()
        mock_node.uuid = "node-123"
        mock_node.name = "Test Entity"
        mock_graphiti.save_entity_node.return_value = mock_node

        response = await add_entity_node(request, mock_graphiti)

        # Verify response
        assert response == mock_node

        # Verify correct call was made
        mock_graphiti.save_entity_node.assert_called_once_with(
            uuid="node-123",
            group_id="group-123",
            name="Test Entity",
            summary="Test entity summary"
        )

    @pytest.mark.asyncio
    async def test_add_entity_node_minimal_data(self):
        """Test entity node creation with minimal data"""
        request = AddEntityNodeRequest(
            uuid="node-123",
            group_id="group-123",
            name="Test Entity"
            # summary uses default empty string
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_node = MagicMock()
        mock_graphiti.save_entity_node.return_value = mock_node

        response = await add_entity_node(request, mock_graphiti)

        assert response == mock_node
        mock_graphiti.save_entity_node.assert_called_once_with(
            uuid="node-123",
            group_id="group-123",
            name="Test Entity",
            summary=""
        )

    @pytest.mark.asyncio
    async def test_add_entity_node_handles_exceptions(self):
        """Test entity node creation error handling"""
        request = AddEntityNodeRequest(
            uuid="node-123",
            group_id="group-123",
            name="Test Entity"
        )

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.save_entity_node.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await add_entity_node(request, mock_graphiti)


class TestDeleteEntityEdgeEndpoint:
    """Test delete entity edge endpoint"""

    @pytest.mark.asyncio
    async def test_delete_entity_edge_success(self):
        """Test successful entity edge deletion"""
        uuid = "edge-123"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        response = await delete_entity_edge(uuid, mock_graphiti)

        # Verify response
        assert isinstance(response, Result)
        assert response.success is True
        assert "Entity Edge deleted" in response.message

        # Verify correct call was made
        mock_graphiti.delete_entity_edge.assert_called_once_with(uuid)

    @pytest.mark.asyncio
    async def test_delete_entity_edge_not_found(self):
        """Test entity edge deletion when edge not found"""
        uuid = "nonexistent-edge"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.delete_entity_edge.side_effect = Exception("Edge not found")

        with pytest.raises(Exception, match="Edge not found"):
            await delete_entity_edge(uuid, mock_graphiti)


class TestDeleteGroupEndpoint:
    """Test delete group endpoint"""

    @pytest.mark.asyncio
    async def test_delete_group_success(self):
        """Test successful group deletion"""
        group_id = "group-123"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        response = await delete_group(group_id, mock_graphiti)

        # Verify response
        assert isinstance(response, Result)
        assert response.success is True
        assert "Group deleted" in response.message

        # Verify correct call was made
        mock_graphiti.delete_group.assert_called_once_with(group_id)

    @pytest.mark.asyncio
    async def test_delete_group_handles_exceptions(self):
        """Test group deletion error handling"""
        group_id = "group-123"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.delete_group.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await delete_group(group_id, mock_graphiti)


class TestDeleteEpisodeEndpoint:
    """Test delete episode endpoint"""

    @pytest.mark.asyncio
    async def test_delete_episode_success(self):
        """Test successful episode deletion"""
        uuid = "episode-123"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        response = await delete_episode(uuid, mock_graphiti)

        # Verify response
        assert isinstance(response, Result)
        assert response.success is True
        assert "Episode deleted" in response.message

        # Verify correct call was made
        mock_graphiti.delete_episodic_node.assert_called_once_with(uuid)

    @pytest.mark.asyncio
    async def test_delete_episode_not_found(self):
        """Test episode deletion when episode not found"""
        uuid = "nonexistent-episode"
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.delete_episodic_node.side_effect = Exception("Episode not found")

        with pytest.raises(Exception, match="Episode not found"):
            await delete_episode(uuid, mock_graphiti)


class TestClearEndpoint:
    """Test clear graph endpoint"""

    @pytest.mark.asyncio
    async def test_clear_success(self):
        """Test successful graph clearing"""
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.driver = MagicMock()

        with patch('graph_service.routers.ingest.clear_data') as mock_clear_data:
            response = await clear(mock_graphiti)

            # Verify response
            assert isinstance(response, Result)
            assert response.success is True
            assert "Graph cleared" in response.message

            # Verify correct calls were made
            mock_clear_data.assert_called_once_with(mock_graphiti.driver)
            mock_graphiti.build_indices_and_constraints.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_handles_clear_data_exception(self):
        """Test clear endpoint when clear_data fails"""
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.driver = MagicMock()

        with patch('graph_service.routers.ingest.clear_data') as mock_clear_data:
            mock_clear_data.side_effect = Exception("Clear failed")

            with pytest.raises(Exception, match="Clear failed"):
                await clear(mock_graphiti)

            # build_indices_and_constraints should not be called if clear_data fails
            mock_graphiti.build_indices_and_constraints.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_handles_indices_exception(self):
        """Test clear endpoint when build_indices_and_constraints fails"""
        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)
        mock_graphiti.driver = MagicMock()

        with patch('graph_service.routers.ingest.clear_data') as mock_clear_data:
            mock_graphiti.build_indices_and_constraints.side_effect = Exception("Index build failed")

            with pytest.raises(Exception, match="Index build failed"):
                await clear(mock_graphiti)

            # clear_data should have been called successfully
            mock_clear_data.assert_called_once_with(mock_graphiti.driver)


class TestIngestRouter:
    """Test ingest router configuration"""

    def test_router_tags(self):
        """Test that router has correct tags"""
        assert "ingest" in router.tags

    def test_router_lifespan(self):
        """Test that router has lifespan configured"""
        assert router.lifespan is not None

    def test_add_messages_endpoint_configuration(self):
        """Test add messages endpoint configuration"""
        for route in router.routes:
            if route.path == "/messages":
                assert status.HTTP_202_ACCEPTED in route.status_code
                assert route.methods == {"POST"}
                break
        else:
            pytest.fail("Add messages endpoint not found")

    def test_add_entity_node_endpoint_configuration(self):
        """Test add entity node endpoint configuration"""
        for route in router.routes:
            if route.path == "/entity-node":
                assert status.HTTP_201_CREATED in route.status_code
                assert route.methods == {"POST"}
                break
        else:
            pytest.fail("Add entity node endpoint not found")

    def test_delete_endpoints_configuration(self):
        """Test delete endpoints configuration"""
        delete_routes = [route for route in router.routes if "DELETE" in route.methods]
        assert len(delete_routes) >= 3  # Should have at least 3 delete endpoints

        for route in delete_routes:
            assert status.HTTP_200_OK in route.status_code

    def test_clear_endpoint_configuration(self):
        """Test clear endpoint configuration"""
        for route in router.routes:
            if route.path == "/clear":
                assert status.HTTP_200_OK in route.status_code
                assert route.methods == {"POST"}
                break
        else:
            pytest.fail("Clear endpoint not found")


class TestIntegrationWithFastAPI:
    """Test integration with FastAPI TestClient"""

    def test_router_can_be_included_in_app(self):
        """Test that ingest router can be included in FastAPI app"""
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)

        # Should not raise any exceptions
        assert test_app is not None

    def test_endpoints_are_accessible_in_test_client(self):
        """Test that endpoints are accessible through TestClient"""
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)

        client = TestClient(test_app)

        # Test that endpoints exist (may return 422 due to missing dependencies)
        response = client.post("/messages", json={})
        assert response.status_code in [422, 500]  # Expected for missing data

        response = client.post("/entity-node", json={})
        assert response.status_code in [422, 500]


class TestMessageFormatting:
    """Test message formatting in add_messages"""

    @pytest.mark.asyncio
    async def test_message_episode_body_formatting(self):
        """Test that message episode body is formatted correctly"""
        worker = AsyncWorker()
        await worker.start()

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        message = Message(
            content="Hello world",
            role_type="user",
            role="TestUser",
            timestamp=datetime.now(timezone.utc)
        )

        request = AddMessagesRequest(
            group_id="test-group",
            messages=[message]
        )

        with patch('graph_service.routers.ingest.async_worker', worker):
            await add_messages(request, mock_graphiti)

        # Wait for job to process
        await asyncio.sleep(0.1)

        # Check episode body format
        call_args = mock_graphiti.add_episode.call_args
        expected_body = f"{message.role}({message.role_type}): {message.content}"
        assert call_args.kwargs['episode_body'] == expected_body

        await worker.stop()

    @pytest.mark.asyncio
    async def test_message_episode_body_with_none_role(self):
        """Test message episode body formatting with None role"""
        worker = AsyncWorker()
        await worker.start()

        mock_graphiti = AsyncMock(spec=ZepGraphitiDep)

        message = Message(
            content="Hello world",
            role_type="user",
            role=None,  # Explicitly None
            timestamp=datetime.now(timezone.utc)
        )

        request = AddMessagesRequest(
            group_id="test-group",
            messages=[message]
        )

        with patch('graph_service.routers.ingest.async_worker', worker):
            await add_messages(request, mock_graphiti)

        await asyncio.sleep(0.1)

        # Check episode body format with None role
        call_args = mock_graphiti.add_episode.call_args
        expected_body = f"{message.role_type}: {message.content}"
        assert call_args.kwargs['episode_body'] == expected_body

        await worker.stop()