"""
Advanced Search Examples for Graphiti Core Library

This file demonstrates how to use the enhanced search functionality
directly with the Graphiti core library, without going through the HTTP API.

Copyright 2025, Zep Software, Inc.

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

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
)
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_RRF,
    COMMUNITY_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    PropertyFilter,
    ComparisonOperator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Database connection configuration
falkor_username = os.environ.get('FALKORDB_USERNAME', None)
falkor_password = os.environ.get('FALKORDB_PASSWORD', None)
falkor_host = os.environ.get('FALKORDB_HOST', 'localhost')
falkor_port = os.environ.get('FALKORDB_PORT', '6379')


async def setup_sample_data(graphiti: Graphiti):
    """
    Set up sample data for demonstration purposes.

    This function creates a sample knowledge graph with various entities
    and relationships to demonstrate the search functionality.
    """
    print("Setting up sample data...")

    # Sample episodes with diverse content
    episodes = [
        {
            'name': 'Tech Company Foundation',
            'content': 'TechCorp was founded in 2015 by Sarah Johnson and Michael Chen. '
                      'The company specializes in artificial intelligence and machine learning solutions. '
                      'Sarah serves as CEO while Michael is the CTO.',
            'type': EpisodeType.text,
            'description': 'company foundation story',
            'reference_time': datetime(2015, 3, 15, tzinfo=timezone.utc)
        },
        {
            'name': 'Product Launch 2020',
            'content': 'In 2020, TechCorp launched their flagship AI platform "NeuralFlow". '
                      'The product gained significant market traction and secured $50M in Series B funding. '
                      'The launch was led by product director Emily Rodriguez.',
            'type': EpisodeType.text,
            'description': 'product launch announcement',
            'reference_time': datetime(2020, 6, 10, tzinfo=timezone.utc)
        },
        {
            'name': 'Acquisition Deal',
            'content': 'In 2023, GlobalTech acquired TechCorp for $500M. '
                      'The acquisition included all intellectual property and key personnel. '
                      'Sarah Johnson continued as CEO of the merged entity.',
            'type': EpisodeType.text,
            'description': 'acquisition details',
            'reference_time': datetime(2023, 9, 1, tzinfo=timezone.utc)
        },
        {
            'name': 'Employee Growth',
            'content': json.dumps({
                'year': 2022,
                'employees': 150,
                'departments': ['Engineering', 'Product', 'Sales', 'Marketing'],
                'engineering_team_size': 60,
                'average_experience_years': 4.5,
                'technologies': ['Python', 'TensorFlow', 'PyTorch', 'Kubernetes']
            }),
            'type': EpisodeType.json,
            'description': 'company statistics',
            'reference_time': datetime(2022, 12, 31, tzinfo=timezone.utc)
        },
        {
            'name': 'Research Publication',
            'content': 'The TechCorp research team published a groundbreaking paper on '
                      'transformer architecture optimization in Nature AI journal. '
                      'The paper was authored by lead researcher Dr. James Wilson.',
            'type': EpisodeType.text,
            'description': 'research achievement',
            'reference_time': datetime(2021, 8, 20, tzinfo=timezone.utc)
        }
    ]

    # Add episodes to the graph
    for episode in episodes:
        await graphiti.add_episode(
            name=episode['name'],
            episode_body=episode['content'],
            source=episode['type'],
            source_description=episode['description'],
            reference_time=episode['reference_time']
        )
        print(f"Added episode: {episode['name']}")

    print("Sample data setup complete.\n")


async def example_1_basic_search(graphiti: Graphiti):
    """
    Example 1: Basic search using the simple search method.
    """
    print("=" * 60)
    print("Example 1: Basic Search")
    print("=" * 60)

    query = "tech company leadership team"
    results = await graphiti.search(query)

    print(f"Query: {query}")
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.fact}")
        print(f"   UUID: {result.uuid}")
        print(f"   Created: {result.created_at}")
        if hasattr(result, 'valid_at') and result.valid_at:
            print(f"   Valid: {result.valid_at}")
        print()


async def example_2_search_with_filters(graphiti: Graphiti):
    """
    Example 2: Search with date and entity filters.
    """
    print("=" * 60)
    print("Example 2: Search with Filters")
    print("=" * 60)

    query = "company milestones and achievements"

    # Create date filters for specific time periods
    date_filters = SearchFilters(
        created_at=[
            [
                DateFilter(
                    date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    comparison_operator=ComparisonOperator.greater_than_equal
                ),
                DateFilter(
                    date=datetime(2023, 12, 31, tzinfo=timezone.utc),
                    comparison_operator=ComparisonOperator.less_than_equal
                )
            ]
        ],
        # Filter for specific entity types
        node_labels=["Company", "Person", "Organization"],
        edge_types=["FOUNDED_BY", "LED_BY", "ACHIEVED"]
    )

    results = await graphiti.search(query, search_filter=date_filters)

    print(f"Query: {query}")
    print(f"Date range: 2020-2023")
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.fact}")
        print(f"   Created: {result.created_at}")
        print()


async def example_3_reranking_methods(graphiti: Graphiti):
    """
    Example 3: Different reranking methods.
    """
    print("=" * 60)
    print("Example 3: Reranking Methods")
    print("=" * 60)

    query = "artificial intelligence and machine learning"

    # Different reranking configurations
    reranking_configs = [
        ("RRF (Default)", EDGE_HYBRID_SEARCH_RRF),
        ("MMR", EDGE_HYBRID_SEARCH_MMR),
        ("Cross-Encoder", EDGE_HYBRID_SEARCH_CROSS_ENCODER),
        ("Node Distance", EDGE_HYBRID_SEARCH_NODE_DISTANCE),
    ]

    for name, config in reranking_configs:
        print(f"\n--- {name} Reranking ---")

        # Copy and customize the configuration
        search_config = config.model_copy(deep=True)
        search_config.limit = 5

        # Use the _search method for advanced configurations
        search_results = await graphiti._search(query, config=search_config)

        # Convert edges to facts for display
        facts = search_results.edges

        print(f"Found {len(facts)} results")

        for i, edge in enumerate(facts, 1):
            fact_str = f"{edge.source_node_name} {edge.name} {edge.target_node_name}"
            print(f"{i}. {fact_str}")


async def example_4_node_search(graphiti: Graphiti):
    """
    Example 4: Node-focused search.
    """
    print("\n" + "=" * 60)
    print("Example 4: Node Search")
    print("=" * 60)

    query = "companies and organizations"

    # Configure node search
    node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_config.limit = 8

    search_results = await graphiti._search(query, config=node_config)

    print(f"Query: {query}")
    print(f"Found {len(search_results.nodes)} nodes\n")

    for i, node in enumerate(search_results.nodes, 1):
        print(f"{i}. {node.name}")
        print(f"   Labels: {', '.join(node.labels)}")
        print(f"   Created: {node.created_at}")
        if node.summary:
            summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f"   Summary: {summary}")
        print()


async def example_5_combined_search(graphiti: Graphiti):
    """
    Example 5: Combined edge and node search.
    """
    print("\n" + "=" * 60)
    print("Example 5: Combined Search")
    print("=" * 60)

    query = "leadership and management"

    # Configure combined search
    combined_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    combined_config.limit = 10

    search_results = await graphiti._search(query, config=combined_config)

    print(f"Query: {query}")
    print(f"Found {len(search_results.edges)} edges and {len(search_results.nodes)} nodes\n")

    print("Edges:")
    for i, edge in enumerate(search_results.edges[:5], 1):
        fact_str = f"{edge.source_node_name} {edge.name} {edge.target_node_name}"
        print(f"{i}. {fact_str}")

    print(f"\nNodes:")
    for i, node in enumerate(search_results.nodes[:5], 1):
        print(f"{i}. {node.name} ({', '.join(node.labels)})")


async def example_6_custom_search_config(graphiti: Graphiti):
    """
    Example 6: Custom search configuration.
    """
    print("\n" + "=" * 60)
    print("Example 6: Custom Search Configuration")
    print("=" * 60)

    query = "technology innovations"

    # Create a completely custom search configuration
    custom_config = SearchConfig(
        limit=8,
        edge_search=EdgeSearchConfig(
            method_list=["bm25", "cosine_similarity"],
            reranker="mmr",
            mmr_lambda=0.7,  # Balance between relevance and diversity
            min_score=0.3
        ),
        node_search=None,  # Only search edges
        include_nodes=False,
        include_edges=True,
        include_episodes=False,
        include_communities=False
    )

    search_results = await graphiti._search(query, config=custom_config)

    print(f"Query: {query}")
    print(f"Search methods: {custom_config.edge_search.method_list}")
    print(f"Reranker: {custom_config.edge_search.reranker}")
    print(f"MMR Lambda: {custom_config.edge_search.mmr_lambda}")
    print(f"Found {len(search_results.edges)} results\n")

    for i, edge in enumerate(search_results.edges, 1):
        fact_str = f"{edge.source_node_name} {edge.name} {edge.target_node_name}"
        print(f"{i}. {fact_str}")


async def example_7_property_filtering(graphiti: Graphiti):
    """
    Example 7: Advanced property-based filtering.
    """
    print("\n" + "=" * 60)
    print("Example 7: Property Filtering")
    print("=" * 60)

    query = "employees and team members"

    # Create advanced property filters
    property_filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name="department",
                property_value="Engineering",
                comparison_operator=ComparisonOperator.equals
            ),
            PropertyFilter(
                property_name="experience_years",
                property_value=3,
                comparison_operator=ComparisonOperator.greater_than
            ),
            PropertyFilter(
                property_name="technologies",
                property_value="Python",
                comparison_operator=ComparisonOperator.is_not_null
            )
        ]
    )

    results = await graphiti.search(query, search_filter=property_filters)

    print(f"Query: {query}")
    print("Filters: department=Engineering, experience_years>3, technologies!=NULL")
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.fact}")
        print()


async def example_8_community_search(graphiti: Graphiti):
    """
    Example 8: Community-based search.
    """
    print("\n" + "=" * 60)
    print("Example 8: Community Search")
    print("=" * 60)

    query = "business relationships and partnerships"

    # Configure community search
    community_config = COMMUNITY_HYBRID_SEARCH_RRF.model_copy(deep=True)
    community_config.limit = 6

    search_results = await graphiti._search(query, config=community_config)

    print(f"Query: {query}")
    print(f"Found {len(search_results.communities)} communities\n")

    for i, community in enumerate(search_results.communities, 1):
        print(f"{i}. Community {community.name}")
        print(f"   Summary: {community.summary[:100]}..." if len(community.summary) > 100 else f"   Summary: {community.summary}")
        if community.members:
            print(f"   Members: {len(community.members)}")
        print()


async def example_9_search_performance_comparison(graphiti: Graphiti):
    """
    Example 9: Performance comparison of different search methods.
    """
    print("\n" + "=" * 60)
    print("Example 9: Performance Comparison")
    print("=" * 60)

    query = "company growth and expansion"

    # Different configurations to compare
    configs = [
        ("Basic Search", None),
        ("RRF Edge Search", EDGE_HYBRID_SEARCH_RRF),
        ("MMR Edge Search", EDGE_HYBRID_SEARCH_MMR),
        ("Cross-Encoder Search", EDGE_HYBRID_SEARCH_CROSS_ENCODER),
    ]

    print(f"Query: {query}\n")

    for name, config in configs:
        start_time = datetime.now()

        if config is None:
            # Basic search
            results = await graphiti.search(query)
            result_count = len(results)
        else:
            # Advanced search
            search_config = config.model_copy(deep=True)
            search_config.limit = 10
            search_results = await graphiti._search(query, config=search_config)
            result_count = len(search_results.edges)

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        print(f"{name:<25}: {result_count:>3} results in {duration_ms:>4}ms")


async def main():
    """
    Main function to demonstrate all advanced search features.
    """
    print("Graphiti Core Advanced Search Examples")
    print("=" * 60)
    print("These examples demonstrate the enhanced search functionality")
    print("directly using the Graphiti core library.\n")

    # Initialize database connection
    try:
        falkor_driver = FalkorDriver(
            host=falkor_host,
            port=falkor_port,
            username=falkor_username,
            password=falkor_password
        )
        graphiti = Graphiti(graph_driver=falkor_driver)

        print("Connected to database successfully.\n")

        # Set up sample data (optional - comment out if you already have data)
        await setup_sample_data(graphiti)

        # Run all examples
        await example_1_basic_search(graphiti)
        await example_2_search_with_filters(graphiti)
        await example_3_reranking_methods(graphiti)
        await example_4_node_search(graphiti)
        await example_5_combined_search(graphiti)
        await example_6_custom_search_config(graphiti)
        await example_7_property_filtering(graphiti)
        await example_8_community_search(graphiti)
        await example_9_search_performance_comparison(graphiti)

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"\nError: {e}")
        print("Make sure FalkorDB is running and accessible.")
    finally:
        if 'graphiti' in locals():
            await graphiti.close()
            print("Database connection closed.")


if __name__ == "__main__":
    """
    To run these examples:

    1. Make sure FalkorDB is running:
       docker run -d --name falkordb -p 6379:6379 falkordb/falkordb

    2. Set up environment variables (optional):
       export FALKORDB_HOST=localhost
       export FALKORDB_PORT=6379
       export FALKORDB_USERNAME=your_username
       export FALKORDB_PASSWORD=your_password

    3. Run this script:
       python advanced_search_core.py
    """
    asyncio.run(main())