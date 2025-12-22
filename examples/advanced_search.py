"""
Advanced Search Examples for Graphiti Memory Retrieval

This file demonstrates comprehensive usage of the enhanced search functionality
including date range filtering, entity filtering, result ranking, and advanced
configuration options.

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
from typing import Any, Dict

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class GraphitiSearchClient:
    """
    Client for interacting with the Graphiti search API.

    This example client demonstrates how to use all the enhanced search features
    including date filtering, entity filtering, and advanced ranking options.
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        """
        Initialize the search client.

        Args:
            base_url: Base URL of the Graphiti API server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {})
        }

    async def search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a search with the given parameters.

        Args:
            search_params: Dictionary containing search parameters

        Returns:
            Search results with metadata and scores
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json=search_params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    async def search_advanced(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform an advanced search (always uses enhanced search features).

        Args:
            search_params: Dictionary containing search parameters

        Returns:
            Search results with metadata and scores
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search-advanced",
                json=search_params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


async def example_1_basic_search():
    """
    Example 1: Basic search (backward compatibility).

    Demonstrates the simplest search query that maintains compatibility
    with existing implementations.
    """
    print("=" * 60)
    print("Example 1: Basic Search")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Basic search query - no advanced parameters
    search_params = {
        "query": "machine learning engineers working at tech companies",
        "group_ids": ["group-123", "group-456"],
        "max_facts": 10
    }

    try:
        results = await client.search(search_params)

        print(f"Query: {search_params['query']}")
        print(f"Found {len(results['facts'])} results")
        print(f"Search time: {results.get('search_time_ms', 'N/A')}ms")

        for i, fact in enumerate(results['facts'][:3], 1):
            print(f"\n{i}. {fact['fact']}")
            print(f"   Created: {fact['created_at']}")
            if results.get('fact_scores'):
                print(f"   Score: {results['fact_scores'][i-1]:.3f}")

    except httpx.HTTPError as e:
        logger.error(f"Search failed: {e}")


async def example_2_date_range_filtering():
    """
    Example 2: Date range filtering.

    Demonstrates how to filter search results by creation date,
    validity period, and other temporal fields.
    """
    print("\n" + "=" * 60)
    print("Example 2: Date Range Filtering")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Define time ranges for filtering
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    sixty_days_ago = now - timedelta(days=60)
    next_week = now + timedelta(weeks=1)

    # Search with date range filters
    search_params = {
        "query": "company acquisitions and mergers",
        "group_ids": ["business-group"],
        "max_facts": 15,

        # Date range filters
        "created_at_start": sixty_days_ago.isoformat(),
        "created_at_end": now.isoformat(),

        "valid_at_start": thirty_days_ago.isoformat(),
        "valid_at_end": next_week.isoformat(),

        # You can also filter by invalid_at and expired_at
        # "invalid_at_start": ...,
        # "invalid_at_end": ...,
        # "expired_at_start": ...,
        # "expired_at_end": ...,
    }

    try:
        results = await client.search(search_params)

        print(f"Query: {search_params['query']}")
        print(f"Date range: {sixty_days_ago.date()} to {now.date()}")
        print(f"Found {len(results['facts'])} results")

        if results.get('total_results'):
            print(f"Total available: {results['total_results']}")

        for i, fact in enumerate(results['facts'][:3], 1):
            print(f"\n{i}. {fact['fact']}")
            print(f"   Created: {fact['created_at']}")
            print(f"   Valid from: {fact.get('valid_at', 'N/A')}")
            print(f"   Valid until: {fact.get('invalid_at', 'N/A')}")
            if results.get('fact_scores'):
                print(f"   Score: {results['fact_scores'][i-1]:.3f}")

    except httpx.HTTPError as e:
        logger.error(f"Search failed: {e}")


async def example_3_entity_filtering():
    """
    Example 3: Entity type filtering.

    Demonstrates how to filter results by node labels and edge types
    to focus on specific types of entities and relationships.
    """
    print("\n" + "=" * 60)
    print("Example 3: Entity Filtering")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Search with entity type filters
    search_params = {
        "query": "executives and their roles",
        "group_ids": ["company-group"],
        "max_facts": 12,

        # Entity filters
        "node_labels": ["Person", "Company", "Organization"],
        "edge_types": ["HAS_ROLE", "WORKS_AT", "EMPLOYED_BY", "MANAGES"],
    }

    try:
        results = await client.search(search_params)

        print(f"Query: {search_params['query']}")
        print(f"Node labels: {', '.join(search_params['node_labels'])}")
        print(f"Edge types: {', '.join(search_params['edge_types'])}")
        print(f"Found {len(results['facts'])} results")

        # Show search configuration used
        if results.get('search_config_used'):
            config = results['search_config_used']
            print(f"Search method: {config.get('edge_search_method', 'N/A')}")
            print(f"Reranker: {config.get('reranker', 'N/A')}")

        for i, fact in enumerate(results['facts'][:4], 1):
            print(f"\n{i}. {fact['fact']}")
            if results.get('fact_scores'):
                print(f"   Score: {results['fact_scores'][i-1]:.3f}")

    except httpx.HTTPError as e:
        logger.error(f"Search failed: {e}")


async def example_4_advanced_ranking():
    """
    Example 4: Advanced ranking options.

    Demonstrates different reranking methods and their parameters
    to optimize result relevance and diversity.
    """
    print("\n" + "=" * 60)
    print("Example 4: Advanced Ranking Options")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Different ranking configurations to compare
    ranking_configs = [
        {
            "name": "MMR (High Diversity)",
            "params": {
                "query": "artificial intelligence and machine learning applications",
                "max_facts": 10,
                "reranker": "mmr",
                "mmr_lambda": 0.3,  # Lower lambda = more diversity
                "min_score": 0.4,
            }
        },
        {
            "name": "MMR (High Relevance)",
            "params": {
                "query": "artificial intelligence and machine learning applications",
                "max_facts": 10,
                "reranker": "mmr",
                "mmr_lambda": 0.8,  # Higher lambda = more relevance
                "min_score": 0.4,
            }
        },
        {
            "name": "Cross-Encoder Reranking",
            "params": {
                "query": "artificial intelligence and machine learning applications",
                "max_facts": 10,
                "reranker": "cross_encoder",
                "reranker_min_score": 0.5,
                "min_score": 0.3,
            }
        },
        {
            "name": "Node Distance Reranking",
            "params": {
                "query": "artificial intelligence and machine learning applications",
                "max_facts": 10,
                "reranker": "node_distance",
                "bfs_max_depth": 3,
                "min_score": 0.3,
            }
        },
        {
            "name": "Reciprocal Rank Fusion (Default)",
            "params": {
                "query": "artificial intelligence and machine learning applications",
                "max_facts": 10,
                "reranker": "reciprocal_rank_fusion",
                "min_score": 0.3,
            }
        }
    ]

    for config in ranking_configs:
        try:
            print(f"\n--- {config['name']} ---")
            results = await client.search(config['params'])

            print(f"Found {len(results['facts'])} results")
            print(f"Ranking method: {results.get('ranking_method', 'N/A')}")
            print(f"Score normalization: {results.get('score_normalization', 'N/A')}")
            print(f"Min score threshold: {results.get('min_score_threshold', 'N/A')}")

            # Show top 2 results for comparison
            for i, fact in enumerate(results['facts'][:2], 1):
                print(f"\n{i}. {fact['fact']}")
                if results.get('fact_scores'):
                    print(f"   Score: {results['fact_scores'][i-1]:.3f}")

        except httpx.HTTPError as e:
            logger.error(f"Search with {config['name']} failed: {e}")


async def example_5_search_methods():
    """
    Example 5: Different search methods.

    Demonstrates how to use different search algorithms
    (BM25, cosine similarity, BFS) for different use cases.
    """
    print("\n" + "=" * 60)
    print("Example 5: Search Methods")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Different search method configurations
    method_configs = [
        {
            "name": "BM25 Only",
            "params": {
                "query": "exact term matching for specific keywords",
                "max_facts": 8,
                "edge_search_methods": ["bm25"],
                "node_search_methods": ["bm25"],
                "reranker": "reciprocal_rank_fusion",
            }
        },
        {
            "name": "Semantic Similarity Only",
            "params": {
                "query": "concepts related to innovation and technology",
                "max_facts": 8,
                "edge_search_methods": ["cosine_similarity"],
                "node_search_methods": ["cosine_similarity"],
                "reranker": "reciprocal_rank_fusion",
            }
        },
        {
            "name": "Breadth-First Search",
            "params": {
                "query": "connected entities and relationships",
                "max_facts": 8,
                "edge_search_methods": ["breadth_first_search"],
                "node_search_methods": ["breadth_first_search"],
                "reranker": "node_distance",
                "bfs_max_depth": 2,
            }
        },
        {
            "name": "Hybrid (Default)",
            "params": {
                "query": "comprehensive search with multiple methods",
                "max_facts": 8,
                # If not specified, defaults to both BM25 and cosine_similarity
                "reranker": "reciprocal_rank_fusion",
            }
        }
    ]

    for config in method_configs:
        try:
            print(f"\n--- {config['name']} ---")
            results = await client.search(config['params'])

            print(f"Found {len(results['facts'])} results")

            # Show search configuration
            if results.get('search_config_used'):
                search_config = results['search_config_used']
                edge_methods = search_config.get('edge_search_method', [])
                node_methods = search_config.get('node_search_method', [])
                print(f"Edge methods: {edge_methods}")
                print(f"Node methods: {node_methods}")

            # Show top 2 results
            for i, fact in enumerate(results['facts'][:2], 1):
                print(f"\n{i}. {fact['fact']}")
                if results.get('fact_scores'):
                    print(f"   Score: {results['fact_scores'][i-1]:.3f}")

        except httpx.HTTPError as e:
            logger.error(f"Search with {config['name']} failed: {e}")


async def example_6_advanced_filtering():
    """
    Example 6: Advanced filtering with complex filters.

    Demonstrates how to use complex property-based filters
    with comparison operators for sophisticated filtering.
    """
    print("\n" + "=" * 60)
    print("Example 6: Advanced Filtering")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Advanced search with complex filters
    search_params = {
        "query": "employees with specific attributes",
        "group_ids": ["hr-group"],
        "max_facts": 12,

        # Advanced filtering using the 'filters' parameter
        "filters": {
            "node_labels": ["Person", "Employee"],
            "edge_types": ["HAS_PROPERTY", "HAS_ATTRIBUTE"],
            "property_filters": [
                {
                    "property_name": "department",
                    "property_value": "Engineering",
                    "comparison_operator": "="
                },
                {
                    "property_name": "experience_years",
                    "property_value": 5,
                    "comparison_operator": ">="
                },
                {
                    "property_name": "salary_grade",
                    "property_value": "L6",
                    "comparison_operator": "<>"
                }
            ],
            # Complex date filters with OR logic
            "created_at": [
                [
                    {
                        "date": datetime.now(timezone.utc) - timedelta(days=90),
                        "comparison_operator": ">="
                    },
                    {
                        "date": datetime.now(timezone.utc) - timedelta(days=30),
                        "comparison_operator": "<="
                    }
                ]
            ]
        },

        # Additional search configuration
        "reranker": "cross_encoder",
        "reranker_min_score": 0.6,
        "include_nodes": True,
        "include_edges": True,
    }

    try:
        results = await client.search_advanced(search_params)  # Use advanced endpoint

        print(f"Query: {search_params['query']}")
        print(f"Found {len(results['facts'])} facts")

        if results.get('nodes'):
            print(f"Found {len(results['nodes'])} nodes")

        if results.get('edges'):
            print(f"Found {len(results['edges'])} edges")

        print(f"Search time: {results.get('search_time_ms', 'N/A')}ms")
        print(f"Ranking method: {results.get('ranking_method', 'N/A')}")

        # Show fact results
        for i, fact in enumerate(results['facts'][:3], 1):
            print(f"\nFact {i}: {fact['fact']}")
            if results.get('fact_scores'):
                print(f"Score: {results['fact_scores'][i-1]:.3f}")

        # Show node results if available
        if results.get('nodes') and results.get('node_scores'):
            print(f"\nNode Results:")
            for i, (node, score) in enumerate(zip(results['nodes'][:3], results['node_scores'][:3]), 1):
                print(f"{i}. {node.get('name', 'Unknown')} (Score: {score:.3f})")
                print(f"   Labels: {', '.join(node.get('labels', []))}")

    except httpx.HTTPError as e:
        logger.error(f"Advanced search failed: {e}")


async def example_7_comprehensive_search():
    """
    Example 7: Comprehensive search combining all features.

    Demonstrates a real-world search that combines date filtering,
    entity filtering, advanced ranking, and multiple result types.
    """
    print("\n" + "=" * 60)
    print("Example 7: Comprehensive Search")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Comprehensive search using all available features
    search_params = {
        "query": "recent developments in renewable energy sector",
        "group_ids": ["energy-group", "technology-group"],
        "max_facts": 15,

        # Date filtering - focus on recent developments
        "created_at_start": (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(),
        "created_at_end": datetime.now(timezone.utc).isoformat(),
        "valid_at_start": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),

        # Entity filtering - focus on relevant entities
        "node_labels": ["Company", "Technology", "Project", "Investment"],
        "edge_types": ["INVESTED_IN", "DEVELOPED", "PARTNERED_WITH", "ACQUIRED"],

        # Search methods - use comprehensive approach
        "edge_search_methods": ["cosine_similarity", "bm25"],
        "node_search_methods": ["cosine_similarity", "bm25"],

        # Advanced ranking for optimal results
        "reranker": "mmr",
        "mmr_lambda": 0.6,  # Balance between relevance and diversity
        "min_score": 0.5,
        "reranker_min_score": 0.4,

        # Include all result types for comprehensive view
        "include_nodes": True,
        "include_edges": True,
        "include_episodes": True,
        "include_communities": False,
    }

    try:
        results = await client.search(search_params)

        print(f"Query: {search_params['query']}")
        print(f"Date range: Last 120 days")
        print(f"Entity filters: {', '.join(search_params['node_labels'])}")
        print(f"Search time: {results.get('search_time_ms', 'N/A')}ms")

        # Summary of results
        print(f"\nResults Summary:")
        print(f"Facts: {len(results['facts'])}")
        if results.get('total_results'):
            print(f"Total available: {results['total_results']}")

        if results.get('nodes'):
            print(f"Nodes: {len(results['nodes'])}")
        if results.get('edges'):
            print(f"Edges: {len(results['edges'])}")
        if results.get('episodes'):
            print(f"Episodes: {len(results['episodes'])}")

        # Search metadata
        print(f"\nSearch Configuration:")
        print(f"Ranking method: {results.get('ranking_method', 'N/A')}")
        print(f"Score normalization: {results.get('score_normalization', 'N/A')}")
        if results.get('min_score_threshold'):
            print(f"Score threshold applied: {results['min_score_threshold']}")

        # Show top fact results
        print(f"\nTop Facts:")
        for i, fact in enumerate(results['facts'][:5], 1):
            print(f"{i}. {fact['fact']}")
            if results.get('fact_scores'):
                print(f"   Score: {results['fact_scores'][i-1]:.3f}")
            print(f"   Created: {fact['created_at'][:10]}")

        # Show top node results
        if results.get('nodes') and results.get('node_scores'):
            print(f"\nTop Nodes:")
            for i, (node, score) in enumerate(zip(results['nodes'][:3], results['node_scores'][:3]), 1):
                print(f"{i}. {node.get('name', 'Unknown')} (Score: {score:.3f})")
                if node.get('labels'):
                    print(f"   Type: {', '.join(node['labels'])}")

    except httpx.HTTPError as e:
        logger.error(f"Comprehensive search failed: {e}")


async def example_8_batch_searches():
    """
    Example 8: Batch searches and performance comparison.

    Demonstrates how to perform multiple searches efficiently
    and compare different search configurations.
    """
    print("\n" + "=" * 60)
    print("Example 8: Batch Searches and Performance")
    print("=" * 60)

    client = GraphitiSearchClient()

    # Define different search configurations to compare
    search_configs = [
        {
            "name": "Basic Search",
            "params": {
                "query": "sustainable technology innovations",
                "max_facts": 10,
            }
        },
        {
            "name": "Date Filtered",
            "params": {
                "query": "sustainable technology innovations",
                "max_facts": 10,
                "created_at_start": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                "created_at_end": datetime.now(timezone.utc).isoformat(),
            }
        },
        {
            "name": "Entity Filtered",
            "params": {
                "query": "sustainable technology innovations",
                "max_facts": 10,
                "node_labels": ["Company", "Technology", "Innovation"],
                "edge_types": ["DEVELOPED", "LAUNCHED", "PATENTED"],
            }
        },
        {
            "name": "Advanced Ranking",
            "params": {
                "query": "sustainable technology innovations",
                "max_facts": 10,
                "reranker": "cross_encoder",
                "reranker_min_score": 0.5,
                "mmr_lambda": 0.7,
            }
        }
    ]

    results_summary = []

    # Execute all searches
    for config in search_configs:
        try:
            print(f"\nRunning: {config['name']}...")

            start_time = datetime.now()
            results = await client.search(config['params'])
            end_time = datetime.now()

            client_time_ms = int((end_time - start_time).total_seconds() * 1000)
            server_time_ms = results.get('search_time_ms', 0)

            summary = {
                "name": config['name'],
                "results_count": len(results['facts']),
                "client_time_ms": client_time_ms,
                "server_time_ms": server_time_ms,
                "avg_score": 0,
                "ranking_method": results.get('ranking_method', 'default'),
            }

            # Calculate average score
            if results.get('fact_scores'):
                summary['avg_score'] = sum(results['fact_scores']) / len(results['fact_scores'])

            results_summary.append(summary)

            print(f"  Results: {summary['results_count']}")
            print(f"  Client time: {summary['client_time_ms']}ms")
            print(f"  Server time: {summary['server_time_ms']}ms")
            print(f"  Avg score: {summary['avg_score']:.3f}")

        except httpx.HTTPError as e:
            logger.error(f"Search '{config['name']}' failed: {e}")
            results_summary.append({
                "name": config['name'],
                "error": str(e)
            })

    # Print comparison table
    print(f"\n{'Configuration':<20} {'Results':<10} {'Client (ms)':<12} {'Server (ms)':<12} {'Avg Score':<12} {'Ranking':<15}")
    print("-" * 85)

    for summary in results_summary:
        if 'error' in summary:
            print(f"{summary['name']:<20} {'ERROR':<10} {'-':<12} {'-':<12} {'-':<12} {'-':<15}")
        else:
            print(f"{summary['name']:<20} {summary['results_count']:<10} "
                  f"{summary['client_time_ms']:<12} {summary['server_time_ms']:<12} "
                  f"{summary['avg_score']:<12.3f} {summary['ranking_method']:<15}")


async def main():
    """
    Main function to run all examples.

    Note: These examples assume a running Graphiti server at http://localhost:8000
    Make sure the server is running before executing these examples.
    """
    print("Graphiti Advanced Search Examples")
    print("=" * 60)
    print("These examples demonstrate the enhanced search functionality")
    print("including date filtering, entity filtering, and advanced ranking.")
    print("\nPrerequisites:")
    print("- Graphiti server running at http://localhost:8000")
    print("- Sample data loaded in the knowledge graph")
    print("- Appropriate group IDs for your data")

    try:
        # Run all examples
        await example_1_basic_search()
        await example_2_date_range_filtering()
        await example_3_entity_filtering()
        await example_4_advanced_ranking()
        await example_5_search_methods()
        await example_6_advanced_filtering()
        await example_7_comprehensive_search()
        await example_8_batch_searches()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nError: {e}")
        print("Make sure the Graphiti server is running and accessible.")


if __name__ == "__main__":
    """
    To run these examples:

    1. Start the Graphiti server:
       cd stack-graphiti/server
       python -m uvicorn graph_service.main:app --reload --host 0.0.0.0 --port 8000

    2. Make sure you have sample data loaded in your knowledge graph

    3. Run this script:
       python advanced_search.py

    4. Modify the group_ids and queries to match your data
    """
    asyncio.run(main())