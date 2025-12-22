# Advanced Search Examples

This directory contains comprehensive examples demonstrating the enhanced search functionality in Graphiti, including date range filtering, entity filtering, and advanced ranking options.

## Files Overview

### 1. `advanced_search.py`
**HTTP API Examples** - Demonstrates how to use the enhanced search endpoints via HTTP requests.

**Features demonstrated:**
- Basic search (backward compatibility)
- Date range filtering (created_at, valid_at, invalid_at, expired_at)
- Entity filtering (node_labels, edge_types)
- Advanced ranking options (MMR, cross-encoder, node distance, RRF)
- Search method configuration (BM25, cosine similarity, BFS)
- Complex property-based filtering
- Batch searches and performance comparison
- Comprehensive search combining all features

### 2. `advanced_search_core.py`
**Core Library Examples** - Shows how to use the enhanced search functionality directly with the Graphiti core library (without HTTP).

**Features demonstrated:**
- Direct usage of SearchFilters and SearchConfig classes
- Advanced search configuration with recipes
- Node, edge, and community searches
- Custom search configurations
- Property-based filtering with comparison operators
- Performance comparison of different methods

## Prerequisites

### Database Setup
You need a running FalkorDB instance:

```bash
# Using Docker
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb

# Or using docker-compose
cd stack-graphiti
docker-compose up -d falkordb
```

### Environment Variables (Optional)
```bash
export FALKORDB_HOST=localhost
export FALKORDB_PORT=6379
export FALKORDB_USERNAME=your_username
export FALKORDB_PASSWORD=your_password
```

### Server Setup (for HTTP API examples)
Start the Graphiti server:

```bash
cd stack-graphiti/server
python -m uvicorn graph_service.main:app --reload --host 0.0.0.0 --port 8000
```

### Python Dependencies
Install required packages:

```bash
cd stack-graphiti
pip install httpx python-dotenv
```

## Running the Examples

### HTTP API Examples

```bash
# Make sure the server is running first
cd stack-graphiti/examples
python advanced_search.py
```

**Output:** The script will run 8 different examples showing various search configurations and their results.

### Core Library Examples

```bash
cd stack-graphiti/examples
python advanced_search_core.py
```

**Output:** The script will demonstrate 9 different search scenarios directly using the core library.

## Key Features Demonstrated

### 1. Date Range Filtering

Filter search results by temporal fields:

```python
# Simple date ranges
search_params = {
    "query": "recent developments",
    "created_at_start": "2024-01-01T00:00:00Z",
    "created_at_end": "2024-12-31T23:59:59Z",
    "valid_at_start": "2024-06-01T00:00:00Z"
}

# Advanced date filters with complex logic
filters = SearchFilters(
    created_at=[
        [
            DateFilter(
                date=datetime(2024, 1, 1),
                comparison_operator=ComparisonOperator.greater_than_equal
            ),
            DateFilter(
                date=datetime(2024, 12, 31),
                comparison_operator=ComparisonOperator.less_than_equal
            )
        ]
    ]
)
```

### 2. Entity Filtering

Filter by node labels and edge types:

```python
# Simple entity filtering
search_params = {
    "query": "company executives",
    "node_labels": ["Person", "Company", "Executive"],
    "edge_types": ["WORKS_AT", "MANAGES", "REPORTS_TO"]
}

# Advanced filtering
filters = SearchFilters(
    node_labels=["Person", "Organization"],
    edge_types=["HAS_ROLE", "EMPLOYED_BY"]
)
```

### 3. Advanced Ranking

Different reranking methods for optimal results:

```python
# MMR for diversity vs relevance
search_params = {
    "query": "technology innovations",
    "reranker": "mmr",
    "mmr_lambda": 0.6,  # 0 = max diversity, 1 = max relevance
    "min_score": 0.4
}

# Cross-encoder for high-quality reranking
search_params = {
    "query": "technology innovations",
    "reranker": "cross_encoder",
    "reranker_min_score": 0.5
}

# Node distance for graph-based ranking
search_params = {
    "query": "technology innovations",
    "reranker": "node_distance",
    "bfs_max_depth": 3
}
```

### 4. Search Methods

Choose different search algorithms:

```python
# BM25 for keyword matching
search_params = {
    "query": "specific keywords",
    "edge_search_methods": ["bm25"],
    "node_search_methods": ["bm25"]
}

# Semantic similarity
search_params = {
    "query": "related concepts",
    "edge_search_methods": ["cosine_similarity"],
    "node_search_methods": ["cosine_similarity"]
}

# Hybrid approach (default)
# Uses both BM25 and cosine_similarity
```

### 5. Complex Property Filtering

Advanced filtering with comparison operators:

```python
# Property filters
search_params = {
    "filters": {
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
                "property_value": "L5",
                "comparison_operator": "<>"
            }
        ]
    }
}
```

### 6. Result Type Configuration

Control what types of results to include:

```python
search_params = {
    "include_nodes": True,        # Include node results
    "include_edges": True,        # Include edge results (default)
    "include_episodes": True,     # Include episode results
    "include_communities": False  # Include community results
}
```

## Search Recipes

The core library provides pre-configured search recipes:

- `EDGE_HYBRID_SEARCH_RRF`: Edge search with Reciprocal Rank Fusion
- `EDGE_HYBRID_SEARCH_MMR`: Edge search with Maximal Marginal Relevance
- `EDGE_HYBRID_SEARCH_CROSS_ENCODER`: Edge search with cross-encoder reranking
- `EDGE_HYBRID_SEARCH_NODE_DISTANCE`: Edge search with graph distance reranking
- `NODE_HYBRID_SEARCH_RRF`: Node-focused search with RRF
- `COMBINED_HYBRID_SEARCH_RRF`: Combined node and edge search
- `COMMUNITY_HYBRID_SEARCH_RRF`: Community-based search

## API Endpoints

### `/search` (Smart Search)
Automatically detects when to use advanced features:

- **Simple queries**: Use optimized basic search
- **Advanced parameters**: Trigger enhanced search capabilities

### `/search-advanced` (Always Advanced)
Always uses enhanced search with all parameters:

- Processes all filtering and ranking options
- Returns comprehensive metadata and scoring information
- Includes all result types (nodes, edges, episodes, communities)

## Response Format

The enhanced search response includes:

```json
{
    "facts": [...],
    "fact_scores": [0.95, 0.87, 0.72, ...],
    "total_results": 42,
    "search_time_ms": 150,
    "search_config_used": {
        "edge_search_method": ["bm25", "cosine_similarity"],
        "reranker": "mmr",
        "mmr_lambda": 0.6
    },
    "ranking_method": "mmr",
    "score_normalization": "min_max",
    "max_score_possible": 1.0,
    "min_score_threshold": 0.3,
    "nodes": [...],
    "edges": [...],
    "episodes": [...],
    "communities": [...]
}
```

## Performance Considerations

### Search Method Performance
- **Basic search**: Fastest, suitable for simple queries
- **RRF reranking**: Good balance of speed and quality
- **MMR reranking**: Moderate speed, good for diverse results
- **Cross-encoder reranking**: Slower but highest quality
- **Node distance reranking**: Fast for graph-based queries

### Filtering Impact
- **Date filters**: Minimal performance impact
- **Entity filters**: Minimal performance impact
- **Property filters**: Variable impact based on complexity
- **Combined filters**: Cumulative impact

### Optimization Tips
1. Use appropriate date ranges to limit result sets
2. Prefer node_labels/edge_types over complex property filters
3. Use reasonable max_facts limits (10-100)
4. Choose simpler rerankers (RRF) for real-time applications
5. Use cross-encoder for offline batch processing

## Troubleshooting

### Common Issues

1. **Server not running**: Make sure the Graphiti server is started
2. **Database connection**: Verify FalkorDB is accessible
3. **No results**: Check if data exists in your knowledge graph
4. **Slow responses**: Consider reducing max_facts or using simpler configurations

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check server logs for detailed error information.

## Next Steps

1. **Adapt to your data**: Modify queries and filters to match your specific use case
2. **Performance tuning**: Experiment with different configurations for optimal performance
3. **Integration**: Use these patterns in your own applications
4. **Advanced use cases**: Combine multiple features for sophisticated search scenarios

## Additional Resources

- [Graphiti Documentation](../README.md)
- [API Reference](../server/graph_service/)
- [Core Library Documentation](../graphiti_core/)
- [Search Configuration Recipes](../graphiti_core/search/search_config_recipes.py)