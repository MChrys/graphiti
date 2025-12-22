import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import initialize_graphiti

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor


def setup_telemetry():
    """Configure OpenTelemetry tracing."""
    otlp_endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4317')
    service_name = os.environ.get('OTEL_SERVICE_NAME', 'graphiti')

    # Skip if disabled
    if os.environ.get('OTEL_ENABLED', 'true').lower() == 'false':
        return

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: '0.1.0',
    })

    # Setup tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Instrument httpx (used for embedding calls)
    HTTPXClientInstrumentor().instrument()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


# Setup telemetry before app creation
setup_telemetry()

app = FastAPI(
    title="Graphiti Knowledge Graph API",
    description="""
    ## Graphiti Knowledge Graph Search API

    A powerful knowledge graph search service with advanced filtering, ranking, and retrieval capabilities.

    ### Key Features

    - **Advanced Search**: Multiple search methods including vector similarity, BM25, and graph traversal
    - **Intelligent Ranking**: RRF, MMR, cross-encoder, and distance-based ranking options
    - **Comprehensive Filtering**: Date ranges, entity types, and property-based filtering
    - **Multiple Result Types**: Facts, nodes, edges, episodes, and communities
    - **Backward Compatibility**: Existing clients continue to work without changes
    - **Performance Optimized**: Automatic detection of query complexity for optimal performance

    ### Search Endpoints

    - **POST /search**: Enhanced search with backward compatibility
    - **POST /search-advanced**: Always uses advanced search for consistent behavior
    - **POST /get-memory**: Contextual memory retrieval for conversational AI

    ### Authentication

    Currently no authentication is configured. Add appropriate authentication for production deployments.

    ### Documentation

    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI JSON: `/openapi.json`

    ### Performance Notes

    - Simple queries automatically use optimized basic search
    - Advanced features (filtering, ranking) trigger enhanced search
    - Consider using date range filters for large datasets
    - Set appropriate `max_facts` limits for your use case
    """,
    version="2.0.0",
    lifespan=lifespan,
    contact={
        "name": "Graphiti API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "search",
            "description": "Knowledge graph search and retrieval operations with advanced filtering and ranking"
        },
        {
            "name": "ingest",
            "description": "Data ingestion and knowledge graph management operations"
        },
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        }
    ]
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get(
    '/healthcheck',
    tags=["health"],
    summary="Health check endpoint",
    description="Simple health check to verify the API is running and responsive.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "healthy"}
                }
            }
        }
    }
)
async def healthcheck():
    """Health check endpoint."""
    return JSONResponse(content={'status': 'healthy'}, status_code=200)


@app.get(
    '/metrics',
    tags=["health"],
    summary="Metrics endpoint",
    description="Prometheus metrics endpoint (placeholder for future implementation).",
    responses={
        200: {
            "description": "Metrics endpoint status",
            "content": {
                "application/json": {
                    "example": {"status": "ok"}
                }
            }
        }
    }
)
async def metrics():
    """Prometheus metrics endpoint (placeholder for future metrics)."""
    return JSONResponse(content={'status': 'ok'}, status_code=200)
