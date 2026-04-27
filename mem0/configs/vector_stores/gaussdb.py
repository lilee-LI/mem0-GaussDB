from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GaussDBConfig(BaseModel):
    database: str = Field("postgres", description="GaussDB database name")
    dbname: Optional[str] = Field(None, description="Alias for database")
    collection_name: str = Field("mem0", description="Name of the collection table")
    embedding_model_dims: int = Field(1536, description="Dimensions of the embedding model")
    user: Optional[str] = Field(None, description="Database user")
    password: Optional[str] = Field(None, description="Database password")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    connection_string: Optional[str] = Field(None, description="GaussDB connection string")
    connection_pool: Optional[Any] = Field(None, description="Existing psycopg2 connection pool")
    minconn: int = Field(1, description="Minimum number of connections in the pool")
    maxconn: int = Field(5, description="Maximum number of connections in the pool")
    sslmode: Optional[str] = Field(None, description="SSL mode")
    sslrootcert: Optional[str] = Field(None, description="SSL root certificate path")
    client_encoding: Optional[str] = Field("UTF8", description="Client encoding used by psycopg2 connections")
    table_storage: str = Field("ustore", description="GaussDB table storage type")
    compatibility_mode: str = Field("A", description="GaussDB compatibility mode")
    gaussdb_version_baseline: str = Field("506", description="Commercial baseline version family")
    id_column_type: str = Field("uuid", description="id column type: uuid or varchar")
    vector_index_type: str = Field("gsdiskann", description="Vector index type: gsdiskann or gsivfflat")
    vector_metric: str = Field("cosine", description="Vector metric: cosine or l2")
    vector_index_maintenance_work_mem: Optional[str] = Field(
        "128MB", description="Session-local maintenance_work_mem used while building vector indexes"
    )
    bm25_enabled: bool = Field(True, description="Enable native GaussDB BM25 keyword search")
    bm25_fail_fast: bool = Field(False, description="Raise BM25 errors instead of returning None")
    bm25_ranking_metric: int = Field(0, description="GaussDB BM25 ranking metric; 0 is BM25_OKAPI")
    bm25_ncandidates: int = Field(128, description="GaussDB BM25 candidate count")
    bm25_dictionary: Optional[str] = Field(None, description="Optional BM25 dictionary name")
    metadata_column_mode: str = Field(
        "jsonb",
        description="Metadata storage mode: jsonb, text, or redundant_columns",
    )
    allowed_filter_keys: Optional[List[str]] = Field(None, description="Optional allowlist for metadata filter keys")
    require_scoped_filters: bool = Field(True, description="Require user_id, agent_id, or run_id on read paths")
    scope_filter_keys: List[str] = Field(
        default_factory=lambda: ["user_id", "agent_id", "run_id"],
        description="Filter keys accepted as tenant scope",
    )
    enable_capability_probe: bool = Field(True, description="Probe GaussDB vector/BM25/type capabilities on init")
    enable_observability: bool = Field(True, description="Enable provider metrics and structured logs")
    slow_query_ms: int = Field(1000, description="Slow query warning threshold")
    retry_attempts: int = Field(2, description="Retry attempts for transient database errors")
    retry_backoff_seconds: float = Field(0.1, description="Initial retry backoff in seconds")
    auto_create: bool = Field(True, description="Create the collection on provider initialization")

    @model_validator(mode="before")
    @classmethod
    def normalize_database_alias(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("dbname") and not values.get("database"):
            values["database"] = values["dbname"]
        return values

    @model_validator(mode="before")
    @classmethod
    def check_auth_and_connection(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("connection_pool") is not None or values.get("connection_string"):
            return values

        missing = [key for key in ("user", "password", "host", "port") if not values.get(key)]
        if missing:
            raise ValueError(
                "GaussDB config requires connection_pool, connection_string, or individual "
                f"connection fields. Missing: {', '.join(missing)}"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                "Extra fields not allowed: "
                f"{', '.join(sorted(extra_fields))}. Please input only the following fields: "
                f"{', '.join(sorted(allowed_fields))}"
            )
        return values

    @field_validator("embedding_model_dims")
    @classmethod
    def validate_embedding_dims(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_model_dims must be greater than 0")
        return value

    @field_validator("minconn", "maxconn", "retry_attempts", "slow_query_ms", "bm25_ncandidates")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("numeric configuration values must be non-negative")
        return value

    @field_validator("maxconn")
    @classmethod
    def validate_pool_bounds(cls, value: int, info) -> int:
        minconn = info.data.get("minconn")
        if minconn is not None and value < minconn:
            raise ValueError("maxconn must be greater than or equal to minconn")
        return value

    @field_validator("table_storage")
    @classmethod
    def validate_table_storage(cls, value: str) -> str:
        normalized = value.lower()
        if normalized != "ustore":
            raise ValueError("GaussDB mem0 provider currently supports only Ustore tables")
        return normalized

    @field_validator("compatibility_mode")
    @classmethod
    def validate_compatibility_mode(cls, value: str) -> str:
        normalized = value.upper()
        if normalized != "A":
            raise ValueError("GaussDB mem0 provider currently targets A compatibility mode")
        return normalized

    @field_validator("id_column_type")
    @classmethod
    def validate_id_column_type(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"uuid", "varchar"}:
            raise ValueError("id_column_type must be 'uuid' or 'varchar'")
        return normalized

    @field_validator("vector_index_type")
    @classmethod
    def validate_vector_index_type(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"gsdiskann", "gsivfflat"}:
            raise ValueError("vector_index_type must be 'gsdiskann' or 'gsivfflat'")
        return normalized

    @field_validator("vector_metric")
    @classmethod
    def validate_vector_metric(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"cosine", "l2"}:
            raise ValueError("vector_metric must be 'cosine' or 'l2'")
        return normalized

    @field_validator("metadata_column_mode")
    @classmethod
    def validate_metadata_column_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"jsonb", "text", "redundant_columns"}:
            raise ValueError("metadata_column_mode must be 'jsonb', 'text', or 'redundant_columns'")
        return normalized

    @field_validator("scope_filter_keys")
    @classmethod
    def validate_scope_filter_keys(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("scope_filter_keys must contain at least one key")
        return value

    model_config = ConfigDict(arbitrary_types_allowed=True)
