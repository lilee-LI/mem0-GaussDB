import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ENV_DEFAULTS = {
    "connection_string": ("GAUSSDB_CONNECTION_STRING", "GAUSSDB_DSN", "GAUSSDB_URL"),
    "host": ("GAUSSDB_HOST",),
    "port": ("GAUSSDB_PORT",),
    "database": ("GAUSSDB_DATABASE", "GAUSSDB_DBNAME"),
    "user": ("GAUSSDB_USER",),
    "password": ("GAUSSDB_PASSWORD",),
    "sslmode": ("GAUSSDB_SSLMODE",),
    "sslrootcert": ("GAUSSDB_SSLROOTCERT",),
}


_METADATA_MODE_MAP = {
    "jsonb": {"payload_storage_mode": "jsonb", "filter_storage_mode": "json_expression"},
    "redundant_columns": {"payload_storage_mode": "jsonb", "filter_storage_mode": "redundant_columns"},
    "compatible": {"payload_storage_mode": "text", "filter_storage_mode": "redundant_columns"},
    "text": {"payload_storage_mode": "text", "filter_storage_mode": "redundant_columns"},
}

_BM25_MODE_MAP = {
    "auto": {"bm25_enabled": True, "bm25_fail_fast": False},
    "required": {"bm25_enabled": True, "bm25_fail_fast": True},
    "disabled": {"bm25_enabled": False, "bm25_fail_fast": False},
}


def _first_env(names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


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
    dsn: Optional[str] = Field(None, description="Alias for connection_string")
    url: Optional[str] = Field(None, description="Alias for connection_string")
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
        description="Legacy combined metadata mode: jsonb, text, or redundant_columns",
    )
    payload_storage_mode: Optional[str] = Field(
        None,
        description="Payload storage mode: jsonb or text. Defaults are derived from metadata_column_mode.",
    )
    filter_storage_mode: Optional[str] = Field(
        None,
        description="Filter storage mode: json_expression or redundant_columns. Defaults are derived from payload mode.",
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
    profile: str = Field("commercial", description="High-level defaults profile: commercial or compatibility")
    metadata_mode: Optional[str] = Field(
        "auto",
        description="High-level metadata mode: auto, jsonb, redundant_columns, compatible, or text",
    )
    bm25_mode: Optional[str] = Field("auto", description="High-level BM25 mode: auto, required, or disabled")

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate_input(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(values or {})
        original_fields = set(values.keys())
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                "Extra fields not allowed: "
                f"{', '.join(sorted(extra_fields))}. Please input only the following fields: "
                f"{', '.join(sorted(allowed_fields))}"
            )

        if values.get("dbname") and not values.get("database"):
            values["database"] = values["dbname"]

        for alias in ("dsn", "url"):
            if values.get(alias) and not values.get("connection_string"):
                values["connection_string"] = values[alias]

        profile = str(values.get("profile", "commercial")).lower()
        if profile == "compatibility":
            values.setdefault("metadata_mode", "compatible")
            values.setdefault("vector_index_type", "gsivfflat")

        explicit_metadata_low_level = bool(
            original_fields & {"metadata_column_mode", "payload_storage_mode", "filter_storage_mode"}
        )
        metadata_mode = values.get("metadata_mode")
        if metadata_mode is None and explicit_metadata_low_level:
            values["metadata_mode"] = None
        elif metadata_mode is not None:
            metadata_mode = str(metadata_mode).lower()
            values["metadata_mode"] = metadata_mode
            if explicit_metadata_low_level:
                if "metadata_mode" in original_fields and metadata_mode not in {"auto", "none"}:
                    raise ValueError(
                        "metadata_mode cannot be combined with metadata_column_mode, "
                        "payload_storage_mode, or filter_storage_mode"
                    )
                values["metadata_mode"] = None
            elif metadata_mode != "auto":
                values.update(_METADATA_MODE_MAP.get(metadata_mode, {}))

        explicit_bm25_low_level = bool(original_fields & {"bm25_enabled", "bm25_fail_fast"})
        bm25_mode = values.get("bm25_mode")
        if bm25_mode is None and explicit_bm25_low_level:
            values["bm25_mode"] = None
        elif bm25_mode is not None:
            bm25_mode = str(bm25_mode).lower()
            values["bm25_mode"] = bm25_mode
            if explicit_bm25_low_level:
                if "bm25_mode" in original_fields and bm25_mode not in {"auto", "none"}:
                    raise ValueError("bm25_mode cannot be combined with bm25_enabled or bm25_fail_fast")
                values["bm25_mode"] = None
            else:
                values.update(_BM25_MODE_MAP.get(bm25_mode, {}))

        for field_name, env_names in _ENV_DEFAULTS.items():
            if not values.get(field_name):
                env_value = _first_env(env_names)
                if env_value:
                    values[field_name] = env_value

        if values.get("connection_pool") is not None or values.get("connection_string"):
            return values

        missing = [key for key in ("user", "password", "host", "port") if not values.get(key)]
        if missing:
            raise ValueError(
                "GaussDB config requires connection_pool, connection_string, GAUSSDB_* environment variables, "
                "or individual connection fields. "
                f"Missing: {', '.join(missing)}"
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

    @field_validator("payload_storage_mode")
    @classmethod
    def validate_payload_storage_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.lower()
        if normalized not in {"jsonb", "text"}:
            raise ValueError("payload_storage_mode must be 'jsonb' or 'text'")
        return normalized

    @field_validator("filter_storage_mode")
    @classmethod
    def validate_filter_storage_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.lower()
        if normalized not in {"json_expression", "redundant_columns"}:
            raise ValueError("filter_storage_mode must be 'json_expression' or 'redundant_columns'")
        return normalized

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"commercial", "compatibility"}:
            raise ValueError("profile must be 'commercial' or 'compatibility'")
        return normalized

    @field_validator("metadata_mode")
    @classmethod
    def validate_metadata_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.lower()
        if normalized not in {"auto", "jsonb", "redundant_columns", "compatible", "text"}:
            raise ValueError("metadata_mode must be 'auto', 'jsonb', 'redundant_columns', 'compatible', or 'text'")
        return normalized

    @field_validator("bm25_mode")
    @classmethod
    def validate_bm25_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.lower()
        if normalized not in {"auto", "required", "disabled"}:
            raise ValueError("bm25_mode must be 'auto', 'required', or 'disabled'")
        return normalized

    @field_validator("scope_filter_keys")
    @classmethod
    def validate_scope_filter_keys(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("scope_filter_keys must contain at least one key")
        return value

    @model_validator(mode="after")
    def validate_storage_mode_combination(self) -> "GaussDBConfig":
        if self.payload_storage_mode == "text" and self.filter_storage_mode == "json_expression":
            raise ValueError("filter_storage_mode='json_expression' requires payload_storage_mode='jsonb'")
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)
