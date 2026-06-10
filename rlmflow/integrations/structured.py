"""Structured-output parsing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel, TypeAdapter, ValidationError


def render_json_schema_hint(schema: Mapping[str, Any] | str) -> str:
    return json.dumps(_load_json_schema(schema), indent=2)


def render_pydantic_schema_hint(schema: type[BaseModel] | TypeAdapter[Any]) -> str:
    return json.dumps(_json_schema_for_pydantic(schema), indent=2)


def json_schema_for(schema: Schema) -> dict[str, Any]:
    if isinstance(schema, Mapping | str):
        json_schema = dict(_load_json_schema(schema))
    else:
        json_schema = _json_schema_for_pydantic(schema)
    jsonschema.validators.validator_for(json_schema).check_schema(json_schema)
    return json_schema


Schema = type[BaseModel] | TypeAdapter[Any] | Mapping[str, Any] | str


class StructuredOutputError(ValueError):
    """Agent-facing structured-output parse/validation failure."""

    def __init__(
        self,
        *,
        content: str,
        schema: Schema,
        cause: Exception,
    ) -> None:
        self.content = content
        self.schema = schema
        self.cause = cause
        super().__init__(_format_error_message(content, schema, cause))


class StructuredOutputParser:
    """Parse and validate a structured-output JSON string.

    ``content`` is the JSON string to validate. ``schema`` is a Pydantic model
    class, TypeAdapter, JSON-schema-like dictionary, or JSON schema string.
    """

    def system_prompt_hint(self, schema: Schema) -> str:
        if isinstance(schema, Mapping | str):
            return render_json_schema_hint(schema)
        return render_pydantic_schema_hint(schema)

    def __call__(self, content: str, schema: Schema) -> Any:
        if isinstance(schema, Mapping | str):
            try:
                return _validate_json_schema(content, schema)
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                raise StructuredOutputError(
                    content=content,
                    schema=schema,
                    cause=exc,
                ) from exc

        adapter = _adapter_for(schema)
        try:
            return adapter.validate_json(content)
        except ValidationError as exc:
            raise StructuredOutputError(
                content=content,
                schema=schema,
                cause=exc,
            ) from exc


def _adapter_for(schema: type[BaseModel] | TypeAdapter[Any]) -> TypeAdapter[Any]:
    if isinstance(schema, TypeAdapter):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return TypeAdapter(schema)
    raise TypeError(
        "schema must be a Pydantic model class, TypeAdapter, "
        "JSON schema dict, or JSON schema string"
    )


def _json_schema_for_pydantic(
    schema: type[BaseModel] | TypeAdapter[Any],
) -> dict[str, Any]:
    if isinstance(schema, TypeAdapter):
        return schema.json_schema()
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    raise TypeError(
        "schema must be a Pydantic model class, TypeAdapter, "
        "JSON schema dict, or JSON schema string"
    )


def _validate_json_schema(content: str, schema: Mapping[str, Any] | str) -> Any:
    json_schema = _load_json_schema(schema)
    value = json.loads(content)
    jsonschema.validate(instance=value, schema=json_schema)
    return value


def _load_json_schema(schema: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(schema, str):
        loaded = json.loads(schema)
        if not isinstance(loaded, Mapping):
            raise TypeError("JSON schema string must decode to an object")
        return loaded
    return schema


def _format_error_message(content: str, schema: Schema, cause: Exception) -> str:
    schema_text = _schema_text(schema)
    return (
        "Structured output is invalid.\n"
        "Hint: call done(value) with a JSON-compatible Python value that matches "
        "the expected schema. Do not pass prose, Markdown fences, or a JSON "
        "string containing JSON.\n\n"
        "Expected JSON Schema:\n"
        f"{schema_text}\n\n"
        "Received JSON:\n"
        f"{_truncate(content)}\n\n"
        "Validation error:\n"
        f"{type(cause).__name__}: {cause}"
    )


def _schema_text(schema: Schema) -> str:
    try:
        return _truncate(json.dumps(json_schema_for(schema), indent=2))
    except Exception:
        return _truncate(repr(schema))


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


__all__ = [
    "Schema",
    "StructuredOutputError",
    "StructuredOutputParser",
    "json_schema_for",
]
