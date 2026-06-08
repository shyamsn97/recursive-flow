"""Structured-output parsing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel, TypeAdapter


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
            return _validate_json_schema(content, schema)
        return _adapter_for(schema).validate_json(content)


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


__all__ = ["Schema", "StructuredOutputParser", "json_schema_for"]
