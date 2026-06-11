from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, TypeAdapter

from rflow.integrations.structured import StructuredOutputError, StructuredOutputParser


class WeatherAdvice(BaseModel):
    cities: list[str]
    packing_list: list[str]
    warnings: list[str]


def test_parser_accepts_json_string_with_pydantic_model_schema():
    parser = StructuredOutputParser()
    content = json.dumps(
        {
            "cities": ["Seattle", "Austin"],
            "packing_list": ["rain jacket", "water bottle"],
            "warnings": [],
        }
    )

    parsed = parser(content, WeatherAdvice)

    assert isinstance(parsed, WeatherAdvice)
    assert parsed.cities == ["Seattle", "Austin"]


def test_parser_accepts_type_adapter_schema():
    parser = StructuredOutputParser()
    adapter = TypeAdapter(dict[str, list[str]])

    parsed = parser('{"blockers": ["missing owner"]}', adapter)

    assert parsed == {"blockers": ["missing owner"]}


def test_parser_rejects_invalid_pydantic_output():
    parser = StructuredOutputParser()

    with pytest.raises(StructuredOutputError) as exc_info:
        parser('{"cities": ["Seattle"], "packing_list": "rain jacket", "warnings": []}', WeatherAdvice)
    message = str(exc_info.value)
    assert "Structured output is invalid" in message
    assert "Expected JSON Schema" in message
    assert "Received JSON" in message
    assert "packing_list" in message


def test_parser_accepts_json_schema_dict():
    parser = StructuredOutputParser()
    schema = {
        "type": "object",
        "properties": {
            "blockers": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["blockers"],
    }

    parsed = parser('{"blockers": ["missing owner"]}', schema)

    assert parsed == {"blockers": ["missing owner"]}


def test_parser_accepts_json_schema_string():
    parser = StructuredOutputParser()
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "owner": {"type": "string"},
            },
            "required": ["confidence"],
        }
    )

    parsed = parser('{"confidence": 0.8, "owner": "infra"}', schema)

    assert parsed == {"confidence": 0.8, "owner": "infra"}


def test_parser_rejects_invalid_json_schema_output():
    parser = StructuredOutputParser()
    schema = {
        "type": "object",
        "properties": {"confidence": {"type": "number"}},
        "required": ["confidence"],
    }

    with pytest.raises(StructuredOutputError) as exc_info:
        parser('{"confidence": "high"}', schema)
    message = str(exc_info.value)
    assert "Structured output is invalid" in message
    assert "Expected JSON Schema" in message
    assert '{"confidence": "high"}' in message


def test_parser_rejects_malformed_json_with_repair_hint():
    parser = StructuredOutputParser()
    schema = {
        "type": "object",
        "properties": {"confidence": {"type": "number"}},
        "required": ["confidence"],
    }

    with pytest.raises(StructuredOutputError) as exc_info:
        parser('```json\n{"confidence": 0.8}\n```', schema)
    message = str(exc_info.value)
    assert "Do not pass prose, Markdown fences" in message
    assert "JSONDecodeError" in message
