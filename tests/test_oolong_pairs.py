from __future__ import annotations

import pytest

from benchmarks.eval import DATASETS
from benchmarks.eval.run import build_parser, config_from_args
from benchmarks.eval.tasks.oolong_pairs import (
    PAPER_CONTEXT_LEN,
    PAPER_QUESTION_IDS,
    OolongPairsDataset,
    OolongPairsPaperDataset,
    _context_at_length,
    _extract_pairs,
)
from benchmarks.eval.types import Example, Prediction


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": str(index),
            "question": f"Return the qualifying pairs for query {index}.",
            "answer": [f"({index}, {index + 20})"],
            "type": "list_of_answers",
        }
        for index in range(1, 21)
    ]


def test_oolong_pairs_exposes_the_twenty_official_queries_with_one_context():
    dataset = OolongPairsDataset()
    dataset._rows = _rows()
    dataset._context = "unlabeled trec_coarse context"

    examples = dataset.examples(split="test", limit=None, seed=0)

    assert len(examples) == 20
    assert {example.metadata["question_id"] for example in examples} == {
        str(index) for index in range(1, 21)
    }
    assert all(
        example.context == {"context": "unlabeled trec_coarse context"}
        for example in examples
    )
    assert all(example.metadata["context_len"] == 32768 for example in examples)


def test_oolong_pairs_limit_selects_a_reproducible_subset():
    dataset = OolongPairsDataset()
    dataset._rows = _rows()
    dataset._context = "context"

    first = dataset.examples(split="test", limit=5, seed=7)
    second = dataset.examples(split="test", limit=5, seed=7)

    assert [example.id for example in first] == [example.id for example in second]
    assert len(first) == 5


def test_oolong_pairs_scores_pair_sets_with_f1():
    dataset = OolongPairsDataset()
    example = Example(
        id="pairs",
        prompt="Return pairs.",
        expected=["(1, 2)", "(3, 4)", "(5, 6)"],
    )

    score = dataset.score(
        example,
        Prediction(answer="Pairs: (1, 2), (3,4), and (9, 10)."),
    )

    assert score.value == pytest.approx(2 / 3)
    assert score.details == {
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "expected_pairs": 3,
        "predicted_pairs": 3,
        "true_positives": 2,
    }
    assert not score.correct


def test_oolong_pairs_normalizes_public_parser_formats_and_handles_empty_sets():
    dataset = OolongPairsDataset()
    pair_example = Example(
        id="pairs",
        prompt="Return pairs.",
        expected=["(1, 2)"],
    )
    empty_example = Example(id="empty", prompt="Return pairs.", expected=[])

    assert dataset.score(pair_example, Prediction(answer="(1, 2)")).correct
    assert dataset.score(pair_example, Prediction(answer="(2, 1)")).correct
    assert dataset.score(pair_example, Prediction(answer="1, 2")).correct
    assert dataset.score(
        pair_example,
        Prediction(answer="<think>wrong: (8, 9)</think>\n(1, 2)"),
    ).correct
    assert dataset.score(empty_example, Prediction(answer="[]")).correct
    assert _extract_pairs(["(1, 2)", "(1, 2)"]) == {(1, 2)}


def test_oolong_pairs_rejects_unknown_context_lengths():
    with pytest.raises(ValueError, match="context_len must be one of"):
        OolongPairsDataset(context_len=12345)


def test_oolong_pairs_selects_the_context_length_regardless_of_window_id():
    rows = [
        {
            "context_len": 8192,
            "context_window_id": 9,
            "context_window_text": "8k context",
        },
        {
            "context_len": 32768,
            "context_window_id": 0,
            "context_window_text": "32k context",
        },
    ]

    assert _context_at_length(rows, 8192) == "8k context"


def test_oolong_pairs_pins_selected_question_ids_in_order():
    dataset = OolongPairsDataset(context_len=8192, question_ids=("4", "11"))
    dataset._rows = _rows()
    dataset._context = "8k context"

    examples = dataset.examples(split="test", limit=None, seed=99)

    assert [example.id for example in examples] == [
        "oolong_pairs_8192_04",
        "oolong_pairs_8192_11",
    ]
    assert [example.metadata["question_id"] for example in examples] == ["4", "11"]


def test_oolong_pairs_rejects_unknown_question_ids():
    dataset = OolongPairsDataset(question_ids=("4", "99"))
    dataset._rows = _rows()
    dataset._context = "context"

    with pytest.raises(ValueError, match="missing question ids: \\['99'\\]"):
        dataset.examples(split="test", limit=None, seed=0)


def test_oolong_pairs_paper_set_pins_all_twenty_queries_in_official_order():
    dataset = OolongPairsPaperDataset()
    dataset._rows = _rows()
    dataset._context = "32k paper context"

    examples = dataset.examples(split="test", limit=None, seed=99)

    assert DATASETS.expand(["oolong-pairs"]) == ["oolong-pairs"]
    assert PAPER_QUESTION_IDS == tuple(str(index) for index in range(1, 21))
    assert [example.metadata["question_id"] for example in examples] == list(
        PAPER_QUESTION_IDS
    )
    assert [example.id for example in examples] == [
        f"oolong_pairs_{PAPER_CONTEXT_LEN}_{int(question_id):02d}"
        for question_id in PAPER_QUESTION_IDS
    ]
    assert all(example.metadata["context_len"] == PAPER_CONTEXT_LEN for example in examples)
    assert dataset.context_len == PAPER_CONTEXT_LEN


def test_oolong_pairs_paper_set_is_its_own_cli_dataset():
    args = build_parser().parse_args(
        [
            "--dataset",
            "oolong-pairs",
            "--runner",
            "fake",
            "--model",
            "fake",
            "--full",
        ]
    )

    config = config_from_args(args)

    assert [spec.name for spec in config.datasets] == ["oolong-pairs"]
    assert config.limit is None
