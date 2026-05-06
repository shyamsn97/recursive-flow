# Recursive Language Models: A graph approach

> [GitHub](https://github.com/shyamsn97/rlmflow) ·
> [PyPI](https://pypi.org/project/rlmflow/) ·
> [Examples](https://github.com/shyamsn97/rlmflow/tree/main/examples) ·
> [Changelog](https://github.com/shyamsn97/rlmflow/blob/main/CHANGELOG.md)

![Hero animation: an rlmflow run unfolding from a single root agent into a tree of typed nodes](rlm_animation.gif)

```bash
pip install rlmflow
```

## tldr

**rlmflow** turns [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) into inspectable execution graphs. It's a Python library for writing RLM agents where every query, action, observation, delegation, wait, resume, and result is a typed, immutable Pydantic node, and a run is just the tree of those snapshots.

The whole engine is one transition: `step(node) → node'`. The trace and the execution are the same data structure — there is no separate "tracing mode" to enable — so the same run renders as a Rich live tree, a Mermaid diagram, a Gantt swimlane, or a Gradio step-through viewer, all from one-line projections of the graph.

That graph allows you to **inspect** each subagent, **replay** from a checkpoint, **fork** from any node, and **edit** a branch before continuing. We'll walk through those moves on a real coding-agent run shipped with the repo.

## Introduction

**Context rot** is the failure mode every practitioner has hit: a
Claude Code session that "gets dumber", a Cursor chat that forgets
the file you opened thirty messages ago, a research agent that can
quote your prompt back but can't *use* it. Anthropic
[defines it](https://www.anthropic.com/news/context-rot) as recall
degrading as the context window grows. Frontier models advertise
200k–1M tokens and in practice degrade long before that — the
tokens fit, the model just can't reason over them all at once. The
easy benchmarks miss this: needle-in-a-haystack tests like RULER are
constant-complexity and frontier models score 90%+, but
[Chroma](https://research.trychroma.com/context-rot),
[OOLONG](https://github.com/oolong-bench/oolong), and
[lost-in-the-middle](https://arxiv.org/abs/2307.03172) all show
real degradation well below the nominal limit.

Existing fixes all bake some decomposition decision into the harness
before the model sees the data. Bigger windows and better positional
encodings (ALiBi, YaRN, ring attention) buy headroom without
addressing rot. Retrieval (vector DBs, BM25, top-k) picks the chunks
for you and falls over on multi-hop. Summarization — Claude Code's
auto-summarize, LangChain's `ConversationSummaryMemory`,
[MemGPT](https://github.com/cpacker/MemGPT) — picks the lossy
compression. The recent **context-folding** thread — [Scaling
Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967),
[AgentFold](https://arxiv.org/abs/2510.24803), [Agentic Context
Engineering](https://arxiv.org/abs/2510.04618) — picks the
branch/return policy. 

Each of these approaches work in practice, but each one is a decomposition
strategy chosen by the system designer in advance, rather than by
the model at run time. This is the pattern [Sutton's *Bitter
Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
describes: methods that hard-code human structure win in the short
run and lose in the long run to general methods that scale with
compute. As model capability improves, a fixed decomposition
strategy becomes the ceiling.

[Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)
flip that. The setup is small: an LLM sits in a Python REPL with
the long context bound **as a variable**, and a single extra
primitive — **`delegate`** — lets it spawn a fresh sub-agent with
its own context window. From there, the model decides for itself
how to peek at the context, slice it, regex through it, or hand a
chunk to a recursive sub-call. Nothing is summarized or delegated
unless the model chooses to. RAG retrieves; RLMs *investigate*. And
because the surface is so small, the strategy itself becomes
another scalar reward — the same RL machinery that taught models to
reason can teach them to manage their own context.

The empirical case is strong: Alex shows RLM(GPT-5-mini) beats raw
GPT-5 drastically on a tough long-context benchmark at roughly the
same API cost, and holds up at 10M+ token corpora that don't fit
any direct baseline. See the
[post](https://alexzhang13.github.io/blog/2025/rlm/),
[paper](https://arxiv.org/abs/2512.24601), and the
[`rlm-minimal`](https://github.com/alexzhang13/rlm-minimal) and
[`verifiers`](https://www.primeintellect.ai/blog/rlm) reference
implementations for the case.

However, as the number of sub-agents grows, that tree becomes much
harder to observe and control: parents spawn children, children
spawn more children, results bubble back up, and a flat transcript
hides almost everything you'd want to ask of the run. But what if there was a better way?

That's where <b>rlmflow</b> comes in -- Representing sprawling trees of recursive agents as inspectable and controllable graphs.

## Acknowledgements

Alex Zhang and Omar Khattab for the RLM paper and post — without
which there is no rlmflow. The
[`rlm-minimal`](https://github.com/alexzhang13/rlm-minimal) and
[`ypi`](https://github.com/rawwerks/ypi) codebases for being
readable, hackable, and right; most of the prompt structure was
learned from them. The OOLONG authors and Prime Intellect's
`verifiers` team for the benchmark environments we wrap. Anthropic's
engineering blog for the harness/session/sandbox vocabulary. And
early users who filed the boids-simulation regressions, the
schema-drift confusions, and the "where is `CONTEXT.fork()`
documented?" issues — those reports are what produced the failure-
shapes section.

---

## Citation

```bibtex
@misc{sudhakaran2026rlmflow,
  author       = {Sudhakaran, Shyam},
  title        = {Recursive Language Models are Graphs},
  year         = {2026},
  howpublished = {\url{https://github.com/shyamsn97/rlmflow}}
}
```

— shyam
