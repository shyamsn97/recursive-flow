# Delegation problem set

This is the spawn-lift spec: **does `launch_subagent` raise the score**. It is not wired as a dataset yet. The live routing set is `delegation` in [Benchmarks](benchmarks.md) (`make eval-delegation`). Do not fold these IDs into `delegation-iteration-ten`.

Pairs already test the RLM loop (`llm_query_batched` + Python join). This set tests the other cut.

The rule is Cursor’s, not “this looks hard.” Spawn when the branch would wreck the parent’s context, needs its own tool loop, or is an independent workstream. Stay in the parent — or use `llm_query` — when the work is one shot, one grep, or step B needs step A’s write in the same namespace.

rlmflow already has Cursor’s two modes. `launch_subagent` returns a handle and the child is running. Awaiting that handle in the same block is **foreground**. Launching, doing other work, then `wait_for_result` / `AGENTS` is **background**. The problems below use that API as it exists.

## How to read a result

Run every item twice on the same model:

| Arm      | Config                                 | What it is allowed to do               |
| -------- | -------------------------------------- | -------------------------------------- |
| No spawn | `max_depth=0`, `use_llm_query=true`    | Parent REPL + one-shot / batched query |
| Spawn    | `max_depth=1` (or 2), same query tools | Plus `launch_subagent`                 |

Delegation **helped** only when the spawn arm’s **grader score** moves and children were actually used on the spawn-tagged items. Child count without a score lift is the explainers suite, not this one.

## Expected routes

| Route           | rlmflow move                                               | Cursor analogue        |
| --------------- | ---------------------------------------------------------- | ---------------------- |
| `local`         | Parent REPL only                                           | Grep / Read / one edit |
| `batched_query` | `llm_query` / `llm_query_batched`                          | Skill / one-shot       |
| `fg_spawn`      | Launch, `await handle.wait_for_result()` in the same block | Foreground subagent    |
| `bg_spawn`      | Launch, continue, join later via handle or `AGENTS`        | Background subagent    |
| `verify`        | Fresh-context child that did not see the parent’s draft    | Reviewer / verifier    |

A problem is tagged with the **cheapest** route that should still score. Spawning on a `local` or `batched_query` item is a miss even if the answer is right.

## Controls — must not spawn

These exist so a model that launches children for everything fails the suite.

### L1 — Gateway port

Reuse `examples/behavior/delegation.py` gateway. One small INI in `INPUTS`. Answer is a single port number.

- **Route:** `local`
- **Why:** One visible line. Cursor would not Task this.
- **A/B:** Both arms score 1. Spawn arm children = 0.

### L2 — Short needle

Reuse `synthetic_needle` at ~20 records. Marker is regex-findable.

- **Route:** `local`
- **Why:** Parent `re.search` over `INPUTS` is the whole task.
- **A/B:** Tie at 1. A child is wasted context isolation.

### L3 — One tiny dossier

One 1–2K file: find a key on line N, apply it to a later line, emit a number. Three REPL steps, one namespace.

- **Route:** `local`
- **Why:** Multi-step is not enough. Cursor stays in the parent when the transcript still fits.
- **A/B:** Tie at 1. Spawn arm children = 0.

### Q1 — Pair join

Reuse `oolong-pairs` (or the two 32K canary IDs). Classify 787 lines, then Python-join.

- **Route:** `batched_query`
- **Why:** Embarrassingly parallel one-shots. This is a skill, not a subagent. Official RLM’s Q3 trace never spawned here either.
- **A/B:** Tie on F1. Spawn arm children = 0. Depth-0 already proved the RLM loop.

### Q2 — Field extract

One 30K log. Gold is a JSON object of ~40 `(request_id → status)` pairs. Each line is independently classifiable.

- **Route:** `batched_query`
- **Why:** No per-chunk REPL state. `llm_query_batched` over slices, then a dict join.
- **A/B:** Tie. Children should not appear.

### Q3 — Trap: three 10-line “modules”

Three named blobs that look like the boids split (`vec`, `rules`, `render`) but each is a 10-line pure function. Ask for the three return values, then a sum.

- **Route:** `local`
- **Why:** Number-stats trap. Separable-looking, free in one REPL.
- **A/B:** Tie at 1. Any child is a routing miss.

### T1 — Trap: 4K multi-hop page

A 2Wiki-sized page with two hops. Gold is the entity name only (no citation product).

- **Route:** `local`
- **Why:** Greppable. This is why the current “subagent” tags never fired.
- **A/B:** Tie. Spawn is wrong.

## Spawn — score should move

Parent `max_iters` must be enough to probe, launch, and join, **not** enough to grind every branch sequentially. That is what creates lift.

### S1 — Three fat dossiers (primary lift)

Three 8–20K dossiers in `INPUTS` (`alpha`, `beta`, `gamma`). Each needs 4–5 REPL turns: locate a rotating key, use it to filter a later section, emit one typed number. Gold is an exact join (sum, or a checksum).

- **Route:** `fg_spawn`
- **Cursor:** Three independent Explore/Bash workstreams in one turn.
- **Why spawn helps:** Depth-0 either blows the turn budget or pulls noisy intermediates into the parent window. `llm_query` cannot hold the key across turns.
- **A/B:** Depth-0 score near 0–0.3. Depth-1 score near 1 with 3 children, one launch batch.
- **Grader:** Exact numeric / tuple. `output_schema` on each child.

### S2 — Noisy search corpora

Four 15K corpora. The question names a rare marker that is **not** a unique regex (many decoy markers, one that survives a two-step predicate). Each corpus needs scan → candidate list → predicate in the REPL.

- **Route:** `fg_spawn`
- **Cursor:** Built-in Explore. Intermediate hits must not enter the parent transcript.
- **A/B:** Depth-0 misses or times out. Depth-1 returns four IDs, children = 4.
- **Grader:** Set equality on the four IDs.

### S3 — Disk catalog, files not in `INPUTS`

Parent `INPUTS` is a file list + sizes. Payloads live in the workdir (DABstep shape). Two large tables plus a manual. Answer is one factoid that requires reading the manual, then aggregating a filtered CSV.

- **Route:** `fg_spawn` (or one child on the CSV if the manual is short enough to keep local)
- **Cursor:** Parent sees the tree; a subagent opens the noisy file.
- **Why:** This is the spawn-shaped item we already have (`delegation_dabstep_11_2536`) once gold is filled and the agent can see `context/`.
- **A/B:** Depth-0 `Not Applicable` or wrong. Depth-1 exact. Repair DABstep rather than inventing a second file bench first.
- **Grader:** Type-aware exact, like DABstep.

### S4 — Three module contracts

Three files, each with a real interface and a hidden test the child must satisfy in its own REPL (write, run, fix). Parent joins the three exports. Smaller than boids; no renderer.

- **Route:** `fg_spawn`
- **Cursor:** Parallel implementation on independent files; parent owns the merge.
- **Reuse:** Slim of `examples/behavior/delegation.py` boids, without the 2k-boid runtime.
- **A/B:** Depth-0 ships a broken interface. Depth-1 children = 3, tests pass.
- **Grader:** Import each file and run the hidden tests. Do not grade child count.

### S5 — Sequential key (foreground, one at a time)

Corpus A hides a key. Corpus B is sealed until that key is known. Each corpus is fat and multi-step.

- **Route:** `fg_spawn` (child A, wait, child B with the key in `inputs`)
- **Cursor:** Sequential foreground. Not one parallel batch — B needs A’s result.
- **A/B:** Depth-0 fails B. Depth-1 children = 2, launched in two turns, not one batch.
- **Anti-pattern:** Launching A and B in the same turn (B has no key yet).

## Research — Cursor Explore

This is the spawn case Cursor built Explore for: long reading, noisy intermediates, parent only wants a cited fact. Do **not** use the live web or `examples/autoresearch` here. Autoresearch is a GPU trial loop; it is the expensive cousin, not a canary. Keep the corpus in `INPUTS` (or the workdir) so both arms see the same papers and the grader is exact.

A research item is just S1 with prose instead of keys — unless each paper needs a **multi-step read** (claim, then split, then caveat in the appendix). If every abstract states the number, it is Q1.

### R0 — Control: one abstract

The question from R1, but `INPUTS` is a single 400-word abstract that states the answer in one paragraph.

- **Route:** `local`
- **Why:** Cursor would not Explore a paragraph.
- **A/B:** Tie at 1. Children = 0.

### R1 — Closed-world literature join

Four synthetic papers, 6–12K each (`paper_a` … `paper_d`). Different titles, tables, and an appendix caveat. Question (fixed wording):

> What is the best reported F1 on TaskX among papers that used the 2023 split and did not leak the test set? Return `{"f1": <number>, "paper": "<title>"}`.

Planted facts:

| Paper |         TaskX F1 | Split | Leak           |
| ----- | ---------------: | ----- | -------------- |
| A     |             91.0 | 2022  | no             |
| B     |             88.0 | 2023  | yes (appendix) |
| C     |             84.0 | 2023  | no             |
| D     | n/a (other task) | 2023  | no             |

Gold: `{"f1": 84.0, "paper": "<C's exact title>"}`.

Each child must: identify the task, find the split, find the leak sentence (not in the abstract), then the number. That is 3–4 REPL turns of search, not one `llm_query`.

- **Route:** `fg_spawn` (four children, one paper each) or `bg_spawn` if the query also asks the parent to print a one-line inclusion protocol before joining
- **Cursor:** Explore per paper; parent synthesizes. Intermediate tables stay out of the root transcript.
- **A/B:** Depth-0 picks 91 or 88 (wrong split or missed appendix). Depth-1 children = 4 (skipping D is allowed if the parent filtered on title), score 1.
- **Grader:** Exact JSON. Do not score citations.

### R2 — Citation chase (sequential research)

Same four papers. Question names a result in paper A (“the 91 F1”) and asks whether a later paper reused that checkpoint on the 2023 split. The pointer is a citation key in A’s related-work; only B’s appendix answers it (no).

- **Route:** `fg_spawn` — child on A, wait, child on B with the citation key in `inputs`
- **Cursor:** Sequential Explore. Not a parallel batch.
- **A/B:** Depth-0 answers from A’s 91. Depth-1 children = 2, two turns. Gold: `{"reused": false, "via": "<citation key>"}`.
- **Anti-pattern:** Opening all four papers in one launch batch.

## Background — join later

These only score if the parent does work **between** launch and wait. Awaiting in the launch block is S1, not background.

### B1 — Launch, inventory, then join

Same three dossiers as S1. The query also asks for a local inventory (key names and sizes) that the parent can compute without children.

- **Route:** `bg_spawn`
- **API:** Launch three handles, print the inventory, then `await` the handles (or wait in a later turn).
- **Cursor:** Background subagents; parent stays useful.
- **A/B:** Depth-1 must include the inventory in the answer. A trace that only `wait_for_result`s in the launch block fails a “did other work” check, even if the join is right.
- **Grader:** Join exact **and** inventory exact.

### B2 — `AGENTS` poll

`Flow(use_agent_tree=True)`. Launch two long children. Next parent turn must read `AGENTS.print_graph` / `AGENTS.get_children()`, wait only on `completed` or via `wait_for_result`, and refuse to guess.

- **Route:** `bg_spawn`
- **Cursor:** Parent checks status instead of blocking the conversation.
- **A/B:** Depth-0 N/A. Depth-1 children = 2. Fail if the parent fabricates results before a child is terminal.
- **Grader:** Join exact. Trace check: an `AGENTS` or handle status read before the join.

## Verify — fresh context

### V1 — Adversarial draft

Parent `INPUTS` has a plausible but wrong itinerary. The raw constraint table is a second, larger input. Ask for the valid plan.

- **Route:** `verify`
- **Cursor:** Reviewer that did not see the parent’s rationalization. Child `inputs` is **only** the constraint table, not the draft.
- **A/B:** Depth-0 often echoes the draft. Depth-1 child returns violations; parent corrects.
- **Grader:** Exact plan parse (pin the output schema). Do not use Natural Plan’s free-prose parser.

### V2 — Dual read, take agreement

One fat dossier, two children, same goal, no shared transcript. Parent finishes only on agreement; otherwise a third local pass.

- **Route:** `verify`
- **Cursor:** Independent verification.
- **A/B:** Depth-0 one noisy read. Depth-1 children = 2. Score is the agreed number.
- **Grader:** Exact. Trace: two children, same `inputs` keys.

## Suggested run of 18

If you only stand up one batch, use this order:

|   # | ID  | Route           | Build from                                 |
| --: | --- | --------------- | ------------------------------------------ |
|   1 | L1  | `local`         | Existing gateway scenario                  |
|   2 | L2  | `local`         | `synthetic_needle` (small)                 |
|   3 | L3  | `local`         | New, 1–2K                                  |
|   4 | Q1  | `batched_query` | `oolong-pairs`                             |
|   5 | Q2  | `batched_query` | New log extract                            |
|   6 | Q3  | `local`         | New 10-line modules                        |
|   7 | T1  | `local`         | Frozen 2Wiki page, answer-only grader      |
|   8 | S1  | `fg_spawn`      | **New generator (first to implement)**     |
|   9 | S2  | `fg_spawn`      | New corpora + predicate                    |
|  10 | S3  | `fg_spawn`      | Repair DABstep 2536                        |
|  11 | S4  | `fg_spawn`      | Slim boids contracts                       |
|  12 | S5  | `fg_spawn`      | New sequential pair                        |
|  13 | R0  | `local`         | One planted abstract                       |
|  14 | R1  | `fg_spawn`      | **Four planted papers (second generator)** |
|  15 | R2  | `fg_spawn`      | Same papers, citation chase                |
|  16 | B1  | `bg_spawn`      | S1 payloads + inventory ask                |
|  17 | B2  | `bg_spawn`      | S1/S2 payloads, `use_agent_tree=True`      |
|  18 | V1  | `verify`        | New constraints + planted draft            |

S1 is the first A/B that must exist. R1 is the first *research* A/B; R0/R2 share its generator. S3 is a packaging fix. V2 can wait.

Open-web research (BrowseComp+, live arXiv) is a later cousin: flaky, expensive, and it confounds “did spawn help” with “did search work.” Plant the library.

## Construction rules (or the lift disappears)

1. **Budget the parent.** If `max_iters` lets the root finish S1 alone, depth-0 will tie and you will conclude delegation does nothing.
1. **State across turns.** Each spawn branch must need a value computed in turn *k* and used in turn *k+1*. Otherwise `llm_query` wins and you have another Q1.
1. **Exact graders.** No citation-product, no prose itinerary parse. `output_schema` on children; Python equality on the join.
1. **Do not grade topology as the answer.** Record children / launch batches / `AGENTS` reads as diagnostics. The score is the factoid.
1. **Keep query tools on both arms.** A depth-0 arm without `llm_query` is not a fair “delegation vs RLM” test.

## What this is not

- DecisionBench (vendor `call_model` routing). That is `llm_query(..., model=peer)`, not a child REPL.
- Another 2Wiki/MuSiQue/Natural Plan slot. Those are greppable or format-trapped.
- Growing `synthetic_needle` until it is long. Regex over a haystack never needs a child.
- Open-web or GPU autoresearch as a canary. Those are product demos. R1 is the research *lift* test.

## Wire-up later

Dataset name when this becomes code: `delegation-lift`. Two implicit runners from one example list: `rlmflow-local` with `max_depth=0` and `max_depth=1`. Compare score, not accuracy-over-citation. Until then this file is the spec; do not add these IDs to `TEN_TASKS`.
