"""Helpers the engine builds on.

:class:`~rflow.flow.RecursiveFlow` is the engine — it owns state and the
loop, and every overridable seam lives there as a method. This
package is its toolbox: pure functions, pure data, and implementation
helpers called by the public ``RecursiveFlow`` methods.

- :mod:`~rflow.engine.actions` — :class:`Action` types
  (:class:`CallLLM` / :class:`Exec` / :class:`Resume`) and the pure
  projection ``Graph -> ActionPlan`` (:func:`act_one` / :func:`act`).
- :mod:`~rflow.engine.replay` — cold-start replay-of-one for
  rebuilding a suspended coroutine after a fork or process restart.
- :mod:`~rflow.engine.scheduler` — :class:`NodeScheduler`: pick the
  agents that can take a step right now (pure top-down walk over a
  :class:`~rflow.graph.Graph`).
- :mod:`~rflow.engine.scheduling` — implementation of the outer
  ``RecursiveFlow.step`` loop and async-child refill policy.
- :mod:`~rflow.engine.transitions` — implementation of action-to-state
  transition handlers behind ``RecursiveFlow.apply_one`` / ``step_exec`` /
  ``step_after_supervising``.
- :mod:`~rflow.engine.helpers` — tiny shared helpers (node appends,
  iteration counts, budget checks, output truncation/formatting, the
  pool factory).
- :mod:`~rflow.engine.config` — :class:`FlowConfig`. Pure data.

If something is a user-facing override seam, it stays as a method on
:class:`~rflow.flow.RecursiveFlow`. Some method implementations delegate here
to keep the façade readable without hiding the public API.
"""
