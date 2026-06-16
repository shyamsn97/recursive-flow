"""Provider-backed sandbox REPL backends (Modal, E2B, …).

Imports are kept lazy at the :mod:`rflow.runtime` level so the optional provider
SDKs are only required when a backend is actually instantiated.
"""
