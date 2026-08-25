"""Import-path (de)serialization for keeping checkpoints JSON-pure.

A checkpoint should reference a class or function by its dotted import path
(e.g. ``"gpsr.beams.NNParticleBeamGenerator"``) rather than by pickling the
object itself -- a stored path is plain JSON, so the checkpoint loads with
``torch.load(..., weights_only=True)``. These two helpers are the single source
of truth for that round-trip, shared by ``builders`` (the beam-generator class)
and ``training`` (the loss function).
"""

import importlib


def to_import_path(obj) -> str:
    """Return the dotted import path of a class or function.

    A string is passed through unchanged, so this is idempotent and accepts an
    already-serialized reference.
    """
    if isinstance(obj, str):
        return obj
    return f"{obj.__module__}.{obj.__qualname__}"


def import_from_path(path: str):
    """Resolve a dotted import path back to the class or function it names.

    Inverse of :func:`to_import_path`. The target module must be importable at
    load time.
    """
    module_path, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)
