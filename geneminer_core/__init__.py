"""GeneMiner core: relevance classification, NER, normalization, pipeline."""

from geneminer_core.devices import ProcessorKind, resolve_torch_device

__all__ = ["ProcessorKind", "resolve_torch_device", "__version__"]

__version__ = "1.0.0"
