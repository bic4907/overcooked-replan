"""Coordination Transformer (CooT) baseline for Overcooked V3.

The implementation is an independent JAX/Flax port of the method described in
Wang et al., "CooT: Learning to Coordinate In-Context with Coordination
Transformers" (arXiv:2506.23549) and its supplementary implementation.
"""

from .model import CooTConfig, CooTTransformer

__all__ = ["CooTConfig", "CooTTransformer"]
