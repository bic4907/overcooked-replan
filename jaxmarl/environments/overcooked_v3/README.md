# Overcooked V3

`overcooked_v3` starts from the `main` branch's `overcooked_v2` implementation
and adds cyclic layout changes. The existing `overcooked` and `overcooked_v2`
environments remain unchanged from `main`.

```python
from jaxmarl import make

env = make("overcooked_v3", layout="dynamic_00")
```

Layout definitions live in `dynamic_layout_data.py` and are validated by
`dynamic_layouts.py`. At phase boundaries, V3 updates the static map, clears
objects on changed cells, and relocates agents that would otherwise be covered
by a newly blocked tile.

The default V3 observation adds two channels to V2's 30-channel grid encoding.
The penultimate channel contains a spatially constant, continuous countdown
that decreases from `1.0` at the start of a phase toward `0.0` immediately
before the next layout change. The final binary channel marks cells whose
static object differs in the next phase. Set both transition options to `False`
when loading a legacy 30-channel policy.
