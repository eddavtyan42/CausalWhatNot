"""Synthetic scenario definitions for phase 3 experiments.

Each entry specifies a known causal edge to remove (``missing``) and
an extra edge to insert (``spurious``) for stress-testing algorithms.
"""

# Mapping of dataset name to perturbation scenarios.
SCENARIOS = {
    "asia": {
        # Smoking is a strong driver of lung cancer in the Asia network.
        "missing": {"edge": ("Smoking", "LungCancer")},
        # Travel history should not directly cause bronchitis; this edge is spurious.
        "spurious": {"edge": ("VisitAsia", "Bronchitis")},
    },
    "sachs": {
        # PKA activates Mek in the canonical signaling cascade.
        "missing": {"edge": ("PKA", "Mek")},
        # No direct biochemical link from PIP2 to PKA, making this edge spurious.
        "spurious": {"edge": ("PIP2", "PKA")},
    },
    "alarm": {
        # Venous oxygen saturation influences arterial oxygen levels.
        "missing": {"edge": ("PVSAT", "SAO2")},
        # A kinked tube does not cause intubation; adding it introduces a false dependency.
        "spurious": {"edge": ("KINKEDTUBE", "INTUBATION")},
    },
    "child": {
        # The underlying disease affects lung parenchyma in the Child model.
        "missing": {"edge": ("Disease", "LungParench")},
        # Age alone should not produce grunting, so this edge is spurious.
        "spurious": {"edge": ("Age", "Grunting")},
    },
}
