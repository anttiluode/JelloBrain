#!/usr/bin/env python3
"""Fine sweep around the S14 retirement-rate transition.

The coarse sweep found qualitatively different attractors near rates 0.005 and
0.02. This probes the interval without changing the mechanism.
"""

from __future__ import annotations

import json

from retirement_sweep import summarize_rate


def main() -> dict:
    rates = (0.0075, 0.0100, 0.0125, 0.0150, 0.0175)
    return {
        "schema": "jellobrain/route-retirement-fine-sweep-v1",
        "rates": [summarize_rate(rate, n_seeds=4) for rate in rates],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
