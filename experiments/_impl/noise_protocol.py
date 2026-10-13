"""Noise protocol numerical routines."""

from __future__ import annotations


from experiments._impl import stochastic as st


SEEDS = tuple(range(10, 30))


MAX_ITER = 900


STEP_MULT = 1.5


LAMBDA_REG = 0.02


def settings():
    return (
        st.build_setting_manifest(["sparse_noisy_moderate"], "mini_batch", [4, 5, 6, 7], [])
        + st.build_setting_manifest(["sparse_noisy_undersampled"], "mini_batch", [3, 4, 5], [])
        + st.build_setting_manifest(["sparse_noisy_moderate"], "additive", [], [0.03, 0.05, 0.08, 0.10])
    )
