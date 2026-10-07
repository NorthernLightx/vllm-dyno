import math

import pytest

from vllm_dyno.harness import position_stats

REF_IDS = [7, 3, 9]
REF_LPS = [math.log(0.7), math.log(0.2), math.log(0.1)]


def test_identical_distributions_have_zero_kl_and_agree():
    kl, same = position_stats(REF_IDS, REF_LPS, dict(zip(REF_IDS, REF_LPS)))
    assert kl == pytest.approx(0)
    assert same


def test_shifted_mass_raises_kl_and_flips_the_top_token():
    cand = {7: math.log(0.3), 3: math.log(0.6), 9: math.log(0.1)}
    kl, same = position_stats(REF_IDS, REF_LPS, cand)
    assert kl == pytest.approx(0.7 * math.log(0.7 / 0.3) + 0.2 * math.log(0.2 / 0.6))
    assert not same


def test_token_missing_from_candidate_top_k_gets_its_lowest_logprob():
    cand = {7: math.log(0.7), 3: math.log(0.2), 5: math.log(0.05)}
    kl, _ = position_stats(REF_IDS, REF_LPS, cand)
    assert kl == pytest.approx(0.1 * math.log(0.1 / 0.05))
