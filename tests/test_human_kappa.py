"""Cohen's kappa for the two-grader referral check (scripts/human_check.py).

Regressions: kappa once hard-coded the labels 0/1/2 in its chance term (A/P/B grades then got pe = 0, i.e. kappa =
raw agreement), and returned 1.0 when both graders used a single shared label, where kappa is 0/0 (not estimable).
A plausible real case: neither grader marks any edge A, so the binary A-vs-not kappa is undefined."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import human_check  # noqa: E402


@pytest.mark.parametrize("label", ["A", "P", "B", True, False, 2])
def test_single_shared_label_is_not_estimable(label):
    k = human_check.kappa([(label, label)] * 20)
    assert k["kappa"] is None and k["estimable"] is False and k["agreement"] == 1.0


def test_binary_a_vs_not_without_any_a_is_not_estimable():
    g1, g2 = list("PPBBPBPB"), list("BPPBPBBP")
    k = human_check.kappa([(x == "A", y == "A") for x, y in zip(g1, g2)])
    assert k["estimable"] is False and k["kappa"] is None


def test_letter_labels_use_their_own_chance_agreement():
    # po = 4/6; marginals A3 P1 B2 vs A3 P2 B1 -> pe = (9 + 2 + 2) / 36
    k = human_check.kappa([("A", "A"), ("A", "P"), ("B", "B"), ("P", "P"), ("A", "A"), ("B", "A")])
    assert k["estimable"] and k["kappa"] == pytest.approx((4 / 6 - 13 / 36) / (1 - 13 / 36))


def test_one_grader_constant_other_varied_is_estimable_zero():
    k = human_check.kappa([("A", "A"), ("A", "B"), ("A", "P"), ("A", "A")])
    assert k["estimable"] and k["kappa"] == pytest.approx(0.0)

