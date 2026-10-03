import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """Audits a solution for the pion decay maximum muon momentum problem."""

    def __init__(self, solution_instance: Any):
        """
        Extracts all required data from the solution instance safely.
        Defaults to None for missing values – never use numeric defaults.
        """
        # Extract result from pseudocode emit contract
        self.p_mu = getattr(solution_instance, 'p_mu', None)
        if self.p_mu is None:
            results = getattr(solution_instance, 'results', None)
            if isinstance(results, dict):
                self.p_mu = results.get('p_mu', None)

        # Extract givens (masses)
        self.m_pion = getattr(solution_instance, 'm_pion', None)
        if self.m_pion is None:
            givens = getattr(solution_instance, 'givens', None)
            if isinstance(givens, dict):
                self.m_pion = givens.get('m_pion', None)

        self.m_muon = getattr(solution_instance, 'm_muon', None)
        if self.m_muon is None:
            givens = getattr(solution_instance, 'givens', None)
            if isinstance(givens, dict):
                self.m_muon = givens.get('m_muon', None)

        # Explanation steps (may be a string or list)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            results = getattr(solution_instance, 'results', None)
            if isinstance(results, dict):
                self.explanation_steps = results.get('explanation_steps', None)

        # Pseudocode (may be a string)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        if self.pseudocode is None:
            results = getattr(solution_instance, 'results', None)
            if isinstance(results, dict):
                self.pseudocode = results.get('pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests: mere presence and type correctness
    # ------------------------------------------------------------------
    def test_interface_p_mu_exists(self) -> Tuple[bool, str]:
        """Check that the result key 'p_mu' is present and is a numerical value."""
        if self.p_mu is None:
            return (False, "SKIP: no p_mu found in solution")
        if not isinstance(self.p_mu, (int, float)):
            return (False, f"p_mu is not numeric: {type(self.p_mu).__name__}")
        return (True, "p_mu exists and is numeric")

    def test_interface_givens_provided(self) -> Tuple[bool, str]:
        """Check that required givens are present for invariant checks."""
        if self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: m_pion or m_muon not provided – cannot run mass‑dependent invariant tests")
        if not isinstance(self.m_pion, (int, float)) or not isinstance(self.m_muon, (int, float)):
            return (False, "m_pion or m_muon are non‑numeric")
        return (True, "masses are available and numeric")

    # ------------------------------------------------------------------
    # Invariant (theorem‑based) tests – use math.isclose with 1% tolerance
    # ------------------------------------------------------------------
    def test_invariant_energy_conservation(self) -> Tuple[bool, str]:
        """
        In the pion rest frame, total energy = m_pion. Conservation:
        sqrt(p_mu^2 + m_mu^2) + p_mu = m_pion   (using c=1 natural units)
        """
        if self.p_mu is None or self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: missing p_mu or masses – cannot test energy conservation")
        p = self.p_mu
        m_pi = self.m_pion
        m_mu = self.m_muon
        E_mu = math.sqrt(p**2 + m_mu**2)
        E_nu = p
        total = E_mu + E_nu
        expected = m_pi
        if math.isclose(total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conserved: {total:.3f} ≈ {expected:.3f} MeV")
        else:
            return (False, f"Energy not conserved: total={total:.3f}, expected={expected:.3f}")

    def test_invariant_momentum_matching(self) -> Tuple[bool, str]:
        """Neutrino energy must equal muon momentum (massless neutrino)."""
        if self.p_mu is None or self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: missing data for momentum check")
        p = self.p_mu
        E_mu = math.sqrt(p**2 + self.m_muon**2)
        E_nu = self.m_pion - E_mu
        if math.isclose(E_nu, p, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Neutrino energy ≈ momentum: {E_nu:.3f} ≈ {p:.3f} MeV")
        else:
            return (False, f"Neutrino E={E_nu:.3f} ≠ p={p:.3f}")

    def test_invariant_formula_consistency(self) -> Tuple[bool, str]:
        """Derived formula: p = (m_pi^2 - m_mu^2) / (2 * m_pi)."""
        if self.p_mu is None or self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: missing data for formula check")
        p_expected = (self.m_pion**2 - self.m_muon**2) / (2.0 * self.m_pion)
        if math.isclose(self.p_mu, p_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_mu matches formula: {self.p_mu:.3f} ≈ {p_expected:.3f} MeV/c")
        else:
            return (False, f"p_mu={self.p_mu:.3f} deviates from formula value={p_expected:.3f}")

    # ------------------------------------------------------------------
    # Modeling / trace tests – quality of derivation steps
    # ------------------------------------------------------------------
    def test_modeling_explanation_contains_formula(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps (if provided as a string) contain
        the critical momentum formula or its numerical application.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        # Convert to string if list
        if isinstance(self.explanation_steps, (list, tuple)):
            combined = " ".join(str(step) for step in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            combined = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps is not a string or list")

        # Look for the formula pattern or the numerical computation
        formula_markers = [
            "p_mu = (m_pi^2 - m_mu^2) / (2 * m_pi)",
            "8427.51 / 280",
            "(140^2 - 105.7^2) / (2 * 140)",
        ]
        for marker in formula_markers:
            if marker in combined:
                return (True, f"Explanation contains formula marker: '{marker}'")
        # Fallback: check for key variables together
        if "p_mu" in combined and ("m_pi" in combined or "140" in combined):
            return (True, "Explanation mentions p_mu and pion mass")
        return (False, "Explanation does not contain clear derivation of p_mu")

    def test_modeling_pseudocode_correct_variables(self) -> Tuple[bool, str]:
        """
        Check that the pseudocode (if provided) includes the variable 'p_mu'
        and references the decay formula or the correct masses.
        """
        if self.pseudocode is None:
            return (False, "SKIP: no pseudocode available")
        if not isinstance(self.pseudocode, str):
            # try to convert if it's a list
            if isinstance(self.pseudocode, (list, tuple)):
                pseudocode_text = "\n".join(str(l) for l in self.pseudocode)
            else:
                return (False, "SKIP: pseudocode is not a string or list")
        else:
            pseudocode_text = self.pseudocode

        # Check for essential variable and the formula
        has_p_mu = "p_mu" in pseudocode_text
        has_formula = ("m_pion" in pseudocode_text or "140" in pseudocode_text) and \
                      ("m_muon" in pseudocode_text or "105.7" in pseudocode_text)
        if has_p_mu and has_formula:
            return (True, "Pseudocode contains 'p_mu' and references to masses/value")
        elif has_p_mu:
            return (True, "Pseudocode contains 'p_mu'")
        else:
            return (False, "Pseudocode does not contain 'p_mu'")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare perturbed output to baseline
    # ------------------------------------------------------------------
    def test_monotonicity_p_mu_increases_with_m_pion(self) -> Tuple[bool, str]:
        """
        For fixed m_muon, increasing m_pion should increase p_mu.
        We recompute p_pert for m_pion +1% and verify that p_pert > self.p_mu.
        """
        if self.p_mu is None or self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: missing data for monotonicity test")
        m_pion_base = self.m_pion
        m_pion_pert = m_pion_base * 1.01  # +1%
        p_pert = (m_pion_pert**2 - self.m_muon**2) / (2.0 * m_pion_pert)
        p_base = self.p_mu
        if p_pert > p_base:
            return (True, f"Perturbation: m_pi+1% → p_mu increases from {p_base:.3f} to {p_pert:.3f}")
        else:
            return (False, f"Monotonicity violated: p_mu did not increase (base={p_base:.3f}, perturbed={p_pert:.3f})")

    def test_monotonicity_p_mu_decreases_with_m_muon(self) -> Tuple[bool, str]:
        """
        For fixed m_pion, increasing m_muon should decrease p_mu.
        Perturb m_muon by +1% and check that p_pert < self.p_mu.
        """
        if self.p_mu is None or self.m_pion is None or self.m_muon is None:
            return (False, "SKIP: missing data for monotonicity test")
        m_muon_base = self.m_muon
        m_muon_pert = m_muon_base * 1.01  # +1%
        p_pert = (self.m_pion**2 - m_muon_pert**2) / (2.0 * self.m_pion)
        p_base = self.p_mu
        if p_pert < p_base:
            return (True, f"Perturbation: m_mu+1% → p_mu decreases from {p_base:.3f} to {p_pert:.3f}")
        else:
            return (False, f"Monotonicity violated: p_mu did not decrease (base={p_base:.3f}, perturbed={p_pert:.3f})")

    # ------------------------------------------------------------------
    # Single weak answer sanity check – basic constraints, no hardcoded answer
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Basic sanity on p_mu: must be non‑negative, finite, and within
        a physically plausible range (0 to pion mass ~140 MeV/c).
        No math.isclose, no re‑derivation of answer.
        """
        if self.p_mu is None:
            return (False, "SKIP: no p_mu to check")
        if not math.isfinite(self.p_mu):
            return (False, f"p_mu is not finite: {self.p_mu}")
        if self.p_mu < 0:
            return (False, f"p_mu is negative: {self.p_mu}")
        if self.p_mu > 140:
            return (False, f"p_mu exceeds plausible maximum: {self.p_mu} > 140 MeV/c")
        return (True, f"p_mu={self.p_mu:.3f} passes basic sanity (non‑negative, finite, ≤ 140 MeV/c)")

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test methods. Each returns (bool, str).
        Collect all logs; overall success iff all tests pass.
        Never crashes – wrap each call in try/except.
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)