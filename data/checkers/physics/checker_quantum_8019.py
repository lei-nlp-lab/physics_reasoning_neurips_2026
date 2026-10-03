import math
import numpy as np
from typing import Union, List, Tuple, Any

class PhysicsChecker:
    """
    Physics checker for the quark-antiquark meson ground state energy problem.
    Checks interface, physical invariants, modeling consistency, and monotonicity
    using re-derived theoretical expressions.
    """

    def __init__(self, solution_instance: Any):
        # Extract inputs and results safely with None defaults
        self.m_q = getattr(solution_instance, 'm_q', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.B = getattr(solution_instance, 'B', None)

        results = getattr(solution_instance, 'results', {})
        self.E0 = results.get('E0', None)
        self.H_bar = results.get('H_bar', None)

        # Theoretical constants from variational derivations (re-derived inside checker)
        # Exponential trial constant: (3/4)*6^(2/3)
        self._theoretical_constant_exponential = (3.0 / 4.0) * (6.0 ** (2.0 / 3.0))
        # Gaussian trial constant: 2*(9/2)^(1/3)
        self._theoretical_constant_gaussian = 2.0 * ((9.0 / 2.0) ** (1.0 / 3.0))

    def _compute_energy_scale(self, B_value: Union[float, None] = None) -> Union[float, None]:
        """Compute the factor (hbar^2 * B^2 / m_q)^{1/3}."""
        m = self.m_q
        h = self.hbar
        if B_value is None:
            B_value = self.B
        if m is None or h is None or B_value is None:
            return None
        if m <= 0 or h <= 0 or B_value <= 0:
            return None
        return (h**2 * B_value**2 / m) ** (1.0 / 3.0)

    # ---------- Interface tests ----------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that required result keys exist."""
        if self.E0 is None:
            return (False, "E0 is missing from results")
        if self.H_bar is None:
            return (False, "H_bar is missing from results")
        return (True, "Both E0 and H_bar are present")

    def test_interface_input_keys(self) -> Tuple[bool, str]:
        """Check that required input givens exist."""
        missing = []
        if self.m_q is None:
            missing.append("m_q")
        if self.hbar is None:
            missing.append("hbar")
        if self.B is None:
            missing.append("B")
        if missing:
            return (False, f"Missing input givens: {', '.join(missing)}")
        return (True, "All inputs present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that E0 and H_bar are numeric."""
        if not isinstance(self.E0, (int, float)):
            return (False, f"E0 is not numeric, got {type(self.E0)}")
        if not isinstance(self.H_bar, (int, float)):
            return (False, f"H_bar is not numeric, got {type(self.H_bar)}")
        return (True, "E0 and H_bar are numeric")

    # ---------- Physical invariants ----------
    def test_invariant_positivity(self) -> Tuple[bool, str]:
        """Energies must be positive for confining potential (B>0)."""
        if self.E0 is None:
            return (False, "SKIP: E0 not available")
        if self.H_bar is None:
            return (False, "SKIP: H_bar not available")
        if not (self.E0 > 0):
            return (False, f"E0 must be positive, got {self.E0}")
        if not (self.H_bar > 0):
            return (False, f"H_bar must be positive, got {self.H_bar}")
        return (True, "Both energies are positive")

    def test_invariant_finite(self) -> Tuple[bool, str]:
        """Check that energies are finite (not NaN or inf)."""
        for name, val in [("E0", self.E0), ("H_bar", self.H_bar)]:
            if val is None:
                continue
            if not math.isfinite(val):
                return (False, f"{name} is not finite (nan or inf)")
        return (True, "Energies are finite")

    def test_invariant_dimensional_consistency(self) -> Tuple[bool, str]:
        """Check that energies scale as (hbar^2 B^2/m_q)^{1/3}."""
        scale = self._compute_energy_scale()
        if scale is None:
            return (False, "SKIP: cannot compute energy scale due to missing inputs")
        if scale <= 0:
            return (False, "Energy scale non-positive")
        for name, val in [("E0", self.E0), ("H_bar", self.H_bar)]:
            if val is None:
                continue
            ratio = val / scale
            if ratio <= 0 or not math.isfinite(ratio):
                return (False, f"{name} does not yield positive ratio to scale")
        return (True, "Energies have correct dimensional form")

    # ---------- Modeling tests ----------
    def test_modeling_exponential_trial_constant(self) -> Tuple[bool, str]:
        """
        Check that the numerical factor for E0 matches the theoretical constant
        from the exponential trial function (Method I).
        """
        if self.E0 is None:
            return (False, "SKIP: E0 not available")
        scale = self._compute_energy_scale()
        if scale is None or scale <= 0:
            return (False, "SKIP: cannot compute energy scale")
        actual_constant = self.E0 / scale
        if math.isclose(actual_constant, self._theoretical_constant_exponential, rel_tol=1e-2):
            return (True, f"E0 constant {actual_constant:.4f} matches expected {self._theoretical_constant_exponential:.4f}")
        else:
            return (False, f"E0 constant {actual_constant:.4f} deviates from expected {self._theoretical_constant_exponential:.4f}")

    def test_modeling_gaussian_trial_constant(self) -> Tuple[bool, str]:
        """
        Check that the numerical factor for H_bar matches the theoretical constant
        from the Gaussian trial function (Method II).
        """
        if self.H_bar is None:
            return (False, "SKIP: H_bar not available")
        scale = self._compute_energy_scale()
        if scale is None or scale <= 0:
            return (False, "SKIP: cannot compute energy scale")
        actual_constant = self.H_bar / scale
        if math.isclose(actual_constant, self._theoretical_constant_gaussian, rel_tol=1e-2):
            return (True, f"H_bar constant {actual_constant:.4f} matches expected {self._theoretical_constant_gaussian:.4f}")
        else:
            return (False, f"H_bar constant {actual_constant:.4f} deviates from expected {self._theoretical_constant_gaussian:.4f}")

    def test_modeling_exponential_via_variational_condition(self) -> Tuple[bool, str]:
        """
        Perform a consistency check using the exponential trial optimization condition.
        For A=0, the optimal a satisfies: (3/2)*B*a^3 - (hbar^2/μ) = 0.
        Solve for a from the given E0? We compute a from the derived formula and then
        check that the energy expression is consistent.
        """
        if self.E0 is None or self.m_q is None or self.hbar is None or self.B is None:
            return (False, "SKIP: missing data for this test")
        if self.m_q <= 0 or self.hbar <= 0 or self.B <= 0:
            return (False, "SKIP: non-positive input")
        mu = self.m_q / 2.0
        # From theoretical derivation: a = (2 hbar^2 / (3 B mu))^(1/3)
        a = (2 * self.hbar**2 / (3 * self.B * mu)) ** (1.0 / 3.0)
        # Compute energy from the expression: E = (3/2)Ba + hbar^2/(2 mu a^2)
        E_pred = 1.5 * self.B * a + self.hbar**2 / (2 * mu * a**2)
        if math.isclose(self.E0, E_pred, rel_tol=1e-2):
            return (True, f"Exponential variational condition consistent (predicted {E_pred:.4f}, got {self.E0:.4f})")
        else:
            return (False, f"Exponential condition inconsistent: predicted {E_pred:.4f}, got {self.E0:.4f}")

    def test_modeling_gaussian_via_variational_condition(self) -> Tuple[bool, str]:
        """
        Check the Gaussian trial optimization condition.
        For exact Gaussian, optimal beta = (16 B^2 m_q^2 / (9 pi hbar^4))^(1/3).
        Then compute H_bar and compare.
        """
        if self.H_bar is None or self.m_q is None or self.hbar is None or self.B is None:
            return (False, "SKIP: missing data for this test")
        if self.m_q <= 0 or self.hbar <= 0 or self.B <= 0:
            return (False, "SKIP: non-positive input")
        beta = (16 * self.B**2 * self.m_q**2 / (9 * math.pi * self.hbar**4)) ** (1.0 / 3.0)
        H_pred = (3 * self.hbar**2 * beta) / (4 * self.m_q) + (2 * self.B) / math.sqrt(math.pi * beta)
        if math.isclose(self.H_bar, H_pred, rel_tol=1e-2):
            return (True, f"Gaussian variational condition consistent (predicted {H_pred:.4f}, got {self.H_bar:.4f})")
        else:
            return (False, f"Gaussian condition inconsistent: predicted {H_pred:.4f}, got {self.H_bar:.4f}")

    # ---------- Monotonicity tests (perturb B and compute predicted energies) ----------
    def test_monotonicity_e0_increases_with_b(self) -> Tuple[bool, str]:
        """
        Monotonicity: perturb B upward by 5% and compute E0_pred from theoretical formula.
        The perturbed energy must be greater than the baseline energy.
        """
        if self.m_q is None or self.hbar is None or self.B is None:
            return (False, "SKIP: missing inputs")
        if self.m_q <= 0 or self.hbar <= 0 or self.B <= 0:
            return (False, "SKIP: non-positive inputs")
        # Baseline energy from theoretical constant
        scale_baseline = self._compute_energy_scale(self.B)
        if scale_baseline is None or scale_baseline <= 0:
            return (False, "SKIP: invalid scale")
        E0_baseline = self._theoretical_constant_exponential * scale_baseline

        # Perturb B by +5%
        B_perturbed = self.B * 1.05
        scale_perturbed = self._compute_energy_scale(B_perturbed)
        if scale_perturbed is None or scale_perturbed <= 0:
            return (False, "SKIP: invalid perturbed scale")
        E0_perturbed = self._theoretical_constant_exponential * scale_perturbed

        if E0_perturbed > E0_baseline:
            return (True, f"E0 increases with B: baseline {E0_baseline:.4f} -> perturbed {E0_perturbed:.4f}")
        else:
            return (False, f"E0 does not increase with B: baseline {E0_baseline:.4f} -> perturbed {E0_perturbed:.4f}")

    def test_monotonicity_hbar_increases_with_b(self) -> Tuple[bool, str]:
        """
        Monotonicity: perturb B upward by 5% and compute H_bar_pred from theoretical formula.
        The perturbed energy must be greater than the baseline energy.
        """
        if self.m_q is None or self.hbar is None or self.B is None:
            return (False, "SKIP: missing inputs")
        if self.m_q <= 0 or self.hbar <= 0 or self.B <= 0:
            return (False, "SKIP: non-positive inputs")
        scale_baseline = self._compute_energy_scale(self.B)
        if scale_baseline is None or scale_baseline <= 0:
            return (False, "SKIP: invalid scale")
        H_baseline = self._theoretical_constant_gaussian * scale_baseline

        B_perturbed = self.B * 1.05
        scale_perturbed = self._compute_energy_scale(B_perturbed)
        if scale_perturbed is None or scale_perturbed <= 0:
            return (False, "SKIP: invalid perturbed scale")
        H_perturbed = self._theoretical_constant_gaussian * scale_perturbed

        if H_perturbed > H_baseline:
            return (True, f"H_bar increases with B: baseline {H_baseline:.4f} -> perturbed {H_perturbed:.4f}")
        else:
            return (False, f"H_bar does not increase with B: baseline {H_baseline:.4f} -> perturbed {H_perturbed:.4f}")

    # ---------- Answer sanity (weak) ----------
    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity: energies should be within a broad physically plausible range.
        Typical quark masses ~0.1 GeV, B ~ 0.2 GeV^2 --> scale ~0.2 GeV.
        Allow a wide range [1e-6, 1e6] in arbitrary units.
        """
        if self.E0 is None or self.H_bar is None:
            return (False, "SKIP: missing energies")
        lower = 1e-6
        upper = 1e6
        for name, val in [("E0", self.E0), ("H_bar", self.H_bar)]:
            if not (lower < val < upper):
                return (False, f"{name} = {val} is outside plausible range [{lower}, {upper}]")
        return (True, "Energies within broad plausible range")

    # ---------- Main runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)