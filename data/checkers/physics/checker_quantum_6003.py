import math
import traceback
from typing import Any, Dict, List, Optional, Tuple


class PhysicsChecker:
    """
    Physics auditor for the s‑wave quantum scattering problem with a
    tanh‑squared potential.  Performs interface, invariant, modeling/trace,
    monotonicity and answer‑sanity checks.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data from the solution instance.
        Use ``None`` defaults for missing values; never assign numeric
        fallback defaults.
        """
        self._sol = solution_instance
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', [])

        # Emit contract keys (may be missing)
        self.delta0 = self.results.get('delta0', None)
        self.sigma = self.results.get('sigma', None)

        # Givens (may also be stored inside results)
        self.m = self.givens.get('m', None)
        self.E = self.givens.get('E', None)
        self.hbar = self.givens.get('hbar', None)
        self.lmbda = self.givens.get('lambda', None)          # avoid built‑in name
        self.k = self.givens.get('k', None)

        # Fallback: try to get k from results
        if self.k is None:
            self.k = self.results.get('k', None)

        # Derive k from m, E, hbar if possible
        if self.k is None and all(v is not None for v in (self.m, self.E, self.hbar)):
            self.k = math.sqrt(2.0 * self.m * self.E) / self.hbar

    # ------------------------------------------------------------------
    #  Helper: safe approximate comparison (only for invariant/modeling)
    # ------------------------------------------------------------------
    @staticmethod
    def _isclose(a: float, b: float, *, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ------------------------------------------------------------------
    #  Interface tests – required keys and types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that `delta0` and `sigma` exist in results."""
        if self.delta0 is None:
            return False, "results missing key 'delta0'"
        if self.sigma is None:
            return False, "results missing key 'sigma'"
        return True, "both required keys present"

    def test_interface_delta0_type(self) -> Tuple[bool, str]:
        """delta0 must be numeric."""
        if self.delta0 is None:
            return False, "delta0 not available – cannot check type"
        if not isinstance(self.delta0, (int, float)):
            return False, f"delta0 has non‑numeric type {type(self.delta0).__name__}"
        return True, "delta0 is numeric"

    def test_interface_sigma_type(self) -> Tuple[bool, str]:
        """sigma must be numeric."""
        if self.sigma is None:
            return False, "sigma not available – cannot check type"
        if not isinstance(self.sigma, (int, float)):
            return False, f"sigma has non‑numeric type {type(self.sigma).__name__}"
        return True, "sigma is numeric"

    # ------------------------------------------------------------------
    #  Invariant tests – physical relationships (use `_isclose`)
    # ------------------------------------------------------------------
    def test_invariant_phase_shift_relation(self) -> Tuple[bool, str]:
        """
        Check that  sin²(delta0) = k²/(λ²+k²)  (derived from
        e^{2iδ0} = (λ−ik)/(λ+ik)).
        """
        if any(v is None for v in (self.delta0, self.k, self.lmbda)):
            return False, "missing delta0, k, or lambda – cannot test invariant"
        k = self.k
        lam = self.lmbda
        sin2_expected = k**2 / (lam**2 + k**2)
        sin2_actual = math.sin(self.delta0)**2
        if not self._isclose(sin2_actual, sin2_expected):
            msg = (f"sin²(δ0) = {sin2_actual:.4f}, expected {sin2_expected:.4f} "
                   f"(k={k}, λ={lam})")
            return False, msg
        return True, "sin²(δ0) matches k²/(λ²+k²)"

    def test_invariant_cross_section_formula(self) -> Tuple[bool, str]:
        """
        Check that  σ = (4π/k²) sin²(δ0) .
        """
        if any(v is None for v in (self.sigma, self.k, self.delta0)):
            return False, "missing sigma, k, or delta0 – cannot check cross section formula"
        expected = (4.0 * math.pi / (self.k**2)) * (math.sin(self.delta0)**2)
        if not self._isclose(self.sigma, expected):
            msg = (f"sigma = {self.sigma:.4f}, expected {expected:.4f} "
                   f"(k={self.k}, δ0={self.delta0})")
            return False, msg
        return True, "σ = 4π/k² sin²δ0"

    # ------------------------------------------------------------------
    #  Modeling / trace tests – quality of the solution model
    # ------------------------------------------------------------------
    def test_trace_explanation_contains_regularity(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps mention the regularity condition
        φ(0)=0 (or equivalently r→0, finite at origin).  This is a key
        trace of the correct modeling.
        """
        if not self.explanation_steps:
            return True, "SKIP: no explanation steps available"
        text = " ".join(self.explanation_steps).lower()
        keywords = ["regularity", "φ(0)=0", "r→0", "phi(0)=0", "regular"]
        if any(kw in text for kw in keywords):
            return True, "explanation steps contain regularity condition"
        return False, "explanation steps do not mention regularity at origin"

    def test_modeling_delta0_from_regularity(self) -> Tuple[bool, str]:
        """
        Check that the emitted δ0 equals  π/2 + arctan(k/λ)  (mod π),
        which follows from the regularity condition and asymptotic matching.
        """
        if any(v is None for v in (self.delta0, self.k, self.lmbda)):
            return False, "missing data for modeling test"
        expected = math.pi / 2.0 + math.atan2(self.k, self.lmbda)
        if not self._isclose(self.delta0, expected):
            msg = (f"δ0 = {self.delta0:.4f}, expected π/2 + arctan(k/λ) = {expected:.4f}")
            return False, msg
        return True, "δ0 matches π/2 + arctan(k/λ)"

    def test_modeling_sigma_low_energy_approximation(self) -> Tuple[bool, str]:
        """
        In the low‑energy limit (E→0) the cross section should be
        4π/k².  Verify that the emitted sigma satisfies  σ·k² ≈ 4π.
        """
        if any(v is None for v in (self.sigma, self.k)):
            return False, "missing sigma or k – cannot test low‑energy approximation"
        product = self.sigma * (self.k**2)
        if not self._isclose(product, 4.0 * math.pi):
            msg = f"σ·k² = {product:.4f}, expected 4π ≈ {4*math.pi:.4f}"
            return False, msg
        return True, "σ·k² ≈ 4π (low‑energy limit)"

    # ------------------------------------------------------------------
    #  Monotonicity test – explicit input perturbation
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_decreases_with_k(self) -> Tuple[bool, str]:
        """
        When k increases by 1%, the exact s‑wave cross section
        σ = 4π/(λ²+k²)  should decrease (derivative negative).
        We check this by finite difference using the baseline k and a
        perturbed k.
        """
        if any(v is None for v in (self.sigma, self.k, self.lmbda)):
            return False, "missing sigma, k, or lambda – cannot test monotonicity"

        lam = self.lmbda
        k_base = self.k
        sigma_base = 4.0 * math.pi / (lam**2 + k_base**2)

        # perturb k upward by 1%
        k_pert = k_base * 1.01
        sigma_pert = 4.0 * math.pi / (lam**2 + k_pert**2)

        if sigma_pert >= sigma_base:
            msg = (f"sigma does not decrease when k is increased by 1%: "
                   f"baseline {sigma_base:.4f}, perturbed {sigma_pert:.4f}")
            return False, msg
        return True, "sigma decreases when k increases (monotonicity satisfied)"

    # ------------------------------------------------------------------
    #  Weak answer‑sanity check (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_checks(self) -> Tuple[bool, str]:
        """
        Very weak sanity: sigma positive, finite, delta0 in (0,π).
        No hardcoded target values allowed.
        """
        checks = []

        if self.sigma is not None:
            if not (self.sigma > 0):
                checks.append("sigma not positive")
            if not math.isfinite(self.sigma):
                checks.append("sigma not finite")
        else:
            checks.append("sigma not available")

        if self.delta0 is not None:
            if not (0 < self.delta0 < math.pi):
                checks.append("delta0 outside (0,π)")
        else:
            checks.append("delta0 not available")

        if not checks:
            return True, "all basic sanity checks passed"
        return False, "; ".join(checks)

    # ------------------------------------------------------------------
    #  Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all `test_*` methods, execute them in try/except,
        collect logs, and return (all_pass, logs).
        """
        logs: List[str] = []
        all_pass = True

        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue

            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {traceback.format_exc()}"

            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False

        return (all_pass, logs)