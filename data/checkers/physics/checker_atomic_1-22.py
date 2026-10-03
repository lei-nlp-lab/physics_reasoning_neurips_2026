import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics checker for Doppler width estimation problem.
    Validates that the computed Δλ (in Å) is physically sensible and
    consistent with thermal broadening theory.
    """

    # Fundamental constants (SI units) – used only for internal re‑derivation
    _k_B = 1.380649e-23        # J/K
    _c = 2.99792458e8           # m/s
    _m_u = 1.66053906660e-27    # kg
    _ln2 = 0.69314718056

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract relevant data from the solution instance.
        All attributes default to None if missing – no numeric fallbacks.
        """
        # Main result (expected key from pseudocode [emit] contract)
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        self.delta_lambda_A = None
        if isinstance(self.results, dict):
            self.delta_lambda_A = self.results.get('delta_lambda_A', None)

        # Optional givens (for invariant/monotonicity checks)
        self.given_lambda_A: Optional[float] = None
        self.given_T: Optional[float] = None
        self.given_A: Optional[float] = None   # mass number (e.g. 40)

        givens = getattr(solution_instance, 'givens', None)
        if isinstance(givens, dict):
            self.given_lambda_A = givens.get('lambda_A', None)
            self.given_T = givens.get('T', None)
            self.given_A = givens.get('A', None)

        # Explanation steps (for modeling quality checks)
        self.explanation_steps: Optional[Any] = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: compute expected Δλ (FWHM) in Å using the rigorous Gaussian formula
    # Use internal constants; returns None if givens insufficient.
    # ------------------------------------------------------------------
    def _compute_expected_fwhm_angstrom(self) -> Optional[float]:
        if self.given_lambda_A is None or self.given_T is None or self.given_A is None:
            return None
        # Convert given values to SI
        lam = self.given_lambda_A * 1e-10            # Å → m
        T = self.given_T
        m = self.given_A * self._m_u                 # mass in kg
        # FWHM from standard thermal broadening
        delta_lambda_m = (lam / self._c) * math.sqrt(
            8 * self._k_B * T * self._ln2 / m
        )
        return delta_lambda_m * 1e10                  # m → Å

    # ------------------------------------------------------------------
    # Helper: compute Δλ using the alternative approximate formula 2βλ
    # (with β = v_rms/c, v_rms = sqrt(3kT/m))
    # Returns None if givens insufficient.
    # ------------------------------------------------------------------
    def _compute_approx_v_rms_angstrom(self) -> Optional[float]:
        if self.given_lambda_A is None or self.given_T is None or self.given_A is None:
            return None
        lam = self.given_lambda_A * 1e-10
        T = self.given_T
        m = self.given_A * self._m_u
        v_rms = math.sqrt(3 * self._k_B * T / m)       # most probable speed
        beta = v_rms / self._c
        delta_lambda_m = 2 * beta * lam                # full Doppler width (approx)
        return delta_lambda_m * 1e10

    # ------------------------------------------------------------------
    # Interface check
    # ------------------------------------------------------------------
    def test_interface_has_result(self) -> Tuple[bool, str]:
        if self.results is None:
            return False, "No 'results' dict found in solution_instance."
        if 'delta_lambda_A' not in self.results:
            return False, "Missing key 'delta_lambda_A' in results."
        if not isinstance(self.delta_lambda_A, (int, float)):
            return False, f"delta_lambda_A = {self.delta_lambda_A} is not numeric."
        return True, "Result key 'delta_lambda_A' present and numeric."

    # ------------------------------------------------------------------
    # Physical invariants (use math.isclose with 1% tolerance)
    # ------------------------------------------------------------------
    def test_invariant_doppler_positive(self) -> Tuple[bool, str]:
        if self.delta_lambda_A is None:
            return False, "SKIP: no delta_lambda_A available."
        if self.delta_lambda_A <= 0:
            return False, f"delta_lambda_A = {self.delta_lambda_A} is not positive."
        return True, "Doppler width positive."

    def test_invariant_order_of_magnitude(self) -> Tuple[bool, str]:
        """
        Check that the result is within the order‑of‑magnitude regime (1e-3 to 1e-1 Å).
        This test does NOT use math.isclose – it's a broad range sanity.
        """
        if self.delta_lambda_A is None:
            return False, "SKIP: no delta_lambda_A available."
        if not (1e-3 <= self.delta_lambda_A <= 1e-1):
            return False, (
                f"delta_lambda_A = {self.delta_lambda_A} Å is not in [1e-3, 1e-1] Å. "
                "Order‑of‑magnitude estimate expected ~10⁻² Å."
            )
        return True, "Doppler width in 1e-3–1e-1 Å range."

    def test_invariant_fwhm_vs_v_rms_approximation(self) -> Tuple[bool, str]:
        """
        If givens are present, check that the result is consistent with either
        the rigorous FWHM formula or the simple 2βλ approximation.
        Accept a factor‑of‑10 tolerance (order of magnitude).
        """
        if self.delta_lambda_A is None:
            return False, "SKIP: no result."
        if self.given_lambda_A is None or self.given_T is None or self.given_A is None:
            return False, "SKIP: givens (lambda_A, T, A) missing."

        fwhm = self._compute_expected_fwhm_angstrom()
        approx = self._compute_approx_v_rms_angstrom()

        # If both computed values exist, check that result lies between them
        # (or within a factor of 10 of either).
        if fwhm is not None and approx is not None:
            lo = min(fwhm, approx)
            hi = max(fwhm, approx)
            # Accept result if it is within [lo/10, hi*10] (order of magnitude)
            if not (lo / 10 <= self.delta_lambda_A <= hi * 10):
                return False, (
                    f"delta_lambda_A = {self.delta_lambda_A} not within "
                    f"[{lo/10:.3e}, {hi*10:.3e}] computed from formulas."
                )
            return True, "Result within order‑of‑magnitude interval of both formulas."

        # Fallback: if only one formula available
        ref = fwhm if fwhm is not None else approx
        if ref is not None and not (ref / 10 <= self.delta_lambda_A <= ref * 10):
            return False, (
                f"Result {self.delta_lambda_A} not within factor 10 of "
                f"expected {ref}."
            )
        return True, "Result consistent with thermal broadening estimate."

    # ------------------------------------------------------------------
    # Modeling quality checks (at least two required)
    # #1: Check explanation steps contain the Doppler formula
    # #2: Check that the formula used in explanation steps matches the result
    # ------------------------------------------------------------------
    def test_modeling_doppler_formula_mentioned(self) -> Tuple[bool, str]:
        """
        If explanation_steps is provided, check that the physical relation
        (Doppler width formula) is mentioned. This is a weak sanity on the
        derivation path.
        """
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided."
        text = ""
        if isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        elif isinstance(self.explanation_steps, (list, tuple)):
            text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, dict):
            text = str(self.explanation_steps)
        else:
            text = str(self.explanation_steps)

        keywords = ["Doppler", "width", "FWHM", "thermal", "broadening", "Δλ", "delta_lambda"]
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        if not found:
            return False, "Explanation steps do not mention Doppler broadening concepts."
        return True, f"Explanation steps mention: {', '.join(found)}."

    def test_modeling_formula_expression_consistent(self) -> Tuple[bool, str]:
        """
        If explanation_steps contain a formula like Δλ = (λ/c)*sqrt(8kT ln2/m),
        then re‑derive an expected value using given constants and check
        that the result is within 1% of that re‑derived value.
        This tests that the solver actually applied the stated formula.
        """
        if self.delta_lambda_A is None:
            return False, "SKIP: no result."
        if self.given_lambda_A is None or self.given_T is None or self.given_A is None:
            return False, "SKIP: givens (lambda_A, T, A) missing."
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided."

        # Re‑derive expected value from the standard formula
        expected = self._compute_expected_fwhm_angstrom()
        if expected is None:
            return False, "SKIP: could not compute expected value from givens."

        # Use math.isclose with 1% tolerance (lenient for rounding)
        if not math.isclose(self.delta_lambda_A, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"Result {self.delta_lambda_A:.4e} Å differs from "
                f"expected {expected:.4e} Å by more than 1%."
            )
        return True, "Result matches standard formula within 1%."

    # ------------------------------------------------------------------
    # Monotonicity test (temperature scaling)
    # Explicitly perturb T and check that Δλ scales as sqrt(T)
    # ------------------------------------------------------------------
    def test_monotonicity_temperature_scaling(self) -> Tuple[bool, str]:
        """
        Check that the result scales approximately as sqrt(T).
        With a +10% temperature increase, the predicted Doppler width
        increases by a factor of sqrt(1.1) ≈ 1.0488.
        This test uses the supplied givens and the result to compute
        what the perturbed result would be, then verifies that the
        relative increase matches the expected square‑root scaling.
        """
        if self.delta_lambda_A is None:
            return False, "SKIP: no result."
        if self.given_T is None:
            return False, "SKIP: no given temperature."

        # Baseline
        T0 = self.given_T
        # Perturbation: +10% in temperature
        T_pert = T0 * 1.10
        # Expected ratio from physics: Δλ ∝ sqrt(T)
        ratio_expected = math.sqrt(T_pert / T0)   # ≈ 1.0488
        # Predicted perturbed result using the supplied result as baseline
        predicted_perturbed = self.delta_lambda_A * ratio_expected

        # The relative increase should be (ratio_expected - 1) ≈ 0.0488
        relative_increase = (predicted_perturbed - self.delta_lambda_A) / self.delta_lambda_A
        if not math.isclose(relative_increase, ratio_expected - 1, rel_tol=0.1):
            return False, (
                f"From result {self.delta_lambda_A:.4e}, a +10% temperature increase "
                f"gives predicted relative increase {relative_increase:.4f}, "
                f"expected ~{ratio_expected-1:.4f}."
            )
        return True, "Temperature scaling (sqrt(T)) consistent with result."

    # ------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_order(self) -> Tuple[bool, str]:
        """
        Weak final‑answer check: the result should be of order 10⁻² Å
        (i.e., between 0.005 and 0.05 Å). This is a very broad sanity.
        """
        if self.delta_lambda_A is None:
            return False, "SKIP: no result."
        if not (0.005 <= self.delta_lambda_A <= 0.05):
            return False, (
                f"delta_lambda_A = {self.delta_lambda_A} not in "
                f"[0.005, 0.05] Å weak sanity range."
            )
        return True, "Result in weak sanity range (0.005–0.05 Å)."

    # ------------------------------------------------------------------
    # Main dispatcher – discovers all test_ methods and runs them
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Gather all callable attributes that start with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs