import math
import sympy
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the relativistic precession of Mercury's orbit.
    Checks givens/results keys, physical invariants, modeling consistency,
    and monotonicity of precession with respect to orbital radius.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract the results and givens containers
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results or {}
        elif isinstance(solution_instance, dict):
            results = solution_instance.get('results', {})
        else:
            results = {}

        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens or {}
        elif isinstance(solution_instance, dict):
            givens = solution_instance.get('givens', {})
        else:
            givens = {}

        self.results = results
        self.alpha = results.get('alpha', None)
        self.r0 = results.get('r0', None)
        self.precession = results.get('precession_arcsec_per_century', None)

        self.G = givens.get('G', None)          # gravitational constant
        self.c = givens.get('c', None)          # speed of light
        self.M_sun = givens.get('M_sun', None)  # solar mass
        self.a = givens.get('a', None)          # mean radius (m)
        self.T = givens.get('T', None)          # orbital period (s)

        # Explanation steps (optional, for context)
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps or []
        elif isinstance(solution_instance, dict):
            self.explanation_steps = solution_instance.get('explanation_steps', [])
        else:
            self.explanation_steps = []

    # ------------------------------------------------------------------
    # INTERFACE TESTS – ensure required keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_has_alpha(self) -> Tuple[bool, str]:
        if self.alpha is None:
            return (False, "Missing required result key: 'alpha'")
        return (True, f"alpha = {self.alpha}")

    def test_interface_has_r0(self) -> Tuple[bool, str]:
        if self.r0 is None:
            return (False, "Missing required result key: 'r0'")
        return (True, f"r0 = {self.r0}")

    def test_interface_has_precession(self) -> Tuple[bool, str]:
        if self.precession is None:
            return (False, "Missing required result key: 'precession_arcsec_per_century'")
        return (True, f"precession = {self.precession} arcsec/century")

    # ------------------------------------------------------------------
    # INVARIANT TESTS – theorem‑based physical sanity checks
    # ------------------------------------------------------------------
    def test_invariant_alpha_close_to_unity(self) -> Tuple[bool, str]:
        """Alpha must be very close to 1 because precession is tiny."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError):
            return (False, f"alpha is not numeric: {self.alpha}")
        if not math.isfinite(alpha):
            return (False, "alpha is not finite")
        if not math.isclose(alpha, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            return (False, f"alpha = {alpha} is not close to 1 (physical expectation)")
        return (True, f"alpha = {alpha} is close to 1")

    def test_invariant_precession_positive(self) -> Tuple[bool, str]:
        """Perihelion advance must be a positive angle."""
        if self.precession is None:
            return (False, "SKIP: precession not available")
        try:
            prec = float(self.precession)
        except (TypeError, ValueError):
            return (False, f"precession is not numeric: {self.precession}")
        if prec <= 0:
            return (False, f"Precession should be positive, got {prec}")
        return (True, f"precession = {prec} > 0 arcsec/century")

    # ------------------------------------------------------------------
    # MODELING / TRACE TESTS – re‑derive numeric values and compare
    # ------------------------------------------------------------------
    def _get_givens_fallback(self):
        """Return problem‑specific numbers if givens are missing."""
        # These are the exact values used in the provided explanation steps
        G = self.G if self.G is not None else 6.67430e-11
        c = self.c if self.c is not None else 2.99792458e8
        M_sun = self.M_sun if self.M_sun is not None else 1.989e30
        a = self.a if self.a is not None else 58e6 * 1000  # km -> m
        T = self.T if self.T is not None else 88 * 86400   # days -> s
        return G, c, M_sun, a, T

    def test_modeling_precession_derivation(self) -> Tuple[bool, str]:
        """
        Re‑compute precession using the formula from explanation steps:
        Δθ_per_orbit ≈ π GM/(a c²), number of orbits per century,
        convert to arcsec.
        Compare with student's result using math.isclose.
        """
        G, c, M_sun, a, T = self._get_givens_fallback()
        GM = G * M_sun   # standard gravitational parameter
        delta_theta_orbit = math.pi * GM / (a * c**2)
        # Number of orbits in 100 years (using 365 days per year, as in explanation)
        N_orbits = (100 * 365) / (T / 86400)   # T in seconds
        total_rad = N_orbits * delta_theta_orbit
        total_arcsec = total_rad * 206265

        if self.precession is None:
            return (False, "SKIP: student precession not available")
        try:
            student_prec = float(self.precession)
        except (TypeError, ValueError):
            return (False, f"student precession not numeric: {self.precession}")

        if not math.isfinite(student_prec):
            return (False, "student precession is not finite")

        if math.isclose(student_prec, total_arcsec, rel_tol=0.02, abs_tol=0.2):
            return (True, f"Re‑derived precession {total_arcsec:.2f} arcsec matches student {student_prec:.2f}")
        else:
            return (False, f"Re‑derived precession {total_arcsec:.4f} arcsec/century, student gave {student_prec:.4f}")

    def test_modeling_alpha_derivation(self) -> Tuple[bool, str]:
        """
        From explanation: α ≈ 1 – ½ (2π a/(T c))².
        Compute this approximate α and compare with student's result.
        """
        _, c, _, a, T = self._get_givens_fallback()
        omega = 2 * math.pi / T
        factor = (a * omega) / c
        alpha_approx = 1.0 - 0.5 * factor**2

        if self.alpha is None:
            return (False, "SKIP: student alpha not available")
        try:
            student_alpha = float(self.alpha)
        except (TypeError, ValueError):
            return (False, f"student alpha not numeric: {self.alpha}")

        if not math.isfinite(student_alpha):
            return (False, "student alpha is not finite")

        if math.isclose(student_alpha, alpha_approx, rel_tol=0.02, abs_tol=1e-5):
            return (True, f"Re‑derived α ≈ {alpha_approx:.10f} matches student {student_alpha:.10f}")
        else:
            return (False, f"Re‑derived α ≈ {alpha_approx:.10f}, student gave {student_alpha:.10f}")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS – direction checks by perturbing input
    # ------------------------------------------------------------------
    def test_monotonicity_precession_decreases_with_a(self) -> Tuple[bool, str]:
        """
        Precession per orbit ∝ 1/a, so increasing a reduces precession.
        Compute baseline and perturbed precession (using re‑derived formula)
        and check the sign of the change.
        """
        G, c, M_sun, a, T = self._get_givens_fallback()
        GM = G * M_sun
        # baseline
        delta_base = math.pi * GM / (a * c**2)
        N_orbits = (100 * 365) / (T / 86400)
        base_rad = N_orbits * delta_base
        base_arcsec = base_rad * 206265

        # perturbed: increase a by 1%
        a_pert = a * 1.01
        delta_pert = math.pi * GM / (a_pert * c**2)
        pert_rad = N_orbits * delta_pert
        pert_arcsec = pert_rad * 206265

        if pert_arcsec >= base_arcsec:
            return (False, f"Precession increased when radius increased: base {base_arcsec:.4f}, perturbed {pert_arcsec:.4f}")
        return (True, f"Precession decreases with a: {base_arcsec:.4f} -> {pert_arcsec:.4f} arcsec/century")

    def test_monotonicity_precession_decreases_with_T(self) -> Tuple[bool, str]:
        """
        Precession per orbit is independent of T, but the number of orbits
        per century ∝ 1/T, so total precession decreases with longer period.
        Perturb T by +1% and check.
        """
        G, c, M_sun, a, T = self._get_givens_fallback()
        GM = G * M_sun
        delta_orbit = math.pi * GM / (a * c**2)
        N_base = (100 * 365) / (T / 86400)   # T in seconds
        base_rad = N_base * delta_orbit
        base_arcsec = base_rad * 206265

        # increase T by 1%
        T_pert = T * 1.01
        N_pert = (100 * 365) / (T_pert / 86400)
        pert_rad = N_pert * delta_orbit
        pert_arcsec = pert_rad * 206265

        if pert_arcsec >= base_arcsec:
            return (False, f"Precession increased when period increased: base {base_arcsec:.4f}, perturbed {pert_arcsec:.4f}")
        return (True, f"Precession decreases with T: {base_arcsec:.4f} -> {pert_arcsec:.4f} arcsec/century")

    # ------------------------------------------------------------------
    # ANSWER SANITY (weak, at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_precession_finite(self) -> Tuple[bool, str]:
        """Precession should be a finite positive number."""
        if self.precession is None:
            return (False, "SKIP: precession not available")
        try:
            prec = float(self.precession)
        except (TypeError, ValueError):
            return (False, f"precession not numeric")
        if not math.isfinite(prec) or prec <= 0:
            return (False, f"precession is not a finite positive number: {prec}")
        # Broad magnitude: between 1 and 100 arcsec/century is plausible
        if prec < 1 or prec > 100:
            return (False, f"precession magnitude {prec} outside plausible range [1,100] arcsec/century")
        return (True, f"precession = {prec} arcsec/century looks plausible")

    # ------------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Collect all bound methods that start with "test_"
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
        return (all_pass, logs)