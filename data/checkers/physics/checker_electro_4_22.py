import math
import cmath
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Robust Physics Checker for anisotropic slab wave problem.
    Tests: interface, invariants, modeling/trace, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # ----- Extract results safely -----
        self.results: Dict = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = getattr(solution_instance, 'output_data', None)
        if not isinstance(self.results, dict):
            self.results = {}

        # ----- Extract givens safely -----
        self.givens: Dict = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = getattr(solution_instance, 'input_data', None)
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Explanation steps (optional for modeling hints)
        self.explanation_steps: List[Dict] = getattr(solution_instance, 'explanation_steps', [])

        # ----- Extract specific result keys with None defaults -----
        self.Erx: Optional[float] = self.results.get('Erx', None)
        self.Ery: Optional[float] = self.results.get('Ery', None)
        self.Tx: Optional[complex] = self.results.get('Tx', None)
        self.Ty: Optional[complex] = self.results.get('Ty', None)
        self.rel_amp: Optional[float] = self.results.get('relative_amplitude', None)
        self.rel_phase: Optional[float] = self.results.get('relative_phase', None)

        # ----- Extract given parameters -----
        self.omega: Optional[float] = self.givens.get('omega', None)
        self.c: Optional[float] = self.givens.get('c', None)
        self.epsilon0: Optional[float] = self.givens.get('epsilon0', None)
        self.nx: Optional[float] = self.givens.get('nx', None)
        self.ny: Optional[float] = self.givens.get('ny', None)
        self.nz: Optional[float] = self.givens.get('nz', None)
        self.d: Optional[float] = self.givens.get('d', None)
        self.E0: Optional[float] = self.givens.get('E0', None)

        # ----- Derived quantities (only if needed givens present) -----
        self.k0: Optional[float] = None
        if self.omega is not None and self.c is not None:
            self.k0 = self.omega / self.c

        # Incident amplitudes for 45° polarisation
        self.E0x: Optional[float] = self.E0 / math.sqrt(2) if self.E0 is not None else None
        self.E0y: Optional[float] = self.E0 / math.sqrt(2) if self.E0 is not None else None

        # Fresnel coefficients (normal incidence) – computed only if nx, ny given
        self.r12x: Optional[float] = None
        self.r12y: Optional[float] = None
        self.t12x: Optional[float] = None
        self.t12y: Optional[float] = None
        if self.nx is not None and self.ny is not None:
            self.r12x = (1 - self.nx) / (1 + self.nx)
            self.r12y = (1 - self.ny) / (1 + self.ny)
            self.t12x = 2 / (1 + self.nx)
            self.t12y = 2 / (1 + self.ny)

        # Theoretical transmission coefficients for slab (if possible)
        self.Tx_theory: Optional[complex] = None
        self.Ty_theory: Optional[complex] = None
        if (self.nx is not None and self.ny is not None and
            self.k0 is not None and self.d is not None):
            phix = cmath.exp(1j * self.k0 * self.nx * self.d)
            phiy = cmath.exp(1j * self.k0 * self.ny * self.d)
            t23x = 2 * self.nx / (1 + self.nx)
            t23y = 2 * self.ny / (1 + self.ny)
            denominator_x = 1 + self.r12x**2 * (phix**2)
            denominator_y = 1 + self.r12y**2 * (phiy**2)
            self.Tx_theory = (self.t12x * t23x * phix) / denominator_x
            self.Ty_theory = (self.t12y * t23y * phiy) / denominator_y

    # ----------------------------------------------------------------------
    #  Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_required_results(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and are not None."""
        required_keys = ['Erx', 'Ery', 'Tx', 'Ty', 'relative_amplitude', 'relative_phase']
        missing = [k for k in required_keys if k not in self.results or self.results[k] is None]
        if missing:
            return (False, f"Missing or None result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that essential given parameters are present."""
        essential = ['omega', 'c', 'nx', 'ny', 'nz', 'E0', 'd']
        missing = [k for k in essential if self.givens.get(k) is None]
        if missing:
            return (False, f"Missing essential given parameters: {missing}")
        return (True, "Essential given parameters present.")

    # ----------------------------------------------------------------------
    #  Invariant / Theorem‑Based Sanity Checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_fresnel_reflection_x(self) -> Tuple[bool, str]:
        """Check reflected x‑component: Erx ≈ r12x * E0x."""
        if self.Erx is None or self.r12x is None or self.E0x is None:
            return (False, "SKIP: Erx, r12x, or E0x missing")
        expected = self.r12x * self.E0x
        if math.isclose(self.Erx, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Erx = {self.Erx} matches r12x*E0x = {expected}")
        else:
            return (False, f"Erx ({self.Erx}) != r12x*E0x ({expected})")

    def test_invariant_fresnel_reflection_y(self) -> Tuple[bool, str]:
        """Check reflected y‑component: Ery ≈ r12y * E0y."""
        if self.Ery is None or self.r12y is None or self.E0y is None:
            return (False, "SKIP: Ery, r12y, or E0y missing")
        expected = self.r12y * self.E0y
        if math.isclose(self.Ery, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ery = {self.Ery} matches r12y*E0y = {expected}")
        else:
            return (False, f"Ery ({self.Ery}) != r12y*E0y ({expected})")

    def test_invariant_continuity_ratio_x(self) -> Tuple[bool, str]:
        """Check continuity relation: (E0x - Erx)/(E0x + Erx) ≈ nx^2 / nz."""
        if self.Erx is None or self.E0x is None or self.nx is None or self.nz is None:
            return (False, "SKIP: missing Erx, E0x, nx, or nz")
        num = self.E0x - self.Erx
        den = self.E0x + self.Erx
        if den == 0:
            return (False, "Denominator zero in continuity ratio x")
        ratio = num / den
        expected = self.nx**2 / self.nz
        if math.isclose(ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"(E0x-Erx)/(E0x+Erx) = {ratio} ≈ {expected}")
        else:
            return (False, f"Ratio {ratio} != expected {expected}")

    def test_invariant_continuity_ratio_y(self) -> Tuple[bool, str]:
        """Check continuity relation: (E0y - Ery)/(E0y + Ery) ≈ ny^2 / nz."""
        if self.Ery is None or self.E0y is None or self.ny is None or self.nz is None:
            return (False, "SKIP: missing Ery, E0y, ny, or nz")
        num = self.E0y - self.Ery
        den = self.E0y + self.Ery
        if den == 0:
            return (False, "Denominator zero in continuity ratio y")
        ratio = num / den
        expected = self.ny**2 / self.nz
        if math.isclose(ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"(E0y-Ery)/(E0y+Ery) = {ratio} ≈ {expected}")
        else:
            return (False, f"Ratio {ratio} != expected {expected}")

    def test_invariant_transmission_coefficient_x(self) -> Tuple[bool, str]:
        """Check Tx against theoretical slab transmission formula (magnitude & phase)."""
        if self.Tx is None or self.Tx_theory is None:
            return (False, "SKIP: Tx or theoretical Tx missing")
        mag_ok = math.isclose(abs(self.Tx), abs(self.Tx_theory), rel_tol=1e-2, abs_tol=1e-2)
        phase_diff = abs(cmath.phase(self.Tx) - cmath.phase(self.Tx_theory)) % (2*math.pi)
        phase_ok = (phase_diff < 0.02) or (abs(phase_diff - 2*math.pi) < 0.02)
        if mag_ok and phase_ok:
            return (True, f"Tx matches theory: |Tx|={abs(self.Tx):.3f}, phase diff={phase_diff:.3f}")
        else:
            return (False, f"Tx mismatch: |Tx|={abs(self.Tx):.3f} vs {abs(self.Tx_theory):.3f}, phase diff={phase_diff:.3f}")

    def test_invariant_transmission_coefficient_y(self) -> Tuple[bool, str]:
        """Check Ty against theoretical slab transmission formula."""
        if self.Ty is None or self.Ty_theory is None:
            return (False, "SKIP: Ty or theoretical Ty missing")
        mag_ok = math.isclose(abs(self.Ty), abs(self.Ty_theory), rel_tol=1e-2, abs_tol=1e-2)
        phase_diff = abs(cmath.phase(self.Ty) - cmath.phase(self.Ty_theory)) % (2*math.pi)
        phase_ok = (phase_diff < 0.02) or (abs(phase_diff - 2*math.pi) < 0.02)
        if mag_ok and phase_ok:
            return (True, f"Ty matches theory: |Ty|={abs(self.Ty):.3f}, phase diff={phase_diff:.3f}")
        else:
            return (False, f"Ty mismatch: |Ty|={abs(self.Ty):.3f} vs {abs(self.Ty_theory):.3f}, phase diff={phase_diff:.3f}")

    def test_invariant_relative_amplitude(self) -> Tuple[bool, str]:
        """Check relative_amplitude = |Tx/Ty|."""
        if self.rel_amp is None or self.Tx is None or self.Ty is None:
            return (False, "SKIP: relative_amplitude or Tx/Ty missing")
        try:
            computed = abs(self.Tx / self.Ty)
        except ZeroDivisionError:
            return (False, "Ty is zero, cannot compute relative amplitude")
        if math.isclose(self.rel_amp, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"relative_amplitude {self.rel_amp} ≈ {computed}")
        else:
            return (False, f"relative_amplitude {self.rel_amp} != computed {computed}")

    def test_invariant_relative_phase(self) -> Tuple[bool, str]:
        """Check relative_phase = arg(Tx) - arg(Ty)."""
        if self.rel_phase is None or self.Tx is None or self.Ty is None:
            return (False, "SKIP: relative_phase or Tx/Ty missing")
        try:
            computed = cmath.phase(self.Tx) - cmath.phase(self.Ty)
        except Exception:
            return (False, "Error computing phase difference")
        diff = (self.rel_phase - computed) % (2*math.pi)
        if diff > math.pi:
            diff -= 2*math.pi
        if abs(diff) < 0.02:
            return (True, f"relative_phase {self.rel_phase:.3f} ≈ {computed:.3f} (diff={diff:.3f})")
        else:
            return (False, f"relative_phase {self.rel_phase:.3f} != {computed:.3f} (diff={diff:.3f})")

    # ----------------------------------------------------------------------
    #  Modeling / Trace Quality Checks
    # ----------------------------------------------------------------------
    def test_modeling_polarization_signs(self) -> Tuple[bool, str]:
        """Check that reflected x and y amplitudes have appropriate signs relative to incident."""
        if self.Erx is None or self.Ery is None or self.E0x is None or self.E0y is None:
            return (False, "SKIP: need Erx, Ery, E0x, E0y")
        # For n>1, reflection coefficient negative => Er / E0 negative.
        if self.nx is not None and self.nx > 1:
            if self.Erx * self.E0x > 0:
                return (False, f"Expected Erx/E0x negative for nx>1, but got sign positive")
        if self.ny is not None and self.ny > 1:
            if self.Ery * self.E0y > 0:
                return (False, f"Expected Ery/E0y negative for ny>1, but got sign positive")
        return (True, "Polarization signs consistent with refraction index >1 (negative reflection)")

    def test_modeling_transmission_phases(self) -> Tuple[bool, str]:
        """Check that the transmitted phases are reasonable (non-zero when k0*nx*d ≠ 0)."""
        if self.Tx is None or self.Ty is None or self.k0 is None or self.nx is None or self.ny is None or self.d is None:
            return (False, "SKIP: missing data for phase check")
        if self.d > 0 and self.k0 * self.nx * self.d > 0.01:
            if abs(cmath.phase(self.Tx)) < 0.001:
                return (False, f"Tx phase near zero, although propagation factor is non‑zero")
        if self.d > 0 and self.k0 * self.ny * self.d > 0.01:
            if abs(cmath.phase(self.Ty)) < 0.001:
                return (False, f"Ty phase near zero, although propagation factor is non‑zero")
        return (True, "Transmission phases appear physically reasonable.")

    # ----------------------------------------------------------------------
    #  Monotonicity Tests (explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_reflection_x_increases_with_nx(self) -> Tuple[bool, str]:
        """
        Perturb nx by +10% and check that |Erx| increases (for nx > 1).
        Baseline |Erx| is taken from the result (or recomputed if missing).
        """
        if self.Erx is None or self.nx is None or self.E0x is None:
            return (False, "SKIP: need Erx, nx, E0x")
        if self.nx <= 1:
            return (False, "SKIP: monotonicity test only valid for nx > 1")
        # Baseline reflected amplitude magnitude
        baseline = abs(self.Erx)
        # Perturb nx by +10%
        nx_pert = self.nx * 1.1
        # Recompute reflection coefficient
        r12x_pert = (1 - nx_pert) / (1 + nx_pert)
        # Perturbed reflected amplitude
        pert_amp = abs(r12x_pert * self.E0x)
        # Expect increase because r12 becomes more negative (larger magnitude)
        if pert_amp > baseline:
            return (True, f"nx increased → |Erx| increased: baseline {baseline:.4f} → perturbed {pert_amp:.4f}")
        else:
            return (False, f"nx increased but |Erx| did not increase: baseline {baseline:.4f} → {pert_amp:.4f}")

    def test_monotonicity_reflection_y_increases_with_ny(self) -> Tuple[bool, str]:
        """
        Perturb ny by +10% and check that |Ery| increases (for ny > 1).
        Baseline |Ery| from result.
        """
        if self.Ery is None or self.ny is None or self.E0y is None:
            return (False, "SKIP: need Ery, ny, E0y")
        if self.ny <= 1:
            return (False, "SKIP: monotonicity test only valid for ny > 1")
        baseline = abs(self.Ery)
        ny_pert = self.ny * 1.1
        r12y_pert = (1 - ny_pert) / (1 + ny_pert)
        pert_amp = abs(r12y_pert * self.E0y)
        if pert_amp > baseline:
            return (True, f"ny increased → |Ery| increased: baseline {baseline:.4f} → perturbed {pert_amp:.4f}")
        else:
            return (False, f"ny increased but |Ery| did not increase: baseline {baseline:.4f} → {pert_amp:.4f}")

    def test_monotonicity_slab_thickness_effect_on_relative_phase(self) -> Tuple[bool, str]:
        """
        Perturb slab thickness d by +10% and check that relative_phase changes.
        Use the theoretical formula to compute perturbed relative phase.
        """
        if (self.rel_phase is None or self.d is None or self.k0 is None or
            self.nx is None or self.ny is None or self.E0x is None or self.E0y is None):
            return (False, "SKIP: missing data for thickness perturbation")
        baseline_phase = self.rel_phase
        # Perturb d by +10%
        d_pert = self.d * 1.1
        # Recompute Tx and Ty with perturbed d
        phix = cmath.exp(1j * self.k0 * self.nx * d_pert)
        phiy = cmath.exp(1j * self.k0 * self.ny * d_pert)
        t23x = 2 * self.nx / (1 + self.nx)
        t23y = 2 * self.ny / (1 + self.ny)
        # (r12 already computed in __init__, use same)
        if self.r12x is None or self.r12y is None:
            return (False, "SKIP: r12x or r12y not available")
        denominator_x = 1 + self.r12x**2 * (phix**2)
        denominator_y = 1 + self.r12y**2 * (phiy**2)
        Tx_pert = (self.t12x * t23x * phix) / denominator_x
        Ty_pert = (self.t12y * t23y * phiy) / denominator_y
        pert_phase = cmath.phase(Tx_pert) - cmath.phase(Ty_pert)
        # Check that the phase changed (at least by a small amount)
        if not math.isclose(baseline_phase, pert_phase, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"Thickness changed → relative phase changed: baseline {baseline_phase:.4f} → perturbed {pert_phase:.4f}")
        else:
            return (False, f"Thickness changed but relative phase unchanged: {baseline_phase:.4f}")

    # ----------------------------------------------------------------------
    #  Weak Answer Sanity (only one such method allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_final_quantities(self) -> Tuple[bool, str]:
        """Sanity checks: relative_amplitude positive and finite; relative_phase finite real."""
        issues = []
        if self.rel_amp is None:
            issues.append("relative_amplitude is None")
        else:
            if self.rel_amp <= 0 or math.isnan(self.rel_amp) or math.isinf(self.rel_amp):
                issues.append(f"relative_amplitude invalid: {self.rel_amp}")
        if self.rel_phase is None:
            issues.append("relative_phase is None")
        else:
            if math.isnan(self.rel_phase) or math.isinf(self.rel_phase):
                issues.append(f"relative_phase invalid: {self.rel_phase}")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Answer sanity passed (positive finite amplitude, finite phase).")

    # ----------------------------------------------------------------------
    #  Run All Tests
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, collect logs."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            if name.startswith("test__"):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {e}"
                import traceback
                msg += "\n" + traceback.format_exc()
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)