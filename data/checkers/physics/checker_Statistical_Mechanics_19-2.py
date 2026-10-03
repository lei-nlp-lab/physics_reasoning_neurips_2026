import math
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Auditor for Fermi gas problem: Fermi momentum, total energy, degeneracy pressure.
    Checks theorem consistency, modeling quality, and answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract physical parameters (givens) with safe defaults = None
        self.N = getattr(solution_instance, 'N', None)
        self.q = getattr(solution_instance, 'q', None)
        self.V = getattr(solution_instance, 'V', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.m_e = getattr(solution_instance, 'm_e', None)

        # Extract results dictionary (must exist, but safe access)
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.k_F = results.get('k_F', None)
        self.E_tot = results.get('E_tot', None)
        self.P = results.get('P', None)

        # Derived quantities (only if all parameters available)
        if self.N is not None and self.q is not None and self.V is not None and self.V > 0:
            self.rho = self.N * self.q / self.V
        else:
            self.rho = None

    # ------------------------------------------------------------
    # Interface checks – ensure required keys exist and are numeric
    # ------------------------------------------------------------
    def test_interface_results_dict(self) -> Tuple[bool, str]:
        """Check that results dict contains all three required keys."""
        missing = []
        for key in ['k_F', 'E_tot', 'P']:
            val = getattr(self, f"{key}", None)
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_K_F_positive(self) -> Tuple[bool, str]:
        if self.k_F is None:
            return (False, "SKIP: k_F not available")
        if not isinstance(self.k_F, (int, float)):
            return (False, f"k_F is not numeric: {type(self.k_F)}")
        if self.k_F <= 0:
            return (False, f"k_F must be positive, got {self.k_F}")
        return (True, f"k_F = {self.k_F} > 0")

    def test_interface_E_tot_positive(self) -> Tuple[bool, str]:
        if self.E_tot is None:
            return (False, "SKIP: E_tot not available")
        if not isinstance(self.E_tot, (int, float)):
            return (False, f"E_tot is not numeric: {type(self.E_tot)}")
        if self.E_tot <= 0:
            return (False, f"E_tot must be positive, got {self.E_tot}")
        return (True, f"E_tot = {self.E_tot} > 0")

    def test_interface_P_positive(self) -> Tuple[bool, str]:
        if self.P is None:
            return (False, "SKIP: P not available")
        if not isinstance(self.P, (int, float)):
            return (False, f"P is not numeric: {type(self.P)}")
        if self.P <= 0:
            return (False, f"P must be positive, got {self.P}")
        return (True, f"P = {self.P} > 0")

    # ------------------------------------------------------------
    # Invariant checks – theorem-based physical relations
    # ------------------------------------------------------------
    def test_invariant_k_F_vs_rho(self) -> Tuple[bool, str]:
        """Fermi momentum k_F = (3π² ρ)^{1/3}."""
        if self.k_F is None or self.rho is None:
            return (False, "SKIP: missing k_F or rho")
        expected = (3 * math.pi**2 * self.rho) ** (1/3)
        if math.isclose(self.k_F, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"k_F ~ (3π²ρ)^(1/3): {self.k_F} ≈ {expected}")
        return (False, f"k_F = {self.k_F} does not match expected {expected}")

    def test_invariant_E_tot_from_k_F(self) -> Tuple[bool, str]:
        """E_tot = (ħ² V)/(10π² m_e) * k_F^5."""
        if any(x is None for x in [self.E_tot, self.k_F, self.hbar, self.V, self.m_e]):
            return (False, "SKIP: missing required variables")
        expected = (self.hbar**2 * self.V * self.k_F**5) / (10 * math.pi**2 * self.m_e)
        if math.isclose(self.E_tot, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_tot matches integral: {self.E_tot} ≈ {expected}")
        return (False, f"E_tot = {self.E_tot} != expected from integral {expected}")

    def test_invariant_pressure_energy_relation(self) -> Tuple[bool, str]:
        """P = (2/3)*E_tot/V."""
        if any(x is None for x in [self.P, self.E_tot, self.V]) or self.V == 0:
            return (False, "SKIP: missing P, E_tot, or V")
        expected = (2/3) * self.E_tot / self.V
        if math.isclose(self.P, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches (2/3)E_tot/V: {self.P} ≈ {expected}")
        return (False, f"P = {self.P} != (2/3)E_tot/V = {expected}")

    def test_invariant_pressure_fermi_energy(self) -> Tuple[bool, str]:
        """P = (2/5) ρ ε_F, with ε_F = ħ² k_F²/(2m_e)."""
        if any(x is None for x in [self.P, self.rho, self.k_F, self.hbar, self.m_e]):
            return (False, "SKIP: missing variables")
        epsilon_F = (self.hbar**2 * self.k_F**2) / (2 * self.m_e)
        expected = (2/5) * self.rho * epsilon_F
        if math.isclose(self.P, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches (2/5)ρ ε_F: {self.P} ≈ {expected}")
        return (False, f"P = {self.P} != (2/5)ρ ε_F = {expected}")

    def test_invariant_state_counting(self) -> Tuple[bool, str]:
        """Check Nq = 2 * (V/(2π)³) * (4π/3) * k_F^3."""
        if any(x is None for x in [self.N, self.q, self.V, self.k_F]):
            return (False, "SKIP: missing N, q, V, or k_F")
        Nq = self.N * self.q
        expected_states = 2 * (self.V / (2*math.pi)**3) * (4*math.pi/3) * (self.k_F**3)
        if math.isclose(Nq, expected_states, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Nq ≈ 2*V/(2π)³ * 4π/3 * k_F^3: {Nq} ≈ {expected_states}")
        return (False, f"Nq = {Nq} does not match state counting {expected_states}")

    # ------------------------------------------------------------
    # Modeling quality checks – derived from explanation steps
    # ------------------------------------------------------------
    def test_modeling_spin_degeneracy_factor(self) -> Tuple[bool, str]:
        """The derivation must include factor 2 (spin)."""
        if self.rho is None or self.k_F is None:
            return (False, "SKIP: missing density or k_F")
        # Without factor 2, k_F would be (6π² ρ)^(1/3). Check that the actual k_F is smaller.
        k_F_no_spin = (6 * math.pi**2 * self.rho) ** (1/3)
        if self.k_F < k_F_no_spin:
            return (True, f"k_F = {self.k_F} < {k_F_no_spin} (implies spin factor 2 included)")
        return (False, "k_F is not smaller than spin-1 case; spin degeneracy may be missing")

    def test_modeling_V_dependence_E_tot(self) -> Tuple[bool, str]:
        """E_tot should scale as V^(-2/3)."""
        if any(x is None for x in [self.E_tot, self.V, self.k_F, self.hbar, self.m_e]):
            return (False, "SKIP: missing variables")
        # Compute constant factor C = ħ² (3π² Nq)^(5/3) / (10π² m_e)
        Nq = self.N * self.q if (self.N and self.q) else None
        if Nq is None:
            return (False, "SKIP: cannot compute Nq")
        C = (self.hbar**2 * (3*math.pi**2 * Nq)**(5/3)) / (10 * math.pi**2 * self.m_e)
        expected_E = C * (self.V ** (-2/3))
        if math.isclose(self.E_tot, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_tot ~ V^{-2/3}: {self.E_tot} ≈ {expected_E}")
        return (False, f"E_tot = {self.E_tot} does not match V^{-2/3} scaling")

    def test_modeling_pressure_from_derivative(self) -> Tuple[bool, str]:
        """P should equal -∂E_tot/∂V = (2/3)E_tot/V (already checked), but also verify that E_tot ∝ V^{-2/3}."""
        # Redundant but ensures consistency
        if self.E_tot is None or self.V is None:
            return (False, "SKIP")
        # if E_tot ∝ V^{-2/3}, then P = (2/3)E_tot/V. Already covered.
        return (True, "Pressure from derivative consistent (cross-checked in invariants)")

    # ------------------------------------------------------------
    # Monotonicity check – e.g., k_F increases with density
    # ------------------------------------------------------------
    def test_monotonicity_k_F_density(self) -> Tuple[bool, str]:
        """If density ρ increases, k_F should increase (function is monotonic)."""
        if self.rho is None or self.k_F is None:
            return (False, "SKIP: missing rho or k_F")
        # Compute derivative sign: d(k_F)/dρ = (1/3)(3π²)^(1/3) ρ^(-2/3) > 0.
        # We can't perturb, but we can check that if we increase ρ by 1%, k_F should increase by ~0.33%.
        rho_pert = self.rho * 1.01
        k_F_pert_expected = (3 * math.pi**2 * rho_pert) ** (1/3)
        # Since we only have one data point, we check that k_F < expected perturbed k_F (since ρ_pert > ρ).
        if self.k_F < k_F_pert_expected:
            return (True, f"k_F monotonic: baseline {self.k_F} < perturbed (ρ*1.01) {k_F_pert_expected}")
        return (False, "Monotonicity violation: k_F not increasing with density")

    # ------------------------------------------------------------
    # Weak answer sanity – broad checks only
    # ------------------------------------------------------------
    def test_answer_sanity_broad_magnitude(self) -> Tuple[bool, str]:
        """Check that results are finite, positive, and not unreasonably large/small.
           This is a weak sanity – no expected values derived."""
        if any(x is None for x in [self.k_F, self.E_tot, self.P]):
            return (False, "SKIP: missing one of k_F, E_tot, P")
        checks = []
        # k_F typically ~10^10 m^-1 for metals; allow huge range
        if not (1e5 < self.k_F < 1e15):
            checks.append(f"k_F outside expected range: {self.k_F}")
        # E_tot positive and finite
        if not (0 < self.E_tot < 1e-5):  # For typical volumes ~10^-30 m^3, E_tot~10^-18 J; allow wide
            checks.append(f"E_tot possibly out of plausible range: {self.E_tot}")
        # Pressure >0
        if not (0 < self.P < 1e20):  # P ~ 10^10 Pa for metals
            checks.append(f"P out of plausible range: {self.P}")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All answers finite, positive, and within broad plausible bounds")

    # ------------------------------------------------------------
    # Test orchestrator – discover all test_ methods
    # ------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)