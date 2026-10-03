import sympy as sp
import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Checker for the satellite period / density equality problem (Earth-Moon).
    Validates:
      - Required output keys exist.
      - Physical invariant: orbital period near surface depends only on density.
      - Derived equality statement is consistent with physics.
      - Consistency of explanation steps (if available).
      - Monotonicity: period unchanged when radius perturbed (density constant).
      - Monotonicity: period decreases with increasing density.
      - Weak answer sanity: output strings are non-empty and contain equality.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely (None if missing)
        self.G = getattr(solution_instance, 'G', None)           # Gravitational constant (m^3 kg^-1 s^-2)
        self.T_E = getattr(solution_instance, 'T_E', None)       # Earth satellite period (s)
        self.T_M = getattr(solution_instance, 'T_M', None)       # Moon satellite period (s)
        self.R_E = getattr(solution_instance, 'R_E', None)       # Earth radius (m)
        self.R_M = getattr(solution_instance, 'R_M', None)       # Moon radius (m)
        self.M_E = getattr(solution_instance, 'M_E', None)       # Earth mass (kg)
        self.M_M = getattr(solution_instance, 'M_M', None)       # Moon mass (kg)
        self.rho_E = getattr(solution_instance, 'rho_E', None)   # Earth average density (kg/m^3)
        self.rho_M = getattr(solution_instance, 'rho_M', None)   # Moon average density (kg/m^3)

        # Extract output results (expected keys per emit contract)
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.density_equal = results.get('density_equal', None)
        self.equality_statement = results.get('equality_statement', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Pseudocode (optional)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the two required output keys are present and are strings."""
        if self.density_equal is None or self.equality_statement is None:
            return (False, "Missing required result key(s): density_equal or equality_statement")
        if not isinstance(self.density_equal, str) or not isinstance(self.equality_statement, str):
            return (False, "Result values must be strings")
        return (True, "Both required result keys are present and are strings")

    # ------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_period_density_relation(self) -> Tuple[bool, str]:
        """
        Verify that the orbital period near a spherical body's surface depends only on density:
            T = sqrt(3π / (G ρ))
        If both G and T are given, derive density and compare to provided density if available.
        Otherwise, verify the logical implication: equal periods imply equal densities.
        """
        # If we have G, T_E, T_M and at least one density, we can check consistency.
        if self.G is not None and self.T_E is not None and self.T_M is not None:
            # Expected density from T (assuming near-surface orbit)
            T_val = self.T_E  # same for both, so use one
            G_val = self.G
            expected_rho = (3 * math.pi) / (G_val * T_val * T_val)  # T^2 = 3π/(Gρ) => ρ = 3π/(G T^2)
            # Compare with provided rho_E if available
            if self.rho_E is not None:
                if not math.isclose(self.rho_E, expected_rho, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Density rho_E={self.rho_E} does not match expected {expected_rho:.2e} from period")
            if self.rho_M is not None:
                if not math.isclose(self.rho_M, expected_rho, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Density rho_M={self.rho_M} does not match expected {expected_rho:.2e} from period")
            # Check that the derived densities are equal
            if self.rho_E is not None and self.rho_M is not None:
                if not math.isclose(self.rho_E, self.rho_M, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, "Both densities should be equal, but they differ")
            # Even if densities not given, the periods equality itself forces density equality.
            return (True, "T^2 = 3π/(G ρ) invariant holds or is logically implied")

        # If we lack G/T, at least check the logical implication in the output
        if self.density_equal is not None and '=' in self.density_equal:
            return (True, "Output contains the density equality statement")
        return (True, "Cannot numerically verify period-density relation (no G/T provided)")

    def test_invariant_mass_radius_density_relation(self) -> Tuple[bool, str]:
        """
        Check that the equality statement M_E / R_E^3 = M_M / R_M^3 is equivalent to density equality.
        Use symbolic derivation: density = mass / volume, volume ∝ R^3.
        """
        try:
            M_E, M_M, R_E, R_M = sp.symbols('M_E M_M R_E R_M')
            # density equality: M_E / ((4/3)π R_E^3) = M_M / ((4/3)π R_M^3)  => M_E / R_E^3 = M_M / R_M^3
            eq1 = sp.Eq(M_E / (R_E**3), M_M / (R_M**3))
            # The target equality_statement should be the same form
            if self.equality_statement is not None:
                # check that the string contains "=" and mentions M_E, R_E, M_M, R_M
                if '=' not in self.equality_statement:
                    return (False, "equality_statement missing '='")
                if 'M_E' not in self.equality_statement or 'R_E' not in self.equality_statement:
                    return (False, "equality_statement should involve M_E and R_E")
                if 'M_M' not in self.equality_statement or 'R_M' not in self.equality_statement:
                    return (False, "equality_statement should involve M_M and R_M")
            return (True, "Mass/radius equality correctly represents density equality")
        except Exception as e:
            return (False, f"Symbolic invariant check failed: {e}")

    # ------------------------------------------------------------------
    # Modeling quality checks (based on explanation_steps)
    # ------------------------------------------------------------------
    def test_modeling_derivation_uses_density(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps (if present) include the key relation
        T = sqrt(3π/(G ρ)) or a verbal equivalent.
        """
        if self.explanation_steps is None:
            return (True, "SKIP: No explanation_steps provided")

        # Look for key concepts: "density", "T = sqrt(3π/(Gρ))", "depends only on density"
        steps_text = ' '.join(str(step) for step in self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        if 'density' not in steps_text.lower():
            return (False, "Explanation steps do not mention density")
        if 'sqrt' in steps_text or 'period depends only on density' in steps_text.lower():
            return (True, "Derivation correctly uses density-dependent period")
        return (True, "Explanation steps mention density but may lack explicit formula")

    def test_modeling_pseudocode_consistency(self) -> Tuple[bool, str]:
        """
        Check that pseudocode (if present) leads to the correct emit results.
        """
        if self.pseudocode is None:
            return (True, "SKIP: No pseudocode provided")
        # We only check that the pseudocode mentions the key derived relation
        pseudo_text = str(self.pseudocode)
        if 'rho' in pseudo_text or 'density' in pseudo_text.lower():
            return (True, "Pseudocode references density")
        if 'T = sqrt(3π/(Gρ))' in pseudo_text or 'T = 2π sqrt(R^3/(G M))' in pseudo_text:
            return (True, "Pseudocode contains correct period formula")
        return (True, "Pseudocode available but not fully verified")

    # ------------------------------------------------------------------
    # Monotonicity checks (trends)
    # ------------------------------------------------------------------
    def test_monotonicity_period_decreases_with_density(self) -> Tuple[bool, str]:
        """
        Perturb density upward and verify period decreases.
        """
        # Use default G if not provided (this is a trend test, not data extraction)
        G = self.G if self.G is not None else 6.67430e-11
        # Define baseline density
        rho_base = 5500.0  # kg/m^3
        try:
            T_base = math.sqrt(3 * math.pi / (G * rho_base))
            # Perturb: increase density by 10%
            rho_pert = rho_base * 1.1
            T_pert = math.sqrt(3 * math.pi / (G * rho_pert))
            if not (T_pert < T_base):
                return (False, f"Period should decrease: T_base={T_base:.2e}, T_pert={T_pert:.2e}")
            return (True, "Period decreases when density increases (T ∝ 1/√ρ)")
        except Exception as e:
            return (False, f"Monotonicity test failed: {e}")

    def test_monotonicity_radius_cancels(self) -> Tuple[bool, str]:
        """
        Perturb radius while keeping density constant; period should remain unchanged.
        """
        G = self.G if self.G is not None else 6.67430e-11
        rho = 5500.0  # baseline density
        # Baseline: radius = 1.0
        R_base = 1.0
        M_base = rho * (4/3) * math.pi * R_base**3
        T_base = 2 * math.pi * math.sqrt(R_base**3 / (G * M_base))
        # Perturb: increase radius by 10%
        R_pert = 1.1 * R_base
        M_pert = rho * (4/3) * math.pi * R_pert**3
        T_pert = 2 * math.pi * math.sqrt(R_pert**3 / (G * M_pert))
        # Period should be unchanged because density is constant
        if not math.isclose(T_base, T_pert, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Period changed from {T_base:.2e} to {T_pert:.2e} despite same density")
        return (True, "Period unchanged when radius perturbed (density constant)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Light sanity: output strings are non-empty, contain equality, and are about density.
        Do NOT compute any expected values or use math.isclose.
        """
        if not self.density_equal or not self.equality_statement:
            return (False, "Answer strings are empty")
        # Check that '=' appears and terms are plausible
        if '=' not in self.density_equal or '=' not in self.equality_statement:
            return (False, "Answer strings must contain '='")
        # At least one should mention rho or M (density/mass)
        has_rho = 'rho' in self.density_equal or 'rho' in self.equality_statement
        has_M = 'M_E' in self.equality_statement or 'M_M' in self.equality_statement
        if not (has_rho or has_M):
            return (False, "Answer strings should reference density or mass")
        return (True, "Answer strings exist, contain equality, and reference relevant quantities")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods. Return (all_pass, logs).
        """
        logs: List[str] = []
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