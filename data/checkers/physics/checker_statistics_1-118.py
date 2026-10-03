import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.delta_H_vap = givens.get('delta_H_vap', None)
        self.rho = givens.get('rho', None)
        self.M = givens.get('M', None)
        self.N_A = givens.get('N_A', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.gamma = results.get('gamma', None)

        # Physical constants (from explanation steps)
        self.surface_thickness = 1e-8  # cm, typical molecular diameter
        # Avogadro constant if not given
        if self.N_A is None:
            self.N_A = 6.022e23

    def run_all_tests(self):
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

    # ----- Interface tests -----
    def test_interface_gamma_key(self):
        """Check that results contain key 'gamma'."""
        if self.gamma is None:
            return (False, "Missing key: gamma in results")
        return (True, "gamma key present")

    def test_interface_delta_H_vap_given(self):
        """Check that delta_H_vap is provided as a given."""
        if self.delta_H_vap is None:
            return (False, "Missing required given: delta_H_vap")
        return (True, "delta_H_vap given")

    def test_interface_rho_given(self):
        """Check that rho is provided (if not, skip but note)."""
        if self.rho is None:
            return (True, "SKIP: rho not given, will assume 1 g/cm³ in molecular test")
        return (True, "rho given")

    # ----- Invariant / sanity tests (physics theorem checks) -----
    def test_invariant_gamma_positive(self):
        """Surface tension must be positive."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if self.gamma <= 0:
            return (False, f"gamma must be positive, got {self.gamma}")
        return (True, "gamma > 0")

    def test_invariant_gamma_finite(self):
        """Surface tension must be finite."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if not math.isfinite(self.gamma):
            return (False, f"gamma is not finite: {self.gamma}")
        return (True, "gamma finite")

    def test_invariant_derived_energy_per_molecule_positive(self):
        """If M and rho are known, E_surf = (ΔH_vap * M)/(2*N_A) should be positive."""
        if self.delta_H_vap is None or self.M is None or self.N_A is None:
            return (True, "SKIP: missing data for energy per molecule")
        E_surf = (self.delta_H_vap * self.M) / (2.0 * self.N_A)
        if E_surf <= 0:
            return (False, f"E_surf <= 0: {E_surf}")
        return (True, "Surface energy per molecule > 0")

    # ----- Modeling / trace tests -----
    def test_modeling_gamma_via_simple_relation(self):
        """
        Use the explanation steps: σ = ΔH_vap * r * ρ, with r=1e-8 cm.
        Compare with reported gamma using isclose (relative tolerance 1%).
        """
        if self.delta_H_vap is None or self.rho is None:
            return (True, "SKIP: missing delta_H_vap or rho for simple relation")
        if self.gamma is None:
            return (False, "Cannot check simple relation: gamma not in results")
        r = self.surface_thickness
        gamma_simple = self.delta_H_vap * r * self.rho
        if not math.isclose(self.gamma, gamma_simple, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma {self.gamma} differs from simple relation {gamma_simple}")
        return (True, f"Simple relation agrees: gamma ~ {gamma_simple:.1f}")

    def test_modeling_gamma_via_molecular_approach(self):
        """
        If M, rho given, compute gamma = (ΔH_vap*M)/(2*N_A) / (M/(ρ N_A))^(2/3)
        and compare with reported gamma using isclose (relative tolerance 1%).
        """
        if self.delta_H_vap is None or self.M is None or self.rho is None:
            return (True, "SKIP: missing M or rho for molecular approach")
        if self.gamma is None:
            return (False, "Cannot check molecular approach: gamma not in results")
        # Use provided N_A; if not given, we already set default in __init__
        gamma_mol = (
            (self.delta_H_vap * self.M) / (2.0 * self.N_A) /
            ((self.M / (self.rho * self.N_A)) ** (2.0 / 3.0))
        )
        if not math.isclose(self.gamma, gamma_mol, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma {self.gamma} differs from molecular approach {gamma_mol}")
        return (True, f"Molecular approach agrees: gamma ~ {gamma_mol:.1f}")

    def test_trace_consistency_simple_vs_molecular(self):
        """
        If both simple and molecular estimates can be computed, they should agree 
        (since they represent the same physical quantity). Use isclose.
        """
        if self.delta_H_vap is None or self.rho is None or self.M is None:
            return (True, "SKIP: insufficient data for cross-check")
        r = self.surface_thickness
        gamma_simple = self.delta_H_vap * r * self.rho
        gamma_mol = (
            (self.delta_H_vap * self.M) / (2.0 * self.N_A) /
            ((self.M / (self.rho * self.N_A)) ** (2.0 / 3.0))
        )
        if not math.isclose(gamma_simple, gamma_mol, rel_tol=1e-1, abs_tol=1e-1):
            return (False, f"Simple {gamma_simple} and molecular {gamma_mol} estimates disagree")
        return (True, "Simple and molecular estimates consistent")

    # ----- Monotonicity tests -----
    def test_monotonicity_gamma_increases_with_Hvap(self):
        """
        Perturb delta_H_vap upward by 10% and recompute gamma via either method.
        The perturbed gamma should be larger.
        """
        if self.delta_H_vap is None or self.rho is None:
            return (True, "SKIP: missing delta_H_vap or rho for monotonicity")
        if self.gamma is None:
            return (True, "SKIP: gamma not in results for comparison")
        # Use simple relation for perturbation (requires rho)
        r = self.surface_thickness
        gamma_base = self.delta_H_vap * r * self.rho
        delta_H_perturbed = self.delta_H_vap * 1.10
        gamma_perturbed = delta_H_perturbed * r * self.rho
        if gamma_perturbed <= gamma_base:
            return (False, f"Perturbed gamma {gamma_perturbed} <= base {gamma_base}")
        return (True, "Monotonicity with ΔH_vap holds")

    def test_monotonicity_gamma_increases_with_rho(self):
        """
        Perturb rho upward by 10% and recompute gamma via simple relation.
        The perturbed gamma should be larger.
        """
        if self.delta_H_vap is None or self.rho is None:
            return (True, "SKIP: missing delta_H_vap or rho for monotonicity")
        if self.gamma is None:
            return (True, "SKIP: gamma not in results for comparison")
        r = self.surface_thickness
        gamma_base = self.delta_H_vap * r * self.rho
        rho_perturbed = self.rho * 1.10
        gamma_perturbed = self.delta_H_vap * r * rho_perturbed
        if gamma_perturbed <= gamma_base:
            return (False, f"Perturbed gamma {gamma_perturbed} <= base {gamma_base}")
        return (True, "Monotonicity with density holds")

    # ----- Weak answer sanity (only one such method) -----
    def test_answer_sanity_gamma_order_of_magnitude(self):
        """
        Weak check: gamma should be of order 100 dyn/cm (typical for liquids),
        say between 10 and 1000. Do NOT use math.isclose; use broad bounds.
        """
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if self.gamma < 10 or self.gamma > 1000:
            return (False, f"gamma = {self.gamma} is far from typical ~100 dyn/cm")
        return (True, f"gamma = {self.gamma:.1f} dyn/cm is within plausible range")