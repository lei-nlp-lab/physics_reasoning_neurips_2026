import math
import sympy as sp
from typing import Any, Dict, Optional, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the virial theorem problem.
    Checks for required result keys, physical consistency of derived conditions,
    and optionally (if explanation_steps are present) modeling quality.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.K_eq_U = self.results.get('K_eq_U', None)
        self.gamma_threshold = self.results.get('gamma_threshold', None)

        # Explanation steps and pseudocode (optional, for modeling checks)
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

        # Optionally fetch givens (if the solution_instance provides them)
        self.gamma: Optional[float] = getattr(solution_instance, 'gamma', None)
        self.K: Optional[float] = getattr(solution_instance, 'K', None)
        self.U: Optional[float] = getattr(solution_instance, 'U', None)
        self.Omega: Optional[float] = getattr(solution_instance, 'Omega', None)

    # ------------------------------------------------------------------
    # Interface tests: check that expected result keys exist and have
    # valid types.
    # ------------------------------------------------------------------
    def test_interface_K_eq_U_exists(self) -> Tuple[bool, str]:
        if self.K_eq_U is None:
            return (False, "Key 'K_eq_U' not found in results.")
        if not isinstance(self.K_eq_U, (bool, int)):
            return (False, f"K_eq_U should be boolean or int, got {type(self.K_eq_U).__name__}.")
        return (True, f"K_eq_U = {self.K_eq_U}")

    def test_interface_gamma_threshold_exists(self) -> Tuple[bool, str]:
        if self.gamma_threshold is None:
            return (False, "Key 'gamma_threshold' not found in results.")
        if not isinstance(self.gamma_threshold, (int, float)):
            return (False, f"gamma_threshold should be numeric, got {type(self.gamma_threshold).__name__}.")
        return (True, f"gamma_threshold = {self.gamma_threshold}")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity.
    # For this purely theoretical problem we derive the expected
    # conditions inside the checker and compare to the student's result.
    # Use sympy for exact symbolic derivation.
    # ------------------------------------------------------------------
    def test_invariant_K_eq_U_derivation(self) -> Tuple[bool, str]:
        """
        Derive that for an ideal gas with γ = 5/3, K = U.
        The student's result K_eq_U should be True.
        """
        if self.K_eq_U is None:
            return (False, "SKIP: K_eq_U not available.")
        # Simulate derivation: we know the physical relation.
        # For γ = 5/3, the ideal gas relation gives K = U.
        # We can re-derive here symbolically.
        gamma_sym = sp.Rational(5, 3)
        R, n, T = sp.symbols('R n T', positive=True)
        # Kinetic energy (translational): K = (3/2) n R T
        K_expr = sp.Rational(3, 2) * n * R * T
        # Internal energy: U = C_v n T, with C_v = R/(γ-1)
        C_v = R / (gamma_sym - 1)
        U_expr = C_v * n * T
        # Simplify the difference
        diff = sp.simplify(K_expr - U_expr)
        # diff should be zero
        if diff == 0:
            expected_bool = True
            msg = f"Derived that for γ=5/3, K = U holds exactly."
        else:
            expected_bool = False
            msg = f"Unexpected: K - U simplifies to {diff} (should be 0)."
        # Compare to student's result
        if self.K_eq_U == expected_bool:
            return (True, msg)
        else:
            return (False, f"Student's K_eq_U is {self.K_eq_U}, but derivation gives {expected_bool}. {msg}")

    def test_invariant_gamma_threshold_derivation(self) -> Tuple[bool, str]:
        """
        Derive the condition γ > 4/3 from the virial theorem and the
        requirement that total energy is negative for a bound configuration.
        The student's gamma_threshold should be exactly 4/3.
        """
        if self.gamma_threshold is None:
            return (False, "SKIP: gamma_threshold not available.")
        # Derive the threshold symbolically.
        gamma = sp.symbols('gamma', real=True, positive=True)
        # Virial theorem: 2K + Ω + 3(γ-1)U = 0  (from physics mapping)
        # Total energy: E = K + U + Ω
        # Eliminate Ω: Ω = -2K - 3(γ-1)U
        # Then E = K + U - 2K - 3(γ-1)U = -K - (3γ - 4)U
        # For bound equilibrium we need E < 0 with K>0, U>0.
        # Hence we require -K - (3γ - 4)U < 0  =>  K + (3γ - 4)U > 0.
        # Since K>0, U>0, this holds when (3γ - 4) > 0, i.e. γ > 4/3.
        threshold = sp.Rational(4, 3)
        derived_threshold = float(threshold)  # convert to float for comparison
        # Use math.isclose for floating-point comparison (allowed in invariant tests)
        if math.isclose(float(self.gamma_threshold), derived_threshold, rel_tol=1e-3):
            return (True, f"Derived threshold γ > {derived_threshold} is consistent with student's value {self.gamma_threshold}.")
        else:
            return (False, f"Student's gamma_threshold = {self.gamma_threshold}, but derived threshold is exactly 4/3 = {derived_threshold}.")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check that explanation steps contain the
    # correct physical equations.
    # ------------------------------------------------------------------
    def test_modeling_virial_equation_in_steps(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps contain the virial theorem in its
        full form: 2K + Omega + 3(gamma-1)U = 0.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available.")
        # Join all steps into one string for searching
        all_steps = " ".join(self.explanation_steps)
        # We look for key terms: "2K", "Omega", "3(gamma-1)U"
        # Use flexible matching.
        has_2K = "2K" in all_steps or "2\\bar{K}" in all_steps
        has_Omega = "Omega" in all_steps or "\\Omega" in all_steps or "V" in all_steps  # potential energy often V
        has_gammaU = "3(gamma-1)U" in all_steps or "3(γ-1)U" in all_steps
        if has_2K and has_Omega and has_gammaU:
            return (True, "Virial equation form appears in explanation steps.")
        else:
            missing = []
            if not has_2K:
                missing.append("2K term")
            if not has_Omega:
                missing.append("Omega (potential energy) term")
            if not has_gammaU:
                missing.append("3(gamma-1)U term")
            return (False, f"Missing in explanation steps: {', '.join(missing)}.")

    def test_modeling_energy_condition_in_steps(self) -> Tuple[bool, str]:
        """
        Check that explanation steps mention the total energy E and the
        condition E < 0 for bound equilibrium.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available.")
        text = " ".join(self.explanation_steps)
        has_energy = "E =" in text or "total energy" in text
        has_bound = "E < 0" in text or "negative" in text or "bound" in text
        if has_energy and has_bound:
            return (True, "Energy condition for bound equilibrium appears.")
        else:
            missing = []
            if not has_energy:
                missing.append("total energy expression")
            if not has_bound:
                missing.append("E < 0 condition")
            return (False, f"Missing: {', '.join(missing)}.")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb the adiabatic index gamma and observe
    # the change in total energy.  The relation E = -K - (3γ-4)U
    # shows that for fixed positive K and U, E is a strictly decreasing
    # function of γ (dE/dγ = -3U < 0).  So increasing γ makes E more
    # negative.
    # ------------------------------------------------------------------
    def test_monotonicity_energy_decreases_with_gamma(self) -> Tuple[bool, str]:
        """
        Verify monotonicity: for fixed K>0, U>0, the total energy E
        decreases (becomes more negative) as γ increases.
        We use arbitrary positive values for K and U (e.g. 1.0 each)
        to demonstrate the monotonic relationship.
        """
        # Baseline: γ1 = 1.2  (value < 4/3)
        gamma_base = 1.2
        # Perturbed: γ2 = 1.5  (value > 4/3)
        gamma_pert = 1.5

        # Choose arbitrary positive K and U (they cancel in the monotonicity)
        K = 1.0
        U = 1.0

        # Compute total energies using the virial-derived formula
        E_base = -K - (3*gamma_base - 4)*U
        E_pert = -K - (3*gamma_pert - 4)*U

        # Check monotonic decrease: E_pert should be less than E_base
        if E_pert < E_base:
            return (True, f"Total energy decreases from {E_base:.3f} (γ={gamma_base}) to {E_pert:.3f} (γ={gamma_pert}), consistent with dE/dγ < 0.")
        else:
            return (False, f"Monotonicity violation: E({gamma_pert}) = {E_pert:.3f} should be less than E({gamma_base}) = {E_base:.3f}.")

    # ------------------------------------------------------------------
    # Weak final answer sanity check (only one, must be weak).
    # For gamma_threshold, check that it is positive and finite.
    # For K_eq_U, check that it is a boolean.
    # ------------------------------------------------------------------
    def test_answer_sanity_gamma_threshold(self) -> Tuple[bool, str]:
        """
        Sanity check: gamma_threshold must be positive and finite.
        """
        if self.gamma_threshold is None:
            return (False, "SKIP: gamma_threshold not available.")
        if not math.isfinite(self.gamma_threshold):
            return (False, "gamma_threshold is not finite.")
        if self.gamma_threshold <= 0:
            return (False, f"gamma_threshold = {self.gamma_threshold} is not positive.")
        # Broad magnitude range: γ threshold should be around 1.33...
        if not (1.0 <= self.gamma_threshold <= 2.0):
            return (False, f"gamma_threshold = {self.gamma_threshold} is outside expected range (1.0–2.0).")
        return (True, f"gamma_threshold = {self.gamma_threshold} passes weak sanity.")

    # ------------------------------------------------------------------
    # Run all tests (as required by architecture).
    # ------------------------------------------------------------------
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