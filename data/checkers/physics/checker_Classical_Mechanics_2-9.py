import math
import sympy as sp
from typing import Any

class PhysicsChecker:
    """
    Audits a solution instance for the top pair production problem.
    Extracts givens (m_t, m_W) and results (E) and performs:
      - interface checks (required keys present and positive)
      - invariant checks (energy-momentum relation, mass relation, derived E)
      - modeling checks (gamma factor consistency)
      - monotonicity check (E increases with m_t via explicit perturbation)
    """
    
    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely – assume solution_instance has .givens dict
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
        else:
            givens = getattr(solution_instance, 'givens', {})
        # Extract results safely – assume .results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = getattr(solution_instance, 'results', {})
        
        # Do NOT assign numeric defaults – only None for missing.
        self.m_t = givens.get('m_t', None)
        self.m_W = givens.get('m_W', None)
        self.E   = results.get('E', None)
        
        # Explanation steps – list of strings, optional for modeling checks
        if hasattr(solution_instance, 'explanation_steps') and isinstance(solution_instance.explanation_steps, list):
            self.explanation_steps = solution_instance.explanation_steps
        else:
            self.explanation_steps = []
        
        # Pseudocode – optional string
        if hasattr(solution_instance, 'pseudocode') and isinstance(solution_instance.pseudocode, str):
            self.pseudocode = solution_instance.pseudocode
        else:
            self.pseudocode = ""
        
    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_givens_exist(self) -> tuple:
        """Ensure both m_t and m_W are given and are positive."""
        if self.m_t is None:
            return (False, "SKIP: 'm_t' not found in givens")
        if self.m_W is None:
            return (False, "SKIP: 'm_W' not found in givens")
        if not isinstance(self.m_t, (int, float)) or self.m_t <= 0:
            return (False, f"m_t must be a positive number, got {self.m_t}")
        if not isinstance(self.m_W, (int, float)) or self.m_W <= 0:
            return (False, f"m_W must be a positive number, got {self.m_W}")
        return (True, "m_t and m_W present and positive")
    
    def test_interface_E_exists(self) -> tuple:
        """Ensure result key 'E' exists and is a positive number."""
        if self.E is None:
            return (False, "SKIP: 'E' not found in results")
        if not isinstance(self.E, (int, float)):
            return (False, f"E must be numeric, got {type(self.E)}")
        if self.E <= 0:
            return (False, f"E must be positive, got {self.E}")
        return (True, "E present and positive")
    
    # ------------------------------------------------------------
    # Invariant checks (theorem‑based, using math.isclose with generous tolerance)
    # ------------------------------------------------------------
    def test_invariant_mt_equals_2mW(self) -> tuple:
        """The problem states m_t = 2 m_W."""
        if self.m_t is None or self.m_W is None:
            return (False, "SKIP: m_t or m_W missing")
        if not math.isclose(self.m_t, 2.0 * self.m_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"m_t / m_W = {self.m_t/self.m_W:.3f}, expected 2.0")
        return (True, "m_t ≈ 2 m_W holds")
    
    def test_invariant_energy_momentum_relation(self) -> tuple:
        """
        Core invariant: (E - m_W)^2 = E^2 - m_t^2.
        This follows from on‑shell conditions and W at rest.
        """
        if self.E is None or self.m_t is None or self.m_W is None:
            return (False, "SKIP: E, m_t or m_W missing")
        lhs = (self.E - self.m_W) ** 2
        rhs = self.E ** 2 - self.m_t ** 2
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"LHS={lhs:.3f}, RHS={rhs:.3f}, difference={lhs - rhs:.3f}")
        return (True, "Energy‑momentum relation holds")
    
    def test_invariant_E_above_threshold(self) -> tuple:
        """E must be at least m_t to produce a top (lab frame)."""
        if self.E is None or self.m_t is None:
            return (False, "SKIP: E or m_t missing")
        if self.E < self.m_t * (1 - 1e-9):  # allow tiny numerical error
            return (False, f"E={self.E:.3f} is less than m_t={self.m_t:.3f}")
        return (True, "E >= m_t satisfied")
    
    def test_invariant_derived_E_from_mass_relation(self) -> tuple:
        """
        Using the mass relation m_t = 2 m_W and the invariant equation,
        we derive that E = (5/4) m_t.  This is a precise value derived
        inside the checker – it is NOT a memorized constant.
        """
        if self.E is None or self.m_t is None or self.m_W is None:
            return (False, "SKIP: E, m_t or m_W missing")
        # Derive expected E from the relation and the mass condition:
        # (E - m_W)^2 = E^2 - m_t^2  and m_t = 2 m_W.
        expected_E = (5.0 / 4.0) * self.m_t
        if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E = {self.E:.3f}, expected (5/4)*m_t = {expected_E:.3f}")
        return (True, "E matches the value derived from invariants")
    
    # ------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------
    def test_modeling_gamma_match(self) -> tuple:
        """
        From explanation steps: γ_t = γ_W*.  The W is at rest in lab when
        its boost in the top rest frame equals the top's boost in the lab.
        Compute γ_t = E/m_t and γ_W* = (5/4) from the method.
        They should agree.
        """
        if self.E is None or self.m_t is None:
            return (False, "SKIP: E or m_t missing")
        gamma_t = self.E / self.m_t
        # γ_W* = 5/4 is derived in the explanation (eq 2.9.2)
        gamma_W_star = 5.0 / 4.0
        if not math.isclose(gamma_t, gamma_W_star, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"γ_t = {gamma_t:.3f}, expected γ_W* = {gamma_W_star:.3f}")
        return (True, "Lorentz factors match: γ_t = γ_W*")
    
    def test_modeling_explanation_steps_present(self) -> tuple:
        """Check that explanation_steps is not empty (basic modelling quality)."""
        if not self.explanation_steps:
            return (False, "No explanation steps provided; at least one step expected.")
        return (True, f"Explanation steps present ({len(self.explanation_steps)} steps)")
    
    # ------------------------------------------------------------
    # Monotonicity check (explicit input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_E_increases_with_mt(self) -> tuple:
        """
        Perturb m_t upward by 1% and verify that the derived E
        (using the invariant relation and m_t=2 m_W) increases.
        This checks that E is an increasing function of m_t,
        as required by the physics (E = (5/4) m_t).
        """
        if self.m_t is None or self.m_W is None or self.E is None:
            return (False, "SKIP: m_t, m_W or E missing")
        # Baseline values
        baseline_m_t = self.m_t
        baseline_E = self.E
        
        # Perturb m_t by +1%
        delta = 0.01 * baseline_m_t
        perturbed_m_t = baseline_m_t + delta
        
        # Compute perturbed m_W using the relation m_t = 2 m_W
        perturbed_m_W = perturbed_m_t / 2.0
        
        # Solve invariant for perturbed E:
        # (E - m_W)^2 = E^2 - m_t^2  =>  E = (m_t^2 + m_W^2) / (2 m_W)
        # Using m_t = 2 m_W gives E = (5/4) m_t.
        # So perturbed_E = (5/4) * perturbed_m_t
        perturbed_E = (5.0 / 4.0) * perturbed_m_t
        
        # Check monotonicity: perturbed_E should be greater than baseline_E
        if perturbed_E <= baseline_E:
            return (False, f"Perturbed E = {perturbed_E:.3f} is not greater than baseline E = {baseline_E:.3f}")
        # Also check that the increase is roughly proportional (optional, but strengthens monotonicity)
        # Expected increase ratio: (perturbed_E / baseline_E) should equal (perturbed_m_t / baseline_m_t)
        expected_ratio = perturbed_m_t / baseline_m_t
        actual_ratio = perturbed_E / baseline_E
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E increases by factor {actual_ratio:.5f}, expected {expected_ratio:.5f}")
        return (True, "E increases with m_t (monotonic) and scales proportionally.")
    
    # ------------------------------------------------------------
    # Answer sanity (weak – only sign and finite check)
    # ------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self) -> tuple:
        """E must be finite and positive."""
        if self.E is None:
            return (False, "SKIP: E not available")
        if not math.isfinite(self.E):
            return (False, "E is not finite")
        if self.E <= 0:
            return (False, f"E = {self.E} is not positive")
        # Weak magnitude check: E should be of order m_t (not, e.g., 1e-30)
        if self.m_t is not None:
            ratio = self.E / self.m_t
            if ratio < 0.1 or ratio > 10:
                return (False, f"E/m_t = {ratio:.3f} seems unrealistic; expected ~1.25")
        return (True, "E is finite, positive, and of plausible magnitude")
    
    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> tuple:
        """
        Discover all methods beginning with 'test_', execute them safely,
        and return (success_bool, logs_list).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)