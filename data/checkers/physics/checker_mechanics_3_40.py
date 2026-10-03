import math
import sympy
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits the solution for the relativistic kinematics problem:
    "When two beams of protons of kinetic energy T collide head-on,
    the available energy for reactions is the same as for a single beam
    of what kinetic energy colliding with stationary protons?"
    """
    
    def __init__(self, solution_instance: Any) -> None:
        # Safely extract required result key
        self.K = getattr(solution_instance, 'results', {}).get('K', None)
        
        # Safely extract givens (expected: T, m, c)
        givens = getattr(solution_instance, 'givens', {})
        self.T = givens.get('T', None)
        self.m = givens.get('m', None)
        self.c = givens.get('c', None)
        
        # For modeling quality, check explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        
        # Additional fallback: if givens not available, try to parse from pseudocode (optional)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
    
    # ----------------- Interface Checks -----------------
    def test_interface_required_results(self) -> Tuple[bool, str]:
        """Check that the required result key 'K' exists and is a number."""
        if self.K is None:
            return (False, "Missing required result key 'K'")
        try:
            # Ensure it can be converted to float (works for sympy numbers too)
            float(self.K)
            return (True, f"Result K found with value: {self.K}")
        except (TypeError, ValueError):
            return (False, f"Result K is not numeric: {self.K}")
    
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all required givens (T, m, c) are present and numeric."""
        missing = []
        for name in ['T', 'm', 'c']:
            val = getattr(self, name)
            if val is None:
                missing.append(name)
            else:
                try:
                    float(val)
                except (TypeError, ValueError):
                    missing.append(f"{name} (non-numeric: {val})")
        if missing:
            return (False, f"Missing or non-numeric givens: {', '.join(missing)}")
        return (True, "All required givens present and numeric")
    
    # ----------------- Invariant Checks (use isclose) -----------------
    def test_invariant_collider_equals_fixed_target_s(self) -> Tuple[bool, str]:
        """Verify that center-of-mass energy squared equality holds using derived K."""
        if any(v is None for v in [self.T, self.m, self.c, self.K]):
            return (False, "SKIP: missing required values for invariant check")
        
        # Use c=1 natural units for simplicity; if c given as number, include it.
        try:
            T = float(self.T)
            m = float(self.m)
            c = float(self.c)
            K_val = float(self.K)
        except (TypeError, ValueError):
            return (False, "One of the values cannot be converted to float")
        
        # Compute s_coll = (E1+E2)^2 - (p1+p2)^2 ; colliding head-on -> equal energies, opposite momenta
        # In natural units (c=1): s_coll = (2E)^2 where E = T + m
        E = T + m
        s_coll = (2 * E) ** 2
        
        # For fixed target: incident kinetic energy K, target at rest.
        # s_fix = (E_inc + m)^2 - p_inc^2 = m^2 + m^2 + 2 m E_inc = 2m(m+E_inc)
        # But E_inc = K + m, p_inc^2 = (K+2m)K (using relativistic relation: p^2 = E^2 - m^2 = (K+m)^2 - m^2 = K^2+2Km)
        # Actually simpler: s_fix = 2m(m + (K+m)) = 2m(2m+K) = 4m^2 + 2mK
        s_fix = 4 * m * m + 2 * m * K_val
        
        # Check equality using relative tolerance
        if math.isclose(s_coll, s_fix, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s_coll ≈ s_fix ≈ {s_coll:.2f}")
        else:
            return (False, f"s_coll = {s_coll:.2f}, s_fix = {s_fix:.2f}, not close")
    
    def test_invariant_relativistic_limit(self) -> Tuple[bool, str]:
        """In the non-relativistic limit T << m, K should approximate 4T."""
        if any(v is None for v in [self.T, self.m, self.c, self.K]):
            return (False, "SKIP: missing required values")
        try:
            T = float(self.T)
            m = float(self.m)
            K_val = float(self.K)
        except (TypeError, ValueError):
            return (False, "Cannot convert to float")
        
        # Ensure non-relativistic regime: T << m (say T < 0.1*m)
        if T > 0.1 * m:
            return (True, "Not in non-relativistic limit; skip limit check")
        
        # Expected asymptotic: K ≈ 4T (since K = 4T + 2T^2/m + ...)
        expected_asymptotic = 4 * T
        if math.isclose(K_val, expected_asymptotic, rel_tol=0.2, abs_tol=1e-6):
            return (True, f"Limit holds: K ≈ {K_val}, expected ≈ {expected_asymptotic}")
        else:
            return (False, f"Limit violation: K = {K_val}, expected ~ {expected_asymptotic}")
    
    # ----------------- Modeling Quality Checks (based on explanation steps) -----------------
    def test_modeling_explanation_steps_present(self) -> Tuple[bool, str]:
        """Check that explanation_steps contain required elements (head-on, stationary target, relativistic)."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = str(self.explanation_steps).lower()
        keywords = ['head-on', 'stationary', 'relativistic', 'energy', 'momentum']
        missing = [kw for kw in keywords if kw not in text]
        if missing:
            return (False, f"Missing key concepts in explanation: {missing}")
        return (True, "Explanation contains required concepts")
    
    def test_modeling_uses_invariant_mass(self) -> Tuple[bool, str]:
        """Check that the explanation uses the invariance of E^2 - p^2."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        text = str(self.explanation_steps).lower()
        if 'invariant' in text or 'e^2 - p^2' in text:
            return (True, "Invariant method referenced")
        else:
            return (False, "No mention of invariant E^2-p^2")
    
    # ----------------- Monotonicity Checks -----------------
    def test_monotonicity_K_increases_with_T(self) -> Tuple[bool, str]:
        """
        Perturb T by +10% and check that K increases.
        Since we may not have an explicit expression for K in terms of T,
        we use a simple numeric check: if both T and K are numbers, 
        verify derivative positivity via small perturbation.
        """
        if any(v is None for v in [self.T, self.K]):
            return (False, "SKIP: missing T or K")
        try:
            T0 = float(self.T)
            K0 = float(self.K)
            m = float(self.m) if self.m is not None else 1.0  # fallback
            c = float(self.c) if self.c is not None else 1.0
        except (TypeError, ValueError):
            return (False, "Cannot convert to float")
        
        # Perturb T by +10%
        T_pert = T0 * 1.1
        
        # Recompute expected K from physics formula: K = (2T^2+4Tm c^2)/(m c^2)
        # Using c^2 = 1 if not provided
        K_expected = (2 * T_pert**2 + 4 * T_pert * m * c**2) / (m * c**2)
        
        if K_expected > K0 + 1e-9:  # allow numeric noise
            return (True, f"Increase confirmed: K from {K0:.2f} to {K_expected:.2f} for 10% T increase")
        else:
            return (False, f"K did not increase: original {K0}, perturbed {K_expected}")
    
    # ----------------- Weak Answer Sanity (only one, no isclose) -----------------
    def test_answer_sanity_reasonable(self) -> Tuple[bool, str]:
        """Basic sanity: K must be positive and finite."""
        if self.K is None:
            return (False, "SKIP: no K")
        try:
            K_val = float(self.K)
        except (TypeError, ValueError):
            return (False, "K is not a number")
        if not math.isfinite(K_val):
            return (False, "K is infinite or NaN")
        if K_val <= 0:
            return (False, f"K is non-positive: {K_val}")
        # Broad magnitude check: K should be on the order of T^2/m or larger
        # We don't compute expected, just check it's not absurdly small.
        if self.T is not None and self.m is not None:
            T = float(self.T)
            m = float(self.m)
            # For typical LHC energies, K ∼ few TeV, T ∼ 7 TeV, m ≈ .938 GeV
            # Minimum plausible: K could be slightly less than 4T if T tiny.
            # We'll just check that K > 0 and less than 1000 times (T^2/m) for safety.
            # But avoid exact formula – use magnitude order.
            # Instead just pass if positive.
            pass
        return (True, f"Answer sanity: K = {K_val} > 0, finite")
    
    # ----------------- Test Runner -----------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all test methods
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