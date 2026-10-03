import sympy as sp
import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract all inputs and results from the solution instance.
        Never access solution_instance.attr directly; use getattr with fallback to results dict.
        """
        # Constants (with safe defaults if missing)
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', 1.0))
        
        # Results (with safe defaults)
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.v_t = getattr(solution_instance, 'v_t', solution_instance.results.get('v_t', 0.0))
        
        # Fixed geometry
        self.theta = 37.0  # degrees
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = 0.6 / 0.8  # = 0.75
        
        # Store original for monotonicity tests
        self._original_m = self.m
        self._original_q = self.q
        self._original_g = self.g
        self._original_H = self.H
        
        # Logs
        self.logs = []

    def _log(self, msg: str):
        """Helper to append log messages."""
        self.logs.append(msg)

    def test_invariants(self) -> bool:
        """
        Verify physics invariants:
        1. Equilibrium condition (step 1): net force along incline = 0.
        2. Newton's 2nd law (step 2): net force = m*a.
        3. Energy/kinematics (step 3): v^2 = 2*a*d.
        """
        try:
            # 1. Equilibrium: q*E*cosθ = m*g*sinθ  => E = (m*g*tanθ)/q
            E_expected = (self.m * self.g * self.tan37) / self.q
            if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL invariant 1 (equilibrium): E={self.E}, expected={E_expected}")
                return False
            self._log(f"PASS invariant 1: E matches equilibrium condition.")
            
            # 2. Newton's 2nd law: a = ((q*E + m*g)*sinθ)/m
            # Note: After electric field is removed? Wait, step 2 uses same E? Let's verify.
            # According to steps: a = ((E*q + m*g)*sin37°)/m
            # But if E is from step1 (qE = mg tanθ), then substitute:
            # a = (mg tanθ + mg) * sinθ / m = g*(tanθ + 1)*sinθ = g*(0.75+1)*0.6 = 1.05g = 21g/20.
            a_expected = ((self.q * self.E + self.m * self.g) * self.sin37) / self.m
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL invariant 2 (Newton 2nd law): a={self.a}, expected={a_expected}")
                return False
            self._log(f"PASS invariant 2: a matches Newton's 2nd law.")
            
            # 3. Kinematics: v_t^2 = 2*a*d, where d = H/sinθ
            d = self.H / self.sin37
            v_t_expected = math.sqrt(2 * self.a * d)
            if not math.isclose(self.v_t, v_t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL invariant 3 (kinematics): v_t={self.v_t}, expected={v_t_expected}")
                return False
            self._log(f"PASS invariant 3: v_t matches kinematics.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Ensure step 1 -> step 2 -> step 3 are consistent.
        """
        try:
            # Step 1: E = (m*g*tan37)/q
            E_calc = (self.m * self.g * self.tan37) / self.q
            if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL trace step1: E={self.E}, calculated={E_calc}")
                return False
            self._log(f"PASS trace step1: E consistent.")
            
            # Step 2: a = ((E*q + m*g)*sin37)/m
            a_calc = ((self.E * self.q + self.m * self.g) * self.sin37) / self.m
            if not math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL trace step2: a={self.a}, calculated={a_calc}")
                return False
            self._log(f"PASS trace step2: a consistent.")
            
            # Step 3: v_t = sqrt(2*a*H/sin37)
            v_t_calc = math.sqrt(2 * self.a * self.H / self.sin37)
            if not math.isclose(self.v_t, v_t_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL trace step3: v_t={self.v_t}, calculated={v_t_calc}")
                return False
            self._log(f"PASS trace step3: v_t consistent.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check qualitative trends.
        We perturb m, q, g, H one at a time, recalculate using formulas, check direction, then restore.
        """
        try:
            # Test 1: Increase m -> E should increase (E ∝ m)
            self.m *= 1.1
            E_new = (self.m * self.g * self.tan37) / self.q
            if not (E_new > self.E * 0.999):  # Allow tiny numerical errors
                self._log(f"FAIL monotonicity: E should increase with m")
                self.m = self._original_m  # restore
                return False
            self.m = self._original_m
            self._log("PASS monotonicity test 1: E increases with m.")
            
            # Test 2: Increase q -> E should decrease (E ∝ 1/q)
            self.q *= 1.1
            E_new = (self.m * self.g * self.tan37) / self.q
            if not (E_new < self.E * 1.001):  # Allow tiny numerical errors
                self._log(f"FAIL monotonicity: E should decrease with q")
                self.q = self._original_q
                return False
            self.q = self._original_q
            self._log("PASS monotonicity test 2: E decreases with q.")
            
            # Test 3: Increase g -> E should increase (E ∝ g)
            self.g *= 1.1
            E_new = (self.m * self.g * self.tan37) / self.q
            if not (E_new > self.E * 0.999):
                self._log(f"FAIL monotonicity: E should increase with g")
                self.g = self._original_g
                return False
            # Also check a: a = ((E*q + m*g)*sin37)/m, but E changes with g.
            # Let's compute a from formulas with new g:
            E_new = (self.m * self.g * self.tan37) / self.q
            a_new = ((E_new * self.q + self.m * self.g) * self.sin37) / self.m
            # Since a ∝ g (from formula a = g*(tan37+1)*sin37), a should increase.
            if not (a_new > self.a * 0.999):
                self._log(f"FAIL monotonicity: a should increase with g")
                self.g = self._original_g
                return False
            self.g = self._original_g
            self._log("PASS monotonicity test 3: E and a increase with g.")
            
            # Test 4: Increase H -> v_t should increase (v_t ∝ sqrt(H))
            self.H *= 1.21  # sqrt(1.21) = 1.1, so v_t should increase by ~10%
            v_t_new = math.sqrt(2 * self.a * self.H / self.sin37)
            if not (v_t_new > self.v_t * 1.099):  # Allow 1% tolerance on 10% increase
                self._log(f"FAIL monotonicity: v_t should increase with H")
                self.H = self._original_H
                return False
            self.H = self._original_H
            self._log("PASS monotonicity test 4: v_t increases with H.")
            
            return True
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore all on error
            self.m = self._original_m
            self.q = self._original_q
            self.g = self._original_g
            self.H = self._original_H
            return False

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all checks and return (success, logs).
        """
        self.logs = []
        try:
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            success = test1 and test2 and test3
            return success, self.logs
        except Exception as e:
            self._log(f"CRITICAL ERROR in run_all_tests: {traceback.format_exc()}")
            return False, self.logs