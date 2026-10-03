import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract constants: try instance attribute, then class attribute
        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            self.m = getattr(type(solution_instance), 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        if self.q is None:
            self.q = getattr(type(solution_instance), 'q', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        if self.E0 is None:
            self.E0 = getattr(type(solution_instance), 'E0', None)
        self.d = getattr(solution_instance, 'd', None)
        if self.d is None:
            self.d = getattr(type(solution_instance), 'd', None)
        self.v0 = getattr(solution_instance, 'v0', None)
        if self.v0 is None:
            self.v0 = getattr(type(solution_instance), 'v0', None)
        
        # Extract results with safe fallback
        self.results = getattr(solution_instance, 'results', {})
        self.v1 = self.results.get('v_1', None)
        self.t1 = self.results.get('t_1', None)
        self.v2 = self.results.get('v_2', None)
        self.t2 = self.results.get('t_2', None)
        self.v = self.results.get('v', None)   # final speed (could be v1 or v2)
        self.t = self.results.get('t', None)   # final time (could be t1 or t2)
        
        # Store logs
        self.logs = []
        
    def _log(self, msg: str):
        self.logs.append(msg)
        
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """Verify work-energy and impulse-momentum theorems."""
        success = True
        
        # Check inputs are available
        if None in (self.m, self.q, self.E0, self.d, self.v0):
            self._log(f"Missing input parameters: m={self.m}, q={self.q}, E0={self.E0}, d={self.d}, v0={self.v0}; skipping invariant tests.")
            return False
        
        # Case 1: exit at x=d (v1 and t1)
        if self.v1 is not None and self.t1 is not None:
            # Work-energy: -q*E0*d = 0.5*m*v1**2 - 0.5*m*v0**2
            lhs_we = -self.q * self.E0 * self.d
            rhs_we = 0.5 * self.m * self.v1**2 - 0.5 * self.m * self.v0**2
            if not self._is_close(lhs_we, rhs_we):
                self._log(f"Work-energy theorem fails for x=d case: {lhs_we} vs {rhs_we}")
                success = False
            else:
                self._log("Work-energy theorem holds for x=d case.")
            
            # Impulse-momentum: -q*E0*t1 = m*v1 - m*v0
            lhs_im = -self.q * self.E0 * self.t1
            rhs_im = self.m * self.v1 - self.m * self.v0
            if not self._is_close(lhs_im, rhs_im):
                self._log(f"Impulse-momentum theorem fails for x=d case: {lhs_im} vs {rhs_im}")
                success = False
            else:
                self._log("Impulse-momentum theorem holds for x=d case.")
        else:
            self._log("v1 or t1 missing; skipping invariants for x=d case.")
        
        # Case 2: exit at x=-2d (v2 and t2)
        if self.v2 is not None and self.t2 is not None:
            # Work-energy: 2*q*E0*d = 0.5*m*v2**2 - 0.5*m*v0**2
            lhs_we2 = 2 * self.q * self.E0 * self.d
            rhs_we2 = 0.5 * self.m * self.v2**2 - 0.5 * self.m * self.v0**2
            if not self._is_close(lhs_we2, rhs_we2):
                self._log(f"Work-energy theorem fails for x=-2d case: {lhs_we2} vs {rhs_we2}")
                success = False
            else:
                self._log("Work-energy theorem holds for x=-2d case.")
            
            # Impulse-momentum: q*E0*t2 = m*v2 + m*v0
            lhs_im2 = self.q * self.E0 * self.t2
            rhs_im2 = self.m * self.v2 + self.m * self.v0
            if not self._is_close(lhs_im2, rhs_im2):
                self._log(f"Impulse-momentum theorem fails for x=-2d case: {lhs_im2} vs {rhs_im2}")
                success = False
            else:
                self._log("Impulse-momentum theorem holds for x=-2d case.")
        else:
            self._log("v2 or t2 missing; skipping invariants for x=-2d case.")
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check that final answers (v, t) match intermediate results."""
        success = True
        
        # Check if v and t match either (v1, t1) or (v2, t2)
        if self.v is not None and self.t is not None:
            matched = False
            if self.v1 is not None and self.t1 is not None:
                if self._is_close(self.v, self.v1) and self._is_close(self.t, self.t1):
                    self._log("Final v and t match the x=d case (v1, t1).")
                    matched = True
            if self.v2 is not None and self.t2 is not None:
                if self._is_close(self.v, self.v2) and self._is_close(self.t, self.t2):
                    self._log("Final v and t match the x=-2d case (v2, t2).")
                    matched = True
            if not matched:
                self._log("Final v and t do not match any intermediate results.")
                success = False
        else:
            self._log("Final v or t missing; skipping trace consistency.")
            
        return success
    
    def test_monotonicity(self) -> bool:
        """Check physical plausibility and monotonic trends via perturbation."""
        success = True
        
        # Check inputs are available
        if None in (self.m, self.q, self.E0, self.d, self.v0):
            self._log(f"Missing input parameters: m={self.m}, q={self.q}, E0={self.E0}, d={self.d}, v0={self.v0}; skipping monotonicity tests.")
            return False
        
        # Physical plausibility checks
        if self.v1 is not None:
            if self.v1 > self.v0 and not self._is_close(self.v1, self.v0):
                self._log(f"v1 ({self.v1}) > v0 ({self.v0}); should be <= v0 for deceleration.")
                success = False
            else:
                self._log("v1 <= v0 (plausible deceleration).")
                
        if self.t1 is not None:
            if self.t1 < 0:
                self._log(f"t1 ({self.t1}) is negative; should be positive.")
                success = False
            else:
                self._log("t1 >= 0 (plausible).")
                
        if self.v2 is not None:
            if self.v2 < self.v0 and not self._is_close(self.v2, self.v0):
                self._log(f"v2 ({self.v2}) < v0 ({self.v0}); should be >= v0 for acceleration in negative direction.")
                success = False
            else:
                self._log("v2 >= v0 (plausible acceleration after turnaround).")
                
        if self.t2 is not None:
            if self.t2 < 0:
                self._log(f"t2 ({self.t2}) is negative; should be positive.")
                success = False
            else:
                self._log("t2 >= 0 (plausible).")
        
        # Perturb mass and check trend using correct formulas
        if self.v1 is not None and self.t1 is not None:
            # Store original mass
            m_orig = self.m
            # Perturb by +1%
            self.m *= 1.01
            # Recalculate v1 and t1 using correct formulas
            v1_new = math.sqrt(max(0, self.v0**2 - 2*self.q*self.E0*self.d/self.m))
            t1_new = (self.m * self.v0 - self.m * v1_new) / (self.q * self.E0)
            # Restore mass
            self.m = m_orig
            # Check trend: as m increases, v1 should increase (since deceleration is weaker)
            if v1_new < self.v1 and not self._is_close(v1_new, self.v1):
                self._log(f"Perturbation: v1 decreased when m increased; unexpected trend.")
                success = False
            else:
                self._log("Perturbation test for v1 passed.")
            # For t1, the trend is not monotonic in general; we skip explicit check
        else:
            self._log("v1 or t1 missing; skipping perturbation for x=d case.")
            
        if self.v2 is not None and self.t2 is not None:
            m_orig = self.m
            self.m *= 1.01
            v2_new = math.sqrt(self.v0**2 + 4*self.q*self.E0*self.d/self.m)
            t2_new = (self.m * self.v0 + self.m * v2_new) / (self.q * self.E0)
            self.m = m_orig
            # As m increases, v2 should decrease (since acceleration is weaker)
            if v2_new > self.v2 and not self._is_close(v2_new, self.v2):
                self._log(f"Perturbation: v2 increased when m increased; unexpected trend.")
                success = False
            else:
                self._log("Perturbation test for v2 passed.")
        else:
            self._log("v2 or t2 missing; skipping perturbation for x=-2d case.")
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        self._log("=== Starting PhysicsChecker ===")
        
        # Test invariants
        inv_success = self.test_invariants()
        overall_success = overall_success and inv_success
        self._log(f"Invariant tests: {'PASS' if inv_success else 'FAIL'}")
        
        # Test trace consistency
        trace_success = self.test_trace_consistency()
        overall_success = overall_success and trace_success
        self._log(f"Trace consistency tests: {'PASS' if trace_success else 'FAIL'}")
        
        # Test monotonicity
        mono_success = self.test_monotonicity()
        overall_success = overall_success and mono_success
        self._log(f"Monotonicity tests: {'PASS' if mono_success else 'FAIL'}")
        
        self._log(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs