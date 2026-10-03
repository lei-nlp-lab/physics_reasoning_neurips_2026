import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.logs = []
        
        # 1. Safe Input Extraction
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', None))
        self.B1 = getattr(solution_instance, 'B1', getattr(solution_instance, 'results', {}).get('B1', None))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'results', {}).get('q', None))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', None))
        
        # 2. Extract Results with specific priority to avoid v1/v2 collision
        res = getattr(solution_instance, 'results', {})
        
        # We explicitly separate the radius for v1 and the radius for v2 trace
        # v1 related:
        self.v_1 = getattr(solution_instance, 'v_1', res.get('v_1', res.get('v1')))
        self.r_1_v1 = getattr(solution_instance, 'r1', res.get('r1')) # Specific for v1
        
        # v2 related:
        self.v_2 = getattr(solution_instance, 'v_2', res.get('v_2', res.get('v2')))
        self.r_2 = getattr(solution_instance, 'r_2', res.get('r_2', res.get('r2')))
        # We look for 'r1_v2' or fallback to 'r_1' only for the trace check
        self.r1_at_v2 = getattr(solution_instance, 'r1_v2', res.get('r1_v2', res.get('r_1')))
        
        # Other parameters
        self.t = getattr(solution_instance, 't', res.get('t'))
        self.eta = getattr(solution_instance, 'eta', res.get('eta'))
        self.T = getattr(solution_instance, 'T', res.get('T'))
        self.sinalpha = getattr(solution_instance, 'sinalpha', res.get('sinalpha'))

        self.has_essentials = all(v is not None for v in [self.L, self.B1, self.q, self.m])
        
    def _log(self, msg, success=None):
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        return success
    
    def test_invariants(self):
        successes = []
        if not self.has_essentials: return self._log("Missing essentials", False)
        
        # Verify v1 radius invariant using r_1_v1 (2L)
        if self.v_1 is not None and self.r_1_v1 is not None:
            r_expected = self.m * self.v_1 / (self.q * self.B1)
            successes.append(self._log("v1 radius Lorentz invariant", math.isclose(self.r_1_v1, r_expected, rel_tol=0.01)))
        
        # Verify final answers
        if self.eta is not None:
            successes.append(self._log("eta check (60%)", math.isclose(self.eta, 0.6, rel_tol=0.01)))
            
        return all(successes)

    def test_trace_consistency(self):
        successes = []
        # Verify geometry for v2 using r1_at_v2 (4L)
        try:
            # Step 6: r1_at_v2 * (sin_alpha - sin_30) = L
            if self.r1_at_v2 is not None and self.sinalpha is not None:
                lhs6 = self.r1_at_v2 * (self.sinalpha - 0.5)
                successes.append(self._log("Step 6 geometry (r1 at v2)", math.isclose(lhs6, self.L, rel_tol=0.01)))
            
            # Step 7: r2 * (1 - sin_alpha) = L/2
            if self.r_2 is not None and self.sinalpha is not None:
                lhs7 = self.r_2 * (1 - self.sinalpha)
                successes.append(self._log("Step 7 geometry (r2 at v2)", math.isclose(lhs7, self.L/2, rel_tol=0.01)))
        except:
            return False
        return all(successes)

    def run_all_tests(self):
        self.logs = []
        s1 = self.test_invariants()
        s2 = self.test_trace_consistency()
        overall = s1 and s2
        self._log(f"Overall Result: {'PASS' if overall else 'FAIL'}")
        return overall, self.logs