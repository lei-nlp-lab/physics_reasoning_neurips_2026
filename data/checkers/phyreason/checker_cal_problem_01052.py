import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults from problem statement
        self.m1 = getattr(solution_instance, 'm1', 20.0)
        self.m2 = getattr(solution_instance, 'm2', 60.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        # Use given exact trigonometric values for 37°
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Safe extraction of results
        self.F_OA = getattr(solution_instance, 'F_OA', 
                            getattr(solution_instance, 'results', {}).get('F_OA', 0.0))
        self.F_OB = getattr(solution_instance, 'F_OB', 
                            getattr(solution_instance, 'results', {}).get('F_OB', 0.0))
        self.f = getattr(solution_instance, 'f', 
                         getattr(solution_instance, 'results', {}).get('f', 0.0))
        
        # Store original for monotonicity test
        self._original_m1 = self.m1
        self._original_results = (self.F_OA, self.F_OB, self.f)
        
        # Initialize logs
        self.logs = []

    def _log(self, message: str):
        self.logs.append(message)

    def test_invariants(self) -> bool:
        """Check fundamental equilibrium conditions."""
        success = True
        
        # 1. Vertical equilibrium for object A: F_OA * cos37 = m1 * g
        lhs1 = self.F_OA * self.cos37
        rhs1 = self.m1 * self.g
        if not math.isclose(lhs1, rhs1, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: Vertical equilibrium violated. F_OA*cos37={lhs1}, m1*g={rhs1}")
            success = False
        else:
            self._log(f"PASS: Vertical equilibrium: {lhs1:.1f} ≈ {rhs1:.1f}")
        
        # 2. Horizontal equilibrium at junction: F_OB = F_OA * sin37
        lhs2 = self.F_OB
        rhs2 = self.F_OA * self.sin37
        if not math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: Horizontal equilibrium violated. F_OB={lhs2}, F_OA*sin37={rhs2}")
            success = False
        else:
            self._log(f"PASS: Horizontal equilibrium: {lhs2:.1f} ≈ {rhs2:.1f}")
        
        # 3. Horizontal equilibrium for person: f = F_OB
        if not math.isclose(self.f, self.F_OB, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: Person's horizontal equilibrium violated. f={self.f}, F_OB={self.F_OB}")
            success = False
        else:
            self._log(f"PASS: Person's horizontal equilibrium: {self.f:.1f} ≈ {self.F_OB:.1f}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """Check that results match step-by-step values from the analysis."""
        success = True
        # Step 1: F_OA = 250 N
        if not math.isclose(self.F_OA, 250.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: F_OA expected ~250N, got {self.F_OA}")
            success = False
        else:
            self._log(f"PASS: F_OA = {self.F_OA:.1f} N (expected 250 N)")
        
        # Step 2: F_OB = 150 N
        if not math.isclose(self.F_OB, 150.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: F_OB expected ~150N, got {self.F_OB}")
            success = False
        else:
            self._log(f"PASS: F_OB = {self.F_OB:.1f} N (expected 150 N)")
        
        # Step 3: f = 150 N
        if not math.isclose(self.f, 150.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"FAIL: f expected ~150N, got {self.f}")
            success = False
        else:
            self._log(f"PASS: f = {self.f:.1f} N (expected 150 N)")
        
        return success

    def test_monotonicity(self) -> bool:
        """Perturb m1 and verify tensions change correctly."""
        success = True
        
        # Store original
        m1_orig = self.m1
        F_OA_orig = self.F_OA
        F_OB_orig = self.F_OB
        f_orig = self.f
        
        # Perturb m1 by +1%
        self.m1 = m1_orig * 1.01
        # Recompute using physical formulas from steps
        F_OA_new = self.m1 * self.g / self.cos37
        F_OB_new = F_OA_new * self.sin37
        f_new = F_OB_new
        
        # Check monotonic increase (since m1 increased)
        if not (F_OA_new > F_OA_orig and F_OB_new > F_OB_orig and f_new > f_orig):
            self._log(f"FAIL: Monotonicity violated. After increasing m1 by 1%:")
            self._log(f"  Original: F_OA={F_OA_orig:.1f}, F_OB={F_OB_orig:.1f}, f={f_orig:.1f}")
            self._log(f"  New:      F_OA={F_OA_new:.1f}, F_OB={F_OB_new:.1f}, f={f_new:.1f}")
            success = False
        else:
            self._log(f"PASS: Monotonicity holds: tensions increase with m1.")
        
        # Restore original values
        self.m1 = m1_orig
        self.F_OA, self.F_OB, self.f = self._original_results
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        overall = inv_ok and trace_ok and mono_ok
        
        # Summary
        self._log("\n" + "="*50)
        self._log(f"Overall result: {'PASS' if overall else 'FAIL'}")
        self._log(f"  Invariants: {'PASS' if inv_ok else 'FAIL'}")
        self._log(f"  Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self._log(f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
        
        return overall, self.logs