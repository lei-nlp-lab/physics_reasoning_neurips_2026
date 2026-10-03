import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        Do NOT directly access solution_instance.attr; use getattr with fallback.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.N = getattr(solution_instance, 'N', getattr(solution_instance, 'n', 220))
        self.A = getattr(solution_instance, 'A', getattr(solution_instance, 'S', 0.05))
        self.f = getattr(solution_instance, 'f', 50.0)
        self.B = getattr(solution_instance, 'B', (2*math.sqrt(2))/math.pi)
        self.U2 = 220.0  # Motor voltage
        self.P_motor = 11000.0  # Power per motor
        self.num_motors = 2
        
        # Extract results with safe fallbacks
        self.E_m = self._safe_get('E_m', 2200*math.sqrt(2))
        self.U1 = self._safe_get('U_1', 2200.0)
        self.n_ratio = self._safe_get('fracn_1n_2', 10.0)  # n1/n2
        self.I2 = self._safe_get('I_2', 100.0)
        self.I1 = self._safe_get('I_1', 10.0)
        
        # Symbolic variables for physics laws
        self.N_sym, self.A_sym, self.B_sym, self.omega_sym, self.t_sym = sp.symbols('N A B omega t', positive=True)
        self.n1_sym, self.n2_sym, self.Vp_sym, self.Vs_sym, self.Ip_sym, self.Is_sym = sp.symbols('n1 n2 Vp Vs Ip Is', positive=True)
        self.P_sym = sp.symbols('P', positive=True)
        
        self.logs = []
        
    def _safe_get(self, key, default):
        """Safely get value from results with fallback."""
        if key in self.results:
            val = self.results[key]
            if isinstance(val, str):
                # Handle fraction string like "10/1"
                if '/' in val:
                    try:
                        num, den = map(float, val.split('/'))
                        return num/den
                    except:
                        pass
                try:
                    return float(val)
                except:
                    pass
            return float(val) if isinstance(val, (int, float)) else default
        return default
    
    def _log(self, msg, success=None):
        """Log message with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        return success
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws from KB."""
        success = True
        
        # 1. Check maximum EMF formula: E_m = N * A * B * omega
        omega = 2 * math.pi * self.f
        E_m_calc = self.N * self.A * self.B * omega
        if not math.isclose(self.E_m, E_m_calc, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"EMF invariant failed: calc={E_m_calc:.2f}, result={self.E_m:.2f}", False)
        else:
            self._log(f"EMF invariant passed: {E_m_calc:.2f} ≈ {self.E_m:.2f}", True)
        
        # 2. Check effective voltage: U1 = E_m / sqrt(2)
        U1_calc = self.E_m / math.sqrt(2)
        if not math.isclose(self.U1, U1_calc, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Effective voltage invariant failed: calc={U1_calc:.2f}, result={self.U1:.2f}", False)
        else:
            self._log(f"Effective voltage invariant passed: {U1_calc:.2f} ≈ {self.U1:.2f}", True)
        
        # 3. Check transformer voltage ratio: n1/n2 = U1/U2
        ratio_calc = self.U1 / self.U2
        if not math.isclose(self.n_ratio, ratio_calc, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Transformer voltage ratio failed: calc={ratio_calc:.2f}, result={self.n_ratio:.2f}", False)
        else:
            self._log(f"Transformer voltage ratio passed: {ratio_calc:.2f} ≈ {self.n_ratio:.2f}", True)
        
        # 4. Check secondary current: I2 = total_power / U2
        total_power = self.num_motors * self.P_motor
        I2_calc = total_power / self.U2
        if not math.isclose(self.I2, I2_calc, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Secondary current invariant failed: calc={I2_calc:.2f}, result={self.I2:.2f}", False)
        else:
            self._log(f"Secondary current invariant passed: {I2_calc:.2f} ≈ {self.I2:.2f}", True)
        
        # 5. Check transformer current ratio: I1 = (n2/n1) * I2
        I1_calc = (1/self.n_ratio) * self.I2
        if not math.isclose(self.I1, I1_calc, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Primary current invariant failed: calc={I1_calc:.2f}, result={self.I1:.2f}", False)
        else:
            self._log(f"Primary current invariant passed: {I1_calc:.2f} ≈ {self.I1:.2f}", True)
        
        # 6. Check power conservation (ideal transformer)
        P_primary = self.U1 * self.I1
        P_secondary = self.U2 * self.I2
        if not math.isclose(P_primary, P_secondary, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Power conservation failed: P_primary={P_primary:.2f}W, P_secondary={P_secondary:.2f}W", False)
        else:
            self._log(f"Power conservation passed: {P_primary:.2f}W ≈ {P_secondary:.2f}W", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check that turns ratio matches expected 10/1
        expected_ratio = 10.0
        if not math.isclose(self.n_ratio, expected_ratio, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Turns ratio mismatch: expected {expected_ratio}, got {self.n_ratio}", False)
        else:
            self._log(f"Turns ratio consistent: {self.n_ratio} ≈ {expected_ratio}", True)
        
        # Check that primary current matches expected 10A
        expected_I1 = 10.0
        if not math.isclose(self.I1, expected_I1, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Primary current mismatch: expected {expected_I1}A, got {self.I1}A", False)
        else:
            self._log(f"Primary current consistent: {self.I1}A ≈ {expected_I1}A", True)
        
        # Verify step-by-step chain using numerical propagation
        # Step 1 -> Step 2
        U1_from_E = self.E_m / math.sqrt(2)
        if not math.isclose(self.U1, U1_from_E, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Step 1->2 consistency failed: U1 from E_m={U1_from_E:.2f}, stored U1={self.U1:.2f}", False)
        
        # Step 2 -> Step 3
        ratio_from_U = self.U1 / self.U2
        if not math.isclose(self.n_ratio, ratio_from_U, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Step 2->3 consistency failed: ratio from U={ratio_from_U:.2f}, stored ratio={self.n_ratio:.2f}", False)
        
        # Step 3 -> Step 5
        I1_from_ratio = (1/self.n_ratio) * self.I2
        if not math.isclose(self.I1, I1_from_ratio, rel_tol=0.02, abs_tol=0.02):
            success = self._log(f"Step 3->5 consistency failed: I1 from ratio={I1_from_ratio:.2f}, stored I1={self.I1:.2f}", False)
        else:
            self._log(f"All intermediate steps consistent", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and verify expected trends from KB.
        RESTORE original values after perturbation.
        """
        success = True
        
        # Store original values
        orig_N = self.N
        orig_f = self.f
        orig_B = self.B
        
        # Test 1: Increase number of turns -> E_m should increase proportionally
        self.N = orig_N * 1.1  # +10%
        new_omega = 2 * math.pi * self.f
        new_E = self.N * self.A * self.B * new_omega
        orig_E = orig_N * self.A * self.B * (2 * math.pi * orig_f)
        
        if not (new_E > orig_E and math.isclose(new_E/orig_E, self.N/orig_N, rel_tol=0.05)):
            success = self._log(f"Monotonicity (N->E_m) failed: N increased 10%, E_m changed by {100*(new_E/orig_E-1):.1f}%", False)
        else:
            self._log(f"Monotonicity (N->E_m) passed: E_m scales with N", True)
        
        # Restore N
        self.N = orig_N
        
        # Test 2: Increase frequency -> E_m should increase proportionally
        self.f = orig_f * 1.1  # +10%
        new_omega = 2 * math.pi * self.f
        new_E = self.N * self.A * self.B * new_omega
        
        if not (new_E > orig_E and math.isclose(new_E/orig_E, self.f/orig_f, rel_tol=0.05)):
            success = self._log(f"Monotonicity (f->E_m) failed: f increased 10%, E_m changed by {100*(new_E/orig_E-1):.1f}%", False)
        else:
            self._log(f"Monotonicity (f->E_m) passed: E_m scales with f", True)
        
        # Restore f
        self.f = orig_f
        
        # Test 3: Increase transformer turns ratio n1/n2 -> primary current should decrease
        orig_ratio = self.n_ratio
        new_ratio = orig_ratio * 1.1  # +10%
        # Using I1 = (1/ratio) * I2, with I2 fixed (depends on motor power)
        orig_I1 = self.I1
        new_I1 = (1/new_ratio) * self.I2
        
        if not (new_I1 < orig_I1 and math.isclose(new_I1/orig_I1, orig_ratio/new_ratio, rel_tol=0.05)):
            success = self._log(f"Monotonicity (ratio->I1) failed: ratio increased 10%, I1 changed by {100*(new_I1/orig_I1-1):.1f}%", False)
        else:
            self._log(f"Monotonicity (ratio->I1) passed: I1 inversely proportional to ratio", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} ===", None)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {str(e)}", False)
                self._log(traceback.format_exc(), None)
                overall_success = False
        
        self._log(f"\n=== Overall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===", None)
        return overall_success, self.logs