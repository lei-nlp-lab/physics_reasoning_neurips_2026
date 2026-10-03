import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results using getattr with fallbacks
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Wave parameters from problem statement (with safe fallbacks)
        self.v = float(self.inputs.get('v', 10.0))  # wave speed in m/s
        self.x_P = float(self.inputs.get('x_P', 1.0))  # position of P in m
        self.x_Q = float(self.inputs.get('x_Q', 6.0))  # position of Q in m
        
        # Extract results with safe access
        self.lmbda = self._safe_get_result('lambda', default=0.0)
        self.t_PQ = self._safe_get_result('t_PQ', default=0.0)
        self.t = self._safe_get_result('t', default=0.0)
        
        # Calculated distances and times for verification
        self.delta_x = abs(self.x_Q - self.x_P)
        
        # Period T calculated from wavelength and wave speed
        if self.lmbda > 0 and self.v > 0:
            self.T = self.lmbda / self.v
        else:
            self.T = 0.0
            
        # Time for Q to reach first positive maximum displacement (T/4)
        self.t_Qmax = self.T / 4.0 if self.T > 0 else 0.0
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def _safe_get_result(self, key: str, default: float = 0.0) -> float:
        """Safely extract a result value with fallback."""
        try:
            value = self.results.get(key, default)
            if isinstance(value, str):
                # Handle string values like "2m" or "0.55s"
                value = value.replace('m', '').replace('s', '').strip()
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return float(default)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        success = True
        
        # 1. Check wave relationship: v = λ * f = λ / T
        if self.lmbda > 0 and self.T > 0:
            v_calc = self.lmbda / self.T
            if not math.isclose(self.v, v_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Wave relationship failed: v={self.v}, λ/T={v_calc}")
                success = False
            else:
                logs.append(f"Wave relationship OK: v=λ/T={v_calc:.3f}")
        
        # 2. Check propagation time: t_PQ = Δx / v
        t_PQ_expected = self.delta_x / self.v
        if not math.isclose(self.t_PQ, t_PQ_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Propagation time mismatch: t_PQ={self.t_PQ}, Δx/v={t_PQ_expected}")
            success = False
        else:
            logs.append(f"Propagation time OK: t_PQ=Δx/v={t_PQ_expected:.3f}")
        
        # 3. Check SHM timing: time to max displacement = T/4
        if self.T > 0:
            t_max_expected = self.T / 4.0
            # From KB: t_max = T/4 should be approximately 0.05s based on steps analysis
            if not math.isclose(t_max_expected, 0.05, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"SHM timing check: T/4={t_max_expected:.3f} should be ~0.05s")
                # This is not necessarily a failure, just a consistency check
            else:
                logs.append(f"SHM timing OK: T/4={t_max_expected:.3f}~0.05s")
        
        # 4. Check total time consistency: t = t_PQ + T/4
        t_expected = self.t_PQ + self.t_Qmax
        if not math.isclose(self.t, t_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Total time mismatch: t={self.t}, t_PQ+T/4={t_expected}")
            success = False
        else:
            logs.append(f"Total time OK: t=t_PQ+T/4={t_expected:.3f}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        # Check wavelength consistency with target answer "2m"
        if not math.isclose(self.lmbda, 2.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Wavelength mismatch: λ={self.lmbda} should be ~2.0")
            success = False
        else:
            logs.append(f"Wavelength OK: λ={self.lmbda}~2.0m")
        
        # Check total time consistency with target answer "0.55s"
        if not math.isclose(self.t, 0.55, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Total time mismatch: t={self.t} should be ~0.55")
            success = False
        else:
            logs.append(f"Total time OK: t={self.t}~0.55s")
        
        # Verify intermediate step values from STEPS ANALYSIS
        # Step 1: λ = 2m
        if self.lmbda > 0 and self.v > 0:
            T_calc = self.lmbda / self.v  # Should be 0.2s
            if math.isclose(T_calc, 0.2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Period OK: T=λ/v={T_calc:.3f}~0.2s")
            else:
                logs.append(f"Period inconsistency: T=λ/v={T_calc:.3f}, expected ~0.2s")
        
        # Step 2: t_PQ = 0.5s
        if math.isclose(self.t_PQ, 0.5, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"t_PQ OK: {self.t_PQ:.3f}~0.5s")
        else:
            logs.append(f"t_PQ inconsistency: {self.t_PQ:.3f}, expected ~0.5s")
        
        # Step 3: T/4 = 0.05s
        if math.isclose(self.t_Qmax, 0.05, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"t_Qmax OK: T/4={self.t_Qmax:.3f}~0.05s")
        else:
            logs.append(f"t_Qmax inconsistency: T/4={self.t_Qmax:.3f}, expected ~0.05s")
        
        # Step 4: t = 0.55s (already checked above)
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test sensitivity to input perturbations."""
        logs = []
        success = True
        
        # Save original values
        original_v = self.v
        original_lmbda = self.lmbda
        original_t = self.t
        
        try:
            # Perturb wave speed by +10%
            self.v = original_v * 1.1
            
            # Recalculate dependent quantities
            if original_lmbda > 0:
                # For fixed wavelength, period changes: T' = λ / v'
                T_prime = original_lmbda / self.v
                t_Qmax_prime = T_prime / 4.0
                t_PQ_prime = self.delta_x / self.v
                t_prime = t_PQ_prime + t_Qmax_prime
                
                # Check monotonic trend: increased v should decrease t
                if t_prime < original_t:
                    logs.append(f"Monotonicity OK: v↑{original_v}→{self.v:.1f}, t↓{original_t:.3f}→{t_prime:.3f}")
                else:
                    logs.append(f"Monotonicity violation: v↑ but t didn't decrease")
                    success = False
            else:
                logs.append("Skipping monotonicity test due to missing wavelength")
        
        finally:
            # Restore original values
            self.v = original_v
        
        # Test perturbation of wavelength (affects period)
        try:
            # Save original T
            original_T = self.T
            
            # Perturb wavelength by +10%
            lmbda_prime = original_lmbda * 1.1
            T_prime = lmbda_prime / self.v
            t_Qmax_prime = T_prime / 4.0
            t_prime = self.t_PQ + t_Qmax_prime
            
            # Check monotonic trend: increased λ → increased T → increased t_Qmax → increased t
            if t_prime > original_t:
                logs.append(f"Monotonicity OK: λ↑{original_lmbda}→{lmbda_prime:.2f}, t↑{original_t:.3f}→{t_prime:.3f}")
            else:
                logs.append(f"Monotonicity violation: λ↑ but t didn't increase")
                success = False
                
        except ZeroDivisionError:
            logs.append("Skipping wavelength perturbation test (v=0)")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                test_success, test_logs = test_func()
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"✗ {test_name} ERROR: {str(e)}")
                overall_success = False
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if overall_success:
            all_logs.append("✓ All physics checks passed!")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return overall_success, all_logs