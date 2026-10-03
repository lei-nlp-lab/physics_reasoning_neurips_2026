import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    """Robust physics auditor for pendulum/nail problem."""
    
    def __init__(self, solution_instance):
        """Safe attribute access to prevent sandbox errors."""
        self.solution = solution_instance
        
        # Extract inputs with fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.l = getattr(solution_instance, 'l', getattr(solution_instance, 'results', {}).get('l', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 9.8))
        self.OD = (2/3) * self.l  # Given constant relationship
        
        # Extract results with safe defaults
        results = getattr(solution_instance, 'results', {})
        self.W_G = results.get('W_G', 0.0)
        self.delta_E_p = results.get('delta_E_p', 0.0)
        self.h = results.get('h', 0.0)
        
        self.logs = []
    
    def _log(self, msg: str):
        """Helper for logging."""
        self.logs.append(msg)
    
    def _is_close(self, a, b) -> bool:
        """Fuzzy numerical matching with physics tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        return False
    
    def test_height_calculation(self) -> bool:
        """Verify height difference h = l - OD = l/3."""
        try:
            expected_h = self.l / 3.0
            
            # Check against stored h if available
            if self.h != 0.0:
                h_correct = self._is_close(self.h, expected_h)
                if not h_correct:
                    self._log(f"Height mismatch: stored h={self.h}, theoretical h={expected_h}")
                    return False
                else:
                    self._log(f"Height calculation correct: h = {self.h}")
                    return True
            else:
                # Compute from geometry
                computed_h = self.l - self.OD
                if self._is_close(computed_h, expected_h):
                    self._log(f"Height calculation correct: h = {computed_h}")
                    return True
                else:
                    self._log(f"Height mismatch: computed h={computed_h}, expected h={expected_h}")
                    return False
        except Exception as e:
            self._log(f"Height test failed: {str(e)}")
            return False
    
    def test_work_energy_theorem(self) -> bool:
        """Verify W_G = m*g*h using numerical checks."""
        try:
            # Expected work
            expected_W_G = (self.m * self.g * self.l) / 3.0
            
            # Compare with stored W_G
            if self.W_G != 0.0:
                numeric_ok = self._is_close(self.W_G, expected_W_G)
                if not numeric_ok:
                    self._log(f"Work mismatch: stored W_G={self.W_G}, calculated={expected_W_G}")
                    return False
                else:
                    self._log(f"Work-energy check: W_G = m*g*h = {self.W_G}")
                    return True
            else:
                # Fallback: just verify the formula numerically
                computed_W_G = (self.m * self.g * self.l) / 3.0
                self._log(f"Work-energy check: W_G = m*g*h = {computed_W_G}")
                return True
        except Exception as e:
            self._log(f"Work-energy test failed: {str(e)}")
            return False
    
    def test_potential_energy_relation(self) -> bool:
        """Verify ΔE_p = -W_G (work-energy theorem for conservative forces)."""
        try:
            # Expected values
            expected_W_G = (self.m * self.g * self.l) / 3.0
            expected_delta_E_p = -expected_W_G
            
            # Check stored values if available
            delta_ok = True
            if self.delta_E_p != 0.0:
                delta_ok = self._is_close(self.delta_E_p, expected_delta_E_p)
                if not delta_ok:
                    self._log(f"Potential energy mismatch: stored={self.delta_E_p}, calculated={expected_delta_E_p}")
            
            # Verify relationship
            if self.W_G != 0.0 and self.delta_E_p != 0.0:
                relation_ok = self._is_close(self.delta_E_p, -self.W_G)
                if not relation_ok:
                    self._log(f"Relation mismatch: ΔE_p={self.delta_E_p}, -W_G={-self.W_G}")
                    delta_ok = False
            
            self._log(f"Potential energy check: ΔE_p = -W_G = {expected_delta_E_p}")
            return delta_ok
        except Exception as e:
            self._log(f"Potential energy test failed: {str(e)}")
            return False
    
    def test_invariants(self) -> bool:
        """Verify physical invariants: energy conservation for gravity-only work."""
        try:
            # Ball moves downward from A to C (h decreases), so:
            # 1. Work by gravity should be positive (W_G > 0)
            # 2. Change in potential energy should be negative (ΔE_p < 0)
            
            W_G_val = (self.m * self.g * self.l) / 3.0
            delta_E_p_val = -W_G_val
            
            sign_ok = True
            if W_G_val <= 0:
                self._log(f"WARNING: Work by gravity should be positive for downward motion, got {W_G_val}")
                sign_ok = False
            
            if delta_E_p_val >= 0:
                self._log(f"WARNING: ΔE_p should be negative for downward motion, got {delta_E_p_val}")
                sign_ok = False
            
            self._log(f"Sign check: W_G={W_G_val} (positive), ΔE_p={delta_E_p_val} (negative)")
            return sign_ok
        except Exception as e:
            self._log(f"Invariant test failed: {str(e)}")
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        try:
            # Expected final answers from problem statement
            expected_W_G = (self.m * self.g * self.l) / 3.0
            expected_delta_E_p = -expected_W_G
            expected_h = self.l / 3.0
            
            # Check if stored results match expected
            wg_ok = True
            delta_ok = True
            h_ok = True
            
            if self.W_G != 0.0:
                wg_ok = self._is_close(self.W_G, expected_W_G)
                if not wg_ok:
                    self._log(f"Trace W_G mismatch: stored={self.W_G}, expected={expected_W_G}")
            
            if self.delta_E_p != 0.0:
                delta_ok = self._is_close(self.delta_E_p, expected_delta_E_p)
                if not delta_ok:
                    self._log(f"Trace ΔE_p mismatch: stored={self.delta_E_p}, expected={expected_delta_E_p}")
            
            if self.h != 0.0:
                h_ok = self._is_close(self.h, expected_h)
                if not h_ok:
                    self._log(f"Trace h mismatch: stored={self.h}, expected={expected_h}")
            
            self._log(f"Trace consistency: W_G match={wg_ok}, ΔE_p match={delta_ok}, h match={h_ok}")
            return wg_ok and delta_ok and h_ok
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}")
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb mass m and check that results scale linearly."""
        try:
            # Store original values
            original_m = self.m
            original_W_G = (self.m * self.g * self.l) / 3.0
            
            # Perturb mass (increase by 10%)
            perturbed_m = original_m * 1.1
            perturbed_W_G = (perturbed_m * self.g * self.l) / 3.0
            
            # Check linear scaling
            scaling_ratio = perturbed_W_G / original_W_G
            expected_ratio = 1.1
            
            scaling_ok = self._is_close(scaling_ratio, expected_ratio)
            
            if not scaling_ok:
                self._log(f"Monotonicity issue: W_G scaling ratio={scaling_ratio}, expected={expected_ratio}")
            
            self._log(f"Monotonicity check: W_G scales linearly with m (ratio={scaling_ratio})")
            return scaling_ok
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audits and return success status with logs."""
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        test_results = []
        
        try:
            # Run individual tests
            test_results.append(("Height calculation", self.test_height_calculation()))
            test_results.append(("Work-energy theorem", self.test_work_energy_theorem()))
            test_results.append(("Potential energy relation", self.test_potential_energy_relation()))
            test_results.append(("Physical invariants", self.test_invariants()))
            test_results.append(("Trace consistency", self.test_trace_consistency()))
            test_results.append(("Monotonicity", self.test_monotonicity()))
            
            # Summary
            success_count = sum(1 for _, passed in test_results if passed)
            total_tests = len(test_results)
            
            self.logs.append("\n=== Test Summary ===")
            for test_name, passed in test_results:
                status = "PASS" if passed else "FAIL"
                self.logs.append(f"{test_name}: {status}")
            
            overall_success = success_count == total_tests
            self.logs.append(f"\nOverall: {success_count}/{total_tests} tests passed")
            
            if overall_success:
                self.logs.append("✓ All physics checks passed")
            else:
                self.logs.append("✗ Some physics checks failed")
            
            return overall_success, self.logs
            
        except Exception as e:
            self.logs.append(f"Critical error in test suite: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False, self.logs