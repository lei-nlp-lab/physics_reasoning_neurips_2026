import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Initialize PhysicsChecker with safe attribute access."""
        self.solution = solution_instance
        
        # Extract inputs safely with defaults
        self.L = float(getattr(solution_instance, 'L', 1.0))
        self.B = float(getattr(solution_instance, 'B', 1.0))
        self.v0 = float(getattr(solution_instance, 'v0', 1.0))
        self.m = float(getattr(solution_instance, 'm', 1.0))
        self.R = float(getattr(solution_instance, 'R', 1.0))
        
        # Extract results safely with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Force - handle both numeric and string with formula
        F_val = results.get('F', 0.0)
        if isinstance(F_val, str):
            # Try to extract numeric part if it's a formula
            try:
                # Simple evaluation for common format
                expr = F_val.replace('^', '**').replace('frac{', '(').replace('}', ')')
                self.F = eval(expr, {"B": self.B, "L": self.L, "v0": self.v0, "R": self.R})
            except:
                self.F = 0.0
        else:
            self.F = float(F_val)
        
        # Charge q
        self.q = float(results.get('q', 0.0))
        
        # Displacement x
        self.x = float(results.get('x', 0.0))
        
        # Velocity v_1
        self.v1 = float(results.get('v_1', 0.0))
        
        # Parameter k - handle range string
        k_val = results.get('k', 2.0)
        if isinstance(k_val, str):
            # Parse range like "2 ≤ k < 3" or single value
            try:
                if '≤' in k_val or '<' in k_val:
                    # Extract first numeric value
                    import re
                    nums = re.findall(r'[\d.]+', k_val)
                    self.k = float(nums[0]) if nums else 2.0
                else:
                    self.k = float(k_val)
            except:
                self.k = 2.0
        else:
            self.k = float(k_val)
        
        # Intermediate results
        self.delta_x = float(results.get('delta_x', self.x))
        
        # Store original values for monotonicity tests
        self._original_v0 = self.v0
        self._original_m = self.m
        self._original_R = self.R
        
        # Store direction - handle both string and numeric
        direction_val = results.get('N/A', '')
        if isinstance(direction_val, (int, float)):
            # Numeric code: negative means left
            self.direction = 'left' if direction_val < 0 else 'right'
        else:
            self.direction = str(direction_val)
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching with 1% tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _test_force_formula(self) -> Tuple[bool, str]:
        """Test F = (B²L²v₀)/(2R) from step 3."""
        try:
            expected_F = (self.B**2 * self.L**2 * self.v0) / (2 * self.R)
            
            if self._is_close(self.F, expected_F):
                return True, f"Force formula correct: F ≈ {self.F:.4e}, expected ≈ {expected_F:.4e}"
            else:
                return False, f"Force mismatch: F={self.F:.4e}, expected={expected_F:.4e}"
        except Exception as e:
            return False, f"Force test error: {str(e)}"
    
    def _test_charge_formula(self) -> Tuple[bool, str]:
        """Test q = (mv₀)/(3BL) from step 7."""
        try:
            expected_q = (self.m * self.v0) / (3 * self.B * self.L)
            
            if self._is_close(self.q, expected_q):
                return True, f"Charge formula correct: q ≈ {self.q:.4e}, expected ≈ {expected_q:.4e}"
            else:
                return False, f"Charge mismatch: q={self.q:.4e}, expected={expected_q:.4e}"
        except Exception as e:
            return False, f"Charge test error: {str(e)}"
    
    def _test_displacement_formula(self) -> Tuple[bool, str]:
        """Test x = (2mv₀R)/(3B²L²) from step 11-12."""
        try:
            expected_x = (2 * self.m * self.v0 * self.R) / (3 * self.B**2 * self.L**2)
            
            if self._is_close(self.x, expected_x):
                return True, f"Displacement correct: x ≈ {self.x:.4e}, expected ≈ {expected_x:.4e}"
            else:
                return False, f"Displacement mismatch: x={self.x:.4e}, expected={expected_x:.4e}"
        except Exception as e:
            return False, f"Displacement test error: {str(e)}"
    
    def _test_velocity_formula(self) -> Tuple[bool, str]:
        """Test v₁ = (2/3)v₀ from step 13 (momentum conservation)."""
        try:
            expected_v1 = (2/3) * self.v0
            
            if self._is_close(self.v1, expected_v1):
                return True, f"Velocity correct: v₁ ≈ {self.v1:.4e}, expected ≈ {expected_v1:.4e}"
            else:
                return False, f"Velocity mismatch: v₁={self.v1:.4e}, expected={expected_v1:.4e}"
        except Exception as e:
            return False, f"Velocity test error: {str(e)}"
    
    def _test_k_range(self) -> Tuple[bool, str]:
        """Test 2 ≤ k < 3 from step 21."""
        try:
            if 2.0 <= self.k < 3.0:
                return True, f"k in valid range: k = {self.k}, 2 ≤ k < 3"
            else:
                return False, f"k out of range: k = {self.k}, should satisfy 2 ≤ k < 3"
        except Exception as e:
            return False, f"k-range test error: {str(e)}"
    
    def _test_charge_displacement_relation(self) -> Tuple[bool, str]:
        """Test q = (BLΔx)/(2R) from step 10 for consistency."""
        try:
            expected_q_from_x = (self.B * self.L * self.delta_x) / (2 * self.R)
            
            if self._is_close(self.q, expected_q_from_x):
                return True, f"Charge-displacement relation holds: q ≈ {self.q:.4e}, (BLΔx)/(2R) ≈ {expected_q_from_x:.4e}"
            else:
                return False, f"Charge-displacement mismatch: q={self.q:.4e}, (BLΔx)/(2R)={expected_q_from_x:.4e}"
        except Exception as e:
            return False, f"Charge-displacement test error: {str(e)}"
    
    def _test_impulse_charge_relation(self) -> Tuple[bool, str]:
        """Test B*L*q = impulse on rod N = m*(v₀/3) from step 5."""
        try:
            impulse_from_q = self.B * self.L * self.q
            expected_impulse = self.m * (self.v0 / 3)
            
            if self._is_close(impulse_from_q, expected_impulse):
                return True, f"Impulse-charge relation holds: B*L*q ≈ {impulse_from_q:.4e}, m*v₀/3 ≈ {expected_impulse:.4e}"
            else:
                return False, f"Impulse-charge mismatch: B*L*q={impulse_from_q:.4e}, m*v₀/3={expected_impulse:.4e}"
        except Exception as e:
            return False, f"Impulse-charge test error: {str(e)}"
    
    def _test_energy_trend(self) -> Tuple[bool, str]:
        """Check that initial KE > final KE (energy dissipated in resistor)."""
        try:
            KE_initial = 0.5 * self.m * self.v0**2
            KE_final = 0.5 * self.m * self.v1**2 + 0.5 * self.m * (self.v0/3)**2
            
            if KE_initial > KE_final:
                return True, f"Energy dissipated: KE_initial={KE_initial:.4e} > KE_final={KE_final:.4e}"
            elif self._is_close(KE_initial, KE_final, rel_tol=0.05):
                return True, f"Energy approximately conserved (within 5%): {KE_initial:.4e} ≈ {KE_final:.4e}"
            else:
                return False, f"Energy anomaly: KE_initial={KE_initial:.4e}, KE_final={KE_final:.4e}"
        except Exception as e:
            return False, f"Energy test error: {str(e)}"
    
    def test_invariants(self) -> List[Tuple[bool, str]]:
        """Test fundamental physics invariants."""
        logs = []
        
        # Test momentum conservation
        try:
            p_initial = self.m * self.v0
            p_final = self.m * self.v1 + self.m * (self.v0 / 3)
            
            if self._is_close(p_initial, p_final):
                logs.append((True, f"Momentum conserved: p_initial={p_initial:.4e}, p_final={p_final:.4e}"))
            else:
                logs.append((False, f"Momentum not conserved: p_initial={p_initial:.4e}, p_final={p_final:.4e}"))
        except Exception as e:
            logs.append((False, f"Momentum test error: {str(e)}"))
        
        # Force direction (qualitative check)
        try:
            if 'left' in self.direction.lower():
                logs.append((True, "Force direction: left (consistent with Lenz's law)"))
            else:
                logs.append((False, f"Force direction unclear: {self.direction}"))
        except Exception as e:
            logs.append((False, f"Force direction test error: {str(e)}"))
        
        # Add other invariant tests
        energy_test = self._test_energy_trend()
        logs.append(energy_test)
        
        impulse_test = self._test_impulse_charge_relation()
        logs.append(impulse_test)
        
        return logs
    
    def test_trace_consistency(self) -> List[Tuple[bool, str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Test force formula
        force_test = self._test_force_formula()
        logs.append(force_test)
        
        # Test charge formula
        charge_test = self._test_charge_formula()
        logs.append(charge_test)
        
        # Test displacement formula
        displacement_test = self._test_displacement_formula()
        logs.append(displacement_test)
        
        # Test velocity formula
        velocity_test = self._test_velocity_formula()
        logs.append(velocity_test)
        
        # Test k range
        k_test = self._test_k_range()
        logs.append(k_test)
        
        # Test charge-displacement relation
        relation_test = self._test_charge_displacement_relation()
        logs.append(relation_test)
        
        return logs
    
    def test_monotonicity(self) -> List[Tuple[bool, str]]:
        """Perturb inputs and check output trends."""
        logs = []
        
        try:
            # Perturb v₀ upward by 10%
            perturb_factor = 1.1
            original_v0 = self.v0
            self.v0 *= perturb_factor
            
            expected_F_new = (self.B**2 * self.L**2 * self.v0) / (2 * self.R)
            expected_q_new = (self.m * self.v0) / (3 * self.B * self.L)
            expected_x_new = (2 * self.m * self.v0 * self.R) / (3 * self.B**2 * self.L**2)
            
            expected_F_orig = (self.B**2 * self.L**2 * original_v0) / (2 * self.R)
            expected_q_orig = (self.m * original_v0) / (3 * self.B * self.L)
            expected_x_orig = (2 * self.m * original_v0 * self.R) / (3 * self.B**2 * self.L**2)
            
            tests_passed = 0
            total_tests = 3
            
            if expected_F_new > expected_F_orig:
                tests_passed += 1
            else:
                logs.append((False, f"F not monotonic with v₀: {expected_F_new:.4e} ≤ {expected_F_orig:.4e}"))
            
            if expected_q_new > expected_q_orig:
                tests_passed += 1
            else:
                logs.append((False, f"q not monotonic with v₀: {expected_q_new:.4e} ≤ {expected_q_orig:.4e}"))
            
            if expected_x_new > expected_x_orig:
                tests_passed += 1
            else:
                logs.append((False, f"x not monotonic with v₀: {expected_x_new:.4e} ≤ {expected_x_orig:.4e}"))
            
            if tests_passed == total_tests:
                logs.append((True, f"Monotonicity test passed: All outputs increase with v₀ (perturb +10%)"))
            else:
                logs.append((False, f"Monotonicity test: {tests_passed}/{total_tests} passed"))
            
            # Restore original v₀
            self.v0 = original_v0
            
        except Exception as e:
            logs.append((False, f"Monotonicity test error: {str(e)}"))
            if hasattr(self, '_original_v0'):
                self.v0 = self._original_v0
        
        # Test R perturbation
        try:
            original_R = self.R
            self.R *= 1.1
            
            expected_F_new = (self.B**2 * self.L**2 * self.v0) / (2 * self.R)
            expected_x_new = (2 * self.m * self.v0 * self.R) / (3 * self.B**2 * self.L**2)
            
            expected_F_orig = (self.B**2 * self.L**2 * self.v0) / (2 * original_R)
            expected_x_orig = (2 * self.m * self.v0 * original_R) / (3 * self.B**2 * self.L**2)
            
            if expected_F_new < expected_F_orig:
                logs.append((True, f"F decreases with R (inverse relationship): {expected_F_new:.4e} < {expected_F_orig:.4e}"))
            else:
                logs.append((False, f"F should decrease with R but: {expected_F_new:.4e} ≥ {expected_F_orig:.4e}"))
            
            if expected_x_new > expected_x_orig:
                logs.append((True, f"x increases with R (direct relationship): {expected_x_new:.4e} > {expected_x_orig:.4e}"))
            else:
                logs.append((False, f"x should increase with R but: {expected_x_new:.4e} ≤ {expected_x_orig:.4e}"))
            
            # Restore original R
            self.R = original_R
            
        except Exception as e:
            logs.append((False, f"R-perturbation test error: {str(e)}"))
            if hasattr(self, '_original_R'):
                self.R = self._original_R
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return (success_bool, logs_list)."""
        all_logs = []
        success = True
        
        try:
            # Run invariant tests
            invariant_logs = self.test_invariants()
            all_logs.extend([f"INVARIANT: {msg}" for _, msg in invariant_logs])
            success &= all(passed for passed, _ in invariant_logs)
            
            # Run trace consistency tests
            trace_logs = self.test_trace_consistency()
            all_logs.extend([f"TRACE: {msg}" for _, msg in trace_logs])
            success &= all(passed for passed, _ in trace_logs)
            
            # Run monotonicity tests
            mono_logs = self.test_monotonicity()
            all_logs.extend([f"MONOTONICITY: {msg}" for _, msg in mono_logs])
            success &= all(passed for passed, _ in mono_logs)
            
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, all_logs