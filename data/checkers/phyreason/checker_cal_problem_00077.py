import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract all required quantities without direct attribute access.
        """
        self.solution = solution_instance
        # Extract parameters with safe defaults
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 1.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 1.0))
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        
        # Extract results with safe defaults
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', 0.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.v1 = getattr(solution_instance, 'v_1', solution_instance.results.get('v_1', 0.0))
        
        # Backup for monotonicity tests
        self._original_params = {
            'm': self.m,
            'M': self.M,
            'R': self.R,
            'omega': self.omega,
            'g': self.g
        }
        
        self.logs = []

    def _log(self, message: str):
        """Helper to log messages."""
        self.logs.append(message)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants:
        1. Conservation of mechanical energy
        2. Newton's 2nd law for circular motion
        """
        success = True
        
        # 1. Energy conservation check
        try:
            # Initial energy: potential energy of weight M*g*H
            E_initial = self.M * self.g * self.H
            
            # Final energy: kinetic energy of weight + 4 balls
            # v1 = R*omega, v = 2R*omega
            K_weight = 0.5 * self.M * (self.v1 ** 2)
            K_balls = 0.5 * 4 * self.m * (self.v ** 2)
            E_final = K_weight + K_balls
            
            if not math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy conservation violated: E_initial={E_initial}, E_final={E_final}")
                success = False
            else:
                self._log("✓ Energy conservation verified")
        except Exception as e:
            self._log(f"Energy conservation check failed: {e}")
            success = False
        
        # 2. Newton's 2nd law for circular motion (check F formula)
        try:
            # Expected centripetal force component: m*v^2/(2R)
            F_cent = self.m * (self.v ** 2) / (2 * self.R)
            # Expected gravitational component: m*g
            F_grav = self.m * self.g
            # Expected magnitude: sqrt(F_cent^2 + F_grav^2)
            F_expected = math.sqrt(F_cent ** 2 + F_grav ** 2)
            
            if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Force magnitude mismatch: F={self.F}, expected={F_expected}")
                success = False
            else:
                self._log("✓ Newton's 2nd law (circular motion) verified")
        except Exception as e:
            self._log(f"Force check failed: {e}")
            success = False
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        Verify relationships between variables.
        """
        success = True
        
        # Check v = 2R*omega
        try:
            v_expected = 2 * self.R * self.omega
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v mismatch: v={self.v}, expected={v_expected}")
                success = False
            else:
                self._log("✓ v = 2Rω verified")
        except Exception as e:
            self._log(f"v check failed: {e}")
            success = False
        
        # Check v1 = R*omega
        try:
            v1_expected = self.R * self.omega
            if not math.isclose(self.v1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v1 mismatch: v1={self.v1}, expected={v1_expected}")
                success = False
            else:
                self._log("✓ v₁ = Rω verified")
        except Exception as e:
            self._log(f"v1 check failed: {e}")
            success = False
        
        # Check H formula: H = (M+16m)R²ω²/(2Mg)
        try:
            numerator = (self.M + 16 * self.m) * (self.R ** 2) * (self.omega ** 2)
            denominator = 2 * self.M * self.g
            H_expected = numerator / denominator
            
            if not math.isclose(self.H, H_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"H mismatch: H={self.H}, expected={H_expected}")
                success = False
            else:
                self._log("✓ H formula verified")
        except Exception as e:
            self._log(f"H check failed: {e}")
            success = False
        
        # Check F formula: F = m*sqrt(4R²ω⁴ + g²)
        try:
            term = 4 * (self.R ** 2) * (self.omega ** 4) + (self.g ** 2)
            F_expected = self.m * math.sqrt(term)
            
            if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"F mismatch: F={self.F}, expected={F_expected}")
                success = False
            else:
                self._log("✓ F formula verified")
        except Exception as e:
            self._log(f"F check failed: {e}")
            success = False
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb parameters and check if results change in the expected direction.
        Tests sensitivity to mass changes.
        """
        success = True
        
        # Test 1: Increase m should increase H (more inertia)
        try:
            original_m = self.m
            perturbed_m = original_m * 1.01  # 1% increase
            
            # Compute H with perturbed m (keeping other parameters constant)
            numerator = (self.M + 16 * perturbed_m) * (self.R ** 2) * (self.omega ** 2)
            denominator = 2 * self.M * self.g
            H_perturbed = numerator / denominator
            
            if not H_perturbed > self.H:
                self._log(f"Monotonicity fail: H should increase when m increases")
                self._log(f"  H(original)={self.H}, H(perturbed)={H_perturbed}")
                success = False
            else:
                self._log("✓ H increases with m (as expected)")
        except Exception as e:
            self._log(f"Monotonicity test (m→H) failed: {e}")
            success = False
        finally:
            self.m = original_m  # Restore original value
        
        # Test 2: Increase M should decrease H (larger driving force)
        try:
            original_M = self.M
            perturbed_M = original_M * 1.01  # 1% increase
            
            # Compute H with perturbed M
            numerator = (perturbed_M + 16 * self.m) * (self.R ** 2) * (self.omega ** 2)
            denominator = 2 * perturbed_M * self.g
            H_perturbed = numerator / denominator
            
            if not H_perturbed < self.H:
                self._log(f"Monotonicity fail: H should decrease when M increases")
                self._log(f"  H(original)={self.H}, H(perturbed)={H_perturbed}")
                success = False
            else:
                self._log("✓ H decreases with M (as expected)")
        except Exception as e:
            self._log(f"Monotonicity test (M→H) failed: {e}")
            success = False
        finally:
            self.M = original_M  # Restore original value
        
        # Test 3: Increase m should increase F (direct proportionality)
        try:
            original_m = self.m
            perturbed_m = original_m * 1.01
            
            # Compute F with perturbed m
            term = 4 * (self.R ** 2) * (self.omega ** 4) + (self.g ** 2)
            F_perturbed = perturbed_m * math.sqrt(term)
            
            if not F_perturbed > self.F:
                self._log(f"Monotonicity fail: F should increase when m increases")
                self._log(f"  F(original)={self.F}, F(perturbed)={F_perturbed}")
                success = False
            else:
                self._log("✓ F increases with m (as expected)")
        except Exception as e:
            self._log(f"Monotonicity test (m→F) failed: {e}")
            success = False
        finally:
            self.m = original_m  # Restore original value
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success status with logs.
        """
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"Test '{test_name}' crashed: {e}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs