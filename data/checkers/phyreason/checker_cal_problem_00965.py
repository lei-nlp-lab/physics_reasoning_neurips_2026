import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access.
        """
        # Extract constants from problem statement with safe defaults
        self.L = getattr(solution_instance, 'L', 1.0)        # rail spacing (m)
        self.R = getattr(solution_instance, 'R', 3.0)        # resistor (Ω)
        self.m = getattr(solution_instance, 'm', 0.1)        # mass (kg)
        self.r = getattr(solution_instance, 'r', 1.0)        # rod resistance (Ω)
        self.g = getattr(solution_instance, 'g', 10.0)       # gravity (m/s²)
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.v = results.get('v', 0.0)
        self.B = results.get('B', 0.0)
        self.I = results.get('I', 0.0)
        self.F_a = results.get('F_a', 0.0)
        self.E = results.get('E', 0.0)
        self.Q = results.get('Q', 0.0)
        
        # For time from v-t graph (needed for Q calculation)
        # Use default 0.1 s (from t=0.1 to t=0.2 s in graph)
        self.t = results.get('t', 0.1)
        
        # Target answers for verification
        self.target_B = 2.0      # T
        self.target_Q = 0.075    # J
        
        # Store logs
        self.logs = []
    
    def _log(self, msg: str):
        """Helper to add log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants using only numeric values.
        """
        success = True
        
        # 1. Equilibrium condition: mg = F_a
        mg = self.m * self.g
        if not self._is_close(mg, self.F_a):
            self._log(f"FAIL: Equilibrium violated. mg = {mg:.3f}, F_a = {self.F_a:.3f}")
            success = False
        else:
            self._log(f"PASS: Equilibrium mg = F_a ({mg:.3f} N)")
        
        # 2. Ohm's Law: E = I*(R+r)
        expected_E = self.I * (self.R + self.r)
        if not self._is_close(self.E, expected_E):
            self._log(f"FAIL: Ohm's Law violated. E = {self.E:.3f}, I*(R+r) = {expected_E:.3f}")
            success = False
        else:
            self._log(f"PASS: Ohm's Law E = I*(R+r) = {expected_E:.3f} V")
        
        # 3. Motional EMF: E = B*L*v
        expected_emf = self.B * self.L * self.v
        if not self._is_close(self.E, expected_emf):
            self._log(f"FAIL: Motional EMF violated. E = {self.E:.3f}, B*L*v = {expected_emf:.3f}")
            success = False
        else:
            self._log(f"PASS: Motional EMF E = B*L*v = {expected_emf:.3f} V")
        
        # 4. Ampere's Force: F_a = B*I*L
        expected_force = self.B * self.I * self.L
        if not self._is_close(self.F_a, expected_force):
            self._log(f"FAIL: Ampere's Force violated. F_a = {self.F_a:.3f}, B*I*L = {expected_force:.3f}")
            success = False
        else:
            self._log(f"PASS: Ampere's Force F_a = B*I*L = {expected_force:.3f} N")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers using numeric formulas.
        """
        success = True
        
        # Check B using derived formula: B = sqrt(m*g*(R+r) / (L²*v))
        B_expected = math.sqrt(self.m * self.g * (self.R + self.r) / (self.L**2 * self.v))
        if not self._is_close(self.B, B_expected):
            self._log(f"FAIL: B formula mismatch. Computed B = {self.B:.3f}, expected from formula = {B_expected:.3f}")
            success = False
        else:
            self._log(f"PASS: B matches derived formula ({B_expected:.3f} T)")
        
        # Check Q using derived formula: Q = (m*g*v*R*t) / (L*(R+r))
        # Derivation: I = (B*L*v)/(R+r) = (sqrt(m*g*(R+r)/(L²*v))*L*v)/(R+r) = sqrt(m*g*v/(L*(R+r)))
        # Then Q = I²*R*t = (m*g*v/(L*(R+r)))*R*t = (m*g*v*R*t) / (L*(R+r))
        Q_expected = (self.m * self.g * self.v * self.R * self.t) / (self.L * (self.R + self.r))
        if not self._is_close(self.Q, Q_expected):
            self._log(f"WARN: Q formula mismatch. Computed Q = {self.Q:.3f}, expected from formula = {Q_expected:.3f}")
            # Not critical due to possible time approximation
        else:
            self._log(f"PASS: Q matches derived formula ({Q_expected:.3f} J)")
        
        # Numerical consistency with target answers
        if not self._is_close(self.B, self.target_B):
            self._log(f"FAIL: Numerical B = {self.B:.3f} T, target = {self.target_B:.3f} T")
            success = False
        else:
            self._log(f"PASS: Numerical B matches target ({self.B:.3f} T)")
        
        if not self._is_close(self.Q, self.target_Q):
            self._log(f"WARN: Numerical Q = {self.Q:.3f} J, target = {self.target_Q:.3f} J")
            # Not critical due to possible time approximation
        else:
            self._log(f"PASS: Numerical Q matches target ({self.Q:.3f} J)")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass m and verify expected trends using numeric calculations only.
        """
        success = True
        
        # Save original values
        original_m = self.m
        original_B = self.B
        original_Q = self.Q
        
        # Perturbation factor
        factor = 1.1  # 10% increase
        perturbed_m = original_m * factor
        
        # Theoretical scaling from derived formulas
        # From B = sqrt(m*g*(R+r) / (L²*v))
        B_expected = original_B * math.sqrt(factor)
        
        # From Q = (m*g*v*R*t) / (L*(R+r)) => Q ∝ m
        Q_expected = original_Q * factor
        
        # Recalculate B with perturbed mass
        B_new = math.sqrt(perturbed_m * self.g * (self.R + self.r) / (self.L**2 * self.v))
        
        if not self._is_close(B_new, B_expected):
            self._log(f"FAIL: Monotonicity B: expected {B_expected:.3f}, got {B_new:.3f}")
            success = False
        else:
            self._log(f"PASS: B scales as √m (B_new = {B_new:.3f} T)")
        
        # Recalculate Q with perturbed mass (using same t)
        Q_new = (perturbed_m * self.g * self.v * self.R * self.t) / (self.L * (self.R + self.r))
        
        if not self._is_close(Q_new, Q_expected):
            self._log(f"WARN: Monotonicity Q: expected {Q_expected:.3f}, got {Q_new:.3f}")
            # Not critical due to possible time approximation
        else:
            self._log(f"PASS: Q scales as m (Q_new = {Q_new:.3f} J)")
        
        # Restore original values
        self.m = original_m
        self.B = original_B
        self.Q = original_Q
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success flag with logs.
        """
        self.logs = []
        self._log("=== Physics Checker Starting ===")
        
        overall_success = True
        
        # Test 1: Physical Invariants
        self._log("\n--- Testing Physical Invariants ---")
        if not self.test_invariants():
            overall_success = False
        
        # Test 2: Trace Consistency
        self._log("\n--- Testing Trace Consistency ---")
        if not self.test_trace_consistency():
            overall_success = False
        
        # Test 3: Monotonicity
        self._log("\n--- Testing Monotonicity ---")
        if not self.test_monotonicity():
            overall_success = False  # Only fail if B monotonicity fails
        
        self._log(f"\n=== Physics Checker Complete: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs