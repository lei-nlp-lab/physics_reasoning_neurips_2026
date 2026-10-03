import math
import sympy as sp
from typing import List, Tuple, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe attribute extraction for physics auditing.
        Uses getattr with fallback to results dict to avoid sandbox errors.
        """
        # Extract inputs with safe defaults
        self.diameter = getattr(solution_instance, 'diameter', 10.0)
        self.height = getattr(solution_instance, 'height', 10.0)
        
        # Extract results with safe defaults (0.0 used for missing keys)
        self.results = getattr(solution_instance, 'results', {})
        self.C = self.results.get('C', 0.0)  # Circumference
        self.s = self.results.get('s', 0.0)  # Shortest path length
        self.x = self.results.get('x', 0.0)  # Displacement magnitude
        
        # Store logs for reporting
        self.logs: List[str] = []
        
    def _log(self, message: str, success: bool) -> None:
        """Helper to append formatted log messages."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """
        Verify geometric invariants:
        1. Circumference calculation: C = π*diameter
        2. Path length: s = sqrt(height² + (C/2)²)  # CORRECTED: half circumference
        3. Displacement: x = sqrt(height² + diameter²)
        """
        success = True
        
        # 1. Circumference invariant
        expected_C = math.pi * self.diameter
        if not self._is_close(self.C, expected_C):
            self._log(f"Circumference mismatch: C={self.C:.3f}, expected π*diameter={expected_C:.3f}", False)
            success = False
        else:
            self._log(f"Circumference correct: C={self.C:.3f} ≈ π*{self.diameter}", True)
        
        # 2. Path length invariant (Pythagorean theorem on unrolled cylinder)
        # CORRECTED: Use half circumference for opposite edge points
        expected_s = math.sqrt(self.height**2 + (self.C/2)**2)
        if not self._is_close(self.s, expected_s):
            self._log(f"Path length mismatch: s={self.s:.3f}, expected √(h²+(C/2)²)={expected_s:.3f}", False)
            success = False
        else:
            self._log(f"Path length correct: s={self.s:.3f} ≈ √({self.height}²+({self.C}/2)²)", True)
        
        # 3. Displacement invariant (3D Pythagorean theorem)
        expected_x = math.sqrt(self.height**2 + self.diameter**2)
        if not self._is_close(self.x, expected_x):
            self._log(f"Displacement mismatch: x={self.x:.3f}, expected √(h²+d²)={expected_x:.3f}", False)
            success = False
        else:
            self._log(f"Displacement correct: x={self.x:.3f} ≈ √({self.height}²+{self.diameter}²)", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        From STEPS ANALYSIS: Check that s ≈ 18.6 cm and x ≈ 14.1 cm.
        """
        success = True
        
        # Check against target answers (with tolerance for rounding)
        if not self._is_close(self.s, 18.6):
            self._log(f"Path length s={self.s:.3f} differs from target 18.6 cm", False)
            success = False
        else:
            self._log(f"Path length matches target: s={self.s:.3f} cm ≈ 18.6 cm", True)
        
        if not self._is_close(self.x, 14.1):
            self._log(f"Displacement x={self.x:.3f} differs from target 14.1 cm", False)
            success = False
        else:
            self._log(f"Displacement matches target: x={self.x:.3f} cm ≈ 14.1 cm", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input (height) and verify expected trends:
        1. Increasing height should increase both s and x.
        2. Path length s should always exceed displacement x.
        """
        original_height = self.height
        success = True
        
        # Test 1: s > x (path longer than displacement)
        if not (self.s > self.x or self._is_close(self.s, self.x)):
            self._log(f"Path length s={self.s:.3f} ≤ displacement x={self.x:.3f} (violates s > x)", False)
            success = False
        else:
            self._log(f"Path length s={self.s:.3f} > displacement x={self.x:.3f}", True)
        
        # Test 2: Perturb height and check monotonicity
        perturbed_height = original_height * 1.1  # +10%
        
        # Recalculate using CORRECTED geometric formulas
        C_original = math.pi * self.diameter
        s_original = math.sqrt(original_height**2 + (C_original/2)**2)  # CORRECTED
        x_original = math.sqrt(original_height**2 + self.diameter**2)
        
        s_perturbed = math.sqrt(perturbed_height**2 + (C_original/2)**2)  # CORRECTED
        x_perturbed = math.sqrt(perturbed_height**2 + self.diameter**2)
        
        # Check monotonic increase
        if not (s_perturbed > s_original or self._is_close(s_perturbed, s_original)):
            self._log(f"Path length non-monotonic: s({perturbed_height})={s_perturbed:.3f} ≤ s({original_height})={s_original:.3f}", False)
            success = False
        else:
            self._log(f"Path length increases with height: s({perturbed_height})={s_perturbed:.3f} > s({original_height})={s_original:.3f}", True)
        
        if not (x_perturbed > x_original or self._is_close(x_perturbed, x_original)):
            self._log(f"Displacement non-monotonic: x({perturbed_height})={x_perturbed:.3f} ≤ x({original_height})={x_original:.3f}", False)
            success = False
        else:
            self._log(f"Displacement increases with height: x({perturbed_height})={x_perturbed:.3f} > x({original_height})={x_original:.3f}", True)
        
        return success
    
    def test_symbolic(self) -> bool:
        """
        Symbolic verification using SymPy.
        Derives CORRECTED formulas for s and x symbolically.
        """
        try:
            d, h = sp.symbols('d h', positive=True)
            C = sp.pi * d
            s_sym = sp.sqrt(h**2 + (C/2)**2)  # CORRECTED: half circumference
            x_sym = sp.sqrt(h**2 + d**2)
            
            # Numerically evaluate with given values
            s_val = float(s_sym.subs({d: self.diameter, h: self.height}))
            x_val = float(x_sym.subs({d: self.diameter, h: self.height}))
            
            success = True
            if not self._is_close(s_val, self.s):
                self._log(f"Symbolic s mismatch: {s_val:.3f} vs computed {self.s:.3f}", False)
                success = False
            else:
                self._log(f"Symbolic s verified: √(h²+(πd/2)²)={s_val:.3f}", True)
            
            if not self._is_close(x_val, self.x):
                self._log(f"Symbolic x mismatch: {x_val:.3f} vs computed {self.x:.3f}", False)
                success = False
            else:
                self._log(f"Symbolic x verified: √(h²+d²)={x_val:.3f}", True)
            
            return success
        except Exception as e:
            self._log(f"Symbolic check failed with error: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all audit tests and return overall success and logs.
        """
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Geometric Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Symbolic Verification", self.test_symbolic)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test crashed: {str(e)}", False)
                overall_success = False
        
        return overall_success, self.logs