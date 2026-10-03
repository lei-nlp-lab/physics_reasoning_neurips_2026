import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Uses getattr with fallback to solution_instance.results dictionary.
        """
        self.solution = solution_instance
        
        # Extract ALL variables from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (should be available in results)
        self.B = self.results.get('B', None)
        self.L = self.results.get('L', None)
        self.R = self.results.get('R', None)
        self.v = self.results.get('v', None)
        
        # Extract target variables
        self.P = self.results.get('P', None)
        self.R_ad = self.results.get('R_ad', None)
        self.F = self.results.get('F', None)
        self.t = self.results.get('t', None)
        self.Q = self.results.get('Q', None)
        self.I = self.results.get('I', None)
        self.F_A = self.results.get('F_A', None)
        self.E = self.results.get('E', None)
        
        self.logs = []
        
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        self.logs.append((success, message))
        
    def _check_numerical(self, actual, expected, name: str, rel_tol=1e-2, abs_tol=1e-2) -> bool:
        """Check if two values are numerically close."""
        if actual is None or expected is None:
            self._log(f"{name}: Missing value (actual={actual}, expected={expected})", False)
            return False
            
        try:
            # Convert to float if possible
            a = float(actual)
            e = float(expected)
            if math.isclose(a, e, rel_tol=rel_tol, abs_tol=abs_tol):
                self._log(f"{name}: ✓ {a:.4g} ≈ {e:.4g}", True)
                return True
            else:
                self._log(f"{name}: ✗ {a:.4g} ≠ {e:.4g} (tolerance: rel={rel_tol}, abs={abs_tol})", False)
                return False
        except (TypeError, ValueError) as err:
            self._log(f"{name}: Cannot compare {actual} and {expected} ({err})", False)
            return False
    
    def test_invariants(self) -> bool:
        """Test fundamental physics laws: Newton's 1st, Energy Conservation."""
        success = True
        
        # 1. Newton's First Law: Constant velocity => Net force = 0
        # From STEPS ANALYSIS: F = F_A (pulling force equals Ampere force)
        if self.F is not None and self.F_A is not None:
            if not self._check_numerical(self.F, self.F_A, "Newton's 1st Law (F = F_A)"):
                success = False
        else:
            self._log("Newton's 1st Law: Missing F or F_A", False)
            success = False
        
        # 2. Energy Conservation: Work by pulling force = TOTAL Joule heat in entire circuit
        # W = P * t, Q_total = I² * R * t (not just side ab)
        if all(x is not None for x in [self.P, self.t, self.I, self.R]):
            W = self.P * self.t  # Work done by pulling force
            Q_total = (self.I ** 2) * self.R * self.t  # Total Joule heat in entire frame
            if not self._check_numerical(W, Q_total, "Energy Conservation (W = I²R*t)"):
                success = False
        else:
            self._log("Energy Conservation: Missing P, t, I, or R", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final answers."""
        success = True
        
        # 1. Check E = B * L * v
        if all(x is not None for x in [self.E, self.B, self.L, self.v]):
            E_expected = self.B * self.L * self.v
            if not self._check_numerical(self.E, E_expected, "E = B*L*v"):
                success = False
        else:
            self._log("E = B*L*v: Missing values", False)
            success = False
        
        # 2. Check I = E / R
        if all(x is not None for x in [self.I, self.E, self.R]):
            I_expected = self.E / self.R
            if not self._check_numerical(self.I, I_expected, "I = E/R"):
                success = False
        else:
            self._log("I = E/R: Missing values", False)
            success = False
        
        # 3. Check F_A = B * I * L
        if all(x is not None for x in [self.F_A, self.B, self.I, self.L]):
            F_A_expected = self.B * self.I * self.L
            if not self._check_numerical(self.F_A, F_A_expected, "F_A = B*I*L"):
                success = False
        else:
            self._log("F_A = B*I*L: Missing values", False)
            success = False
        
        # 4. Check P = F * v = F_A * v
        if all(x is not None for x in [self.P, self.F, self.v]):
            P_expected1 = self.F * self.v
            if not self._check_numerical(self.P, P_expected1, "P = F*v"):
                success = False
        
        # 5. Check P = B² * L² * v² / R (from step 6)
        if all(x is not None for x in [self.P, self.B, self.L, self.v, self.R]):
            P_expected2 = (self.B**2 * self.L**2 * self.v**2) / self.R
            if not self._check_numerical(self.P, P_expected2, "P = B²L²v²/R"):
                success = False
        
        # 6. Check t = L / v
        if all(x is not None for x in [self.t, self.L, self.v]):
            t_expected = self.L / self.v
            if not self._check_numerical(self.t, t_expected, "t = L/v"):
                success = False
        
        # 7. Check R_ad = R / 4
        if all(x is not None for x in [self.R_ad, self.R]):
            R_ad_expected = self.R / 4
            if not self._check_numerical(self.R_ad, R_ad_expected, "R_ad = R/4"):
                success = False
        
        # 8. Check Q = I² * R_ad * t (Joule heat in side ab)
        if all(x is not None for x in [self.Q, self.I, self.R_ad, self.t]):
            Q_expected1 = (self.I**2) * self.R_ad * self.t
            if not self._check_numerical(self.Q, Q_expected1, "Q = I² * R_ad * t"):
                success = False
        
        # 9. Check Q = B² * L³ * v / (4 * R) (from step 11)
        if all(x is not None for x in [self.Q, self.B, self.L, self.v, self.R]):
            Q_expected2 = (self.B**2 * self.L**3 * self.v) / (4 * self.R)
            if not self._check_numerical(self.Q, Q_expected2, "Q = B²L³v/(4R)"):
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check trends.
        Uses only numeric values, no SymPy symbols.
        """
        success = True
        
        # Store original values
        orig_values = {
            'B': self.B, 'L': self.L, 'R': self.R, 'v': self.v,
            'P': self.P, 'Q': self.Q, 'E': self.E
        }
        
        # Check that all required values are present
        for key in ['B', 'L', 'R', 'v', 'P', 'Q', 'E']:
            if orig_values[key] is None:
                self._log(f"Monotonicity: Missing {key}", False)
                return False
        
        # Helper to calculate expected changes
        def calculate_derivatives(param, delta_factor):
            """Calculate expected multiplicative factors for E, P, Q when param changes."""
            if param == 'B':
                # E ∝ B, P ∝ B², Q ∝ B²
                return delta_factor, delta_factor**2, delta_factor**2
            elif param == 'L':
                # E ∝ L, P ∝ L², Q ∝ L³
                return delta_factor, delta_factor**2, delta_factor**3
            elif param == 'v':
                # E ∝ v, P ∝ v², Q ∝ v
                return delta_factor, delta_factor**2, delta_factor
            elif param == 'R':
                # E independent of R, P ∝ 1/R, Q ∝ 1/R
                return 1.0, 1.0/delta_factor, 1.0/delta_factor
            return 1.0, 1.0, 1.0
        
        # Test each parameter
        test_params = [('B', 1.01), ('L', 1.01), ('v', 1.01), ('R', 1.01)]
        
        for param_name, factor in test_params:
            # Perturb the parameter
            new_param_val = orig_values[param_name] * factor
            
            # Recalculate using formulas with numeric values
            B_val = new_param_val if param_name == 'B' else orig_values['B']
            L_val = new_param_val if param_name == 'L' else orig_values['L']
            R_val = new_param_val if param_name == 'R' else orig_values['R']
            v_val = new_param_val if param_name == 'v' else orig_values['v']
            
            # Calculate expected values after perturbation
            E_new = B_val * L_val * v_val
            P_new = (B_val**2 * L_val**2 * v_val**2) / R_val
            Q_new = (B_val**2 * L_val**3 * v_val) / (4 * R_val)  # Heat in side ab
            
            # Get original values
            E_orig = orig_values['E']
            P_orig = orig_values['P']
            Q_orig = orig_values['Q']
            
            # Calculate expected factors
            exp_E_factor, exp_P_factor, exp_Q_factor = calculate_derivatives(param_name, factor)
            
            # Check actual vs expected ratios
            if E_orig != 0:
                E_ratio = E_new / E_orig
                if not math.isclose(E_ratio, exp_E_factor, rel_tol=1e-2):
                    self._log(f"Monotonicity {param_name}: E ratio {E_ratio:.4g} ≠ expected {exp_E_factor:.4g}", False)
                    success = False
            
            if P_orig != 0:
                P_ratio = P_new / P_orig
                if not math.isclose(P_ratio, exp_P_factor, rel_tol=1e-2):
                    self._log(f"Monotonicity {param_name}: P ratio {P_ratio:.4g} ≠ expected {exp_P_factor:.4g}", False)
                    success = False
            
            if Q_orig != 0:
                Q_ratio = Q_new / Q_orig
                if not math.isclose(Q_ratio, exp_Q_factor, rel_tol=1e-2):
                    self._log(f"Monotonicity {param_name}: Q ratio {Q_ratio:.4g} ≠ expected {exp_Q_factor:.4g}", False)
                    success = False
        
        if success:
            self._log("Monotonicity tests: All trends as expected", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return (success, logs)."""
        self.logs = []
        
        # Run tests
        test_functions = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in test_functions:
            self._log(f"=== {test_name} ===", True)
            if not test_func():
                overall_success = False
        
        # Summary
        if overall_success:
            self._log("\n✅ All tests passed!", True)
        else:
            self._log("\n❌ Some tests failed!", False)
        
        # Convert logs to readable format
        log_messages = []
        for success, message in self.logs:
            prefix = "✓" if success else "✗"
            log_messages.append(f"{prefix} {message}")
        
        return overall_success, log_messages