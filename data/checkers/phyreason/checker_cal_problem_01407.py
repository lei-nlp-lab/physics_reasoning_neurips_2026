import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        All attributes default to None if not found in solution_instance or results.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Extract inputs with safe fallbacks
        self.m = self._safe_get('m')
        self.q = self._safe_get('q')
        self.theta = self._safe_get('theta')
        self.E = self._safe_get('E')
        self.g = self._safe_get('g', default=9.8)  # Default gravitational acceleration
        self.v0 = self._safe_get('v0')
        
        # Extract results with safe fallbacks
        self.mu = self._safe_get('mu') or self._safe_get('μ')
        self.F_f = self._safe_get('F_f')
        self.F_N = self._safe_get('F_N')  # Normal force
        
        # For symbolic computations
        self.m_sym, self.q_sym, self.theta_sym, self.E_sym, self.g_sym, self.v0_sym = sp.symbols('m q theta E g v0', positive=True)
        self.mu_sym = sp.symbols('mu', positive=True)
        
    def _safe_get(self, key: str, default=None):
        """Safely get attribute from solution instance or results dict."""
        try:
            # First try direct attribute
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) and str(val).replace('.', '', 1).isdigit() else val
            # Then try results dict
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(key, None)
                if val is not None:
                    return float(val) if isinstance(val, (int, float, str)) and str(val).replace('.', '', 1).isdigit() else val
        except:
            pass
        return default

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test Newton's First Law and friction definition."""
        logs = []
        success = True
        
        try:
            # Check 1: Newton's First Law - net force along incline should be zero
            # From STEPS ANALYSIS: F_f + mg sinθ = qE cosθ
            if all(v is not None for v in [self.F_f, self.m, self.g, self.theta, self.q, self.E]):
                lhs = self.F_f + self.m * self.g * math.sin(math.radians(self.theta) if self.theta > 10 else self.theta)
                rhs = self.q * self.E * math.cos(math.radians(self.theta) if self.theta > 10 else self.theta)
                if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Newton's 1st Law along incline. LHS={lhs:.3f}, RHS={rhs:.3f}")
                    success = False
                else:
                    logs.append("PASS: Net force along incline is zero (Newton's 1st Law)")
            else:
                logs.append("SKIP: Newton's 1st Law check - missing data")
            
            # Check 2: Normal force decomposition
            # From STEPS ANALYSIS: mg cosθ + qE sinθ = F_N
            if all(v is not None for v in [self.F_N, self.m, self.g, self.theta, self.q, self.E]):
                lhs = self.m * self.g * math.cos(math.radians(self.theta) if self.theta > 10 else self.theta) + \
                      self.q * self.E * math.sin(math.radians(self.theta) if self.theta > 10 else self.theta)
                if not math.isclose(lhs, self.F_N, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Normal force decomposition. Calculated={lhs:.3f}, Reported F_N={self.F_N:.3f}")
                    success = False
                else:
                    logs.append("PASS: Normal force decomposition consistent")
            else:
                logs.append("SKIP: Normal force check - missing data")
            
            # Check 3: Kinetic friction definition F_f = μ * F_N
            if all(v is not None for v in [self.F_f, self.mu, self.F_N]):
                expected_friction = self.mu * self.F_N
                if not math.isclose(self.F_f, expected_friction, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Friction definition F_f = μ*F_N. F_f={self.F_f:.3f}, μ*F_N={expected_friction:.3f}")
                    success = False
                else:
                    logs.append("PASS: Kinetic friction definition satisfied")
            elif self.F_f is not None and self.mu is not None:
                # Try to compute F_N from decomposition if not provided
                if all(v is not None for v in [self.m, self.g, self.theta, self.q, self.E]):
                    F_N_calc = self.m * self.g * math.cos(math.radians(self.theta) if self.theta > 10 else self.theta) + \
                              self.q * self.E * math.sin(math.radians(self.theta) if self.theta > 10 else self.theta)
                    expected_friction = self.mu * F_N_calc
                    if not math.isclose(self.F_f, expected_friction, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: Friction definition with computed F_N. F_f={self.F_f:.3f}, μ*F_N_calc={expected_friction:.3f}")
                        success = False
                    else:
                        logs.append("PASS: Kinetic friction definition satisfied (with computed F_N)")
                else:
                    logs.append("SKIP: Friction definition check - insufficient data")
            else:
                logs.append("SKIP: Friction definition check - missing μ or F_f")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Check if derived formula matches target answer
            # Target formula: μ = (qE cosθ - mg sinθ) / (mg cosθ + qE sinθ)
            if all(v is not None for v in [self.m, self.q, self.theta, self.E, self.g, self.mu]):
                numerator = self.q * self.E * math.cos(math.radians(self.theta) if self.theta > 10 else self.theta) - \
                           self.m * self.g * math.sin(math.radians(self.theta) if self.theta > 10 else self.theta)
                denominator = self.m * self.g * math.cos(math.radians(self.theta) if self.theta > 10 else self.theta) + \
                             self.q * self.E * math.sin(math.radians(self.theta) if self.theta > 10 else self.theta)
                
                # Avoid division by zero
                if abs(denominator) > 1e-12:
                    mu_calc = numerator / denominator
                    if not math.isclose(self.mu, mu_calc, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: Formula consistency. Reported μ={self.mu:.4f}, Calculated={mu_calc:.4f}")
                        success = False
                    else:
                        logs.append(f"PASS: Formula consistency check. μ={self.mu:.4f}")
                else:
                    logs.append("SKIP: Formula consistency - denominator near zero")
            else:
                logs.append("SKIP: Formula consistency check - missing parameters")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb mass m and check trend in μ."""
        logs = []
        success = True
        
        try:
            if all(v is not None for v in [self.m, self.q, self.theta, self.E, self.g, self.mu]):
                # Store original values
                m_orig = self.m
                
                # Perturb mass by +10%
                m_perturbed = self.m * 1.1
                
                # Calculate μ for perturbed mass using the formula
                theta_rad = math.radians(self.theta) if self.theta > 10 else self.theta
                num_orig = self.q * self.E * math.cos(theta_rad) - self.m * self.g * math.sin(theta_rad)
                den_orig = self.m * self.g * math.cos(theta_rad) + self.q * self.E * math.sin(theta_rad)
                
                num_pert = self.q * self.E * math.cos(theta_rad) - m_perturbed * self.g * math.sin(theta_rad)
                den_pert = m_perturbed * self.g * math.cos(theta_rad) + self.q * self.E * math.sin(theta_rad)
                
                if abs(den_orig) > 1e-12 and abs(den_pert) > 1e-12:
                    mu_orig = num_orig / den_orig
                    mu_pert = num_pert / den_pert
                    
                    # Check trend: For fixed q, E, θ, increasing m should decrease μ
                    # because numerator decreases and denominator increases
                    if mu_pert < mu_orig:
                        logs.append(f"PASS: Monotonicity - μ decreases with increased mass (μ_orig={mu_orig:.4f}, μ_pert={mu_pert:.4f})")
                    else:
                        logs.append(f"WARN: Unexpected monotonicity trend - μ not decreasing with increased mass")
                        # This might not be a failure if other parameters dominate
                else:
                    logs.append("SKIP: Monotonicity - denominator near zero")
                
                # Restore original value
                self.m = m_orig
            else:
                logs.append("SKIP: Monotonicity check - missing parameters")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Restore original value if we can
            if 'm_orig' in locals():
                self.m = m_orig
            
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and combined logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"RESULT: {test_name} FAILED")
                else:
                    all_logs.append(f"RESULT: {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== OVERALL RESULT: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs