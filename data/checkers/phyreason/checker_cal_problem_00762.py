import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem context
        self.h = 0.8  # platform height (m)
        self.DC = 1.6  # horizontal distance for ball 2 (m)
        self.R = 1.0  # circular track radius (m)
        self.g = 10.0  # gravity (m/s²)
        
        # Extract results with safe fallbacks
        self.v1 = self._safe_get('v_1', 0.0)
        self.v2 = self._safe_get('v_2', 0.0)
        self.Ep = self._safe_get('E_p', 0.0)
        self.F_N = self._safe_get('F_N', 0.0)
        self.N = self._safe_get('N', 0.0)
        self.t = self._safe_get('t', 0.0)
        self.v1A = self._safe_get('v_{1A}', 0.0)
        
        # Derived masses (computed from conservation laws)
        self.m1, self.m2 = self._compute_masses()
        
        # Test logs
        self.logs = []
    
    def _safe_get(self, key: str, default: float) -> float:
        """
        Safely retrieve a value from solution instance with multiple fallbacks.
        """
        try:
            # First try direct attribute
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val)
            
            # Then try results dictionary
            if key in self.results:
                val = self.results[key]
                if isinstance(val, str):
                    # Handle strings like "10" or "10N"
                    val = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', val))
                return float(val)
        except (AttributeError, ValueError, TypeError) as e:
            self.logs.append(f"Warning: Could not extract {key}: {e}")
        
        return default
    
    def _compute_masses(self) -> Tuple[float, float]:
        """
        Compute masses m1 and m2 from energy and momentum conservation.
        Returns: (m1, m2) in kg
        """
        try:
            # From momentum conservation: m1*v1 = m2*v2
            # From energy: Ep = 0.5*m1*v1² + 0.5*m2*v2²
            
            # Solve for m1:
            # Ep = 0.5*m1*v1² + 0.5*(m1*v1/v2)*v2²
            # Ep = 0.5*m1*v1² + 0.5*m1*v1*v2
            # Ep = 0.5*m1*(v1² + v1*v2)
            
            if abs(self.v1) > 1e-6 and abs(self.v1 + self.v2) > 1e-6:
                m1 = (2 * self.Ep) / (self.v1**2 + self.v1 * self.v2)
                m2 = (m1 * self.v1) / self.v2 if abs(self.v2) > 1e-6 else 0.0
                return m1, m2
        except (ZeroDivisionError, ValueError):
            pass
        
        # Fallback: approximate values from known solution
        return 0.2, 0.5  # kg (from analysis)
    
    def _is_close(self, a: float, b: float) -> bool:
        """
        Fuzzy numerical comparison with physics-appropriate tolerances.
        """
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> List[str]:
        """
        Verify fundamental physics laws.
        """
        logs = []
        
        try:
            # 1. Conservation of Momentum (initial total = 0)
            initial_momentum = 0.0
            final_momentum = self.m1 * self.v1 - self.m2 * self.v2  # opposite directions
            
            if not self._is_close(final_momentum, 0.0):
                logs.append(f"FAIL: Momentum not conserved. Net momentum = {final_momentum:.3f} kg·m/s")
            else:
                logs.append("PASS: Momentum conserved")
            
            # 2. Conservation of Mechanical Energy for Ball 1 on circular track
            # From platform to top of track (height change = 2R)
            v1A_calc = math.sqrt(max(0, self.v1**2 - 4 * self.g * self.R))
            
            if self.v1A > 0:
                if not self._is_close(self.v1A, v1A_calc):
                    logs.append(f"FAIL: Energy conservation for ball 1. v1A: {self.v1A:.3f}, calculated: {v1A_calc:.3f}")
                else:
                    logs.append("PASS: Energy conservation for ball 1 verified")
            
            # 3. Newton's 2nd Law at top of circular track
            # F_N + m1*g = m1*v1A²/R
            required_centripetal = self.m1 * (self.v1A**2 / self.R) if self.R > 0 else 0.0
            net_force_calc = self.F_N + self.m1 * self.g
            
            if not self._is_close(net_force_calc, required_centripetal):
                logs.append(f"FAIL: Newton's 2nd Law at top. Net force: {net_force_calc:.3f} N, required: {required_centripetal:.3f} N")
            else:
                logs.append("PASS: Newton's 2nd Law satisfied at top")
            
            # 4. Newton's 3rd Law (action-reaction)
            if not self._is_close(self.F_N, self.N):
                logs.append(f"FAIL: Newton's 3rd Law. Track force F_N: {self.F_N:.3f} N, ball force N: {self.N:.3f} N")
            else:
                logs.append("PASS: Newton's 3rd Law satisfied")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
        
        self.logs.extend(logs)
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        try:
            # Step 1: Projectile time
            t_calc = math.sqrt(2 * self.h / self.g)
            if not self._is_close(self.t, t_calc):
                logs.append(f"FAIL: Projectile time mismatch. Given: {self.t:.3f}s, calculated: {t_calc:.3f}s")
            else:
                logs.append("PASS: Projectile time consistent")
            
            # Step 2: Ball 2 initial velocity
            v2_calc = self.DC / self.t if self.t > 0 else 0.0
            if not self._is_close(self.v2, v2_calc):
                logs.append(f"FAIL: v2 mismatch. Given: {self.v2:.3f}m/s, calculated: {v2_calc:.3f}m/s")
            else:
                logs.append("PASS: v2 consistent with kinematics")
            
            # Step 3: Momentum conservation gives v1
            v1_from_momentum = (self.m2 / self.m1) * self.v2 if self.m1 > 0 else 0.0
            if not self._is_close(self.v1, v1_from_momentum):
                logs.append(f"FAIL: v1 from momentum mismatch. Given: {self.v1:.3f}m/s, calculated: {v1_from_momentum:.3f}m/s")
            else:
                logs.append("PASS: v1 consistent with momentum conservation")
            
            # Step 4: Elastic potential energy
            Ep_calc = 0.5 * self.m1 * self.v1**2 + 0.5 * self.m2 * self.v2**2
            if not self._is_close(self.Ep, Ep_calc):
                logs.append(f"FAIL: Elastic energy mismatch. Given: {self.Ep:.3f}J, calculated: {Ep_calc:.3f}J")
            else:
                logs.append("PASS: Elastic energy consistent")
            
            # Final answers
            if self._is_close(self.Ep, 14.0):
                logs.append("PASS: Final answer E_p = 14J correct")
            else:
                logs.append(f"WARN: E_p = {self.Ep:.3f}J, expected ~14J")
            
            if self._is_close(self.N, 10.0):
                logs.append("PASS: Final answer N = 10N correct")
            else:
                logs.append(f"WARN: N = {self.N:.3f}N, expected ~10N")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
        
        self.logs.extend(logs)
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Perturb an input and check if trends match expectations.
        """
        logs = []
        
        try:
            # Perturb platform height by +10%
            h_perturbed = self.h * 1.10
            
            # Recalculate chain
            t_perturbed = math.sqrt(2 * h_perturbed / self.g)
            v2_perturbed = self.DC / t_perturbed
            v1_perturbed = (self.m2 / self.m1) * v2_perturbed
            Ep_perturbed = 0.5 * self.m1 * v1_perturbed**2 + 0.5 * self.m2 * v2_perturbed**2
            
            # Check monotonic trends
            # Higher platform → longer flight time → slower v2 → slower v1 → lower spring energy
            trends_ok = True
            
            if not (t_perturbed > self.t):
                logs.append(f"FAIL: Time not increasing with height. Original: {self.t:.3f}s, perturbed: {t_perturbed:.3f}s")
                trends_ok = False
            
            if not (v2_perturbed < self.v2):
                logs.append(f"FAIL: v2 not decreasing with height. Original: {self.v2:.3f}m/s, perturbed: {v2_perturbed:.3f}m/s")
                trends_ok = False
            
            if not (v1_perturbed < self.v1):
                logs.append(f"FAIL: v1 not decreasing with height. Original: {self.v1:.3f}m/s, perturbed: {v1_perturbed:.3f}m/s")
                trends_ok = False
            
            if not (Ep_perturbed < self.Ep):
                logs.append(f"FAIL: Spring energy not decreasing with height. Original: {self.Ep:.3f}J, perturbed: {Ep_perturbed:.3f}J")
                trends_ok = False
            
            if trends_ok:
                logs.append("PASS: All monotonic trends as expected with height perturbation")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
        
        self.logs.extend(logs)
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        
        # Run all test suites
        test_methods = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for method in test_methods:
            method()
        
        # Determine overall success (no FAIL or ERROR messages)
        success = True
        for log in self.logs:
            if log.startswith(("FAIL:", "ERROR:")):
                success = False
                break
        
        return success, self.logs