import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using robust access patterns."""
        # Extract input parameters with safe fallbacks
        self.E = self._safe_get(solution_instance, 'E')
        self.B = self._safe_get(solution_instance, 'B')
        self.L = self._safe_get(solution_instance, 'L')
        self.g = self._safe_get(solution_instance, 'g')
        self.theta = self._safe_get(solution_instance, 'theta')
        
        # Extract results with safe fallbacks
        self.fracqm = self._safe_get(solution_instance, 'fracqm')  # q/m
        self.v = self._safe_get(solution_instance, 'v')
        self.Bprime = self._safe_get(solution_instance, "B'")  # B'
        
        # Helper variables for computations
        self.q = sp.Symbol('q')
        self.m = sp.Symbol('m')
    
    def _safe_get(self, obj, key):
        """Safely get attribute from instance or results dict with fallback."""
        try:
            # Try direct attribute
            val = getattr(obj, key, None)
            if val is not None:
                return val
            # Try results dict
            if hasattr(obj, 'results'):
                val = obj.results.get(key, None)
                if val is not None:
                    return val
            # Return None if not found
            return None
        except:
            return None
    
    def _is_close(self, a, b):
        """Robust numerical comparison with physics-appropriate tolerances."""
        try:
            # Convert to float if possible
            a_val = float(a) if not isinstance(a, sp.Basic) else float(a.evalf())
            b_val = float(b) if not isinstance(b, sp.Basic) else float(b.evalf())
            return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except:
            # Symbolic comparison fallback
            try:
                diff = sp.simplify(a - b)
                return diff == 0 or abs(diff.evalf()) < 1e-2
            except:
                return False
    
    def test_invariants(self):
        """Test fundamental physics invariants from KB."""
        logs = []
        success = True
        
        # 1. Test force equilibrium (only electric field case)
        if self.E is not None and self.g is not None and self.fracqm is not None:
            # From equilibrium: qE = mg => q/m = g/E
            expected_fracqm = self.g / self.E
            if not self._is_close(self.fracqm, expected_fracqm):
                logs.append(f"FAIL: Force equilibrium violated. q/m = {self.fracqm}, expected g/E = {expected_fracqm}")
                success = False
            else:
                logs.append(f"PASS: Force equilibrium q/m = g/E holds")
        
        # 2. Test Lorentz force as centripetal force
        if all(x is not None for x in [self.v, self.B, self.fracqm]):
            # From qvB = mv²/r => r = v/((q/m)*B)
            # Also q/m = v/(r*B) => r = v/((q/m)*B)
            r_from_lorentz = self.v / (self.fracqm * self.B)
            
            # From geometry: L = r*sinθ => r = L/sinθ
            if self.L is not None and self.theta is not None:
                r_from_geometry = self.L / sp.sin(self.theta)
                
                if not self._is_close(r_from_lorentz, r_from_geometry):
                    logs.append(f"FAIL: Lorentz/geometry mismatch. r(Lorentz)={r_from_lorentz}, r(geometry)={r_from_geometry}")
                    success = False
                else:
                    logs.append(f"PASS: Lorentz force matches geometric constraints")
        
        # 3. Test velocity expression from combined constraints
        if all(x is not None for x in [self.v, self.L, self.g, self.B, self.E, self.theta]):
            expected_v = (self.L * self.g * self.B) / (self.E * sp.sin(self.theta))
            if not self._is_close(self.v, expected_v):
                logs.append(f"FAIL: Velocity mismatch. v={self.v}, expected={expected_v}")
                success = False
            else:
                logs.append(f"PASS: Velocity expression v = LgB/(E sinθ) holds")
        
        # 4. Test B' relationship
        if all(x is not None for x in [self.Bprime, self.B, self.theta]):
            expected_Bprime = self.B / sp.sin(self.theta)
            if not self._is_close(self.Bprime, expected_Bprime):
                logs.append(f"FAIL: B' mismatch. B'={self.Bprime}, expected B/sinθ={expected_Bprime}")
                success = False
            else:
                logs.append(f"PASS: B' = B/sinθ relationship holds")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        # Build symbolic relationships from steps
        q, m, v, B, L, θ, E, g = sp.symbols('q m v B L θ E g')
        
        # From step 1: q/m = g/E
        step1_eq = sp.Eq(q/m, g/E)
        
        # From step 2 & 3: qvB = mv²/r and L = r sinθ => v = (q/m)*B*L/sinθ
        step2_3_eq = sp.Eq(v, (q/m) * B * L / sp.sin(θ))
        
        # Substitute step1 into step2_3
        v_expected = (g/E) * B * L / sp.sin(θ)
        
        if all(x is not None for x in [self.v, self.L, self.g, self.B, self.E, self.theta]):
            v_calc = (self.g/self.E) * self.B * self.L / sp.sin(self.theta)
            if not self._is_close(self.v, v_calc):
                logs.append(f"FAIL: Trace inconsistency. v from q/m substitution = {v_calc}, given v = {self.v}")
                success = False
            else:
                logs.append(f"PASS: Velocity consistent through substitution chain")
        
        # Check B' derivation chain
        if all(x is not None for x in [self.Bprime, self.v, self.L, self.fracqm]):
            # From tangent condition: qvB' = mv²/L => B' = (m/q)*(v/L)
            Bprime_from_v = (1/self.fracqm) * (self.v / self.L)
            
            # Also B' = B/sinθ from geometry
            if self.B is not None and self.theta is not None:
                Bprime_from_B = self.B / sp.sin(self.theta)
                
                # Check all three match
                checks = [
                    (self.Bprime, Bprime_from_v, "B' vs from v/L"),
                    (self.Bprime, Bprime_from_B, "B' vs B/sinθ"),
                    (Bprime_from_v, Bprime_from_B, "v/L vs B/sinθ")
                ]
                
                for a, b, desc in checks:
                    if not self._is_close(a, b):
                        logs.append(f"FAIL: Trace inconsistency in {desc}: {a} ≠ {b}")
                        success = False
                
                if success:
                    logs.append("PASS: B' derivation chain consistent")
        
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        success = True
        
        # Store original values
        orig_values = {
            'B': self.B,
            'E': self.E,
            'L': self.L,
            'theta': self.theta,
            'g': self.g
        }
        
        # Test 1: Increase B by 10%, check v increases proportionally
        if all(x is not None for x in [self.B, self.E, self.L, self.g, self.theta, self.v]):
            B_perturbed = self.B * 1.1
            v_expected_perturbed = (self.L * self.g * B_perturbed) / (self.E * sp.sin(self.theta))
            v_ratio = v_expected_perturbed / self.v
            
            if not self._is_close(v_ratio, 1.1):
                logs.append(f"FAIL: v not proportional to B. Ratio = {v_ratio}, expected 1.1")
                success = False
            else:
                logs.append(f"PASS: v ∝ B monotonicity holds")
        
        # Test 2: Increase E, check v decreases
        if all(x is not None for x in [self.B, self.E, self.L, self.g, self.theta, self.v]):
            E_perturbed = self.E * 1.1
            v_expected_perturbed = (self.L * self.g * self.B) / (E_perturbed * sp.sin(self.theta))
            v_ratio = v_expected_perturbed / self.v
            
            if not self._is_close(v_ratio, 1/1.1):  # v ∝ 1/E
                logs.append(f"FAIL: v not inversely proportional to E. Ratio = {v_ratio}, expected {1/1.1}")
                success = False
            else:
                logs.append(f"PASS: v ∝ 1/E monotonicity holds")
        
        # Test 3: Increase θ, check v and B' decrease
        if all(x is not None for x in [self.B, self.E, self.L, self.g, self.theta, self.v, self.Bprime]):
            # Small perturbation to avoid singularities
            if float(self.theta.evalf() if hasattr(self.theta, 'evalf') else self.theta) > 0.2:
                theta_perturbed = self.theta * 0.9  # Decrease slightly
                
                v_expected_perturbed = (self.L * self.g * self.B) / (self.E * sp.sin(theta_perturbed))
                Bprime_expected_perturbed = self.B / sp.sin(theta_perturbed)
                
                # Both should increase when θ decreases
                if not (v_expected_perturbed > self.v and Bprime_expected_perturbed > self.Bprime):
                    logs.append("FAIL: v and B' monotonicity w.r.t θ violated")
                    success = False
                else:
                    logs.append("PASS: v and B' decrease with increasing θ")
        
        # Test 4: Check B' ∝ 1/sinθ directly
        if all(x is not None for x in [self.Bprime, self.B, self.theta]):
            # Compute expected from different θ values
            test_angles = [self.theta * 0.8, self.theta * 1.2]
            for test_theta in test_angles:
                Bprime_test = self.B / sp.sin(test_theta)
                # Check ordering is consistent
                if test_theta < self.theta:
                    if not (Bprime_test > self.Bprime):
                        logs.append(f"FAIL: B' not decreasing with increasing θ at θ={test_theta}")
                        success = False
                        break
                else:
                    if not (Bprime_test < self.Bprime):
                        logs.append(f"FAIL: B' not increasing with decreasing θ at θ={test_theta}")
                        success = False
                        break
            else:
                logs.append("PASS: B' ∝ 1/sinθ monotonicity holds")
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
                all_logs.append("")  # Empty line for readability
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs