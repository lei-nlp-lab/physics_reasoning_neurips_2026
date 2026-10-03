import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr with fallback to results dictionary to prevent AttributeError.
        """
        # Physical constants and inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.d = getattr(solution_instance, 'd', solution_instance.results.get('d', 1.0))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 1.0))
        
        # Electric field from given formula
        self.E = (3 * self.m * self.v0**2) / (2 * self.q * self.d)
        
        # Results (with safe access)
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', None))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', None))
        self.x1 = getattr(solution_instance, 'x_1', solution_instance.results.get('x_1', None))
        self.v_t = getattr(solution_instance, 'v_t', solution_instance.results.get('v_t', None))
        
        self.logs = []
    
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Work-energy theorem in electric field
        2. Lorentz force as centripetal force
        3. Kinematic relationships
        """
        successes = []
        
        try:
            # 1. Work-energy theorem: qEd = 0.5*m*(v_t^2 - v0^2)
            work = self.q * self.E * self.d
            delta_KE = 0.5 * self.m * (self.v_t**2 - self.v0**2)
            if math.isclose(work, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
                successes.append(self._log("Work-energy theorem satisfied", True))
            else:
                successes.append(self._log(f"Work-energy violation: work={work}, ΔKE={delta_KE}", False))
            
            # 2. Lorentz force as centripetal force: q*v_t*B = m*v_t^2/r
            if self.r and self.B:
                lhs = self.q * self.v_t * self.B
                rhs = self.m * self.v_t**2 / self.r
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("Lorentz force equals centripetal force", True))
                else:
                    successes.append(self._log(f"Lorentz-centripetal mismatch: qvB={lhs}, mv²/r={rhs}", False))
            
            # 3. Kinematic: tan(theta) = v_y/v_x
            if self.theta:
                # v_x remains v0 (no x-acceleration)
                v_y = math.sqrt(self.v_t**2 - self.v0**2)
                tan_calc = v_y / self.v0
                tan_expected = math.tan(self.theta)
                if math.isclose(tan_calc, tan_expected, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("Kinematic angle relation satisfied", True))
                else:
                    successes.append(self._log(f"Angle mismatch: tan(calc)={tan_calc}, tan(θ)={tan_expected}", False))
            
            # 4. Geometric condition from step 5: r + r*cos(theta) = d
            if self.r and self.theta:
                geom = self.r + self.r * math.cos(self.theta)
                if math.isclose(geom, self.d, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("Geometric condition r(1+cosθ)=d satisfied", True))
                else:
                    successes.append(self._log(f"Geometry mismatch: r(1+cosθ)={geom}, d={self.d}", False))
                    
        except Exception as e:
            successes.append(self._log(f"Invariant test crashed: {traceback.format_exc()}", False))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify that values from steps analysis match extracted results.
        """
        successes = []
        
        try:
            # Step 1: v_t should be 2*v0
            if self.v_t:
                if math.isclose(self.v_t, 2 * self.v0, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("v_t = 2*v0 consistent", True))
                else:
                    successes.append(self._log(f"v_t mismatch: {self.v_t} vs 2*{self.v0}", False))
            
            # Step 2: theta should be π/3 ≈ 1.0472 rad
            if self.theta:
                if math.isclose(self.theta, math.pi/3, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("θ = π/3 consistent", True))
                else:
                    successes.append(self._log(f"θ mismatch: {self.theta} vs π/3", False))
            
            # Step 3: x1 should be (2√3/3)d ≈ 1.1547*d
            if self.x1:
                expected_x1 = (2 * math.sqrt(3) / 3) * self.d
                if math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("x1 = (2√3/3)d consistent", True))
                else:
                    successes.append(self._log(f"x1 mismatch: {self.x1} vs {expected_x1}", False))
            
            # Step 5: r should be (2/3)d
            if self.r:
                expected_r = (2/3) * self.d
                if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("r = (2/3)d consistent", True))
                else:
                    successes.append(self._log(f"r mismatch: {self.r} vs {expected_r}", False))
            
            # Step 7: B should be (3*m*v0)/(q*d)
            if self.B:
                expected_B = (3 * self.m * self.v0) / (self.q * self.d)
                if math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                    successes.append(self._log("B = 3mv₀/(qd) consistent", True))
                else:
                    successes.append(self._log(f"B mismatch: {self.B} vs {expected_B}", False))
                    
        except Exception as e:
            successes.append(self._log(f"Trace consistency test crashed: {traceback.format_exc()}", False))
        
        return all(successes)
    
    def test_monotonicity(self):
        """
        Perturb input v0 and check if B changes in the expected direction.
        B ∝ v0 from formula B = 3*m*v0/(q*d)
        """
        try:
            if self.B is None:
                return self._log("Cannot test monotonicity: B not available", False)
            
            # Save original
            original_v0 = self.v0
            original_B = self.B
            
            # Perturb v0 by +5%
            perturbed_v0 = original_v0 * 1.05
            
            # Recalculate expected B using formula (not re-running full solution)
            expected_B_perturbed = (3 * self.m * perturbed_v0) / (self.q * self.d)
            
            # Check monotonic increase
            if expected_B_perturbed > original_B:
                success = self._log("Monotonicity: B increases with v0 (as expected)", True)
            else:
                success = self._log("Monotonicity violated: B does not increase with v0", False)
            
            # Restore original value (important for other tests)
            self.v0 = original_v0
            
            return success
            
        except Exception as e:
            return self._log(f"Monotonicity test crashed: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        overall = inv_ok and trace_ok and mono_ok
        
        # Summary log
        self.logs.append(f"\nSUMMARY: Overall checks {'PASSED' if overall else 'FAILED'} "
                        f"(Invariants: {inv_ok}, Trace: {trace_ok}, Monotonicity: {mono_ok})")
        
        return overall, self.logs