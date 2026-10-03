import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.G = 10.0  # Given weight in Newtons
        self.α = math.radians(30)  # Fixed angle in radians
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Expected keys: F_1, β, F_2 (converting to float if possible)
        self.F1 = float(getattr(solution_instance, 'F_1', self.results.get('F_1', 0.0)))
        self.β_deg = float(getattr(solution_instance, 'β', self.results.get('β', 0.0)))
        self.F2 = float(getattr(solution_instance, 'F_2', self.results.get('F_2', 0.0)))
        
        # Convert β to radians for calculations
        self.β_rad = math.radians(self.β_deg) if self.β_deg != 0 else 0.0
        
        self.logs = []

    def _log(self, message):
        self.logs.append(message)

    def _check_equilibrium(self, F_BO_val, β_val):
        """Check static equilibrium conditions for given β and F_BO."""
        try:
            # Force equilibrium equations (2D, assuming AO to left, BO to right of vertical)
            # x-direction: F_AO*sin(α) = F_BO*sin(β)
            # y-direction: F_AO*cos(α) + F_BO*cos(β) = G
            
            # Solve for F_AO from x-equation
            F_AO_calc = F_BO_val * math.sin(β_val) / math.sin(self.α)
            
            # Check y-equation
            y_sum = F_AO_calc * math.cos(self.α) + F_BO_val * math.cos(β_val)
            
            if math.isclose(y_sum, self.G, rel_tol=1e-2, abs_tol=1e-2):
                return True, F_AO_calc
            else:
                self._log(f"Equilibrium failure: ΣF_y = {y_sum:.3f} ≠ G = {self.G}")
                return False, 0.0
        except Exception as e:
            self._log(f"Equilibrium check error: {str(e)}")
            return False, 0.0

    def test_invariants(self):
        """Verify static equilibrium conditions for both scenarios."""
        try:
            # Scenario 1: β = 30°, F_BO = F1
            if self.β_deg != 0:  # Only check if β was provided
                valid1, F_AO1 = self._check_equilibrium(self.F1, math.radians(30))
                if valid1:
                    self._log("PASS: Equilibrium satisfied for β=30°, F_BO=F1")
                else:
                    self._log("FAIL: Equilibrium violated for β=30°, F_BO=F1")
            
            # Scenario 2: β = β_deg (should be 60°), F_BO = F2
            if self.β_deg != 0 and self.F2 != 0:
                valid2, F_AO2 = self._check_equilibrium(self.F2, self.β_rad)
                if valid2:
                    self._log("PASS: Equilibrium satisfied for minimum tension scenario")
                else:
                    self._log("FAIL: Equilibrium violated for minimum tension scenario")
            
            # Net force should be zero (vector sum)
            # For scenario 1:
            F_AO1_calc = self.F1 * math.sin(math.radians(30)) / math.sin(self.α)
            F_net_x1 = F_AO1_calc * math.sin(self.α) - self.F1 * math.sin(math.radians(30))
            F_net_y1 = F_AO1_calc * math.cos(self.α) + self.F1 * math.cos(math.radians(30)) - self.G
            F_net1 = math.sqrt(F_net_x1**2 + F_net_y1**2)
            
            if math.isclose(F_net1, 0, rel_tol=1e-2, abs_tol=1e-2):
                self._log("PASS: Net force ≈ 0 for β=30° (Newton's 1st Law)")
            else:
                self._log(f"FAIL: Net force = {F_net1:.3f} N ≠ 0 for β=30°")
            
            return True
        except Exception as e:
            self._log(f"test_invariants error: {str(e)}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check F1 formula: F1 = (G/2) / cos(30°)
            expected_F1 = (self.G / 2) / math.cos(math.radians(30))
            if math.isclose(self.F1, expected_F1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"PASS: F1 = {self.F1:.3f} ≈ (G/2)/cos30° = {expected_F1:.3f}")
            else:
                self._log(f"FAIL: F1 = {self.F1:.3f} ≠ (G/2)/cos30° = {expected_F1:.3f}")
            
            # Check β = 60° for minimum tension
            if math.isclose(self.β_deg, 60, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"PASS: β = {self.β_deg}° ≈ 60°")
            else:
                self._log(f"FAIL: β = {self.β_deg}° ≠ 60°")
            
            # Check F2 formula: F2 = G*sin(30°)
            expected_F2 = self.G * math.sin(math.radians(30))
            if math.isclose(self.F2, expected_F2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"PASS: F2 = {self.F2:.3f} ≈ G*sin30° = {expected_F2:.3f}")
            else:
                self._log(f"FAIL: F2 = {self.F2:.3f} ≠ G*sin30° = {expected_F2:.3f}")
            
            # Check that F2 is indeed minimum by verifying equilibrium at β=60°
            if self.β_deg != 0 and self.F2 != 0:
                valid, _ = self._check_equilibrium(self.F2, self.β_rad)
                if valid:
                    self._log("PASS: F2 satisfies equilibrium at β=60°")
                else:
                    self._log("FAIL: F2 does not satisfy equilibrium at β=60°")
            
            return True
        except Exception as e:
            self._log(f"test_trace_consistency error: {str(e)}")
            return False

    def test_monotonicity(self):
        """Perturb β to verify F_BO is minimized at β=60°."""
        try:
            # Function to compute F_BO for given β using equilibrium
            def compute_F_BO(beta_deg):
                beta_rad = math.radians(beta_deg)
                # From Lami's theorem: G/sin(180-α-β) = F_BO/sin(α)
                # So F_BO = G*sin(α)/sin(α+β)
                return self.G * math.sin(self.α) / math.sin(self.α + beta_rad)
            
            # Test values around 60°
            test_angles = [50, 55, 60, 65, 70]
            tensions = [compute_F_BO(angle) for angle in test_angles]
            
            # Find minimum
            min_idx = tensions.index(min(tensions))
            min_angle = test_angles[min_idx]
            
            if math.isclose(min_angle, 60, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"PASS: Minimum tension occurs at β ≈ {min_angle}° (close to 60°)")
            else:
                self._log(f"FAIL: Minimum tension at β = {min_angle}°, not 60°")
            
            # Verify F2 corresponds to minimum
            min_tension = min(tensions)
            if math.isclose(self.F2, min_tension, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"PASS: F2 = {self.F2:.3f} ≈ minimum tension {min_tension:.3f}")
            else:
                self._log(f"FAIL: F2 = {self.F2:.3f} ≠ minimum tension {min_tension:.3f}")
            
            # Check monotonicity: should decrease to 60°, then increase
            decreasing = all(tensions[i] >= tensions[i+1] for i in range(min_idx))
            increasing = all(tensions[i] <= tensions[i+1] for i in range(min_idx, len(tensions)-1))
            
            if decreasing and increasing:
                self._log("PASS: Tension decreases to minimum then increases (convex)")
            else:
                self._log("FAIL: Tension not monotonically changing around minimum")
            
            return True
        except Exception as e:
            self._log(f"test_monotonicity error: {str(e)}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        try:
            self._log("=== Physics Checker Started ===")
            
            # Run all tests
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Check for failures
            failures = [log for log in self.logs if log.startswith("FAIL")]
            
            if not failures:
                success = True
                self._log("=== All checks PASSED ===")
            else:
                success = False
                self._log(f"=== {len(failures)} check(s) FAILED ===")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in run_all_tests: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg)
            return False, self.logs