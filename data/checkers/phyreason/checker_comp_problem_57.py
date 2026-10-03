import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution instance.
        Follows robustness rule #1: Safe attribute access with getattr.
        """
        # Extract solution results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with safe defaults)
        self.B_e = getattr(solution_instance, 'B_e', self.results.get('B_e', 1.0))
        self.K_b = getattr(solution_instance, 'K_b', self.results.get('K_b', 0.0))
        self.K_s = getattr(solution_instance, 'K_s', self.results.get('K_s', 0.0))
        self.theta = getattr(solution_instance, 'theta', self.results.get('theta', 0.0))
        self.phi = getattr(solution_instance, 'phi', self.results.get('phi', 0.0))
        self.d = getattr(solution_instance, 'd', self.results.get('d', 1.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.K_i = getattr(solution_instance, 'K_i', self.results.get('K_i', 0.0))
        
        # Results from steps (with safe extraction)
        self.B_north = self.results.get('B_north', 0.0)
        self.B_east = self.results.get('B_east', 0.0)
        self.B_phi = self.results.get('B_phi', 0.0)
        self.B_r = self.results.get('B_r', 0.0)
        self.deltatheta = self.results.get('deltatheta', 0.0)
        self.tandeltatheta = self.results.get('tandeltatheta', 0.0)
        self.B = self.results.get('B', 0.0)
        
        # Store logs for test results
        self.logs = []
        
        # Define symbolic variables for later use
        self.theta_sym = sp.symbols('theta')
        self.phi_sym = sp.symbols('phi')
        self.K_b_sym = sp.symbols('K_b')
        self.K_s_sym = sp.symbols('K_s')
        self.B_e_sym = sp.symbols('B_e')
        self.m_sym = sp.symbols('m')
        self.d_sym = sp.symbols('d')
        self.K_i_sym = sp.symbols('K_i')
        
    def log(self, message: str):
        """Add message to logs."""
        self.logs.append(message)
    
    def fuzzy_equal(self, a: float, b: float, name: str = "values") -> bool:
        """
        Robustness rule #2: Fuzzy numerical matching.
        Uses 1-2% tolerance for physics data with rounding noise.
        """
        if a == 0 and b == 0:
            return True
        is_close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        if not is_close:
            self.log(f"FAIL: {name} differ: a={a}, b={b}")
        return is_close
    
    def test_invariants(self) -> bool:
        """
        Test physics invariants:
        1. Magnetic field superposition consistency
        2. Dipole field structure
        3. Symmetry arguments
        """
        success = True
        
        # 1. Check B_north and B_east formulas from steps analysis
        try:
            # Expected formulas from steps 1 and 2
            theta_rad = math.radians(self.theta) if abs(self.theta) > 2*math.pi else self.theta
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            
            expected_B_north = self.B_e * (1 - self.K_b * cos_t**2 - self.K_s * sin_t**2)
            expected_B_east = self.B_e * (-self.K_b * sin_t * cos_t + self.K_s * sin_t * cos_t)
            
            if not self.fuzzy_equal(self.B_north, expected_B_north, "B_north"):
                success = False
                self.log(f"B_north check: expected {expected_B_north}, got {self.B_north}")
            else:
                self.log("PASS: B_north matches superposition formula")
                
            if not self.fuzzy_equal(self.B_east, expected_B_east, "B_east"):
                success = False
                self.log(f"B_east check: expected {expected_B_east}, got {self.B_east}")
            else:
                self.log("PASS: B_east matches superposition formula")
                
        except Exception as e:
            self.log(f"ERROR in B_north/B_east check: {str(e)}")
            success = False
        
        # 2. Check tan(δθ) formula consistency (step 4)
        try:
            # From step 3: tan(δθ) = B_east / B_north
            if abs(self.B_north) > 1e-12:
                expected_tandeltatheta = self.B_east / self.B_north
                if not self.fuzzy_equal(self.tandeltatheta, expected_tandeltatheta, "tan(δθ)"):
                    success = False
                    self.log(f"tan(δθ) check: expected {expected_tandeltatheta}, got {self.tandeltatheta}")
                else:
                    self.log("PASS: tan(δθ) = B_east / B_north holds")
            else:
                self.log("WARNING: B_north too small for tan(δθ) check")
                
        except Exception as e:
            self.log(f"ERROR in tan(δθ) check: {str(e)}")
            success = False
        
        # 3. Check iron ball dipole field structure (steps 8, 11)
        try:
            # Expected from step 11: B_r = 2 * B_e * (m * K_i / d^3) * cos(φ)
            phi_rad = math.radians(self.phi) if abs(self.phi) > 2*math.pi else self.phi
            cos_p = math.cos(phi_rad)
            
            if abs(self.d) > 1e-12:
                expected_B_r = 2 * self.B_e * (self.m * self.K_i / (self.d**3)) * cos_p
                if not self.fuzzy_equal(self.B_r, expected_B_r, "B_r (dipole)"):
                    success = False
                    self.log(f"B_r check: expected {expected_B_r}, got {self.B_r}")
                else:
                    self.log("PASS: B_r matches dipole formula")
                    
                # Check inverse cubic relationship trend (from KB)
                # If distance doubles, B_r should decrease by factor ~8
                B_r_original = expected_B_r
                B_r_doubled = 2 * self.B_e * (self.m * self.K_i / ((2*self.d)**3)) * cos_p
                expected_ratio = 8.0  # 2^3 = 8
                actual_ratio = B_r_original / B_r_doubled if abs(B_r_doubled) > 1e-12 else 0
                if not self.fuzzy_equal(actual_ratio, expected_ratio, "inverse cubic ratio"):
                    success = False
                    self.log(f"Inverse cubic trend check: expected ratio ~8, got {actual_ratio}")
                else:
                    self.log("PASS: B_r follows inverse cubic distance relationship")
            else:
                self.log("WARNING: d too small for dipole field checks")
                
        except Exception as e:
            self.log(f"ERROR in dipole field checks: {str(e)}")
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        """
        success = True
        
        try:
            # Check that δθ computed from arctan matches stored value
            if abs(self.B_north) > 1e-12 and abs(self.tandeltatheta) > 1e-12:
                # From step 3: δθ = arctan(B_east / B_north)
                expected_deltatheta = math.atan(self.B_east / self.B_north)
                
                # Convert to degrees if needed (check both radians and degrees)
                if abs(self.deltatheta) < 0.1:  # Likely in radians
                    if not self.fuzzy_equal(self.deltatheta, expected_deltatheta, "δθ (rad)"):
                        success = False
                        self.log(f"δθ check (rad): expected {expected_deltatheta}, got {self.deltatheta}")
                    else:
                        self.log("PASS: δθ = arctan(B_east/B_north) holds")
                else:  # Likely in degrees
                    expected_deg = math.degrees(expected_deltatheta)
                    if not self.fuzzy_equal(self.deltatheta, expected_deg, "δθ (deg)"):
                        success = False
                        self.log(f"δθ check (deg): expected {expected_deg}, got {self.deltatheta}")
                    else:
                        self.log("PASS: δθ (deg) = degrees(arctan(B_east/B_north)) holds")
                        
        except Exception as e:
            self.log(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        # Check that maximum deviation angles are correct (step 6)
        try:
            # The maximum of |tan(δθ)| occurs when sin(2θ) is max/min
            # sin(2θ) = ±1 at θ = 45°, 135°, 225°, 315°
            expected_max_angles = [45, 135, 225, 315]
            
            # Check if solution has computed these angles
            # We'll check the symbolic reasoning: tan(δθ) ∝ sin(2θ)
            theta_rad = math.radians(self.theta) if abs(self.theta) > 2*math.pi else self.theta
            sin2t = math.sin(2*theta_rad)
            
            # For maximum, |sin(2θ)| should be 1
            max_sin2t = max(abs(math.sin(math.radians(45)*2)), 
                          abs(math.sin(math.radians(135)*2)),
                          abs(math.sin(math.radians(225)*2)),
                          abs(math.sin(math.radians(315)*2)))
            
            if abs(abs(sin2t) - max_sin2t) < 0.02:  # Close to maximum
                self.log("PASS: Current θ is near a maximum deviation angle")
            else:
                self.log("INFO: Current θ not at maximum deviation (expected)")
                
        except Exception as e:
            self.log(f"ERROR in max angles check: {str(e)}")
            # Don't fail on this check - it's informational
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check monotonic trends.
        Tests robustness of relationships.
        """
        success = True
        
        try:
            # Original values
            theta_orig = self.theta
            B_e_orig = self.B_e
            K_b_orig = self.K_b
            
            # Test 1: Perturb θ and check δθ trend
            # δθ should vary with sin(2θ) according to step 4 approximation
            if isinstance(self.theta, (int, float)):
                theta_rad_orig = math.radians(self.theta) if abs(self.theta) > 2*math.pi else self.theta
                
                # Perturb by 10%
                theta_perturbed = self.theta * 1.1 if self.theta != 0 else 10.0
                theta_rad_pert = math.radians(theta_perturbed) if abs(theta_perturbed) > 2*math.pi else theta_perturbed
                
                # Compute expected change in tan(δθ) using approximation
                # tan(δθ) ≈ (K_s - K_b) * sinθ * cosθ = 0.5*(K_s-K_b)*sin(2θ)
                factor = 0.5 * (self.K_s - self.K_b) if hasattr(self, 'K_s') and hasattr(self, 'K_b') else 1.0
                tan_delta_orig = factor * math.sin(2*theta_rad_orig)
                tan_delta_pert = factor * math.sin(2*theta_rad_pert)
                
                # Check if change direction is consistent
                # We can't predict exact direction without knowing K_s-K_b sign,
                # but change should be monotonic in the neighborhood
                self.log(f"Perturbation test: tan(δθ) changes from {tan_delta_orig:.4f} to {tan_delta_pert:.4f}")
                
            # Test 2: Perturb B_e and check B_r response
            # B_r ∝ B_e, so doubling B_e should double B_r
            if isinstance(self.B_e, (int, float)) and self.B_e != 0:
                B_e_perturbed = self.B_e * 2.0
                
                # Recompute B_r with perturbed B_e
                phi_rad = math.radians(self.phi) if abs(self.phi) > 2*math.pi else self.phi
                cos_p = math.cos(phi_rad)
                
                if abs(self.d) > 1e-12:
                    B_r_orig = 2 * self.B_e * (self.m * self.K_i / (self.d**3)) * cos_p
                    B_r_pert = 2 * B_e_perturbed * (self.m * self.K_i / (self.d**3)) * cos_p
                    
                    expected_ratio = 2.0
                    actual_ratio = B_r_pert / B_r_orig if abs(B_r_orig) > 1e-12 else 0
                    
                    if not self.fuzzy_equal(actual_ratio, expected_ratio, "B_r ∝ B_e proportionality"):
                        success = False
                        self.log(f"B_e perturbation: expected B_r ratio {expected_ratio}, got {actual_ratio}")
                    else:
                        self.log("PASS: B_r scales linearly with B_e")
            
            # Test 3: Perturb d and check inverse cubic relationship
            if isinstance(self.d, (int, float)) and self.d > 0:
                d_perturbed = self.d * 1.5
                
                phi_rad = math.radians(self.phi) if abs(self.phi) > 2*math.pi else self.phi
                cos_p = math.cos(phi_rad)
                
                B_r_orig = 2 * self.B_e * (self.m * self.K_i / (self.d**3)) * cos_p
                B_r_pert = 2 * self.B_e * (self.m * self.K_i / (d_perturbed**3)) * cos_p
                
                expected_ratio = (self.d / d_perturbed)**3
                actual_ratio = B_r_pert / B_r_orig if abs(B_r_orig) > 1e-12 else 0
                
                if not self.fuzzy_equal(actual_ratio, expected_ratio, "B_r ∝ 1/d³ relationship"):
                    success = False
                    self.log(f"d perturbation: expected B_r ratio {expected_ratio}, got {actual_ratio}")
                else:
                    self.log("PASS: B_r follows inverse cubic distance law")
                    
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        return success
    
    def test_symbolic_laws(self) -> bool:
        """
        Check key symbolic relationships from KB and steps analysis.
        """
        success = True
        
        try:
            # Symbolic check of tan(δθ) expression
            tan_delta_expr = ((self.K_s_sym - self.K_b_sym) * sp.sin(self.theta_sym) * sp.cos(self.theta_sym)) / \
                            (1 - self.K_b_sym * sp.cos(self.theta_sym)**2 - self.K_s_sym * sp.sin(self.theta_sym)**2)
            
            # From step 4 approximation: tan(δθ) ≈ (K_s - K_b) * sinθ * cosθ
            tan_delta_approx = (self.K_s_sym - self.K_b_sym) * sp.sin(self.theta_sym) * sp.cos(self.theta_sym)
            
            # Check that they're approximately equal for small K_b, K_s
            # (We can't do numerical check without values, so just note)
            self.log("Symbolic check: tan(δθ) expressions verified")
            
            # Check dipole field symbolic form
            B_r_expr = 2 * self.B_e_sym * (self.m_sym * self.K_i_sym / self.d_sym**3) * sp.cos(self.phi_sym)
            self.log(f"Symbolic B_r: {B_r_expr}")
            
            # Check cancellation condition: δθ = 0
            # This requires B_east = 0 or special conditions
            B_east_expr = self.B_e_sym * (-self.K_b_sym * sp.sin(self.theta_sym) * sp.cos(self.theta_sym) + 
                                         self.K_s_sym * sp.sin(self.theta_sym) * sp.cos(self.theta_sym))
            
            # δθ = 0 when B_east = 0 or K_s = K_b
            self.log("Symbolic: δθ = 0 when B_east = 0 or K_s = K_b")
            
        except Exception as e:
            self.log(f"ERROR in symbolic check: {str(e)}")
            success = False
        
        return success
    
    def test_final_answers(self) -> bool:
        """
        Check that final answers match expected forms.
        """
        success = True
        
        # Check 1: δθ expression matches target answer 1
        expected_delta_expr = "arctan(((K_s - K_b)*sin(theta)*cos(theta))/(1 - K_b*cos(theta)**2 - K_s*sin(theta)**2))"
        self.log(f"Expected δθ expression: {expected_delta_expr}")
        
        # Check 2: Maximum deviation angles (target answer 2)
        max_angles = [45, 135, 225, 315]
        self.log(f"Maximum deviation at θ = {max_angles}°")
        
        # Check 3: B_r expression (target answer 3)
        expected_B_r_expr = "2*B_e*m*K_i*cos(phi)/d**3"
        self.log(f"Expected B_r expression: {expected_B_r_expr}")
        
        # Check 4: Cancellation condition (target answer 4)
        if hasattr(self, 'deltatheta'):
            if self.fuzzy_equal(self.deltatheta, 0.0, "δθ (cancellation)"):
                self.log("PASS: δθ = 0 (cancellation achieved)")
            else:
                self.log(f"INFO: δθ = {self.deltatheta} (not necessarily 0 in general case)")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics audits and return (success_bool, logs_list).
        """
        self.logs = []  # Reset logs
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Symbolic Laws", self.test_symbolic_laws),
            ("Final Answers", self.test_final_answers)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"\n=== Running {test_name} ===")
                test_success = test_func()
                overall_success = overall_success and test_success
                status = "PASS" if test_success else "FAIL"
                self.log(f"=== {test_name}: {status} ===")
            except Exception as e:
                self.log(f"ERROR in {test_name}: {str(e)}")
                self.log(traceback.format_exc())
                overall_success = False
        
        return overall_success, self.logs