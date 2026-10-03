import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely
        self.R = getattr(solution_instance, 'R', 1.0)  # Default R=1 for calculations
        self.theta_deg = 30.0  # Given in problem
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results safely
        self.beta1_deg = self._safe_get('β1', 0.0)
        self.beta2_deg = self._safe_get('β2', 0.0)
        self.C1_deg = self._safe_get('C1', 0.0)
        self.theta_min_deg = self._safe_get('theta_min', 0.0)
        
        # Convert to radians for calculations
        self.beta1_rad = math.radians(self.beta1_deg)
        self.beta2_rad = math.radians(self.beta2_deg)
        self.C1_rad = math.radians(self.C1_deg)
        self.theta_min_rad = math.radians(self.theta_min_deg)
        
        # Physical constants
        self.n1 = math.sqrt(2)
        self.n2 = math.sqrt(3)
        self.n_air = 1.0
        
        # Expected values from steps analysis (corrected)
        self.expected_beta1_deg = 45.0
        self.expected_beta2_deg = 60.0
        self.expected_C1_deg = 45.0
        self.expected_theta_min_deg = 45.0
        
        # Expected width expression: R*(tan(60°) - tan(45°)) = R*(√3 - 1) ≈ 0.7321R
        self.expected_width_expr = self.R * (math.sqrt(3) - 1)
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get value from results with multiple fallbacks"""
        if key in self.results:
            val = self.results[key]
            if isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, str):
                try:
                    return float(val)
                except:
                    return default
        return default
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics laws"""
        logs = []
        success = True
        
        try:
            # 1. Test Snell's Law for both refractive indices (CORRECTED)
            # For n1: n1 * sin(theta) = n_air * sin(beta1)
            lhs1 = self.n1 * math.sin(self.theta_rad)
            rhs1 = self.n_air * math.sin(self.beta1_rad)
            
            if not math.isclose(lhs1, rhs1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Snell's law violated for n1. LHS={lhs1:.4f}, RHS={rhs1:.4f}")
                success = False
            else:
                logs.append(f"PASS: Snell's law satisfied for n1")
            
            # For n2: n2 * sin(theta) = n_air * sin(beta2)
            lhs2 = self.n2 * math.sin(self.theta_rad)
            rhs2 = self.n_air * math.sin(self.beta2_rad)
            
            if not math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Snell's law violated for n2. LHS={lhs2:.4f}, RHS={rhs2:.4f}")
                success = False
            else:
                logs.append(f"PASS: Snell's law satisfied for n2")
            
            # 2. Test critical angle formula: sin(C1) = 1/n1
            expected_sin_C1 = 1 / self.n1
            actual_sin_C1 = math.sin(self.C1_rad)
            
            if not math.isclose(expected_sin_C1, actual_sin_C1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Critical angle formula violated. Expected sin(C1)={expected_sin_C1:.4f}, Actual={actual_sin_C1:.4f}")
                success = False
            else:
                logs.append(f"PASS: Critical angle formula satisfied")
            
            # 3. Test that theta_min equals C1 (from step 8)
            if not math.isclose(self.theta_min_deg, self.C1_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: theta_min ({self.theta_min_deg:.2f}°) should equal C1 ({self.C1_deg:.2f}°)")
                success = False
            else:
                logs.append(f"PASS: theta_min = C1 as expected")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with expected values"""
        logs = []
        success = True
        
        try:
            # Check beta1 matches expected 45°
            if not math.isclose(self.beta1_deg, self.expected_beta1_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: beta1={self.beta1_deg:.2f}°, expected {self.expected_beta1_deg}°")
                success = False
            else:
                logs.append(f"PASS: beta1={self.beta1_deg:.2f}° matches expected 45°")
            
            # Check beta2 matches expected 60°
            if not math.isclose(self.beta2_deg, self.expected_beta2_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: beta2={self.beta2_deg:.2f}°, expected {self.expected_beta2_deg}°")
                success = False
            else:
                logs.append(f"PASS: beta2={self.beta2_deg:.2f}° matches expected 60°")
            
            # Check C1 matches expected 45°
            if not math.isclose(self.C1_deg, self.expected_C1_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: C1={self.C1_deg:.2f}°, expected {self.expected_C1_deg}°")
                success = False
            else:
                logs.append(f"PASS: C1={self.C1_deg:.2f}° matches expected 45°")
            
            # Check theta_min matches expected 45°
            if not math.isclose(self.theta_min_deg, self.expected_theta_min_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: theta_min={self.theta_min_deg:.2f}°, expected {self.expected_theta_min_deg}°")
                success = False
            else:
                logs.append(f"PASS: theta_min={self.theta_min_deg:.2f}° matches expected 45°")
                
            # Check width calculation consistency
            # From corrected physics: width = R*(tan(beta2) - tan(beta1))
            width_calc = self.R * (math.tan(self.beta2_rad) - math.tan(self.beta1_rad))
            # Expected: (√3 - 1)R ≈ 0.7321R
            expected_width = self.expected_width_expr
            
            if not math.isclose(width_calc, expected_width, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Width calculation inconsistent. Calculated={width_calc:.4f}, Expected={expected_width:.4f}")
                success = False
            else:
                logs.append(f"PASS: Width calculation consistent: {width_calc:.4f} ≈ {expected_width:.4f}")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test that increasing refractive index increases refraction angle"""
        logs = []
        success = True
        
        try:
            # Test 1: n2 > n1 should imply beta2 > beta1
            if not (self.n2 > self.n1 and self.beta2_deg > self.beta1_deg):
                logs.append(f"FAIL: n2={self.n2:.3f} > n1={self.n1:.3f} but beta2={self.beta2_deg:.2f}° not > beta1={self.beta1_deg:.2f}°")
                success = False
            else:
                logs.append(f"PASS: n2 > n1 implies beta2 > beta1")
            
            # Test 2: Perturb n1 and check beta1 trend
            # Increase n1 by 10%
            n1_perturbed = self.n1 * 1.1
            # Calculate new beta1 using Snell's law: n1_perturbed * sin(theta) = sin(beta1_perturbed)
            sin_beta1_perturbed = n1_perturbed * math.sin(self.theta_rad)
            if sin_beta1_perturbed <= 1:  # Valid refraction
                beta1_perturbed = math.degrees(math.asin(sin_beta1_perturbed))
                if beta1_perturbed > self.beta1_deg:
                    logs.append(f"PASS: Increasing n1 increases beta1 (perturbed: {beta1_perturbed:.2f}° > original: {self.beta1_deg:.2f}°)")
                else:
                    logs.append(f"FAIL: Increasing n1 should increase beta1, but {beta1_perturbed:.2f}° ≤ {self.beta1_deg:.2f}°")
                    success = False
            else:
                logs.append(f"INFO: Perturbed n1 causes total internal reflection")
            
            # Test 3: Perturb incidence angle and check beta trend
            theta_perturbed = math.radians(self.theta_deg * 1.1)  # Increase by 10%
            sin_beta1_theta_perturbed = self.n1 * math.sin(theta_perturbed)
            if sin_beta1_theta_perturbed <= 1:
                beta1_theta_perturbed = math.degrees(math.asin(sin_beta1_theta_perturbed))
                if beta1_theta_perturbed > self.beta1_deg:
                    logs.append(f"PASS: Increasing theta increases beta1")
                else:
                    logs.append(f"FAIL: Increasing theta should increase beta1")
                    success = False
            else:
                logs.append(f"INFO: Perturbed theta causes total internal reflection")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def test_total_internal_reflection(self) -> Tuple[bool, List[str]]:
        """Test total internal reflection conditions"""
        logs = []
        success = True
        
        try:
            # For n1, check if theta_min = C1 is indeed the critical angle
            # Critical angle condition: n1 * sin(C1) = n_air * sin(90°) = 1
            lhs = self.n1 * math.sin(self.C1_rad)
            
            if not math.isclose(lhs, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: At C1, n1*sin(C1) should equal 1, but got {lhs:.4f}")
                success = False
            else:
                logs.append(f"PASS: C1 satisfies critical angle condition")
            
            # Check that for theta > C1, we get total internal reflection
            theta_test = self.C1_rad + 0.1  # Slightly above critical
            sin_theta_t = (self.n_air / self.n1) * math.sin(theta_test)
            
            if sin_theta_t > 1:  # Mathematically impossible, indicates TIR
                logs.append(f"PASS: For theta > C1, sin(theta_t) > 1 indicates TIR")
            else:
                # Calculate refraction angle
                theta_t = math.asin(sin_theta_t)
                if theta_t > math.pi/2:
                    logs.append(f"PASS: theta_t > 90° indicates TIR")
                else:
                    logs.append(f"INFO: For theta just above C1, refraction still possible")
                    
        except Exception as e:
            logs.append(f"ERROR in test_total_internal_reflection: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs"""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity),
            ("Total Internal Reflection Test", self.test_total_internal_reflection)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Running {test_name}")
            all_logs.append(f"{'='*50}")
            
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: {test_name} FAILED")
                else:
                    all_logs.append(f"Result: {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n{'='*50}")
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"{'='*50}")
        
        return overall_success, all_logs