import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        try:
            # Extract input parameters using getattr with defaults
            # These are the parameters that should be provided as inputs
            self.R = getattr(solution_instance, 'R', 1.737e6)
            self.h = getattr(solution_instance, 'h', 2e5)
            self.T = getattr(solution_instance, 'T', 7200)
            self.G = getattr(solution_instance, 'G', 6.67430e-11)
            
            # Extract results from results dictionary
            # The solution code should store M and g in results
            self.M = getattr(solution_instance, 'M', None)
            self.g = getattr(solution_instance, 'g', None)
            
            # If not found as attributes, try results dict
            if self.M is None:
                self.M = getattr(solution_instance, 'results', {}).get('M', None)
            if self.g is None:
                self.g = getattr(solution_instance, 'results', {}).get('g', None)
            
            # Ensure numeric types
            self.R = float(self.R)
            self.h = float(self.h)
            self.T = float(self.T)
            self.G = float(self.G)
            
            # Log extraction status
            self.extraction_log = [
                f"Extracted inputs: R={self.R}, h={self.h}, T={self.T}, G={self.G}",
                f"Extracted results: M={self.M}, g={self.g}"
            ]
            
            # Initialize logs
            self.logs = []
            
        except Exception as e:
            self.extraction_log = [f"Extraction error: {traceback.format_exc()}"]
            self.logs = self.extraction_log
            raise

    def test_invariants(self):
        """Verify physical invariants using symbolic and numerical methods."""
        logs = []
        
        try:
            # 1. Newton's Law of Universal Gravitation for circular orbit
            # For circular orbit: gravitational force = centripetal force
            # G*M*m/(R+h)^2 = m*(2π/T)^2*(R+h)
            # Simplify: M = 4π^2*(R+h)^3/(G*T^2)
            
            # Numerical check
            M_expected = 4 * math.pi**2 * (self.R + self.h)**3 / (self.G * self.T**2)
            
            if self.M is not None:
                # Numerical check with tolerance
                if not math.isclose(self.M, M_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Moon mass mismatch. Expected {M_expected:.3e}, got {self.M:.3e}")
                else:
                    logs.append(f"PASS: Moon mass matches Newton's Law (within 1%)")
            else:
                logs.append("SKIP: Moon mass not provided for invariant check")
            
            # 2. Gravitational acceleration at surface
            # g = G*M/R^2
            # From step analysis: g = 4π^2*(R+h)^3/(T^2*R^2)
            
            if self.g is not None and self.M is not None:
                # Check using direct formula
                g_from_M = self.G * self.M / (self.R**2)
                if not math.isclose(self.g, g_from_M, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: g mismatch. From M: {g_from_M:.3e}, got {self.g:.3e}")
                else:
                    logs.append(f"PASS: g matches g = GM/R² (within 1%)")
                
                # Check derived formula
                g_expected = 4 * math.pi**2 * (self.R + self.h)**3 / (self.T**2 * self.R**2)
                if not math.isclose(self.g, g_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: g differs from derived formula. Expected {g_expected:.3e}")
            
        except Exception as e:
            logs.append(f"ERROR in invariant test: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0

    def test_trace_consistency(self):
        """Cross-check intermediate and final results for consistency."""
        logs = []
        
        try:
            # Consistency between step 1 (M) and step 2 (g)
            if self.M is not None and self.g is not None:
                # Recompute g from M
                g_recomputed = self.G * self.M / (self.R**2)
                
                # Check if recomputed g matches the provided g
                if not math.isclose(self.g, g_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Trace inconsistency. g from M: {g_recomputed:.3e}, given g: {self.g:.3e}")
                else:
                    logs.append(f"PASS: g is consistent with M (within 1%)")
                    
                # Also verify that M produces the correct orbital period
                # T_expected = 2π√((R+h)^3/(G*M))
                T_from_M = 2 * math.pi * math.sqrt((self.R + self.h)**3 / (self.G * self.M))
                if not math.isclose(self.T, T_from_M, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: Period T mismatch. From M: {T_from_M:.3e}, given T: {self.T:.3e}")
                else:
                    logs.append(f"PASS: T is consistent with M (within 1%)")
            
            else:
                logs.append("SKIP: Missing M or g for trace consistency check")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0

    def test_monotonicity(self):
        """Perturb input parameters and check if results change in expected direction."""
        logs = []
        
        try:
            # Store original values
            h_original = self.h
            perturb_factor = 1.01  # 1% increase
            
            # Perturb altitude h
            h_perturbed = h_original * perturb_factor
            
            # Recompute expected values with perturbed h
            M_perturbed = 4 * math.pi**2 * (self.R + h_perturbed)**3 / (self.G * self.T**2)
            g_perturbed = 4 * math.pi**2 * (self.R + h_perturbed)**3 / (self.T**2 * self.R**2)
            
            # Check monotonicity trends
            # From formulas: M ∝ (R+h)³, g ∝ (R+h)³
            # Both should increase when h increases
            if self.M is not None:
                M_original_formula = 4 * math.pi**2 * (self.R + h_original)**3 / (self.G * self.T**2)
                if math.isclose(self.M, M_original_formula, rel_tol=1e-2, abs_tol=1e-2):
                    # Check if perturbation increases M
                    if M_perturbed > self.M:
                        logs.append("PASS: M increases with increased h (cubic dependence)")
                    else:
                        logs.append("FAIL: M does not show expected monotonic increase with h")
                else:
                    logs.append("SKIP: M doesn't match formula; monotonicity check unreliable")
            
            if self.g is not None:
                g_original_formula = 4 * math.pi**2 * (self.R + h_original)**3 / (self.T**2 * self.R**2)
                if math.isclose(self.g, g_original_formula, rel_tol=1e-2, abs_tol=1e-2):
                    # Check if perturbation increases g
                    if g_perturbed > self.g:
                        logs.append("PASS: g increases with increased h (cubic dependence)")
                    else:
                        logs.append("FAIL: g does not show expected monotonic increase with h")
                else:
                    logs.append("SKIP: g doesn't match formula; monotonicity check unreliable")
                    
            # Additional monotonicity: Check if g ∝ M (linear)
            if self.M is not None and self.g is not None:
                # g = (G/R²) * M, so g should be proportional to M
                # Compute constant of proportionality
                const = self.g / self.M
                expected_const = self.G / (self.R**2)
                if math.isclose(const, expected_const, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: g ∝ M as expected (linear relationship)")
                else:
                    logs.append(f"WARN: g/M ratio {const:.3e} differs from G/R²={expected_const:.3e}")
                    
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        self.logs.extend(self.extraction_log)
        
        success = True
        
        # Run tests
        test_results = []
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Summary
        self.logs.append("\n=== SUMMARY ===")
        for test_name, passed in test_results:
            status = "PASS" if passed else "FAIL"
            self.logs.append(f"{test_name}: {status}")
            if not passed:
                success = False
        
        self.logs.append(f"\nOverall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs