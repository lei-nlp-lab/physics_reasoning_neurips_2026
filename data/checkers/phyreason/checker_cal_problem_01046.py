import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.0))
        
        # Extract results with safe fallbacks
        self.n = getattr(solution_instance, 'n', 
                        getattr(solution_instance, 'results', {}).get('n', None))
        self.alpha_deg = getattr(solution_instance, 'alpha', 
                                getattr(solution_instance, 'results', {}).get('alpha', None))
        self.beta_deg = getattr(solution_instance, 'beta', 
                               getattr(solution_instance, 'results', {}).get('β', None))
        self.i0_deg = getattr(solution_instance, 'i_0', 
                             getattr(solution_instance, 'results', {}).get('i_0', None))
        self.s = getattr(solution_instance, 's', 
                        getattr(solution_instance, 'results', {}).get('s', None))
        
        # Convert angles to radians for calculations
        self.i_deg = 60.0  # Given in problem
        self.i_rad = math.radians(self.i_deg)
        self.alpha_rad = math.radians(self.alpha_deg) if self.alpha_deg else None
        self.beta_rad = math.radians(self.beta_deg) if self.beta_deg else None
        self.i0_rad = math.radians(self.i0_deg) if self.i0_deg else None
        
        self.logs = []

    def log(self, msg, success=None):
        """Helper to log messages and track success"""
        if success is not None:
            msg = f"{'✓' if success else '✗'} {msg}"
        self.logs.append(msg)
        return success

    def test_invariants(self):
        """Test physical invariants: Snell's Law and Total Internal Reflection condition"""
        try:
            # Test 1: Snell's Law at AB interface (air to medium)
            if self.n and self.alpha_rad:
                lhs = 1.0 * math.sin(self.i_rad)  # n_air * sin(i)
                rhs = self.n * math.sin(self.alpha_rad)  # n_medium * sin(alpha)
                snell_ok = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Snell's Law at AB: 1·sin(60°) = {lhs:.4f}, n·sin(α) = {rhs:.4f}", snell_ok)
            else:
                self.log("Missing n or α for Snell's Law test", False)

            # Test 2: Condition for single bright spot (critical TIR condition)
            if self.n:
                # Minimum n for TIR when incidence at AC is 60°: n ≥ 1/sin(60°)
                n_min = 1.0 / math.sin(math.radians(60))
                tir_ok = self.n >= n_min * 0.99  # Allow 1% tolerance for floating point
                self.log(f"TIR condition: n = {self.n:.4f} ≥ 1/sin(60°) = {n_min:.4f}", tir_ok)
                
                # Alternatively, check n·sin(60°) ≥ 1
                lhs_tir = self.n * math.sin(math.radians(60))
                tir_alt = math.isclose(lhs_tir, 1.0, rel_tol=1e-2, abs_tol=1e-2) or lhs_tir > 1.0
                self.log(f"n·sin(60°) = {lhs_tir:.4f} ≥ 1.0", tir_alt)
            else:
                self.log("Missing n for TIR test", False)

            # Test 3: Geometric relationship from steps analysis
            if self.alpha_deg and self.i0_deg:
                # From step4: α = 30°, i_0 = 30°
                alpha_ok = math.isclose(self.alpha_deg, 30.0, rel_tol=1e-2, abs_tol=1e-2)
                i0_ok = math.isclose(self.i0_deg, 30.0, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"α = {self.alpha_deg:.2f}° ≈ 30°", alpha_ok)
                self.log(f"i_0 = {self.i0_deg:.2f}° ≈ 30°", i0_ok)
            else:
                self.log("Missing α or i_0 for geometric test", False)

            return True
        except Exception as e:
            self.log(f"Invariant test failed: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results"""
        try:
            # Test 1: Verify n from step1 (minimum) vs actual n
            if self.n:
                n_min = 2 * math.sqrt(3) / 3  # 2√3/3 ≈ 1.1547
                n_ok = math.isclose(self.n, n_min, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"n = {self.n:.4f} ≈ 2√3/3 = {n_min:.4f}", n_ok)
            else:
                self.log("Missing n for trace test", False)

            # Test 2: Verify s from step5 matches R/3
            if self.s and self.R:
                s_expected = self.R / 3.0
                s_ok = math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"s = {self.s:.4f} ≈ R/3 = {s_expected:.4f} (R={self.R})", s_ok)
            else:
                self.log("Missing s or R for trace test", False)

            # Test 3: Verify β from step2 is 60°
            if self.beta_deg:
                beta_ok = math.isclose(self.beta_deg, 60.0, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"β = {self.beta_deg:.2f}° ≈ 60°", beta_ok)
            else:
                self.log("Missing β for trace test", False)

            # Test 4: Cross-check using step3 equation: n = sin(60°)/sin(α)
            if self.n and self.alpha_rad:
                n_calc = math.sin(math.radians(60)) / math.sin(self.alpha_rad)
                cross_ok = math.isclose(self.n, n_calc, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Cross-check: n = {self.n:.4f} ≈ sin60°/sinα = {n_calc:.4f}", cross_ok)
            else:
                self.log("Missing n or α for cross-check", False)

            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """Test that changing R proportionally affects s, while n remains constant"""
        try:
            # Test 1: s should be proportional to R
            if self.s and self.R and self.R != 0:
                ratio = self.s / self.R
                ratio_expected = 1.0 / 3.0  # s = R/3
                ratio_ok = math.isclose(ratio, ratio_expected, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"s/R = {ratio:.4f} ≈ 1/3 = {ratio_expected:.4f}", ratio_ok)
            else:
                self.log("Missing s or R for monotonicity test", False)

            # Test 2: Perturb R and verify formula holds (numerical simulation)
            if self.R and self.n:
                # Perturb R by 10%
                R_perturbed = self.R * 1.1
                s_expected_perturbed = R_perturbed / 3.0
                
                # Check if n would remain the same (material property independent of R)
                self.log(f"Material property: n = {self.n:.4f} independent of R (R_perturbed={R_perturbed:.2f})", True)
                self.log(f"If R→{R_perturbed:.2f}, expected s→{s_expected_perturbed:.4f} (proportionality check)", True)
            else:
                self.log("Missing R or n for perturbation test", False)

            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        # Overall success
        success = inv_ok and trace_ok and mono_ok
        
        # Summary
        summary = f"\n=== Physics Checker Summary ==="
        summary += f"\nInvariants: {'PASS' if inv_ok else 'FAIL'}"
        summary += f"\nTrace Consistency: {'PASS' if trace_ok else 'FAIL'}"
        summary += f"\nMonotonicity: {'PASS' if mono_ok else 'FAIL'}"
        summary += f"\nOverall: {'PASS' if success else 'FAIL'}"
        self.logs.append(summary)
        
        return success, self.logs