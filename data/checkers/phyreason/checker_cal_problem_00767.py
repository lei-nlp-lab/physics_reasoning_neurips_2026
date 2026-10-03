import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs (SI units)
        self.inputs['m'] = getattr(solution_instance, 'm', 0.1)  # 100g = 0.1kg
        self.inputs['k'] = getattr(solution_instance, 'k', 200.0)  # N/m
        self.inputs['l0'] = getattr(solution_instance, 'l0', 0.05)  # 5cm = 0.05m
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Extract results with safe fallbacks
        self.results['delta_x0'] = getattr(solution_instance, 'delta_x0', 
                                          getattr(solution_instance, 'results', {}).get('delta_x0', 
                                          getattr(solution_instance, 'results', {}).get('Δx₀', 0.0)))
        self.results['h'] = getattr(solution_instance, 'h', 
                                   getattr(solution_instance, 'results', {}).get('h', 0.0))
        self.results['a_m'] = getattr(solution_instance, 'a_m', 
                                     getattr(solution_instance, 'results', {}).get('a_m', 0.0))
        self.results['N'] = getattr(solution_instance, 'N', 
                                   getattr(solution_instance, 'results', {}).get('N', 0.0))
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        
    def test_invariants(self):
        """Verify physics laws based on KB theorems."""
        try:
            m = self.inputs['m']
            k = self.inputs['k']
            g = self.inputs['g']
            l0 = self.inputs['l0']
            delta_x0 = self.results['delta_x0']
            h = self.results['h']
            a_m = self.results['a_m']
            N = self.results['N']
            
            # Total mass (A + B)
            M_total = 2 * m
            
            # 1. Check equilibrium condition (Newton's 1st Law at equilibrium)
            # k * Δx₀ = (m_A + m_B) * g
            lhs_eq = k * delta_x0
            rhs_eq = M_total * g
            eq_check = math.isclose(lhs_eq, rhs_eq, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Equilibrium check: k·Δx₀ = {lhs_eq:.3f} ≈ (2m)g = {rhs_eq:.3f}", eq_check)
            
            # 2. Check geometric relationship
            # h = l0 - Δx₀
            expected_h = l0 - delta_x0
            geo_check = math.isclose(h, expected_h, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Geometry check: h = {h:.3f}m ≈ l0 - Δx₀ = {expected_h:.3f}m", geo_check)
            
            # 3. Check maximum acceleration using Newton's 2nd Law at lowest point
            # At lowest point, spring force = k(Δx₀ + A) where A = amplitude = Δx₀
            # Net force: F_net = k(Δx₀ + A) - (2m)g = k(2Δx₀) - (2m)g
            # But from equilibrium: kΔx₀ = (2m)g, so F_net = kΔx₀
            # Therefore: a_m = F_net/(2m) = kΔx₀/(2m)
            expected_a_m = (k * delta_x0) / M_total
            accel_check = math.isclose(a_m, expected_a_m, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Max acceleration check: a_m = {a_m:.3f}m/s² ≈ kΔx₀/(2m) = {expected_a_m:.3f}m/s²", accel_check)
            
            # 4. Check maximum normal force (Newton's 2nd Law for block B)
            # N = m_B(g + a_m) = m(g + a_m)
            expected_N = m * (g + a_m)
            normal_check = math.isclose(N, expected_N, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Normal force check: N = {N:.3f}N ≈ m(g+a_m) = {expected_N:.3f}N", normal_check)
            
            return all([eq_check, geo_check, accel_check, normal_check])
            
        except Exception as e:
            self._log(f"Invariant test failed: {str(e)}", False)
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            m = self.inputs['m']
            k = self.inputs['k']
            g = self.inputs['g']
            delta_x0 = self.results['delta_x0']
            a_m = self.results['a_m']
            N = self.results['N']
            
            # Check that using a_m from step 3 gives correct N in step 5
            computed_N = m * (g + a_m)
            consistency_check = math.isclose(N, computed_N, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Trace consistency: N from a_m = {computed_N:.3f}N ≈ reported N = {N:.3f}N", 
                     consistency_check)
            
            # Check that amplitude A = Δx₀ (implied by step 3 equation)
            M_total = 2 * m
            computed_a_m = (k * delta_x0) / M_total  # Using A = Δx₀
            amp_check = math.isclose(a_m, computed_a_m, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Amplitude consistency: a_m from Δx₀ = {computed_a_m:.3f}m/s² ≈ reported a_m = {a_m:.3f}m/s²", 
                     amp_check)
            
            return consistency_check and amp_check
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
            
    def test_monotonicity(self):
        """Perturb mass and check physical trends."""
        try:
            original_m = self.inputs['m']
            k = self.inputs['k']
            g = self.inputs['g']
            l0 = self.inputs['l0']
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            M_total_perturbed = 2 * perturbed_m
            
            # Recalculate expected values using physics formulas
            delta_x0_perturbed = (M_total_perturbed * g) / k
            h_perturbed = l0 - delta_x0_perturbed
            a_m_perturbed = (k * delta_x0_perturbed) / M_total_perturbed
            N_perturbed = perturbed_m * (g + a_m_perturbed)
            
            # Get original values
            delta_x0_orig = self.results['delta_x0']
            h_orig = self.results['h']
            a_m_orig = self.results['a_m']
            N_orig = self.results['N']
            
            # Check monotonic trends
            trends_ok = True
            
            # 1. Δx₀ should increase with mass (spring compresses more)
            if delta_x0_perturbed <= delta_x0_orig:
                self._log(f"Monotonicity: Δx₀ should increase with mass (got {delta_x0_perturbed:.4f} ≤ {delta_x0_orig:.4f})", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: Δx₀ increases with mass ✓", True)
                
            # 2. h should decrease with mass (equilibrium position lowers)
            if h_perturbed >= h_orig:
                self._log(f"Monotonicity: h should decrease with mass (got {h_perturbed:.4f} ≥ {h_orig:.4f})", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: h decreases with mass ✓", True)
                
            # 3. a_m should remain constant (since a_m = kΔx₀/(2m) = g, independent of m!)
            # Actually: a_m = (kΔx₀)/(2m) = (k*(2mg/k))/(2m) = g
            # So a_m should equal g regardless of mass
            a_m_check = math.isclose(a_m_perturbed, g, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity: a_m constant = g = {g:.1f}m/s² ✓", a_m_check)
            trends_ok = trends_ok and a_m_check
            
            # 4. N should increase with mass (N = m(g + a_m) = 2mg)
            if N_perturbed <= N_orig:
                self._log(f"Monotonicity: N should increase with mass (got {N_perturbed:.3f} ≤ {N_orig:.3f})", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: N increases with mass ✓", True)
            
            return trends_ok
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
            
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        try:
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = test1 and test2 and test3
            
            self.logs.append(f"\nSummary: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
            self.logs.append(f"  Invariants: {'PASS' if test1 else 'FAIL'}")
            self.logs.append(f"  Trace consistency: {'PASS' if test2 else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if test3 else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"\nCritical error: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
            
        return success, self.logs