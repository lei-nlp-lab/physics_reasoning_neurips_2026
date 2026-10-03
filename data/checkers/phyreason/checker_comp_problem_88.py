import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using robust attribute access."""
        # Extract inputs with safe fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.mu0 = getattr(solution_instance, 'mu0', getattr(solution_instance, 'inputs', {}).get('mu0', 0.5))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'inputs', {}).get('mu', 0.3))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'inputs', {}).get('k', 100.0))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'inputs', {}).get('v0', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.81))
        
        # Extract results with safe fallbacks using the provided mapping
        results = getattr(solution_instance, 'results', {})
        self.s = getattr(solution_instance, 's', results.get('s', 0.0))
        self.delta_l = getattr(solution_instance, 'delta_l', results.get('delta_l', 0.0))
        self.delta_x = getattr(solution_instance, 'delta_x', results.get('delta_x', 0.0))
        self.t0 = getattr(solution_instance, 't0', results.get('t0', 0.0))
        self.delta_t = getattr(solution_instance, 'delta_t', results.get('delta_t', 0.0))
        self.a_expr = getattr(solution_instance, 'a', results.get('a', None))
        
        # Check if we have symbolic expressions
        self.xt_expr = getattr(solution_instance, 'xt', results.get('xt', None))
        self.vt_expr = getattr(solution_instance, 'vt', results.get('vt', None))
        
        # Compute derived quantities
        self.omega = math.sqrt(self.k / self.m) if self.m > 0 else 0.0
        self.r = self.mu / self.mu0 if self.mu0 > 0 else 0.0
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check static friction condition: k*s = μ0*m*g
            lhs_static = self.k * self.s
            rhs_static = self.mu0 * self.m * self.g
            if not math.isclose(lhs_static, rhs_static, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Static friction condition failed. k*s = {lhs_static}, μ0*m*g = {rhs_static}")
                success = False
            else:
                logs.append(f"PASS: Static friction condition: k*s = μ0*m*g")
            
            # 2. Check Newton's Second Law consistency - only if we have expressions
            # Skip if expressions are not available or are numeric
            if self.a_expr is not None and self.xt_expr is not None:
                # Check if they are numeric (float/int) or symbolic
                a_is_numeric = isinstance(self.a_expr, (int, float))
                xt_is_numeric = isinstance(self.xt_expr, (int, float))
                
                if not a_is_numeric or not xt_is_numeric:
                    logs.append("INFO: Skipping Newton check - requires symbolic expressions")
                else:
                    # Both are numeric, check at t=0
                    t_val = 0.0
                    x_val = self.xt_expr  # xt is already evaluated at t=0
                    a_expected = (self.k/self.m)*(self.s + self.v0*t_val - x_val) - self.mu*self.g
                    if not math.isclose(self.a_expr, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Newton's Law mismatch at t=0: {self.a_expr} vs {a_expected}")
                        success = False
                    else:
                        logs.append(f"PASS: Newton's Law holds at t=0")
            
            # 3. Check kinetic friction magnitude
            f_k_expected = self.mu * self.m * self.g
            logs.append(f"INFO: Kinetic friction magnitude = {f_k_expected} N")
            
            # 4. Check initial conditions - only if we have expressions
            if self.xt_expr is not None and self.vt_expr is not None:
                x0_is_numeric = isinstance(self.xt_expr, (int, float))
                v0_is_numeric = isinstance(self.vt_expr, (int, float))
                
                if x0_is_numeric and v0_is_numeric:
                    if not math.isclose(self.xt_expr, 0.0, abs_tol=1e-2):
                        logs.append(f"FAIL: Initial position not zero: x(0) = {self.xt_expr}")
                        success = False
                    if not math.isclose(self.vt_expr, 0.0, abs_tol=1e-2):
                        logs.append(f"FAIL: Initial velocity not zero: v(0) = {self.vt_expr}")
                        success = False
                    if math.isclose(self.xt_expr, 0.0, abs_tol=1e-2) and math.isclose(self.vt_expr, 0.0, abs_tol=1e-2):
                        logs.append(f"PASS: Initial conditions: x(0)=0, v(0)=0")
                else:
                    logs.append("INFO: Skipping initial condition check - requires numeric values")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate results and final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check spring stretch at stopping: Δl = (2r-1)s
            delta_l_expected = (2*self.r - 1) * self.s
            if not math.isclose(self.delta_l, delta_l_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Spring stretch mismatch: Δl = {self.delta_l}, expected (2r-1)s = {delta_l_expected}")
                success = False
            else:
                logs.append(f"PASS: Spring stretch at stopping: Δl = (2r-1)s")
            
            # 2. Check stopping time t0 via alpha
            if self.t0 > 0:
                alpha = math.atan((1 - self.r) * self.s * self.omega / self.v0)
                t0_expected = (2*math.pi - 2*alpha) / self.omega
                
                if not math.isclose(self.t0, t0_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Stopping time mismatch: t0 = {self.t0}, expected = {t0_expected}")
                    success = False
                else:
                    logs.append(f"PASS: Stopping time t0 = (2π-2α)/ω with α = arctan((1-r)sω/v0)")
            
            # 3. Check rest time Δt = 2(1-r)s/v0
            delta_t_expected = 2 * (1 - self.r) * self.s / self.v0 if self.v0 > 0 else 0.0
            if not math.isclose(self.delta_t, delta_t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Rest time mismatch: Δt = {self.delta_t}, expected 2(1-r)s/v0 = {delta_t_expected}")
                success = False
            else:
                logs.append(f"PASS: Rest time Δt = 2(1-r)s/v0")
            
            # 4. Check distance needed to move again: Δx = 2(1-r)s
            delta_x_expected = 2 * (1 - self.r) * self.s
            if not math.isclose(self.delta_x, delta_x_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Distance mismatch: Δx = {self.delta_x}, expected 2(1-r)s = {delta_x_expected}")
                success = False
            else:
                logs.append(f"PASS: Distance to move again: Δx = 2(1-r)s")
            
            # 5. Verify velocity at t0 is zero - skip if vt_expr is numeric
            if self.vt_expr is not None and self.t0 > 0:
                if isinstance(self.vt_expr, (int, float)):
                    logs.append("INFO: Skipping velocity at t0 check - vt is numeric, not a function")
                else:
                    logs.append("INFO: Could not check velocity at t0 - vt is not numeric")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic trends by perturbing inputs."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase μ0 → s should increase
            mu0_original = self.mu0
            mu0_perturbed = mu0_original * 1.1  # +10%
            s_original = self.s
            s_perturbed = (mu0_perturbed * self.m * self.g) / self.k
            
            if s_perturbed <= s_original:
                logs.append(f"FAIL: Monotonicity - s should increase with μ0")
                success = False
            else:
                logs.append(f"PASS: s increases with μ0 (as expected)")
            
            # Test 2: Increase k → s should decrease
            k_original = self.k
            k_perturbed = k_original * 1.1
            s_k_perturbed = (self.mu0 * self.m * self.g) / k_perturbed
            
            if s_k_perturbed >= s_original:
                logs.append(f"FAIL: Monotonicity - s should decrease with k")
                success = False
            else:
                logs.append(f"PASS: s decreases with k (as expected)")
            
            # Test 3: Increase v0 → t0 should decrease? Actually, α increases with v0 in denominator, so t0 changes
            # Check that the relationship makes sense physically
            v0_original = self.v0
            v0_perturbed = v0_original * 1.2
            alpha_original = math.atan((1 - self.r) * self.s * self.omega / v0_original)
            alpha_perturbed = math.atan((1 - self.r) * self.s * self.omega / v0_perturbed)
            
            # As v0 increases, α decreases (since denominator increases), so t0 should increase
            t0_original = (2*math.pi - 2*alpha_original) / self.omega
            t0_perturbed = (2*math.pi - 2*alpha_perturbed) / self.omega
            
            if t0_perturbed <= t0_original:
                logs.append(f"WARN: t0 behavior with v0 may be non-monotonic - check carefully")
            else:
                logs.append(f"PASS: t0 increases with v0 (as α decreases)")
            
            # Test 4: Increase μ (kinetic friction) → Δt should decrease
            mu_original = self.mu
            mu_perturbed = min(mu_original * 1.2, self.mu0 * 0.99)  # Keep less than μ0
            r_perturbed = mu_perturbed / self.mu0
            delta_t_original = 2 * (1 - self.r) * self.s / self.v0
            delta_t_perturbed = 2 * (1 - r_perturbed) * self.s / self.v0
            
            if delta_t_perturbed >= delta_t_original:
                logs.append(f"FAIL: Monotonicity - Δt should decrease with μ")
                success = False
            else:
                logs.append(f"PASS: Δt decreases with μ (as expected)")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        try:
            # Run each test
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== Running {test_name} Test ===")
                try:
                    success, logs = test_method()
                    for log in logs:
                        all_logs.append(log)
                    if not success:
                        overall_success = False
                except Exception as e:
                    all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                    overall_success = False
            
            # Summary
            all_logs.append(f"\n=== SUMMARY ===")
            all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
            
        except Exception as e:
            all_logs.append(f"FATAL ERROR: {traceback.format_exc()}")
            overall_success = False
        
        return overall_success, all_logs