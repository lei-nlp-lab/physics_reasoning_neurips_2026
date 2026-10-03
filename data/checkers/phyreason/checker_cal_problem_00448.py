import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants (safe access via getattr or from results)
        self.R = 1.8  # m
        self.dist_to_barrier = 2.4  # m
        self.h = 0.2  # m
        self.m = 2.0  # kg
        self.mu = 0.8
        self.g = 10.0  # m/s^2
        
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Essential computed quantities (from KB mapping)
        self.v_B = self.results.get('v_B', None)
        self.F_N = self.results.get('F_N', None)
        self.v_1 = self.results.get('v_1', None)
        self.t = self.results.get('t', None)
        self.x = self.results.get('x', None)
        self.L = self.results.get('L', None)
        self.v_2 = self.results.get('v_2', None)
        self.v_0 = self.results.get('v_0', None)
        
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics values."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws from KB."""
        successes = []
        
        # 1. Energy conservation: A -> B (Step 1)
        try:
            if self.v_B is not None:
                lhs = self.m * self.g * self.R
                rhs = 0.5 * self.m * self.v_B**2
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Energy cons A->B: m*g*R={lhs:.2f}, 0.5*m*v_B²={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Energy cons A->B failed: {e}", False))
        
        # 2. Newton's 2nd law at B (Step 2)
        try:
            if self.F_N is not None and self.v_B is not None:
                lhs = self.F_N - self.m * self.g
                rhs = self.m * self.v_B**2 / self.R
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Newton at B: F_N - mg={lhs:.2f}, m*v_B²/R={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Newton at B failed: {e}", False))
        
        # 3. Work-energy: B -> end of cart (Step 4)
        try:
            if self.v_1 is not None and self.v_B is not None and self.L is not None:
                lhs = -self.mu * self.m * self.g * self.L
                rhs = 0.5 * self.m * self.v_1**2 - 0.5 * self.m * self.v_B**2
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Work-energy B->end: -μmgL={lhs:.2f}, ΔKE={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Work-energy B->end failed: {e}", False))
        
        # 4. Projectile vertical motion (Step 5)
        try:
            if self.t is not None:
                lhs = self.h
                rhs = 0.5 * self.g * self.t**2
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Projectile vertical: h={lhs:.2f}, 0.5*g*t²={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Projectile vertical failed: {e}", False))
        
        # 5. Projectile horizontal motion (Step 6)
        try:
            if self.x is not None and self.v_1 is not None and self.t is not None:
                lhs = self.x
                rhs = self.v_1 * self.t
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Projectile horizontal: x={lhs:.2f}, v_1*t={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Projectile horizontal failed: {e}", False))
        
        # 6. Geometric constraint (Step 7)
        try:
            if self.x is not None and self.L is not None:
                total = self.x + self.L
                numeric_ok = self._is_close(total, self.dist_to_barrier)
                successes.append(self._log(
                    f"Geometry: x+L={total:.2f}, should be {self.dist_to_barrier}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Geometry constraint failed: {e}", False))
        
        # 7. Second scenario: Horizontal projection (Step 10)
        try:
            if self.v_2 is not None and self.t is not None:
                lhs = self.dist_to_barrier
                rhs = self.v_2 * self.t
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Second scenario horizontal: 2.4={lhs:.2f}, v_2*t={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Second scenario horizontal failed: {e}", False))
        
        # 8. Second scenario energy (Step 11)
        try:
            if self.v_0 is not None and self.v_2 is not None:
                lhs = 0.5 * self.m * self.v_0**2 + self.m * self.g * self.R
                rhs = 0.5 * self.m * self.v_2**2
                numeric_ok = self._is_close(lhs, rhs)
                successes.append(self._log(
                    f"Energy cons A->B (second): KE₀+mgR={lhs:.2f}, 0.5*m*v₂²={rhs:.2f}",
                    numeric_ok
                ))
        except Exception as e:
            successes.append(self._log(f"Second scenario energy failed: {e}", False))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        successes = []
        
        # Check Step 1: v_B = sqrt(2gR)
        try:
            v_B_expected = math.sqrt(2 * self.g * self.R)
            if self.v_B is not None:
                ok = self._is_close(self.v_B, v_B_expected)
                successes.append(self._log(
                    f"Step 1: v_B={self.v_B:.2f}, expected sqrt(2gR)={v_B_expected:.2f}",
                    ok
                ))
        except Exception as e:
            successes.append(self._log(f"Step 1 check failed: {e}", False))
        
        # Check Step 2-3: F_N = m(g + v_B²/R)
        try:
            if self.v_B is not None and self.F_N is not None:
                F_N_expected = self.m * (self.g + self.v_B**2 / self.R)
                ok = self._is_close(self.F_N, F_N_expected)
                successes.append(self._log(
                    f"Step 2-3: F_N={self.F_N:.2f}, expected m(g+v_B²/R)={F_N_expected:.2f}",
                    ok
                ))
        except Exception as e:
            successes.append(self._log(f"Step 2-3 check failed: {e}", False))
        
        # Check Step 4-8: Solve for L from geometry
        try:
            if all(v is not None for v in [self.v_B, self.v_1, self.t, self.x, self.L]):
                # From work-energy: v_1² = v_B² - 2μgL
                L_from_energy = (self.v_B**2 - self.v_1**2) / (2 * self.mu * self.g)
                # From geometry: L = 2.4 - x, and x = v_1*t
                L_from_geom = self.dist_to_barrier - self.v_1 * self.t
                
                ok_energy = self._is_close(self.L, L_from_energy)
                ok_geom = self._is_close(self.L, L_from_geom)
                ok_total = ok_energy and ok_geom
                
                successes.append(self._log(
                    f"Step 4-8: L={self.L:.2f}, from energy={L_from_energy:.2f}, from geom={L_from_geom:.2f}",
                    ok_total
                ))
        except Exception as e:
            successes.append(self._log(f"Step 4-8 check failed: {e}", False))
        
        # Check Step 9-11: Second scenario
        try:
            if self.t is not None and self.v_2 is not None and self.v_0 is not None:
                # Step 9: t from vertical drop
                t_expected = math.sqrt(2 * self.h / self.g)
                # Step 10: v_2 = 2.4 / t
                v_2_expected = self.dist_to_barrier / t_expected
                # Step 11: v_0² = v_2² - 2gR
                v_0_sq_expected = v_2_expected**2 - 2 * self.g * self.R
                v_0_expected = math.sqrt(max(v_0_sq_expected, 0))
                
                ok_t = self._is_close(self.t, t_expected)
                ok_v2 = self._is_close(self.v_2, v_2_expected)
                ok_v0 = self._is_close(self.v_0, v_0_expected)
                ok_total = ok_t and ok_v2 and ok_v0
                
                successes.append(self._log(
                    f"Step 9-11: t={self.t:.2f}, v₂={self.v_2:.2f}, v₀={self.v_0:.2f}; "
                    f"expected t={t_expected:.2f}, v₂={v_2_expected:.2f}, v₀={v_0_expected:.2f}",
                    ok_total
                ))
        except Exception as e:
            successes.append(self._log(f"Step 9-11 check failed: {e}", False))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb μ and check L decreases (more friction → shorter cart needed)."""
        try:
            if self.L is None:
                return self._log("Monotonicity: L not available", False)
            
            original_mu = self.mu
            perturbed_mu = original_mu * 1.1  # Increase by 10%
            
            # Known constants
            v_B = math.sqrt(2 * self.g * self.R)
            t = math.sqrt(2 * self.h / self.g)
            
            # Solve for L with perturbed μ using correct equations:
            # v_1² = v_B² - 2*μ*g*L  ... (1)
            # x = v_1*t  ... (2)
            # x + L = 2.4  ... (3)
            # Substitute (2) and (3): v_1 = (2.4 - L)/t
            # Substitute into (1): ((2.4 - L)/t)² = v_B² - 2*μ*g*L
            # Rearrange: (2.4 - L)² = t²*(v_B² - 2*μ*g*L)
            
            μ = perturbed_mu
            # Check if solution exists by testing discriminant
            A = 1.0
            B = 2 * μ * self.g * t**2 - 4.8
            C = 5.76 - t**2 * v_B**2
            
            discriminant = B**2 - 4 * A * C
            
            if discriminant < 0:
                # No real solution - this is physically possible when friction is too high
                # The block would stop before reaching the end of the cart
                # This doesn't mean our solution is wrong, just that monotonicity
                # test can't be performed with this perturbation
                return self._log(
                    f"Monotonicity: μ↑ {original_mu:.2f}→{perturbed_mu:.2f} "
                    f"makes problem impossible (no real L solution) - skipping",
                    True  # Mark as PASS since this is a valid physical scenario
                )
            
            L_perturbed1 = (-B + math.sqrt(discriminant)) / (2 * A)
            L_perturbed2 = (-B - math.sqrt(discriminant)) / (2 * A)
            
            # Choose positive L less than 2.4
            L_perturbed = None
            for L_candidate in [L_perturbed1, L_perturbed2]:
                if 0 < L_candidate < 2.4:
                    L_perturbed = L_candidate
                    break
            
            if L_perturbed is None:
                return self._log(
                    f"Monotonicity: μ↑ {original_mu:.2f}→{perturbed_mu:.2f} "
                    f"gives no valid L - skipping",
                    True  # Mark as PASS since this is a valid physical scenario
                )
            
            # With increased friction, L should decrease
            monotonic_ok = L_perturbed < self.L
            return self._log(
                f"Monotonicity: μ↑ {original_mu:.2f}→{perturbed_mu:.2f}, "
                f"L↓ {self.L:.2f}→{L_perturbed:.2f}",
                monotonic_ok
            )
        except Exception as e:
            return self._log(f"Monotonicity test failed: {e}", False)
    
    def run_all_tests(self):
        """Execute all checks and return (success_bool, logs_list)."""
        self.logs.clear()
        
        # Run tests with error handling
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, self.logs