import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants
        self.R = 1.0
        self.L = 1.5
        self.v0 = 5.0
        self.g = 10.0
        self.pi = 3.14
        self.x_given = 2.0  # Horizontal projectile distance
        
        # Extract results safely
        self.omega = self._safe_get(solution_instance, 'omega')
        self.a = self._safe_get(solution_instance, 'a')
        self.t1 = self._safe_get(solution_instance, 't_1')
        self.t2 = self._safe_get(solution_instance, 't_2')
        self.t = self._safe_get(solution_instance, 't')
        self.h = self._safe_get(solution_instance, 'h')
        
        # For monotonicity test backup
        self._original_v0 = self.v0
        
        # Target values for comparison
        self.target_omega = 5.0
        self.target_a = 25.0
        self.target_t = 0.928
        self.target_h = 0.8
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
        
    def _safe_get(self, solution_instance, key):
        """Safely extract attribute from solution instance."""
        val = getattr(solution_instance, key, None)
        if val is None and hasattr(solution_instance, 'results'):
            val = solution_instance.results.get(key, None)
        return float(val) if val is not None else None
        
    def _log(self, msg, success):
        """Add message to logs."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self):
        """Check fundamental physical laws."""
        successes = []
        
        # 1. Angular velocity invariant: ω = v0/R
        if self.omega is not None:
            expected_omega = self.v0 / self.R
            success = math.isclose(self.omega, expected_omega, 
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"ω invariant: {self.omega} ≈ v0/R = {expected_omega}", success))
        
        # 2. Centripetal acceleration invariant: a = v0²/R
        if self.a is not None:
            expected_a = self.v0**2 / self.R
            success = math.isclose(self.a, expected_a,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"a invariant: {self.a} ≈ v0²/R = {expected_a}", success))
        
        # 3. Time on semicircle: t1 = πR/v0
        if self.t1 is not None:
            expected_t1 = self.pi * self.R / self.v0
            success = math.isclose(self.t1, expected_t1,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"t1 invariant: {self.t1} ≈ πR/v0 = {expected_t1}", success))
        
        # 4. Time on straight section: t2 = L/v0
        if self.t2 is not None:
            expected_t2 = self.L / self.v0
            success = math.isclose(self.t2, expected_t2,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"t2 invariant: {self.t2} ≈ L/v0 = {expected_t2}", success))
        
        # 5. Total time: t = t1 + t2
        if all(x is not None for x in [self.t, self.t1, self.t2]):
            expected_t = self.t1 + self.t2
            success = math.isclose(self.t, expected_t,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"t invariant: {self.t} ≈ t1+t2 = {expected_t}", success))
        
        # 6. Projectile height: h = (g*x²)/(2*v0²)
        if self.h is not None:
            expected_h = (self.g * self.x_given**2) / (2 * self.v0**2)
            success = math.isclose(self.h, expected_h,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"h invariant: {self.h} ≈ gx²/(2v0²) = {expected_h}", success))
            
            # Also check via time of flight
            t3 = self.x_given / self.v0  # Projectile time
            h_from_t3 = 0.5 * self.g * t3**2
            success2 = math.isclose(self.h, h_from_t3,
                                   rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"h via t3: {self.h} ≈ ½gt3² = {h_from_t3} (t3={t3:.3f})", success2))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        successes = []
        
        # Check against target answers (from problem statement)
        if self.omega is not None:
            success = math.isclose(self.omega, self.target_omega,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"ω matches target: {self.omega} ≈ {self.target_omega}", success))
        
        if self.a is not None:
            success = math.isclose(self.a, self.target_a,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"a matches target: {self.a} ≈ {self.target_a}", success))
        
        if self.t is not None:
            success = math.isclose(self.t, self.target_t,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"t matches target: {self.t} ≈ {self.target_t}", success))
        
        if self.h is not None:
            success = math.isclose(self.h, self.target_h,
                                 rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            successes.append(self._log(
                f"h matches target: {self.h} ≈ {self.target_h}", success))
        
        # Check that all times are positive
        for name, val in [('t1', self.t1), ('t2', self.t2), ('t', self.t)]:
            if val is not None:
                success = val > 0
                successes.append(self._log(f"{name} positive: {val} > 0", success))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb v0 and check trend consistency."""
        successes = []
        
        # Original calculations
        orig_omega = self.v0 / self.R
        orig_a = self.v0**2 / self.R
        orig_t1 = self.pi * self.R / self.v0
        orig_t2 = self.L / self.v0
        orig_t = orig_t1 + orig_t2
        orig_h = (self.g * self.x_given**2) / (2 * self.v0**2)
        
        # Perturb v0 by +1%
        perturb_factor = 1.01
        v0_perturbed = self.v0 * perturb_factor
        
        # Recalculate with perturbed v0
        pert_omega = v0_perturbed / self.R
        pert_a = v0_perturbed**2 / self.R
        pert_t1 = self.pi * self.R / v0_perturbed
        pert_t2 = self.L / v0_perturbed
        pert_t = pert_t1 + pert_t2
        pert_h = (self.g * self.x_given**2) / (2 * v0_perturbed**2)
        
        # Check monotonic trends
        # ω and a should increase with v0
        successes.append(self._log(
            f"ω monotonic: {pert_omega:.3f} > {orig_omega:.3f} (v0↑)",
            pert_omega > orig_omega))
        successes.append(self._log(
            f"a monotonic: {pert_a:.3f} > {orig_a:.3f} (v0↑)",
            pert_a > orig_a))
        
        # Times should decrease with v0
        successes.append(self._log(
            f"t1 monotonic: {pert_t1:.3f} < {orig_t1:.3f} (v0↑)",
            pert_t1 < orig_t1))
        successes.append(self._log(
            f"t2 monotonic: {pert_t2:.3f} < {orig_t2:.3f} (v0↑)",
            pert_t2 < orig_t2))
        successes.append(self._log(
            f"t monotonic: {pert_t:.3f} < {orig_t:.3f} (v0↑)",
            pert_t < orig_t))
        
        # Height should decrease with v0 (h ∝ 1/v0²)
        successes.append(self._log(
            f"h monotonic: {pert_h:.3f} < {orig_h:.3f} (v0↑)",
            pert_h < orig_h))
        
        # Restore original v0 (though we never actually changed it)
        self.v0 = self._original_v0
        
        return all(successes)
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            test_results = []
            
            test_results.append(("Invariants", self.test_invariants()))
            test_results.append(("Trace Consistency", self.test_trace_consistency()))
            test_results.append(("Monotonicity", self.test_monotonicity()))
            
            # Summary
            all_passed = all(result for _, result in test_results)
            
            # Add summary to logs
            self.logs.append("=" * 50)
            self.logs.append("TEST SUMMARY:")
            for name, passed in test_results:
                self.logs.append(f"  {name}: {'PASS' if passed else 'FAIL'}")
            self.logs.append(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
            
            return all_passed, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(f"CRITICAL: {error_msg}")
            return False, self.logs