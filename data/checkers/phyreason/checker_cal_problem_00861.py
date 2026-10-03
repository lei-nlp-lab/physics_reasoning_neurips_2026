import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr with fallback to .get() on a results dictionary.
        """
        # Extract inputs with safe defaults
        self.B = getattr(solution_instance, 'B', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.n = getattr(solution_instance, 'n', 1)  # integer, default 1
        # U might be an input or a result; we try both
        self.U_input = getattr(solution_instance, 'U', None)
        
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract result quantities with safe fallbacks
        self.r = self.results.get('r', 0.0)
        self.U = self.results.get('U', 0.0)
        self.T = self.results.get('T', 0.0)
        self.t_min = self.results.get('t_min', 0.0)
        self.theta = self.results.get('theta', 0.0)  # in radians
        self.tan_half_theta = self.results.get('tanfractheta2', 0.0)  # from mapping
        self.kinetic_energy = self.results.get('mv^2/2', 0.0)  # KE = 1/2 m v^2
        
        # If U_input is provided and U is not in results, use U_input
        if self.U_input is not None and self.U == 0.0:
            self.U = self.U_input
            
        # Compute derived quantities for checking
        self.v = None
        if self.kinetic_energy > 0:
            self.v = math.sqrt(2 * self.kinetic_energy / self.m) if self.m > 0 else 0.0
        elif self.r > 0 and self.B > 0 and self.q != 0:
            # from r = m v / (q B) => v = q B r / m
            self.v = self.q * self.B * self.r / self.m if self.m > 0 else 0.0
        else:
            self.v = 0.0
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Work-Energy Theorem: qU = 1/2 m v^2
            expected_ke = self.q * self.U
            if self.kinetic_energy > 0:
                if not self._is_close(self.kinetic_energy, expected_ke):
                    logs.append(f"Work-Energy violation: KE from work {expected_ke} != KE stored {self.kinetic_energy}")
                    success = False
                else:
                    logs.append("Work-Energy Theorem satisfied.")
            else:
                logs.append("No kinetic energy data to check Work-Energy.")
                
            # 2. Lorentz force as centripetal force: q v B = m v^2 / r
            if self.v > 0 and self.r > 0:
                lhs = self.q * self.v * self.B
                rhs = self.m * self.v**2 / self.r
                if not self._is_close(lhs, rhs):
                    logs.append(f"Lorentz force violation: qvB = {lhs} != mv^2/r = {rhs}")
                    success = False
                else:
                    logs.append("Lorentz force as centripetal force satisfied.")
                    
            # 3. Period of circular motion: T = 2πm/(qB)
            expected_T = 2 * math.pi * self.m / (self.q * self.B) if self.q * self.B != 0 else 0
            if self.T > 0 and not self._is_close(self.T, expected_T):
                logs.append(f"Period violation: T expected {expected_T} != T given {self.T}")
                success = False
            elif self.T > 0:
                logs.append("Period formula satisfied.")
                
            # 4. Time from angle for MINIMUM TIME CASE: t_min should use θ = π/4, not self.theta
            # For minimum time (one collision), θ = π/4
            theta_min = math.pi / 4
            expected_t_min_min = (theta_min / math.pi) * self.T
            if not self._is_close(self.t_min, expected_t_min_min):
                logs.append(f"Minimum time violation: t_min expected {expected_t_min_min} != t_min given {self.t_min}")
                success = False
            else:
                logs.append("Minimum time correct (θ = π/4).")
                    
            # 5. Geometric relation: tan(θ/2) = R / r
            if self.r > 0 and self.theta > 0:
                expected_tan = math.tan(self.theta / 2)
                if not self._is_close(self.tan_half_theta, expected_tan):
                    logs.append(f"Geometric relation violation: tan(θ/2) computed {expected_tan} != given {self.tan_half_theta}")
                    success = False
                else:
                    logs.append("Geometric relation satisfied.")
                    
            # 6. For n=1 in GENERAL CASE, θ should be 3π/8, not π/4
            # π/4 is for the minimum time specific case, not for the general formula
            if self.n == 1:
                expected_theta_general = 3 * math.pi / 8  # 3π/(4*(1+1)) = 3π/8
                if not self._is_close(self.theta, expected_theta_general):
                    logs.append(f"For n=1 in general case, theta should be 3π/8 ({expected_theta_general}) but got {self.theta}")
                    success = False
                else:
                    logs.append("Theta matches 3π/8 for n=1 general case.")
                    
        except Exception as e:
            logs.append(f"Error in invariant tests: {e}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        logs = []
        success = True
        
        try:
            # Check that radius from step_2 (r = m v / (q B)) and step_3 (r = (1/B)*sqrt(2Um/q)) are consistent
            if self.v > 0:
                r_from_step2 = self.m * self.v / (self.q * self.B) if self.q * self.B != 0 else 0
                r_from_step3 = (1/self.B) * math.sqrt(2 * self.U * self.m / self.q) if self.q > 0 else 0
                if not self._is_close(r_from_step2, r_from_step3):
                    logs.append(f"Radius inconsistency: from Lorentz {r_from_step2} != from energy {r_from_step3}")
                    success = False
                else:
                    logs.append("Radius consistent between step_2 and step_3.")
                    
            # Check that kinetic energy from step_1 (qU) matches the velocity computed from radius and B
            if self.r > 0 and self.B > 0 and self.q != 0:
                v_from_r = self.q * self.B * self.r / self.m if self.m > 0 else 0
                ke_from_v = 0.5 * self.m * v_from_r**2
                if not self._is_close(self.kinetic_energy, ke_from_v):
                    logs.append(f"KE inconsistency: from work {self.kinetic_energy} != from v(r) {ke_from_v}")
                    success = False
                else:
                    logs.append("Kinetic energy consistent with radius and magnetic field.")
                    
            # Check that U from the geometric relation matches the given U
            # From tan(θ/2) = R/r and r = (1/B)*sqrt(2Um/q)
            # We get: U = (B²R²q)/(2m) * cot²(θ/2)
            if self.theta > 0:
                expected_U = (self.B**2 * self.R**2 * self.q) / (2 * self.m) * (1 / math.tan(self.theta / 2)**2)
                if not self._is_close(self.U, expected_U):
                    logs.append(f"U inconsistency: given {self.U} != formula from geometry {expected_U}")
                    success = False
                else:
                    logs.append("U consistent with geometric formula.")
                
        except Exception as e:
            logs.append(f"Error in trace consistency tests: {e}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb an input and check the trend of results."""
        logs = []
        success = True
        
        try:
            # Perturb magnetic field B by +10%
            original_B = self.B
            perturbed_B = original_B * 1.1
            
            # Recompute expected changes using formulas
            # Period T = 2πm/(qB) -> should decrease by ~9.1%
            original_T = 2 * math.pi * self.m / (self.q * original_B) if self.q * original_B != 0 else 0
            perturbed_T = 2 * math.pi * self.m / (self.q * perturbed_B) if self.q * perturbed_B != 0 else 0
            if not perturbed_T < original_T:
                logs.append(f"Monotonicity violation: T did not decrease when B increased (original {original_T}, perturbed {perturbed_T})")
                success = False
            else:
                logs.append(f"T decreases with B: {original_T} -> {perturbed_T}")
                
            # Radius r = (1/B)*sqrt(2Um/q). For fixed U, r ∝ 1/B
            # But U itself depends on B: U = (B²R²q)/(2m) * cot²(θ/2) ∝ B²
            # So r = (1/B)*sqrt(2 * [(B²R²q)/(2m) * cot²(θ/2)] * m/q) = R * cot(θ/2)
            # So r should be independent of B! Let's check:
            original_r = self.r
            # For perturbed B, recompute U first
            perturbed_U = (perturbed_B**2 * self.R**2 * self.q) / (2 * self.m) * (1 / math.tan(self.theta / 2)**2)
            perturbed_r = (1/perturbed_B) * math.sqrt(2 * perturbed_U * self.m / self.q) if self.q > 0 else 0
            # r should be constant: r = R * cot(θ/2)
            expected_r = self.R / math.tan(self.theta / 2)
            if not self._is_close(original_r, expected_r) or not self._is_close(perturbed_r, expected_r):
                logs.append(f"Radius should be constant R*cot(θ/2) = {expected_r}, but got original {original_r}, perturbed {perturbed_r}")
                success = False
            else:
                logs.append(f"Radius independent of B as expected: {original_r}")
                
        except Exception as e:
            logs.append(f"Error in monotonicity tests: {e}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"{test_name} FAILED.")
                else:
                    all_logs.append(f"{test_name} passed.")
            except Exception as e:
                all_logs.append(f"Exception in {test_name}: {traceback.format_exc()}")
                overall_success = False
                
        all_logs.append("="*40)
        if overall_success:
            all_logs.append("All physics checks passed.")
        else:
            all_logs.append("Some physics checks failed.")
            
        return overall_success, all_logs