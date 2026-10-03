import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs using getattr
        self.L = getattr(solution_instance, 'L', 0.5)  # m
        self.B = getattr(solution_instance, 'B', 2.0)  # T
        self.m = getattr(solution_instance, 'm', 1.0)  # kg
        self.G = getattr(solution_instance, 'G', 3.0)  # N
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.E = getattr(solution_instance, 'E', 10.0)  # V
        self.r = getattr(solution_instance, 'r', 0.1)  # Ω
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²

        # Safely extract results with fallback
        sol_results = getattr(solution_instance, 'results', {})
        self.R1 = sol_results.get('R₁', sol_results.get('R1', None))
        self.R2 = sol_results.get('R₂', sol_results.get('R2', None))
        self.I1 = sol_results.get('I₁', sol_results.get('I1', None))
        self.I2 = sol_results.get('I₂', sol_results.get('I2', None))
        self.R_range = sol_results.get('R', None)  # Could be tuple, list, or string
        
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((success, msg))
    
    def _is_close(self, a, b):
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def _extract_range(self):
        """Extract numerical min/max from range data."""
        if self.R_range is None:
            return None, None
        
        # Handle different formats
        if isinstance(self.R_range, (tuple, list)) and len(self.R_range) >= 2:
            return float(self.R_range[0]), float(self.R_range[1])
        elif isinstance(self.R_range, str):
            import re
            nums = re.findall(r'[\d.]+', self.R_range)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
        return None, None
    
    def test_invariants(self):
        """Check force balance and Ohm's law."""
        success = True
        
        # 1. Force balance for I1 (about to slide right): G - μmg - B*I1*L = 0
        mu_mg = self.mu * self.m * self.g
        expected_I1 = (self.G - mu_mg) / (self.B * self.L)
        if self.I1 is not None:
            if not self._is_close(self.I1, expected_I1):
                self._log(f"Force balance for I1 failed: expected {expected_I1:.3f}, got {self.I1}", False)
                success = False
            else:
                self._log(f"Force balance for I1 correct: I1 = {self.I1:.3f} A", True)
        
        # 2. Force balance for I2 (about to slide left): B*I2*L - G - μmg = 0
        expected_I2 = (self.G + mu_mg) / (self.B * self.L)
        if self.I2 is not None:
            if not self._is_close(self.I2, expected_I2):
                self._log(f"Force balance for I2 failed: expected {expected_I2:.3f}, got {self.I2}", False)
                success = False
            else:
                self._log(f"Force balance for I2 correct: I2 = {self.I2:.3f} A", True)
        
        # 3. Ohm's law for R1: E = I1*(R1 + r)
        if self.I1 is not None and self.R1 is not None:
            expected_R1 = self.E / self.I1 - self.r
            if not self._is_close(self.R1, expected_R1):
                self._log(f"Ohm's law for R1 failed: expected {expected_R1:.3f}, got {self.R1}", False)
                success = False
            else:
                self._log(f"Ohm's law for R1 correct: R1 = {self.R1:.3f} Ω", True)
        
        # 4. Ohm's law for R2: E = I2*(R2 + r)
        if self.I2 is not None and self.R2 is not None:
            expected_R2 = self.E / self.I2 - self.r
            if not self._is_close(self.R2, expected_R2):
                self._log(f"Ohm's law for R2 failed: expected {expected_R2:.3f}, got {self.R2}", False)
                success = False
            else:
                self._log(f"Ohm's law for R2 correct: R2 = {self.R2:.3f} Ω", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results against final range."""
        success = True
        R_min, R_max = self._extract_range()
        
        # Check R1 and R2 against range
        if R_min is not None and R_max is not None:
            if self.R1 is not None and not self._is_close(self.R1, R_max):
                self._log(f"R1 inconsistent with range max: R1={self.R1}, range max={R_max}", False)
                success = False
            if self.R2 is not None and not self._is_close(self.R2, R_min):
                self._log(f"R2 inconsistent with range min: R2={self.R2}, range min={R_min}", False)
                success = False
            
            if self.R1 is not None and self.R2 is not None:
                if self.R1 < self.R2:
                    self._log(f"Range ordering wrong: R1={self.R1} < R2={self.R2}", False)
                    success = False
                else:
                    self._log(f"Range ordering correct: {self.R2:.1f} ≤ R ≤ {self.R1:.1f}", True)
        
        # Compute expected range from physics
        mu_mg = self.mu * self.m * self.g
        expected_I1 = (self.G - mu_mg) / (self.B * self.L)
        expected_I2 = (self.G + mu_mg) / (self.B * self.L)
        expected_R1 = self.E / expected_I1 - self.r
        expected_R2 = self.E / expected_I2 - self.r
        
        if R_min is not None and R_max is not None:
            if not (self._is_close(R_min, expected_R2) and self._is_close(R_max, expected_R1)):
                self._log(f"Range mismatch: expected [{expected_R2:.1f}, {expected_R1:.1f}], got [{R_min}, {R_max}]", False)
                success = False
            else:
                self._log(f"Range consistent with physics: {R_min:.1f} Ω ≤ R ≤ {R_max:.1f} Ω", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb G and check trend in resistance range."""
        success = True
        original_G = self.G
        
        # Perturb G by +10%
        perturbed_G = original_G * 1.1
        
        # Recompute expected range with perturbed G
        mu_mg = self.mu * self.m * self.g
        perturbed_I1 = (perturbed_G - mu_mg) / (self.B * self.L)
        perturbed_I2 = (perturbed_G + mu_mg) / (self.B * self.L)
        perturbed_R1 = self.E / perturbed_I1 - self.r
        perturbed_R2 = self.E / perturbed_I2 - self.r
        
        # Original expected range
        original_I1 = (original_G - mu_mg) / (self.B * self.L)
        original_I2 = (original_G + mu_mg) / (self.B * self.L)
        original_R1 = self.E / original_I1 - self.r
        original_R2 = self.E / original_I2 - self.r
        
        # Check trend: increasing G should decrease both R1 and R2
        # (since current increases, resistance decreases for fixed voltage)
        if perturbed_R1 >= original_R1:
            self._log(f"Monotonicity violated: R1 did NOT decrease when G increased", False)
            success = False
        else:
            self._log(f"Monotonicity: R1 decreases when G increases (from {original_R1:.2f} to {perturbed_R1:.2f})", True)
        
        if perturbed_R2 >= original_R2:
            self._log(f"Monotonicity violated: R2 did NOT decrease when G increased", False)
            success = False
        else:
            self._log(f"Monotonicity: R2 decreases when G increases (from {original_R2:.2f} to {perturbed_R2:.2f})", True)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {str(e)}", False)
                overall_success = False
        
        return overall_success, self.logs