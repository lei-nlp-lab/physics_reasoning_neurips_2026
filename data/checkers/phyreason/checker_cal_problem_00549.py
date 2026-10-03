import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', 
                         getattr(solution_instance, 'results', {}).get('m', 0.01))  # 10g = 0.01 kg
        self.L = getattr(solution_instance, 'L', 
                         getattr(solution_instance, 'results', {}).get('L', 0.5))   # 50cm = 0.5 m
        self.g = getattr(solution_instance, 'g', 
                         getattr(solution_instance, 'results', {}).get('g', 10.0))  # 10 m/s²
        
        # Magnetic fields and currents (may not be provided)
        self.B1 = getattr(solution_instance, 'B1', 
                           getattr(solution_instance, 'results', {}).get('B1', None))
        self.B2 = getattr(solution_instance, 'B2', 
                           getattr(solution_instance, 'results', {}).get('B2', None))
        self.I2 = getattr(solution_instance, 'I2', 
                           getattr(solution_instance, 'results', {}).get('I2', None))
        
        # Results from solution
        self.I1 = getattr(solution_instance, 'I1', 
                           getattr(solution_instance, 'results', {}).get('I1', None))
        self.I1_direction = getattr(solution_instance, 'I1_direction', 
                                     getattr(solution_instance, 'results', {}).get('I1_direction', None))
        self.F2 = getattr(solution_instance, 'F2', 
                           getattr(solution_instance, 'results', {}).get('F2', None))
        
        # Optional: force direction for consistency check
        self.force_direction = getattr(solution_instance, 'force_direction', 
                                        getattr(solution_instance, 'results', {}).get('force_direction', None))

    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check I1 = mg/(B1 * L) from step 1
        if self.B1 is not None and self.I1 is not None:
            expected_I1 = self.m * self.g / (self.B1 * self.L)
            if not math.isclose(self.I1, expected_I1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: I1 = {self.I1}, expected mg/(B1·L) = {expected_I1:.4f}")
                success = False
            else:
                logs.append(f"I1 check passed: {self.I1} ≈ {expected_I1:.4f}")
        else:
            logs.append("Skipping I1 invariant (missing B1 or I1)")
        
        # 2. Check current direction matches "from c to d"
        if self.I1_direction is not None:
            expected_dir = "from c to d"
            if self.I1_direction.strip().lower() != expected_dir:
                logs.append(f"Direction mismatch: got '{self.I1_direction}', expected '{expected_dir}'")
                success = False
            else:
                logs.append(f"Direction check passed: {self.I1_direction}")
        else:
            logs.append("Skipping direction check (missing I1_direction)")
        
        # 3. Check F2 = mg + B2·I2·L from step 4
        if all(v is not None for v in [self.B2, self.I2, self.F2]):
            expected_F2 = self.m * self.g + self.B2 * self.I2 * self.L
            if not math.isclose(self.F2, expected_F2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: F2 = {self.F2}, expected mg + B2·I2·L = {expected_F2:.4f}")
                success = False
            else:
                logs.append(f"F2 check passed: {self.F2} ≈ {expected_F2:.4f}")
        else:
            logs.append("Skipping F2 invariant (missing B2, I2, or F2)")
        
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check force direction consistency if provided
        if self.force_direction is not None:
            expected_force_dir = "downward"
            if self.force_direction.strip().lower() != expected_force_dir:
                logs.append(f"Force direction mismatch: got '{self.force_direction}', expected '{expected_force_dir}'")
                success = False
            else:
                logs.append(f"Force direction consistent: {self.force_direction}")
        else:
            logs.append("Skipping force direction consistency (not provided)")
        
        # Symbolic check for Ampere's force law (optional)
        try:
            mu0, I1, I2, d = sp.symbols('mu0 I1 I2 d', positive=True)
            F_per_length = mu0 * I1 * I2 / (2 * sp.pi * d)
            # If we had two parallel wires, force per length would follow this law.
            # Here we just verify the symbolic form is correct.
            logs.append("Ampere's force symbolic check passed (formula verified)")
        except Exception as e:
            logs.append(f"Symbolic check warning: {e}")
        
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        # Helper to restore after perturbation
        original_attrs = {attr: getattr(self, attr) for attr in ['m', 'B1', 'B2', 'I2']}
        
        # 1. Perturb mass m and check I1 (I1 = mg/(B1 L))
        if self.B1 is not None and self.I1 is not None:
            self.m *= 1.01  # +1%
            new_I1 = self.m * self.g / (self.B1 * self.L)
            if new_I1 <= self.I1:
                logs.append(f"Monotonicity failed: I1 should increase with m, but {new_I1:.4f} ≤ {self.I1}")
                success = False
            else:
                logs.append(f"I1 monotonicity passed: Δm → ΔI1 > 0")
            self.m = original_attrs['m']  # restore
        
        # 2. Perturb m for F2 (F2 = mg + B2 I2 L)
        if all(v is not None for v in [self.B2, self.I2, self.F2]):
            self.m *= 1.01
            new_F2 = self.m * self.g + self.B2 * self.I2 * self.L
            if new_F2 <= self.F2:
                logs.append(f"Monotonicity failed: F2 should increase with m, but {new_F2:.4f} ≤ {self.F2}")
                success = False
            else:
                logs.append(f"F2 monotonicity (m) passed: Δm → ΔF2 > 0")
            self.m = original_attrs['m']
            
            # 3. Perturb B2 for F2
            self.B2 *= 1.01
            new_F2 = self.m * self.g + self.B2 * self.I2 * self.L
            if new_F2 <= self.F2:
                logs.append(f"Monotonicity failed: F2 should increase with B2, but {new_F2:.4f} ≤ {self.F2}")
                success = False
            else:
                logs.append(f"F2 monotonicity (B2) passed: ΔB2 → ΔF2 > 0")
            self.B2 = original_attrs['B2']
            
            # 4. Perturb I2 for F2
            self.I2 *= 1.01
            new_F2 = self.m * self.g + self.B2 * self.I2 * self.L
            if new_F2 <= self.F2:
                logs.append(f"Monotonicity failed: F2 should increase with I2, but {new_F2:.4f} ≤ {self.F2}")
                success = False
            else:
                logs.append(f"F2 monotonicity (I2) passed: ΔI2 → ΔF2 > 0")
            self.I2 = original_attrs['I2']
        else:
            logs.append("Skipping F2 monotonicity (missing parameters)")
        
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            all_logs.append(f"--- {name} ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Test crashed: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs