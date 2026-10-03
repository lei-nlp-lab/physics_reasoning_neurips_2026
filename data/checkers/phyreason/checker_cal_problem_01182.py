import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = {}
        
        # Extract given constants
        self.m_A = 0.3  # kg
        self.g = 10.0   # m/s²
        self.cos37 = 0.8
        self.sin37 = 0.6
        self.cos60 = 0.5  # cos60° = 0.5
        
        # Safely extract results from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract numeric values with defaults
        self.F_1 = float(self.results.get('F_1', 0.0))
        self.m_B = float(self.results.get('m_B', 0.0))
        self.f = float(self.results.get('f', 0.0))
        # Direction may be stored as numeric (1.0 for left, -1.0 for right) or string
        direction_raw = self.results.get('direction', 0.0)
        if isinstance(direction_raw, str):
            # If it's a string, try to interpret
            dir_str = direction_raw.lower()
            if "left" in dir_str:
                self.direction_numeric = 1.0
            elif "right" in dir_str:
                self.direction_numeric = -1.0
            else:
                self.direction_numeric = 0.0
        else:
            # Assume it's numeric
            self.direction_numeric = float(direction_raw)
        
        self.logs = []

    def test_invariants(self):
        """Check fundamental physics laws: force equilibrium."""
        try:
            # Test 1: Check equilibrium of movable pulley (Step 1)
            expected_F1 = (self.m_A * self.g) / (2 * self.cos60)
            if not math.isclose(self.F_1, expected_F1, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: F_1 should be {expected_F1:.3f} N, got {self.F_1:.3f} N")
                return False

            # Test 2: Check equilibrium at knot N in horizontal direction
            expected_m_B = self.F_1 / (self.g * self.sin37)
            if not math.isclose(self.m_B, expected_m_B, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: m_B should be {expected_m_B:.3f} kg, got {self.m_B:.3f} kg")
                return False

            # Test 3: Check horizontal equilibrium on block C
            expected_f = self.m_B * self.g * self.cos37 - self.F_1
            if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: f should be {expected_f:.3f} N, got {self.f:.3f} N")
                return False

            # Test 4: Check direction of friction (numeric encoding)
            # For the correct physics, f should be positive (1.0 N), meaning friction acts left.
            # So direction_numeric should be positive (1.0 for left).
            if self.f > 0:
                if not (self.direction_numeric > 0):
                    self.logs.append(f"FAIL: Friction is positive ({self.f:.3f} N) but direction numeric is not positive ({self.direction_numeric})")
                    return False
            elif self.f < 0:
                if not (self.direction_numeric < 0):
                    self.logs.append(f"FAIL: Friction is negative ({self.f:.3f} N) but direction numeric is not negative ({self.direction_numeric})")
                    return False
            # If f == 0, direction can be anything (or 0)

            self.logs.append("PASS: All equilibrium conditions satisfied")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False

    def test_trace_consistency(self):
        """Check intermediate steps from solution trace."""
        try:
            # Step 1: 2F_1*cos60° = m_A*g → F_1 = 3 N
            step1_F1 = (self.m_A * self.g) / (2 * self.cos60)
            if not math.isclose(self.F_1, step1_F1, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 1 yields F_1 = {step1_F1:.3f} N, but got {self.F_1:.3f} N")
                return False

            # Step 2: F_1 = m_B*g*sin37° → m_B = 0.5 kg
            step2_mB = self.F_1 / (self.g * self.sin37)
            if not math.isclose(self.m_B, step2_mB, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 2 yields m_B = {step2_mB:.3f} kg, but got {self.m_B:.3f} kg")
                return False

            # Step 3: F_1 + f = m_B*g*cos37° → f = 1 N
            step3_f = self.m_B * self.g * self.cos37 - self.F_1
            if not math.isclose(self.f, step3_f, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 3 yields f = {step3_f:.3f} N, but got {self.f:.3f} N")
                return False

            self.logs.append("PASS: All trace steps are consistent")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False

    def test_monotonicity(self):
        """Perturb m_A and check if results change consistently."""
        try:
            original_m_A = self.m_A
            
            # Perturb m_A by +10%
            perturbed_m_A = original_m_A * 1.10
            
            # Recalculate expected values using physics equations
            new_F1 = (perturbed_m_A * self.g) / (2 * self.cos60)
            new_mB = new_F1 / (self.g * self.sin37)
            new_f = new_mB * self.g * self.cos37 - new_F1
            
            # Check monotonicity: if m_A increases, all quantities should increase
            if not (new_F1 > self.F_1 and new_mB > self.m_B and new_f > self.f):
                self.logs.append(f"FAIL: Monotonicity broken with m_A +10%")
                self.logs.append(f"  Original: F1={self.F_1:.3f}, mB={self.m_B:.3f}, f={self.f:.3f}")
                self.logs.append(f"  Perturbed: F1={new_F1:.3f}, mB={new_mB:.3f}, f={new_f:.3f}")
                return False
            
            # Negative perturbation: -5%
            perturbed_m_A2 = original_m_A * 0.95
            new_F1_2 = (perturbed_m_A2 * self.g) / (2 * self.cos60)
            new_mB_2 = new_F1_2 / (self.g * self.sin37)
            new_f_2 = new_mB_2 * self.g * self.cos37 - new_F1_2
            
            if not (new_F1_2 < self.F_1 and new_mB_2 < self.m_B and new_f_2 < self.f):
                self.logs.append(f"FAIL: Monotonicity broken with m_A -5%")
                return False
            
            self.logs.append("PASS: Monotonicity holds for m_A perturbations")
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False

    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.logs.append(f"✗ {test_name} failed")
            else:
                self.logs.append(f"✓ {test_name} passed")
        
        self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs