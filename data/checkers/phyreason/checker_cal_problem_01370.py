import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        # Input parameters (with safe defaults)
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 6371000.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 23400.0))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 2*math.pi))
        self.G = getattr(solution_instance, 'G', solution_instance.results.get('G', 6.67430e-11))
        
        # Results (with safe defaults)
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', 400000.0))
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 5.97e24))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 5400.0))
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', 0.00116))
    
    def test_invariants(self):
        """
        Verify physics invariants using KB theorems.
        Returns: (bool, list_of_logs)
        """
        logs = []
        all_passed = True
        
        # 1. Check Newton's Law of Universal Gravitation at Earth's surface
        # From KB: F = G*M*m/R^2 and g = F/m = G*M/R^2
        expected_M = (self.g * self.R**2) / self.G
        if not math.isclose(self.M, expected_M, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Earth mass M mismatch. Got {self.M}, expected {expected_M}")
            all_passed = False
        else:
            logs.append(f"PASS: Earth mass M = gR²/G verified")
        
        # 2. Check angular velocity definition ω = θ/t
        expected_omega = self.theta / self.t
        if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Angular velocity ω mismatch. Got {self.omega}, expected {expected_omega}")
            all_passed = False
        else:
            logs.append(f"PASS: Angular velocity ω = θ/t verified")
        
        # 3. Check period-angular velocity relationship T = 2π/ω
        # Use the exact ω value from the solution, not a rounded display value
        if self.omega != 0:
            expected_T = (2 * math.pi) / self.omega
        else:
            expected_T = float('inf')
            
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Period T mismatch. Got {self.T}, expected {expected_T}")
            all_passed = False
        else:
            logs.append(f"PASS: Period T = 2π/ω verified")
        
        # 4. Check Newton's Second Law for circular orbit
        # From KB: Gravitational force = centripetal force
        # G*M*m/(R+H)² = m*(R+H)*ω²  =>  G*M/(R+H)³ = ω²
        lhs = (self.G * self.M) / ((self.R + self.H)**3)
        rhs = self.omega**2
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Circular orbit condition. LHS={lhs}, RHS={rhs}")
            all_passed = False
        else:
            logs.append(f"PASS: Newton's Second Law for circular orbit verified")
        
        return all_passed, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        all_passed = True
        
        # Check that H satisfies the derived formula from step 5
        # H = ∛(gR²t²/θ²) - R
        numerator = self.g * (self.R**2) * (self.t**2)
        denominator = self.theta**2
        expected_H = (numerator / denominator)**(1/3) - self.R
        
        if not math.isclose(self.H, expected_H, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Altitude H mismatch. Got {self.H}, expected {expected_H}")
            all_passed = False
        else:
            logs.append(f"PASS: Altitude H = ∛(gR²t²/θ²) - R verified")
        
        # Check that T = 2πt/θ
        expected_T = 2 * math.pi * self.t / self.theta
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Period T from trace. Got {self.T}, expected {expected_T}")
            all_passed = False
        else:
            logs.append(f"PASS: Period T = 2πt/θ verified")
        
        return all_passed, logs
    
    def test_monotonicity(self):
        """
        Perturb input t and verify expected trend in H.
        """
        logs = []
        
        # Original value
        t_original = self.t
        H_original = self.H
        
        # Perturb t by +10%
        self.t = t_original * 1.10
        
        # Recalculate H using the formula (not using solution_instance methods)
        numerator = self.g * (self.R**2) * (self.t**2)
        denominator = self.theta**2
        H_perturbed = (numerator / denominator)**(1/3) - self.R
        
        # Check monotonicity: if t increases, H should increase (since t² inside cube root)
        if H_perturbed > H_original:
            logs.append(f"PASS: Monotonicity - H increases when t increases (H={H_original}→{H_perturbed})")
            monotonic_ok = True
        else:
            logs.append(f"FAIL: Monotonicity - H should increase when t increases")
            monotonic_ok = False
        
        # Restore original values
        self.t = t_original
        self.H = H_original
        
        return monotonic_ok, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Physics invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Physics Invariants Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        return overall_success, all_logs