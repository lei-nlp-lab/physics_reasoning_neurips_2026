import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from problem statement
        self.F = 4200.0  # buoyant force (N)
        self.m = 400.0   # total mass (kg)
        self.g = 10.0    # gravity (m/s²)
        self.k = 40.0    # air resistance coefficient (N·s/m)
        
        # Extract results safely
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.v_prime = getattr(solution_instance, 'v_prime', 
                              self.results.get("v'", self.results.get('v_prime', 0.0)))
        
        # Expected values from steps analysis
        self.v_given = 2.5  # velocity for acceleration calculation
        
        # Compute expected values for verification
        self.a_expected = (self.F - self.m*self.g - self.k*self.v_given) / self.m
        self.v_prime_expected = (self.F - self.m*self.g) / self.k
        
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Newton's Second Law at v=2.5 m/s
            net_force = self.F - self.m*self.g - self.k*self.v_given
            lhs = net_force
            rhs = self.m * self.a
            
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton's 2nd Law failed at v={self.v_given}:")
                logs.append(f"  LHS (F_net) = {lhs:.3f} N")
                logs.append(f"  RHS (m*a) = {self.m*self.a:.3f} N")
                logs.append(f"  a = {self.a:.3f} m/s² (expected {self.a_expected:.3f})")
                return False, logs
            
            logs.append(f"✓ Newton's 2nd Law satisfied at v={self.v_given} m/s")
            
            # 2. Newton's First Law (terminal velocity condition)
            net_force_terminal = self.F - self.m*self.g - self.k*self.v_prime
            if not math.isclose(net_force_terminal, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Newton's 1st Law failed at terminal velocity:")
                logs.append(f"  F_net = {net_force_terminal:.3f} N (should be 0)")
                logs.append(f"  v' = {self.v_prime:.3f} m/s (expected {self.v_prime_expected:.3f})")
                return False, logs
            
            logs.append(f"✓ Newton's 1st Law satisfied at terminal velocity")
            
            # 3. Energy check: work-energy theorem (CORRECTED)
            # At terminal velocity, net power = (F - mg)*v' should equal drag power = k*v'²
            net_power = (self.F - self.m*self.g) * self.v_prime
            drag_power = self.k * self.v_prime**2
            if not math.isclose(net_power, drag_power, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Energy balance check failed:")
                logs.append(f"  Net power (F-mg)*v' = {net_power:.3f} W")
                logs.append(f"  Drag power k*v'² = {drag_power:.3f} W")
                return False, logs
            
            logs.append("✓ Energy balance satisfied at terminal velocity")
            
            return True, logs
            
        except Exception as e:
            logs.append(f"Error in invariant tests: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # Check if acceleration value matches expected from given velocity
            if not math.isclose(self.a, self.a_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Inconsistent acceleration value:")
                logs.append(f"  Computed: {self.a:.3f} m/s²")
                logs.append(f"  Expected from v={self.v_given}: {self.a_expected:.3f} m/s²")
                return False, logs
            
            logs.append(f"✓ Acceleration consistent with given velocity v={self.v_given} m/s")
            
            # Check if terminal velocity matches expected
            if not math.isclose(self.v_prime, self.v_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Inconsistent terminal velocity:")
                logs.append(f"  Computed: {self.v_prime:.3f} m/s")
                logs.append(f"  Expected from force balance: {self.v_prime_expected:.3f} m/s")
                return False, logs
            
            logs.append("✓ Terminal velocity consistent with force balance")
            
            # Check that terminal velocity > given velocity (system is accelerating up)
            if self.v_prime <= self.v_given:
                logs.append("Logic error: terminal velocity should be greater than initial velocity")
                logs.append(f"  v' = {self.v_prime:.3f} m/s, v = {self.v_given:.3f} m/s")
                return False, logs
            
            logs.append("✓ Terminal velocity > given velocity (consistent with acceleration)")
            
            return True, logs
            
        except Exception as e:
            logs.append(f"Error in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def test_monotonicity(self):
        logs = []
        
        try:
            # Perturb mass by +10%
            m_perturbed = self.m * 1.10
            
            # Recalculate expected values with perturbed mass
            a_perturbed = (self.F - m_perturbed*self.g - self.k*self.v_given) / m_perturbed
            v_prime_perturbed = (self.F - m_perturbed*self.g) / self.k
            
            # Check monotonicity trends
            # 1. Increasing mass should decrease acceleration at fixed velocity
            if a_perturbed >= self.a_expected:
                logs.append("Monotonicity violation: Acceleration should decrease with increased mass")
                logs.append(f"  Original a: {self.a_expected:.3f} m/s²")
                logs.append(f"  Perturbed a: {a_perturbed:.3f} m/s² (mass +10%)")
                return False, logs
            
            # 2. Increasing mass should decrease terminal velocity
            if v_prime_perturbed >= self.v_prime_expected:
                logs.append("Monotonicity violation: Terminal velocity should decrease with increased mass")
                logs.append(f"  Original v': {self.v_prime_expected:.3f} m/s")
                logs.append(f"  Perturbed v': {v_prime_perturbed:.3f} m/s (mass +10%)")
                return False, logs
            
            logs.append("✓ Monotonicity checks passed:")
            logs.append(f"  - Acceleration decreases with increased mass")
            logs.append(f"  - Terminal velocity decreases with increased mass")
            
            return True, logs
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run invariant tests
        test_name = "Physical Invariants"
        all_logs.append(f"\n=== {test_name} ===")
        invariant_success, invariant_logs = self.test_invariants()
        all_logs.extend(invariant_logs)
        if not invariant_success:
            success = False
            all_logs.append(f"✗ {test_name} FAILED")
        else:
            all_logs.append(f"✓ {test_name} PASSED")
        
        # Run trace consistency tests
        test_name = "Trace Consistency"
        all_logs.append(f"\n=== {test_name} ===")
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        if not trace_success:
            success = False
            all_logs.append(f"✗ {test_name} FAILED")
        else:
            all_logs.append(f"✓ {test_name} PASSED")
        
        # Run monotonicity tests
        test_name = "Monotonicity"
        all_logs.append(f"\n=== {test_name} ===")
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        if not mono_success:
            success = False
            all_logs.append(f"✗ {test_name} FAILED")
        else:
            all_logs.append(f"✓ {test_name} PASSED")
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if success:
            all_logs.append("All physics checks PASSED ✓")
        else:
            all_logs.append("Some physics checks FAILED ✗")
        
        return success, all_logs