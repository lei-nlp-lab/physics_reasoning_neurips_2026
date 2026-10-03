import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with defaults
        self.M = getattr(solution_instance, 'M', 900.0)
        self.m = getattr(solution_instance, 'm', 300.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.dv = getattr(solution_instance, 'dv', 6.0)
        self.dt = getattr(solution_instance, 'dt', 3.0)
        self.v_safe = getattr(solution_instance, 'v_safe', 4.0)
        self.H0 = getattr(solution_instance, 'H0', 400.0)
        
        # Extract results with safe access - try both 'h' and "h'" keys
        self.a1 = self.results.get('a_1', 0.0)
        self.h_prime = self.results.get('h', self.results.get("h'", 0.0))
        
        # Derived quantities for checking
        self.F_b = None  # Buoyant force
        self.v0_jettison = None  # Velocity at jettison
        self.a2 = None  # Deceleration after jettison
        
    def compute_derived_quantities(self):
        """Compute intermediate physics quantities from inputs."""
        try:
            # Step 1: Acceleration before jettison (from given data)
            # a1 = Δv/Δt = 6/3 = 2 m/s²
            self.a1_calc = self.dv / self.dt
            
            # Step 2: Buoyant force from Newton's 2nd Law
            # M*g - F_b = M*a1 => F_b = M*(g - a1)
            self.F_b = self.M * (self.g - self.a1_calc)
            
            # Step 3: Velocity at jettison from kinematics
            # v0² = 2*a1*(H0 - h') - but we need h' first
            # We'll compute this in test_trace_consistency
            
            # Step 4: Deceleration after jettison
            # New mass: M' = M - m = 600 kg
            # Newton's 2nd: (M-m)*g - F_b = (M-m)*a2 (downward positive)
            # This gives a2 = g - F_b/(M-m)
            M_prime = self.M - self.m
            self.a2 = ((M_prime * self.g) - self.F_b) / M_prime  # Should be negative
            
        except Exception as e:
            return f"Error in compute_derived_quantities: {str(e)}"
        return None
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Check Newton's 2nd Law before jettison
        try:
            # ΣF = M*a1
            net_force_before = self.M * self.g - self.F_b
            expected_net = self.M * self.a1_calc
            if not math.isclose(net_force_before, expected_net, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL Newton 2nd (before): ΣF={net_force_before:.2f} != M*a1={expected_net:.2f}")
            else:
                logs.append(f"PASS Newton 2nd (before): ΣF = M*a1 = {expected_net:.2f} N")
        except Exception as e:
            logs.append(f"ERROR in Newton 2nd (before): {str(e)}")
        
        # 2. Check Newton's 2nd Law after jettison
        try:
            M_prime = self.M - self.m
            # Net force after jettison: (M-m)*g - F_b (downward positive)
            net_force_after = M_prime * self.g - self.F_b
            expected_net_after = M_prime * self.a2
            if not math.isclose(net_force_after, expected_net_after, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL Newton 2nd (after): ΣF={net_force_after:.2f} != M'*a2={expected_net_after:.2f}")
            else:
                logs.append(f"PASS Newton 2nd (after): ΣF = M'*a2 = {net_force_after:.2f} N")
        except Exception as e:
            logs.append(f"ERROR in Newton 2nd (after): {str(e)}")
        
        # 3. Check energy conservation (work-energy theorem) - CORRECTED
        try:
            # From jettison to landing: ΔKE = W_net
            M_prime = self.M - self.m
            
            # Velocity at jettison from kinematics: v0² = 2*a1*(H0 - h')
            v0_sq = 2 * self.a1_calc * (self.H0 - self.h_prime)
            v0 = math.sqrt(v0_sq) if v0_sq >= 0 else 0
            
            # Change in kinetic energy: final KE - initial KE
            ΔKE = 0.5 * M_prime * (self.v_safe**2 - v0**2)
            
            # Work done by net force: W_net = F_net * distance
            # Net force after jettison: (M-m)*g - F_b (downward positive)
            F_net = M_prime * self.g - self.F_b
            W_net = F_net * self.h_prime
            
            if not math.isclose(ΔKE, W_net, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL Work-Energy: ΔKE={ΔKE:.2f} J != W_net={W_net:.2f} J")
            else:
                logs.append(f"PASS Work-Energy: ΔKE = W_net = {W_net:.2f} J")
        except Exception as e:
            logs.append(f"ERROR in Work-Energy: {str(e)}")
        
        # 4. Check direction consistency
        try:
            if self.a1_calc <= 0:
                logs.append("WARNING: a1 should be positive (downward acceleration)")
            if self.a2 >= 0:
                logs.append("WARNING: a2 should be negative (upward deceleration)")
            if self.F_b >= self.M * self.g:
                logs.append("WARNING: F_b should be less than weight for downward acceleration")
        except Exception as e:
            logs.append(f"ERROR in direction check: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        """Verify intermediate results lead to final answer."""
        logs = []
        
        try:
            # Reconstruct h' from physics equations
            # Step 1: Get velocity at jettison from first phase
            # v0² = 2*a1*(H0 - h') => h' = H0 - v0²/(2*a1)
            
            # Step 2: From second phase kinematics with uniform deceleration:
            # v_safe² = v0² + 2*a2*h'  (a2 is negative)
            
            # Solve these simultaneously
            # From step 1: v0² = 2*a1*(H0 - h')
            # From step 2: v_safe² = v0² + 2*a2*h'
            # Substitute: v_safe² = 2*a1*(H0 - h') + 2*a2*h'
            # Rearrange: v_safe² = 2*a1*H0 - 2*a1*h' + 2*a2*h'
            # v_safe² = 2*a1*H0 + 2*h'*(a2 - a1)
            # h' = (v_safe² - 2*a1*H0) / (2*(a2 - a1))
            
            a1 = self.dv / self.dt
            M_prime = self.M - self.m
            F_b = self.M * (self.g - a1)
            a2 = ((M_prime * self.g) - F_b) / M_prime  # Should be negative
            
            # Use the correct formula that doesn't have division by zero issues
            numerator = self.v_safe**2 - 2 * a1 * self.H0
            denominator = 2 * (a2 - a1)
            
            # Avoid division by zero
            if abs(denominator) < 1e-10:
                logs.append("WARNING: Denominator near zero in h' calculation")
                h_prime_calc = 0.0
            else:
                h_prime_calc = numerator / denominator
            
            if not math.isclose(self.h_prime, h_prime_calc, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL Trace Consistency: h'={self.h_prime:.2f} != recomputed={h_prime_calc:.2f}")
            else:
                logs.append(f"PASS Trace Consistency: h'={self.h_prime:.2f} matches physics derivation")
                
            # Also check a1 matches given data
            if not math.isclose(self.a1, a1, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL a1 consistency: {self.a1:.2f} != {a1:.2f}")
            else:
                logs.append(f"PASS a1 = {a1:.2f} m/s² matches Δv/Δt")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """Check that perturbing mass m changes h' in expected direction."""
        logs = []
        
        original_m = self.m
        original_h_prime = self.h_prime
        
        try:
            # Perturb m upward by 10%
            delta_m = original_m * 0.1
            perturbed_m = original_m + delta_m
            
            # Recompute with perturbed m (using physics formulas)
            a1 = self.dv / self.dt  # Unchanged
            F_b = self.M * (self.g - a1)  # Unchanged
            
            # Original case
            M_prime_orig = self.M - original_m
            a2_orig = ((M_prime_orig * self.g) - F_b) / M_prime_orig
            
            # Use the correct formula: h' = (v_safe² - 2*a1*H0) / (2*(a2 - a1))
            numerator = self.v_safe**2 - 2 * a1 * self.H0
            denominator_orig = 2 * (a2_orig - a1)
            if abs(denominator_orig) < 1e-10:
                h_prime_orig = 0.0
            else:
                h_prime_orig = numerator / denominator_orig
            
            # Perturbed case
            M_prime_pert = self.M - perturbed_m
            a2_pert = ((M_prime_pert * self.g) - F_b) / M_prime_pert
            denominator_pert = 2 * (a2_pert - a1)
            if abs(denominator_pert) < 1e-10:
                h_prime_pert = 0.0
            else:
                h_prime_pert = numerator / denominator_pert
            
            # Expected: Increasing m makes deceleration stronger (more negative a2)
            # Since a2 becomes more negative, denominator = 2*(a2 - a1) becomes more negative
            # So h' should decrease (since numerator is negative)
            if h_prime_pert < h_prime_orig:
                logs.append(f"PASS Monotonicity: Increasing m from {original_m} to {perturbed_m} "
                           f"decreases h' from {h_prime_orig:.2f} to {h_prime_pert:.2f}")
            else:
                logs.append(f"FAIL Monotonicity: h' did not decrease when m increased")
                
            # Restore original values
            self.m = original_m
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}\n{traceback.format_exc()}")
            # Restore on error
            self.m = original_m
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        
        # Compute derived quantities first
        err = self.compute_derived_quantities()
        if err:
            all_logs.append(f"SETUP ERROR: {err}")
            return False, all_logs
        
        # Run tests
        all_logs.append("=== TESTING INVARIANTS (Physics Laws) ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== TESTING TRACE CONSISTENCY ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== TESTING MONOTONICITY ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
            # Warnings don't fail the test
        
        return success, all_logs