import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters with safe defaults
        self.h = self.results.get('h', 3.0)
        self.R = self.results.get('R', 2.0)
        self.mu = self.results.get('mu', 0.2)
        self.g = self.results.get('g', 10.0)
        
        # Angle in degrees (60° for incline)
        self.theta_deg = 60.0
        self.theta = math.radians(self.theta_deg)
        self.cos_theta = math.cos(self.theta)
        
        # Extract the computed s value
        self.s = self.results.get('s', 0.0)
        
        # Logs for reporting
        self.logs = []
        
    def test_invariants(self):
        """Verify physical invariants via Work-Energy Theorem."""
        logs = []
        
        try:
            # Work-energy equation per unit mass:
            # Initial PE: g*h
            # Final PE at arc top: g*R*(1 - cosθ)
            # Work lost to friction: μ*g*cosθ*s
            # Equation: g*h = μ*g*cosθ*s + g*R*(1 - cosθ)
            
            lhs = self.g * self.h
            rhs = self.mu * self.g * self.cos_theta * self.s + self.g * self.R * (1 - self.cos_theta)
            
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Work-Energy invariant satisfied")
            else:
                logs.append(f"✗ Work-Energy mismatch: LHS={lhs:.3f}, RHS={rhs:.3f}")
            
            # Check energy at arc entry (should be non-negative)
            energy_at_arc = self.g * self.h - self.mu * self.g * self.cos_theta * self.s
            if energy_at_arc >= -1e-2:
                logs.append(f"✓ Energy at arc entry: {energy_at_arc:.3f} J/kg")
            else:
                logs.append(f"✗ Negative energy at arc entry: {energy_at_arc:.3f} J/kg")
                
        except Exception as e:
            logs.append(f"✗ Invariant check failed: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        try:
            # Direct formula from physics
            s_expected = (self.h - self.R * (1 - self.cos_theta)) / \
                         (self.mu * self.cos_theta)
            
            if math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Trace consistency: s={self.s:.3f} matches formula")
            else:
                logs.append(f"✗ Trace mismatch: s={self.s:.3f}, expected={s_expected:.3f}")
                
            # Check that height drop is correct
            height_drop = self.h - self.R * (1 - self.cos_theta)
            if height_drop > 0:
                logs.append(f"✓ Height drop to arc: {height_drop:.3f} m")
            else:
                logs.append(f"✗ Negative height drop: {height_drop:.3f} m")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency failed: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def test_monotonicity(self):
        """Perturb inputs and check trend, then restore."""
        logs = []
        
        try:
            # Store original values
            original_h = self.h
            original_mu = self.mu
            original_s = self.s
            
            # Perturb h by +10%
            self.h = original_h * 1.10
            s_perturbed_h = (self.h - self.R * (1 - self.cos_theta)) / \
                           (self.mu * self.cos_theta)
            
            # Check monotonicity: increasing h should increase s
            if s_perturbed_h > original_s:
                logs.append("✓ Monotonicity: s increases with h (as expected)")
            else:
                logs.append(f"✗ Monotonicity violated with h: s={s_perturbed_h:.3f} <= {original_s:.3f}")
            
            # Restore h and perturb mu
            self.h = original_h
            self.mu = original_mu * 1.10
            s_perturbed_mu = (self.h - self.R * (1 - self.cos_theta)) / \
                            (self.mu * self.cos_theta)
            
            if s_perturbed_mu < original_s:
                logs.append("✓ Monotonicity: s decreases with μ (as expected)")
            else:
                logs.append(f"✗ Monotonicity violated with μ: s={s_perturbed_mu:.3f} >= {original_s:.3f}")
            
            # Restore original values
            self.h = original_h
            self.mu = original_mu
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed: {traceback.format_exc()}")
            # Restore on error
            self.h = original_h
            self.mu = original_mu
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs = []
        success = True
        
        # Run tests
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False
        
        # Summary
        if success:
            self.logs.append("✅ All physics checks passed")
        else:
            self.logs.append("❌ Physics checks failed")
        
        return success, self.logs