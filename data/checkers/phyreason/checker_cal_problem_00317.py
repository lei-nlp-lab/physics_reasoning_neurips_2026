import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Get input parameters from results dictionary with sensible defaults
        self.m = self.results.get('m', 1.0)
        self.L = self.results.get('L', 1.0)
        self.omega = self.results.get('omega', 1.0)
        self.mu = self.results.get('mu', 0.5)
        self.R = self.results.get('R', 1.0)
        
        # Get G from results, but handle the case where it might be stored as 0.0
        # due to floating point precision issues
        G_from_results = self.results.get('G', 6.6743e-11)
        if abs(G_from_results) < 1e-20:  # If it's essentially zero
            self.G = 6.6743e-11  # Use default
        else:
            self.G = G_from_results
        
        # Get output variables
        self.T = self.results.get('T', 0.0)
        self.g = self.results.get('g', 0.0)
        self.v = self.results.get('v', 0.0)
        self.rho = self.results.get('ρ', self.results.get('rho', 0.0))
    
    def _compare(self, a, b):
        """Compare two values with fuzzy tolerance for physics data."""
        try:
            # Convert to float if possible
            a_val = float(a) if hasattr(a, '__float__') else a
            b_val = float(b) if hasattr(b, '__float__') else b
            return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            # If comparison fails, return False
            return False
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        try:
            # Check force equations for objects A and B
            # Object A at distance L: μmg - T = mω²L (inward positive)
            lhs_A = self.mu * self.m * self.g - self.T
            rhs_A = self.m * self.omega**2 * self.L
            if not self._compare(lhs_A, rhs_A):
                logs.append(f"FAIL: Newton's law for object A: μmg - T = {lhs_A:.3f}, mω²L = {rhs_A:.3f}")
            
            # Object B at distance 2L: T + μmg = mω²(2L)
            lhs_B = self.T + self.mu * self.m * self.g
            rhs_B = self.m * self.omega**2 * 2 * self.L
            if not self._compare(lhs_B, rhs_B):
                logs.append(f"FAIL: Newton's law for object B: T + μmg = {lhs_B:.3f}, 2mω²L = {rhs_B:.3f}")
            
            # Check gravitational relation g = GM/R²
            # Use a small epsilon to avoid division issues
            if abs(self.G) < 1e-20:
                logs.append("WARNING: G is very small, using default for gravitational checks")
                G_effective = 6.6743e-11
            else:
                G_effective = self.G
                
            M_from_g = self.g * self.R**2 / G_effective
            # Check density consistency: ρ = 3g/(4πGR)
            rho_from_g = 3 * self.g / (4 * math.pi * G_effective * self.R)
            if not self._compare(self.rho, rho_from_g):
                logs.append(f"FAIL: Density inconsistent with g: ρ = {self.rho:.3e}, 3g/(4πGR) = {rho_from_g:.3e}")
            
            # Check first cosmic velocity: v = √(gR)
            v_from_g = math.sqrt(self.g * self.R)
            if not self._compare(self.v, v_from_g):
                logs.append(f"FAIL: Velocity inconsistent: v = {self.v:.3f}, √(gR) = {v_from_g:.3f}")
            
            if not logs:
                logs.append("PASS: All invariants satisfied")
                return True, logs
            else:
                return False, logs
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            return False, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Check g from step_2: g = 3ω²L/(2μ)
            g_expected = 3 * self.omega**2 * self.L / (2 * self.mu)
            if not self._compare(self.g, g_expected):
                logs.append(f"FAIL: g mismatch: {self.g:.3f} vs expected {g_expected:.3f}")
            
            # Check T from step_4: T = ½ m ω² L
            T_expected = 0.5 * self.m * self.omega**2 * self.L
            if not self._compare(self.T, T_expected):
                logs.append(f"FAIL: T mismatch: {self.T:.3f} vs expected {T_expected:.3f}")
            
            # Check ρ from step_7: ρ = 9Lω²/(8μπGR)
            if abs(self.G) < 1e-20:
                logs.append("WARNING: G is very small, using default for density check")
                G_effective = 6.6743e-11
            else:
                G_effective = self.G
                
            rho_expected = 9 * self.L * self.omega**2 / (8 * self.mu * math.pi * G_effective * self.R)
            if not self._compare(self.rho, rho_expected):
                logs.append(f"FAIL: ρ mismatch: {self.rho:.3e} vs expected {rho_expected:.3e}")
            
            # Check v from step_10: v = ω√(3RL/(2μ))
            v_expected = self.omega * math.sqrt(3 * self.R * self.L / (2 * self.mu))
            if not self._compare(self.v, v_expected):
                logs.append(f"FAIL: v mismatch: {self.v:.3f} vs expected {v_expected:.3f}")
            
            if not logs:
                logs.append("PASS: All trace consistencies satisfied")
                return True, logs
            else:
                return False, logs
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            return False, logs
    
    def test_monotonicity(self):
        """Test that perturbing ω changes results as expected."""
        logs = []
        
        try:
            # Store original values
            omega_orig = self.omega
            g_orig = self.g
            T_orig = self.T
            rho_orig = self.rho
            v_orig = self.v
            
            # Perturb ω by +1%
            omega_perturbed = omega_orig * 1.01
            
            # Recalculate expected values using formulas with perturbed ω
            g_new = 3 * omega_perturbed**2 * self.L / (2 * self.mu)
            T_new = 0.5 * self.m * omega_perturbed**2 * self.L
            
            if abs(self.G) < 1e-20:
                logs.append("WARNING: G is very small, using default for monotonicity check")
                G_effective = 6.6743e-11
            else:
                G_effective = self.G
                
            rho_new = 9 * self.L * omega_perturbed**2 / (8 * self.mu * math.pi * G_effective * self.R)
            v_new = omega_perturbed * math.sqrt(3 * self.R * self.L / (2 * self.mu))
            
            # Check monotonic trends (all should increase with ω)
            if not (g_new > g_orig * 0.999):  # Allow small numerical noise
                logs.append(f"FAIL: g not monotonic with ω: {g_new:.3f} <= {g_orig:.3f}")
            if not (T_new > T_orig * 0.999):
                logs.append(f"FAIL: T not monotonic with ω: {T_new:.3f} <= {T_orig:.3f}")
            if not (rho_new > rho_orig * 0.999):
                logs.append(f"FAIL: ρ not monotonic with ω: {rho_new:.3e} <= {rho_orig:.3e}")
            if not (v_new > v_orig * 0.999):
                logs.append(f"FAIL: v not monotonic with ω: {v_new:.3f} <= {v_orig:.3f}")
            
            if not logs:
                logs.append("PASS: All monotonicity trends correct")
                return True, logs
            else:
                return False, logs
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            return False, logs
    
    def run_all_tests(self):
        """Run all checks and return summary."""
        all_logs = []
        success = True
        
        # Test 1: Invariants
        s1, logs1 = self.test_invariants()
        all_logs.extend(["=== Invariants Test ==="] + logs1)
        success = success and s1
        
        # Test 2: Trace Consistency
        s2, logs2 = self.test_trace_consistency()
        all_logs.extend(["", "=== Trace Consistency Test ==="] + logs2)
        success = success and s2
        
        # Test 3: Monotonicity
        s3, logs3 = self.test_monotonicity()
        all_logs.extend(["", "=== Monotonicity Test ==="] + logs3)
        success = success and s3
        
        # Overall summary
        all_logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs