import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.sol = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants (SI units)
        self.p0 = 1.0e5  # Pa
        self.rho = 1000.0  # kg/m³
        self.g = 10.0  # m/s²
        self.S = 5e-4  # m² (5 cm²)
        self.h0 = 0.20  # m (20 cm)
        self.L1_initial = 0.25  # m (25 cm)
        
        # Extracted results with safe access
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))
        self.L2 = getattr(solution_instance, 'L_2', self.results.get('L_2', None))
        self.p1 = getattr(solution_instance, 'p_1', self.results.get('p_1', None))
        self.p2 = getattr(solution_instance, 'p_2', self.results.get('p_2', None))
        self.Deltah = getattr(solution_instance, 'Deltah', self.results.get('Deltah', None))
        
        # Convert to float if they exist
        if self.m is not None:
            self.m = float(self.m)
        if self.L2 is not None:
            self.L2 = float(self.L2)
        if self.p1 is not None:
            self.p1 = float(self.p1)
        if self.p2 is not None:
            self.p2 = float(self.p2)
        if self.Deltah is not None:
            self.Deltah = float(self.Deltah)
        
        # Initialize logs
        self.logs = []

    def test_invariants(self):
        """Verify physical invariants: fluid pressure, Boyle's law, force equilibrium."""
        try:
            # 1. Check initial gas pressure p1 = p0 + ρ*g*h
            p1_expected = self.p0 + self.rho * self.g * self.h0
            if self.p1 is not None:
                if math.isclose(self.p1, p1_expected, rel_tol=0.02, abs_tol=100.0):
                    self.logs.append("✓ Initial pressure p1 matches fluid pressure formula")
                else:
                    self.logs.append(f"✗ p1 mismatch: {self.p1} vs expected {p1_expected}")
            else:
                self.logs.append("⚠ p1 not available for invariant test")
            
            # 2. Check final gas pressure p2 = p0 (equilibrium)
            if self.p2 is not None:
                if math.isclose(self.p2, self.p0, rel_tol=0.02, abs_tol=100.0):
                    self.logs.append("✓ Final pressure p2 equals atmospheric pressure")
                else:
                    self.logs.append(f"✗ p2 mismatch: {self.p2} vs p0 {self.p0}")
            else:
                self.logs.append("⚠ p2 not available for invariant test")
            
            # 3. Check Boyle's law: p1*L1 = p2*L2 (area S cancels)
            if all(v is not None for v in [self.p1, self.p2, self.L2]):
                lhs = self.p1 * self.L1_initial
                rhs = self.p2 * self.L2
                if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=1.0):
                    self.logs.append("✓ Boyle's law (p1*L1 = p2*L2) satisfied")
                else:
                    self.logs.append(f"✗ Boyle's law violation: {lhs} vs {rhs}")
            else:
                self.logs.append("⚠ Missing values for Boyle's law test")
            
            # 4. Check geometric relation: Δh = h/2 + (L2 - L1)
            if all(v is not None for v in [self.Deltah, self.L2]):
                Deltah_expected = self.h0/2.0 + (self.L2 - self.L1_initial)
                if math.isclose(self.Deltah, Deltah_expected, rel_tol=0.02, abs_tol=0.002):
                    self.logs.append("✓ Geometric relationship for Δh satisfied")
                else:
                    self.logs.append(f"✗ Δh mismatch: {self.Deltah} vs expected {Deltah_expected}")
            else:
                self.logs.append("⚠ Missing values for Δh geometric test")
            
            # 5. Force equilibrium for piston mass (from step 3)
            if self.m is not None:
                m_expected = self.rho * self.h0 * self.S  # 1000 * 0.20 * 5e-4 = 0.10 kg
                if math.isclose(self.m, m_expected, rel_tol=0.02, abs_tol=0.002):
                    self.logs.append("✓ Piston mass matches force equilibrium condition")
                else:
                    self.logs.append(f"✗ Mass mismatch: {self.m} vs expected {m_expected}")
            else:
                self.logs.append("⚠ Mass not available for force equilibrium test")
                
        except Exception as e:
            self.logs.append(f"Error in invariant tests: {str(e)}")

    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        try:
            # Mass consistency check (should match step 3)
            if self.m is not None:
                m_calc = self.rho * self.h0 * self.S
                if math.isclose(self.m, m_calc, rel_tol=0.02, abs_tol=0.002):
                    self.logs.append("✓ Mass consistent with input parameters")
                else:
                    self.logs.append(f"✗ Mass inconsistent: {self.m} vs calculated {m_calc}")
            
            # Pressure consistency
            if self.p1 is not None and self.p2 is not None:
                # p1 should be greater than p2 (since p1 includes water column)
                if self.p1 > self.p2:
                    self.logs.append("✓ p1 > p2 as expected (initial gas under extra pressure)")
                else:
                    self.logs.append(f"✗ Pressure ordering unexpected: p1={self.p1}, p2={self.p2}")
            
            # L2 should be > L1 (gas expands when pressure decreases)
            if self.L2 is not None:
                if self.L2 > self.L1_initial:
                    self.logs.append("✓ L2 > L1 as expected (gas expands)")
                else:
                    self.logs.append(f"✗ Length unexpected: L2={self.L2} should be > L1={self.L1_initial}")
            
            # Δh should be positive (piston rises)
            if self.Deltah is not None:
                if self.Deltah > 0:
                    self.logs.append("✓ Δh positive (piston rises)")
                else:
                    self.logs.append(f"✗ Δh non-positive: {self.Deltah}")
                    
        except Exception as e:
            self.logs.append(f"Error in trace consistency: {str(e)}")

    def test_monotonicity(self):
        """Perturb input h and check expected trends, then restore."""
        original_h = self.h0
        
        try:
            # Test 1: Increase h by 10%
            perturbed_h = original_h * 1.10
            
            # Recalculate expected values based on perturbed h
            m_perturbed = self.rho * perturbed_h * self.S
            p1_perturbed = self.p0 + self.rho * self.g * perturbed_h
            L2_perturbed = (p1_perturbed * self.L1_initial) / self.p0  # Boyle's law with p2 = p0
            Deltah_perturbed = perturbed_h/2.0 + (L2_perturbed - self.L1_initial)
            
            # Check monotonic trends
            if self.m is not None:
                if m_perturbed > self.m:
                    self.logs.append("✓ Mass increases with h (monotonic)")
                else:
                    self.logs.append("✗ Mass monotonicity violated")
            
            if self.p1 is not None:
                if p1_perturbed > self.p1:
                    self.logs.append("✓ p1 increases with h (monotonic)")
                else:
                    self.logs.append("✗ p1 monotonicity violated")
            
            if self.L2 is not None:
                if L2_perturbed > self.L2:
                    self.logs.append("✓ L2 increases with h (monotonic)")
                else:
                    self.logs.append("✗ L2 monotonicity violated")
            
            if self.Deltah is not None:
                if Deltah_perturbed > self.Deltah:
                    self.logs.append("✓ Δh increases with h (monotonic)")
                else:
                    self.logs.append("✗ Δh monotonicity violated")
                    
        except Exception as e:
            self.logs.append(f"Error in monotonicity test: {str(e)}")
        finally:
            # Restore original value
            self.h0 = original_h

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = ["=== Physics Audit Report ==="]
        
        # Run tests
        self.test_invariants()
        self.logs.append("---")
        self.test_trace_consistency()
        self.logs.append("---")
        self.test_monotonicity()
        self.logs.append("---")
        
        # Check target answers
        target_mass = 0.10  # kg
        target_deltah = 0.105  # m (10.5 cm)
        
        mass_ok = (self.m is not None and 
                   math.isclose(self.m, target_mass, rel_tol=0.02, abs_tol=0.002))
        deltah_ok = (self.Deltah is not None and 
                     math.isclose(self.Deltah, target_deltah, rel_tol=0.02, abs_tol=0.002))
        
        if mass_ok:
            self.logs.append(f"✓ Mass matches target: {self.m:.3f} kg ≈ {target_mass} kg")
        else:
            self.logs.append(f"✗ Mass mismatch: {self.m} vs target {target_mass}")
        
        if deltah_ok:
            self.logs.append(f"✓ Δh matches target: {self.Deltah:.3f} m ≈ {target_deltah} m")
        else:
            self.logs.append(f"✗ Δh mismatch: {self.Deltah} vs target {target_deltah}")
        
        # Overall success
        success = mass_ok and deltah_ok
        self.logs.append(f"\nOverall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs