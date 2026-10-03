import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr with fallbacks
        self.solution = solution_instance
        
        # Extract inputs from problem context (safely)
        self.m = 80.0  # kg
        self.F_per = 450.0  # N
        self.h1 = 0.20  # m (20 cm converted)
        self.h2 = 0.03  # m (3 cm converted)
        # Use g = 10 m/s² to match the target answers
        self.g = 10.0  # m/s²
        
        # Extract results safely from results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        self.F_total = results_dict.get('F_total', 0.0)
        self.v = results_dict.get('v', 0.0)
        self.F_impact = results_dict.get('F_impact', 0.0)
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
    
    def test_invariants(self):
        """Test fundamental physics laws from KB."""
        tests_passed = []
        
        try:
            # 1. Test vector addition (Step 1)
            expected_F_total = 4 * self.F_per  # 4 * 450 = 1800 N
            if math.isclose(self.F_total, expected_F_total, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Force addition: F_total = {self.F_total:.1f} ≈ 4×450 = 1800 N", True))
            else:
                tests_passed.append(self._log(
                    f"Force addition failed: {self.F_total:.1f} ≠ 1800 N", False))
            
            # 2. Test work-energy theorem during lifting (Step 2)
            # Work done by forces: W = F_total * h1
            # Kinetic energy gain: ΔK = 0.5 * m * v²
            work_done = self.F_total * self.h1
            kinetic_energy = 0.5 * self.m * self.v**2
            
            if math.isclose(work_done, kinetic_energy, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Work-energy (lift): F_total·h1 = {work_done:.1f} J ≈ ½mv² = {kinetic_energy:.1f} J", True))
            else:
                tests_passed.append(self._log(
                    f"Work-energy (lift) failed: {work_done:.1f} ≠ {kinetic_energy:.1f}", False))
            
            # 3. Test work-energy theorem during impact (Step 3)
            # Initial energy at start of impact: KE + PE = 0.5*m*v² + m*g*h2
            # Work done by impact force: W_impact = F_impact * h2 (negative since force opposes motion)
            initial_energy = 0.5 * self.m * self.v**2 + self.m * self.g * self.h2
            work_impact = self.F_impact * self.h2
            
            if math.isclose(initial_energy, work_impact, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Work-energy (impact): ½mv²+mgh₂ = {initial_energy:.1f} J ≈ F_impact·h₂ = {work_impact:.1f} J", True))
            else:
                tests_passed.append(self._log(
                    f"Work-energy (impact) failed: {initial_energy:.1f} ≠ {work_impact:.1f}", False))
            
            # 4. CORRECTED: Check energy conservation over entire motion
            # Total work by external forces: F_total*h1 (positive during lift) - F_impact*h2 (negative during impact)
            # Change in potential energy: from ground (PE=0) to depth h2 (PE = -m*g*h2)
            total_work = self.F_total * self.h1 - self.F_impact * self.h2
            delta_PE = -self.m * self.g * self.h2  # Object ends at depth h2 below ground
            
            if math.isclose(total_work, delta_PE, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Energy balance: F_total·h1 - F_impact·h₂ = {total_work:.1f} ≈ -mgh₂ = {delta_PE:.1f} J", True))
            else:
                tests_passed.append(self._log(
                    f"Energy balance failed: {total_work:.1f} ≠ {delta_PE:.1f}", False))
                
        except Exception as e:
            tests_passed.append(self._log(f"Invariant test failed with error: {str(e)}", False))
            traceback.print_exc()
        
        return all(tests_passed)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = []
        
        try:
            # 1. Check v from F_total (consistent with Step 2)
            v_calculated = math.sqrt(2 * self.F_total * self.h1 / self.m)
            if math.isclose(self.v, v_calculated, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Velocity consistency: v = {self.v:.2f} m/s ≈ √(2F_total·h1/m) = {v_calculated:.2f} m/s", True))
            else:
                tests_passed.append(self._log(
                    f"Velocity inconsistent: {self.v:.2f} ≠ {v_calculated:.2f}", False))
            
            # 2. Check F_impact from v (consistent with Step 3)
            F_impact_calculated = (0.5 * self.m * self.v**2 / self.h2) + self.m * self.g
            if math.isclose(self.F_impact, F_impact_calculated, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(
                    f"Impact force consistency: F_impact = {self.F_impact:.0f} N ≈ (½mv²/h₂)+mg = {F_impact_calculated:.0f} N", True))
            else:
                tests_passed.append(self._log(
                    f"Impact force inconsistent: {self.F_impact:.0f} ≠ {F_impact_calculated:.0f}", False))
            
            # 3. Check with target answers
            if math.isclose(self.v, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(f"v matches target: {self.v:.2f} ≈ 3.0 m/s", True))
            else:
                tests_passed.append(self._log(f"v off target: {self.v:.2f} ≠ 3.0 m/s", False))
                
            if math.isclose(self.F_impact, 12800.0, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed.append(self._log(f"F_impact matches target: {self.F_impact:.0f} ≈ 12800 N", True))
            else:
                tests_passed.append(self._log(f"F_impact off target: {self.F_impact:.0f} ≠ 12800 N", False))
                
        except Exception as e:
            tests_passed.append(self._log(f"Trace consistency test failed: {str(e)}", False))
            traceback.print_exc()
        
        return all(tests_passed)
    
    def test_monotonicity(self):
        """Perturb mass and check physical trends."""
        original_m = self.m
        tests_passed = []
        
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1  # 88 kg
            
            # Recalculate using physics formulas
            F_total = 4 * self.F_per  # unchanged
            v_perturbed = math.sqrt(2 * F_total * self.h1 / perturbed_m)
            F_impact_perturbed = (0.5 * perturbed_m * v_perturbed**2 / self.h2) + perturbed_m * self.g
            
            # Check trends:
            # 1. v should decrease with increased mass (v ∝ 1/√m)
            if v_perturbed < self.v:
                tests_passed.append(self._log(
                    f"Monotonicity (v): v decreased from {self.v:.2f} to {v_perturbed:.2f} when m increased", True))
            else:
                tests_passed.append(self._log(
                    f"Monotonicity (v) failed: v should decrease with increased mass", False))
            
            # 2. F_impact should increase with increased mass (F_impact = constant/h₂ + m*g)
            if F_impact_perturbed > self.F_impact:
                tests_passed.append(self._log(
                    f"Monotonicity (F_impact): increased from {self.F_impact:.0f} to {F_impact_perturbed:.0f} when m increased", True))
            else:
                tests_passed.append(self._log(
                    f"Monotonicity (F_impact) failed: should increase with increased mass", False))
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            tests_passed.append(self._log(f"Monotonicity test failed: {str(e)}", False))
            traceback.print_exc()
            self.m = original_m  # Ensure restoration
        
        return all(tests_passed)
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = []
        success = True
        
        # Run tests
        try:
            if not self.test_invariants():
                success = False
            
            if not self.test_trace_consistency():
                success = False
                
            if not self.test_monotonicity():
                success = False
                
        except Exception as e:
            self._log(f"Test suite crashed: {str(e)}", False)
            success = False
            traceback.print_exc()
        
        # Final summary
        if success:
            self._log("All physics checks passed!", True)
        else:
            self._log("Some physics checks failed.", False)
        
        return success, self.logs