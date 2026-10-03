import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ALL values from results dictionary with proper defaults
        results = getattr(solution_instance, 'results', {})
        
        self.B = results.get('B', 1.2)
        self.F_given = results.get('F', 9.6e-14)
        self.q = results.get('q', 1.6e-19)
        self.v = results.get('v', None)
        self.direction = results.get('direction', None)
        
        # Store logs
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self):
        """Test fundamental physical invariants."""
        results = []
        
        # 1. Check force formula F = q*v*B*sin(theta)
        if self.v is not None:
            # Theta = 90° since v (down) perpendicular to B (north)
            F_calc = self.q * self.v * self.B * math.sin(math.pi/2)
            if math.isclose(F_calc, self.F_given, rel_tol=1e-2, abs_tol=1e-2):
                results.append(self._log(f"Force formula invariant: F = qvB = {F_calc:.2e} ≈ given {self.F_given:.2e}", True))
            else:
                results.append(self._log(f"Force formula failed: qvB = {F_calc:.2e} ≠ given {self.F_given:.2e}", False))
        
        # 2. Check Lorentz force direction (simplified numeric check)
        if self.v is not None:
            # Using coordinate system: north = +y, east = +x, down = -z
            # B = [0, B, 0], v = [0, 0, -v]
            # F = q * (v × B) = q * det([[i,j,k],[0,0,-v],[0,B,0]])
            # = q * (i*(0*0 - (-v)*B) - j*(0*0 - (-v)*0) + k*(0*B - 0*0))
            # = q * (i*(v*B) - j*(0) + k*(0)) = [q*v*B, 0, 0]
            F_x = self.q * self.v * self.B
            F_mag_calc = abs(F_x)
            
            # Check magnitude
            if math.isclose(F_mag_calc, self.F_given, rel_tol=1e-2, abs_tol=1e-2):
                results.append(self._log(f"Lorentz force magnitude: |q(v×B)| = {F_mag_calc:.2e} ≈ given {self.F_given:.2e}", True))
            else:
                results.append(self._log(f"Lorentz force magnitude mismatch: |q(v×B)| = {F_mag_calc:.2e} ≠ given {self.F_given:.2e}", False))
            
            # Check direction (eastward = positive x)
            if F_x > 0:
                results.append(self._log("Lorentz force direction: eastward (positive x) as expected", True))
            else:
                results.append(self._log(f"Lorentz force direction incorrect: got F_x = {F_x}, expected positive", False))
        
        # 3. Check magnetic force does no work (F ⟂ v)
        if self.v is not None:
            # F = [q*v*B, 0, 0], v = [0, 0, -v]
            work = 0.0  # Dot product is zero since perpendicular
            if math.isclose(work, 0, abs_tol=1e-2):
                results.append(self._log(f"Magnetic force does no work: F·v = {work:.2e} ≈ 0", True))
            else:
                results.append(self._log(f"Magnetic force does work: F·v = {work:.2e} ≠ 0", False))
        
        return all(results)
    
    def test_trace_consistency(self):
        """Check consistency with solution steps and final answers."""
        results = []
        
        # Check speed from step_2: v = F/(q*B)
        if self.q != 0 and self.B != 0:
            v_expected = self.F_given / (self.q * self.B)
            if self.v is not None:
                if math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    results.append(self._log(f"Speed consistency: v = F/(qB) = {v_expected:.2e} ≈ result {self.v:.2e}", True))
                else:
                    results.append(self._log(f"Speed inconsistency: v = F/(qB) = {v_expected:.2e} ≠ result {self.v:.2e}", False))
        
        # Check direction string
        if self.direction is not None:
            # Case-insensitive check for eastward deflection
            if "east" in str(self.direction).lower():
                results.append(self._log(f"Direction string '{self.direction}' contains 'east' as expected", True))
            else:
                results.append(self._log(f"Direction string '{self.direction}' doesn't indicate eastward deflection", False))
        
        return all(results)
    
    def test_monotonicity(self):
        """Test that increasing inputs produces expected trends."""
        results = []
        
        # Store original values
        orig_F = self.F_given
        orig_v = self.v
        
        if self.q != 0 and self.B != 0 and orig_v is not None:
            # Perturb force by +10%
            perturbed_F = orig_F * 1.1
            perturbed_v = perturbed_F / (self.q * self.B)
            
            if perturbed_v > orig_v:
                results.append(self._log(f"Monotonicity: F↑ {orig_F:.2e}→{perturbed_F:.2e} => v↑ {orig_v:.2e}→{perturbed_v:.2e}", True))
            else:
                results.append(self._log(f"Monotonicity violation: F↑ but v didn't increase", False))
        
        # Restore original values
        self.F_given = orig_F
        self.v = orig_v
        
        return all(results)
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.logs.append(f"ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs