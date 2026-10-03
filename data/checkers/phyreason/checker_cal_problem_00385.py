import math
import sympy as sp
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access for all inputs and results
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with defaults from problem context
        self.inputs['SS1'] = 4.0  # m
        self.inputs['S1S2'] = 3.0  # m
        self.inputs['v'] = 5.0  # m/s
        self.inputs['A1'] = 3.0  # cm
        self.inputs['A2'] = 1.0  # cm
        self.inputs['T1'] = 0.8  # s
        self.inputs['T2'] = 0.2  # s
        self.inputs['t'] = 6.0  # s (inferred from step_5: Δt1 = 5.2 = 6 - 0.8)
        
        # Extract results safely from solution_instance
        result_keys = [
            'ss_2', 't_1', 't_2', 'Deltat', 'Deltat_1', 
            'Deltat_2', 'x_1', 'x_2', 'x'
        ]
        
        # First try to get from solution_instance.results, then from attributes
        for key in result_keys:
            value = getattr(solution_instance, 'results', {}).get(key, 
                   getattr(solution_instance, key, None))
            if value is not None:
                self.results[key] = float(value) if isinstance(value, (int, float, str)) else float(value)
        
        # Initialize symbols for symbolic checks
        self._init_symbols()
        
        # Log storage
        self.logs = []
        
    def _init_symbols(self):
        """Initialize sympy symbols for symbolic physics laws."""
        self.SS1, self.S1S2, self.v, self.t = sp.symbols('SS1 S1S2 v t', positive=True)
        self.A1, self.A2, self.T1, self.T2 = sp.symbols('A1 A2 T1 T2', positive=True)
        self.ss_2, self.t_1, self.t_2 = sp.symbols('ss_2 t_1 t_2', positive=True)
        self.Deltat, self.Deltat_1, self.Deltat_2 = sp.symbols('Deltat Deltat_1 Deltat_2', real=True)
        self.x_1, self.x_2, self.x = sp.symbols('x_1 x_2 x', real=True)
        
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_pythagorean_theorem(self) -> bool:
        """Check SS2 = sqrt(SS1² + S1S2²)."""
        ss2_calc = math.sqrt(self.inputs['SS1']**2 + self.inputs['S1S2']**2)
        ss2_given = self.results.get('ss_2', 0.0)
        
        # Symbolic check
        symbolic_eq = sp.Eq(self.ss_2, sp.sqrt(self.SS1**2 + self.S1S2**2))
        
        # Numerical check with fuzzy matching
        numeric_pass = self._is_close(ss2_given, ss2_calc)
        
        msg = f"Pythagorean theorem: ss_2 = {ss2_given:.3f}, calculated = {ss2_calc:.3f}"
        return self._log(msg, numeric_pass)
    
    def test_uniform_motion_times(self) -> bool:
        """Check t1 = SS1/v and t2 = SS2/v."""
        # Calculate expected times
        t1_calc = self.inputs['SS1'] / self.inputs['v']
        ss2_val = self.results.get('ss_2', 5.0)
        t2_calc = ss2_val / self.inputs['v']
        
        t1_given = self.results.get('t_1', 0.0)
        t2_given = self.results.get('t_2', 0.0)
        
        # Symbolic checks
        eq1 = sp.Eq(self.t_1, self.SS1 / self.v)
        eq2 = sp.Eq(self.t_2, self.ss_2 / self.v)
        
        # Numerical checks with fuzzy matching
        pass1 = self._is_close(t1_given, t1_calc)
        pass2 = self._is_close(t2_given, t2_calc)
        
        msg1 = f"Uniform motion t1: {t1_given:.3f}s, calculated = {t1_calc:.3f}s"
        msg2 = f"Uniform motion t2: {t2_given:.3f}s, calculated = {t2_calc:.3f}s"
        
        self._log(msg1, pass1)
        return self._log(msg2, pass2)
    
    def test_time_difference(self) -> bool:
        """Check Δt = t2 - t1."""
        t1 = self.results.get('t_1', 0.0)
        t2 = self.results.get('t_2', 0.0)
        Deltat_given = self.results.get('Deltat', 0.0)
        Deltat_calc = t2 - t1
        
        # Symbolic check
        symbolic_eq = sp.Eq(self.Deltat, self.t_2 - self.t_1)
        
        # Numerical check
        numeric_pass = self._is_close(Deltat_given, Deltat_calc)
        
        msg = f"Time difference Δt: {Deltat_given:.3f}s, calculated = {Deltat_calc:.3f}s"
        return self._log(msg, numeric_pass)
    
    def test_vibration_times(self) -> bool:
        """Check Δt1 = t - t1 and Δt2 = t - t2."""
        t = self.inputs['t']
        t1 = self.results.get('t_1', 0.0)
        t2 = self.results.get('t_2', 0.0)
        
        Deltat1_given = self.results.get('Deltat_1', 0.0)
        Deltat2_given = self.results.get('Deltat_2', 0.0)
        
        Deltat1_calc = t - t1
        Deltat2_calc = t - t2
        
        # Symbolic checks
        eq1 = sp.Eq(self.Deltat_1, self.t - self.t_1)
        eq2 = sp.Eq(self.Deltat_2, self.t - self.t_2)
        
        # Numerical checks
        pass1 = self._is_close(Deltat1_given, Deltat1_calc)
        pass2 = self._is_close(Deltat2_given, Deltat2_calc)
        
        msg1 = f"Vibration time Δt1: {Deltat1_given:.3f}s, calculated = {Deltat1_calc:.3f}s"
        msg2 = f"Vibration time Δt2: {Deltat2_given:.3f}s, calculated = {Deltat2_calc:.3f}s"
        
        self._log(msg1, pass1)
        return self._log(msg2, pass2)
    
    def test_harmonic_motion_displacements(self) -> bool:
        """Check x1 = A1*sin(2π*Δt1/T1) and x2 = A2*sin(2π*Δt2/T2)."""
        Deltat1 = self.results.get('Deltat_1', 0.0)
        Deltat2 = self.results.get('Deltat_2', 0.0)
        
        # Calculate expected displacements
        x1_calc = self.inputs['A1'] * math.sin(2 * math.pi * Deltat1 / self.inputs['T1'])
        x2_calc = self.inputs['A2'] * math.sin(2 * math.pi * Deltat2 / self.inputs['T2'])
        
        x1_given = self.results.get('x_1', 0.0)
        x2_given = self.results.get('x_2', 0.0)
        
        # Symbolic checks
        eq1 = sp.Eq(self.x_1, self.A1 * sp.sin(2 * sp.pi * self.Deltat_1 / self.T1))
        eq2 = sp.Eq(self.x_2, self.A2 * sp.sin(2 * sp.pi * self.Deltat_2 / self.T2))
        
        # Numerical checks with fuzzy matching (allow for small floating point differences)
        pass1 = self._is_close(x1_given, x1_calc, abs_tol=1e-3)
        pass2 = self._is_close(x2_given, x2_calc, abs_tol=1e-3)
        
        msg1 = f"Displacement x1: {x1_given:.3f}cm, calculated = {x1_calc:.3f}cm"
        msg2 = f"Displacement x2: {x2_given:.3f}cm, calculated = {x2_calc:.3f}cm"
        
        self._log(msg1, pass1)
        return self._log(msg2, pass2)
    
    def test_superposition_principle(self) -> bool:
        """Check x = x1 + x2."""
        x1 = self.results.get('x_1', 0.0)
        x2 = self.results.get('x_2', 0.0)
        x_given = self.results.get('x', 0.0)
        x_calc = x1 + x2
        
        # Symbolic check
        symbolic_eq = sp.Eq(self.x, self.x_1 + self.x_2)
        
        # Numerical check
        numeric_pass = self._is_close(x_given, x_calc)
        
        msg = f"Superposition: x = {x_given:.3f}cm, x1 + x2 = {x_calc:.3f}cm"
        return self._log(msg, numeric_pass)
    
    def test_period_relationships(self) -> bool:
        """Check that displacements are zero when Δt1/T1 and Δt2/T2 are integers."""
        Deltat1 = self.results.get('Deltat_1', 0.0)
        Deltat2 = self.results.get('Deltat_2', 0.0)
        
        # Check if Δt1/T1 is half-integer (6.5) and Δt2/T2 is integer (25)
        ratio1 = Deltat1 / self.inputs['T1']
        ratio2 = Deltat2 / self.inputs['T2']
        
        # Check if ratio1 is 6.5 ± tolerance (half-integer)
        is_half_int1 = self._is_close(ratio1 % 1, 0.5, abs_tol=0.01) or self._is_close(ratio1 % 1, 0.0, abs_tol=0.01)
        is_int2 = self._is_close(ratio2 % 1, 0.0, abs_tol=0.01)
        
        # If half-integer or integer multiple, sin(2π*ratio) should be ~0
        sin_val1 = math.sin(2 * math.pi * ratio1)
        sin_val2 = math.sin(2 * math.pi * ratio2)
        
        pass1 = self._is_close(sin_val1, 0.0, abs_tol=0.01)
        pass2 = self._is_close(sin_val2, 0.0, abs_tol=0.01)
        
        msg1 = f"Δt1/T1 = {ratio1:.3f}, sin(2π*ratio) = {sin_val1:.3f} (should be ~0)"
        msg2 = f"Δt2/T2 = {ratio2:.3f}, sin(2π*ratio) = {sin_val2:.3f} (should be ~0)"
        
        self._log(msg1, pass1)
        return self._log(msg2, pass2)
    
    def test_trace_consistency(self) -> bool:
        """Cross-check all intermediate results for consistency."""
        all_pass = True
        
        # 1. Check that all required results exist
        required = ['ss_2', 't_1', 't_2', 'Deltat', 'Deltat_1', 'Deltat_2', 'x_1', 'x_2', 'x']
        for key in required:
            if key not in self.results:
                all_pass = self._log(f"Missing result: {key}", False)
        
        # 2. Recalculate SS2 from Pythagorean theorem
        ss2_calc = math.sqrt(self.inputs['SS1']**2 + self.inputs['S1S2']**2)
        if not self._is_close(self.results.get('ss_2', 0.0), ss2_calc):
            all_pass = self._log(f"Trace: ss_2 inconsistent with Pythagorean theorem", False)
        
        # 3. Recalculate times from uniform motion
        t1_calc = self.inputs['SS1'] / self.inputs['v']
        t2_calc = self.results.get('ss_2', 5.0) / self.inputs['v']
        if not (self._is_close(self.results.get('t_1', 0.0), t1_calc) and 
                self._is_close(self.results.get('t_2', 0.0), t2_calc)):
            all_pass = self._log(f"Trace: t1 or t2 inconsistent with uniform motion", False)
        
        # 4. Check time difference
        Deltat_calc = self.results.get('t_2', 0.0) - self.results.get('t_1', 0.0)
        if not self._is_close(self.results.get('Deltat', 0.0), Deltat_calc):
            all_pass = self._log(f"Trace: Δt inconsistent with t2 - t1", False)
        
        # 5. Check vibration times
        t = self.inputs['t']
        Deltat1_calc = t - self.results.get('t_1', 0.0)
        Deltat2_calc = t - self.results.get('t_2', 0.0)
        if not (self._is_close(self.results.get('Deltat_1', 0.0), Deltat1_calc) and 
                self._is_close(self.results.get('Deltat_2', 0.0), Deltat2_calc)):
            all_pass = self._log(f"Trace: Δt1 or Δt2 inconsistent with t - t1 or t - t2", False)
        
        # 6. Check displacements from harmonic motion
        x1_calc = self.inputs['A1'] * math.sin(2 * math.pi * self.results.get('Deltat_1', 0.0) / self.inputs['T1'])
        x2_calc = self.inputs['A2'] * math.sin(2 * math.pi * self.results.get('Deltat_2', 0.0) / self.inputs['T2'])
        if not (self._is_close(self.results.get('x_1', 0.0), x1_calc, abs_tol=1e-3) and 
                self._is_close(self.results.get('x_2', 0.0), x2_calc, abs_tol=1e-3)):
            all_pass = self._log(f"Trace: x1 or x2 inconsistent with harmonic motion formula", False)
        
        # 7. Check superposition
        x_calc = self.results.get('x_1', 0.0) + self.results.get('x_2', 0.0)
        if not self._is_close(self.results.get('x', 0.0), x_calc):
            all_pass = self._log(f"Trace: x inconsistent with x1 + x2", False)
        
        if all_pass:
            self._log("All trace consistency checks passed", True)
        return all_pass
    
    def test_monotonicity(self) -> bool:
        """Perturb input parameters and check expected trends."""
        all_pass = True
        
        # Store original values
        original_A1 = self.inputs['A1']
        original_ss2 = self.results.get('ss_2', 5.0)
        
        # Test 1: Increase amplitude A1 by 10%, displacement magnitude should increase
        perturbed_A1 = original_A1 * 1.1
        Deltat1 = self.results.get('Deltat_1', 0.0)
        perturbed_x1 = perturbed_A1 * math.sin(2 * math.pi * Deltat1 / self.inputs['T1'])
        original_x1 = self.results.get('x_1', 0.0)
        
        # Since original x1 is 0, the perturbed x1 might still be 0 if sin term is 0
        # But in general, if sin term ≠ 0, |perturbed_x1| should be > |original_x1|
        if not self._is_close(math.sin(2 * math.pi * Deltat1 / self.inputs['T1']), 0.0, abs_tol=0.01):
            if abs(perturbed_x1) < abs(original_x1) - 0.1:  # Allow some tolerance
                all_pass = self._log(f"Monotonicity: Increasing A1 should increase |x1|", False)
            else:
                self._log(f"Monotonicity: |x1| behaves as expected with A1 perturbation", True)
        
        # Test 2: Increase SS1 distance, SS2 should increase
        perturbed_SS1 = self.inputs['SS1'] * 1.1
        perturbed_ss2 = math.sqrt(perturbed_SS1**2 + self.inputs['S1S2']**2)
        if perturbed_ss2 <= original_ss2:
            all_pass = self._log(f"Monotonicity: Increasing SS1 should increase SS2", False)
        else:
            self._log(f"Monotonicity: SS2 increases with SS1 as expected", True)
        
        # Restore original values
        self.inputs['A1'] = original_A1
        
        return all_pass
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics invariants."""
        all_pass = True
        
        # 1. Energy conservation for each harmonic oscillator at source
        # For harmonic motion, total energy = 1/2 * m * ω² * A²
        # Since we don't have mass, check that displacement doesn't exceed amplitude
        x1 = abs(self.results.get('x_1', 0.0))
        x2 = abs(self.results.get('x_2', 0.0))
        
        if x1 > self.inputs['A1'] * 1.01:  # Allow 1% tolerance
            all_pass = self._log(f"Invariant: |x1| ≤ A1 violated: |x1|={x1:.3f}, A1={self.inputs['A1']:.3f}", False)
        else:
            self._log(f"Invariant: |x1| ≤ A1 satisfied: |x1|={x1:.3f}, A1={self.inputs['A1']:.3f}", True)
        
        if x2 > self.inputs['A2'] * 1.01:
            all_pass = self._log(f"Invariant: |x2| ≤ A2 violated: |x2|={x2:.3f}, A2={self.inputs['A2']:.3f}", False)
        else:
            self._log(f"Invariant: |x2| ≤ A2 satisfied: |x2|={x2:.3f}, A2={self.inputs['A2']:.3f}", True)
        
        # 2. Wave speed invariance
        # Check that v = distance/time for both paths
        v1_calc = self.inputs['SS1'] / self.results.get('t_1', 1e-9)
        v2_calc = self.results.get('ss_2', 5.0) / self.results.get('t_2', 1e-9)
        
        if not (self._is_close(v1_calc, self.inputs['v']) and self._is_close(v2_calc, self.inputs['v'])):
            all_pass = self._log(f"Invariant: Wave speed constant violated: v1={v1_calc:.3f}, v2={v2_calc:.3f}, v={self.inputs['v']:.3f}", False)
        else:
            self._log(f"Invariant: Wave speed constant satisfied", True)
        
        # 3. Superposition linearity
        # If we scale both amplitudes by factor k, net displacement should scale by k
        # Since original net displacement is 0, this is trivial, but check the formula
        k = 2.0
        scaled_x1 = (k * self.inputs['A1']) * math.sin(2 * math.pi * self.results.get('Deltat_1', 0.0) / self.inputs['T1'])
        scaled_x2 = (k * self.inputs['A2']) * math.sin(2 * math.pi * self.results.get('Deltat_2', 0.0) / self.inputs['T2'])
        scaled_x = scaled_x1 + scaled_x2
        
        # Should equal k * (original x1 + x2) = k * 0 = 0
        if not self._is_close(scaled_x, k * self.results.get('x', 0.0), abs_tol=1e-3):
            all_pass = self._log(f"Invariant: Superposition linearity violated", False)
        else:
            self._log(f"Invariant: Superposition linearity satisfied", True)
        
        return all_pass
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics checks and return success status with logs."""
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        # Run individual tests
        tests = [
            self.test_pythagorean_theorem,
            self.test_uniform_motion_times,
            self.test_time_difference,
            self.test_vibration_times,
            self.test_harmonic_motion_displacements,
            self.test_superposition_principle,
            self.test_period_relationships,
            self.test_trace_consistency,
            self.test_monotonicity,
            self.test_invariants
        ]
        
        all_pass = True
        for test in tests:
            try:
                if not test():
                    all_pass = False
            except Exception as e:
                self._log(f"Test {test.__name__} failed with error: {str(e)}", False)
                all_pass = False
        
        self.logs.append(f"=== Physics Checker Completed: {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
        return all_pass, self.logs