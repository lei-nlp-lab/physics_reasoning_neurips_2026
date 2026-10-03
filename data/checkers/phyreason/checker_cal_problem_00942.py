import math
import sympy as sp
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.logs = []
        self.inputs = {}
        self.results = {}
        
        # Problem constants
        self.P_out = 10000  # W (10 kW)
        self.U1 = 400       # V
        self.U2 = 2000      # V (2 kV)
        self.U4 = 220       # V
        self.P_rec = 9500   # W (9.5 kW)
        
        # Safe extraction of results from solution instance
        safe_get = lambda key: getattr(solution_instance, key, 
                                     getattr(solution_instance, 'results', {}).get(key, None))
        
        self.results['P_loss'] = safe_get('$P_loss$')
        self.results['n1_n2'] = safe_get('$fracn_1n_2$')
        self.results['U3'] = safe_get('U_3')
        self.results['R'] = safe_get('R')
        self.results['n3_n4'] = safe_get('$fracn_3n_4$')
        self.results['I'] = safe_get('I')
        
        # Convert string fractions to float if needed
        for key in ['n1_n2', 'n3_n4']:
            if isinstance(self.results[key], str):
                if '/' in self.results[key]:
                    num, den = map(float, self.results[key].split('/'))
                    self.results[key] = num / den
                else:
                    self.results[key] = float(self.results[key])
        
        # Expected values (from steps analysis)
        self.expected = {
            'P_loss': 500.0,
            'n1_n2': 1/5,
            'I': 5.0,
            'R': 20.0,
            'U3': 1900.0,
            'n3_n4': 95/11
        }
        
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _check_result(self, name, value, expected):
        """Check if result matches expected value."""
        if value is None:
            self.logs.append(f"WARNING: {name} not found in results")
            return False
        if self._isclose(value, expected):
            self.logs.append(f"✓ {name} = {value} (expected {expected})")
            return True
        else:
            self.logs.append(f"✗ {name} = {value} but expected {expected}")
            return False
    
    def test_invariants(self) -> bool:
        """Verify physical invariants using Ohm's and Joule's laws."""
        success = True
        
        # 1. Check power conservation: P_out = P_rec + P_loss
        if self.results['P_loss'] is not None:
            total = self.results['P_loss'] + self.P_rec
            if not self._isclose(total, self.P_out):
                self.logs.append(f"✗ Power conservation: P_loss + P_rec = {total} ≠ P_out = {self.P_out}")
                success = False
            else:
                self.logs.append(f"✓ Power conservation: {self.results['P_loss']} + {self.P_rec} ≈ {self.P_out}")
        
        # 2. Check transformer ratios
        if self.results['n1_n2'] is not None:
            expected_ratio = self.U1 / self.U2
            if not self._isclose(self.results['n1_n2'], expected_ratio):
                self.logs.append(f"✗ Transformer ratio n1/n2: {self.results['n1_n2']} ≠ U1/U2 = {expected_ratio}")
                success = False
            else:
                self.logs.append(f"✓ n1/n2 = U1/U2 = {self.U1}/{self.U2}")
        
        if self.results['U3'] is not None and self.results['n3_n4'] is not None:
            expected_ratio = self.results['U3'] / self.U4
            if not self._isclose(self.results['n3_n4'], expected_ratio):
                self.logs.append(f"✗ Transformer ratio n3/n4: {self.results['n3_n4']} ≠ U3/U4 = {expected_ratio}")
                success = False
            else:
                self.logs.append(f"✓ n3/n4 = U3/U4 = {self.results['U3']}/{self.U4}")
        
        # 3. Check Ohm's law and Joule's law consistency
        if all(k in self.results and self.results[k] is not None 
               for k in ['I', 'R', 'P_loss']):
            # P_loss = I²R
            computed = self.results['I']**2 * self.results['R']
            if not self._isclose(self.results['P_loss'], computed):
                self.logs.append(f"✗ Joule's law: P_loss = {self.results['P_loss']} ≠ I²R = {computed}")
                success = False
            else:
                self.logs.append(f"✓ Joule's law: P_loss = I²R = {computed}")
            
            # I = P_out / U2 (assuming ideal transformer)
            computed_I = self.P_out / self.U2
            if not self._isclose(self.results['I'], computed_I):
                self.logs.append(f"✗ Current: I = {self.results['I']} ≠ P_out/U2 = {computed_I}")
                success = False
            else:
                self.logs.append(f"✓ Current: I = P_out/U2 = {computed_I}")
        
        # 4. Check voltage drop: U3 = U2 - I*R
        if all(k in self.results and self.results[k] is not None 
               for k in ['U3', 'I', 'R']):
            computed_U3 = self.U2 - self.results['I'] * self.results['R']
            if not self._isclose(self.results['U3'], computed_U3):
                self.logs.append(f"✗ Voltage drop: U3 = {self.results['U3']} ≠ U2 - IR = {computed_U3}")
                success = False
            else:
                self.logs.append(f"✓ Voltage drop: U3 = U2 - IR = {computed_U3}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check each result against expected value
        for key, expected in self.expected.items():
            if key in self.results:
                if not self._check_result(key, self.results[key], expected):
                    success = False
        
        # Additional consistency: R computed from different formulas
        if all(k in self.results and self.results[k] is not None 
               for k in ['P_loss', 'I']):
            R_from_loss = self.results['P_loss'] / (self.results['I']**2)
            if self.results['R'] is not None:
                if not self._isclose(self.results['R'], R_from_loss):
                    self.logs.append(f"✗ R consistency: R = {self.results['R']} ≠ P_loss/I² = {R_from_loss}")
                    success = False
                else:
                    self.logs.append(f"✓ R consistent via P_loss/I² = {R_from_loss}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input and check physical trends."""
        success = True
        
        # Save original values
        orig_R = self.results['R']
        
        if orig_R is None:
            self.logs.append("Skipping monotonicity test: R not available")
            return True
        
        # Test 1: Increase resistance, power loss should increase
        new_R = orig_R * 1.1  # +10%
        new_P_loss = self.results['I']**2 * new_R
        if new_P_loss <= self.results['P_loss']:
            self.logs.append(f"✗ Monotonicity: Increasing R should increase P_loss")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: P_loss increases with R")
        
        # Test 2: Increase voltage U2, current should decrease (for fixed power)
        new_U2 = self.U2 * 1.1
        new_I = self.P_out / new_U2
        if new_I >= self.results['I']:
            self.logs.append(f"✗ Monotonicity: Increasing U2 should decrease I")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: I decreases with U2")
        
        # Test 3: Decrease received power, power loss should increase?
        # Actually, for fixed R and U2, if P_out decreases, I decreases, so P_loss decreases
        new_P_rec = self.P_rec * 0.9
        new_P_loss2 = self.P_out - new_P_rec
        new_I2 = self.P_out / self.U2
        new_P_loss3 = new_I2**2 * orig_R
        
        # P_loss should remain the same since P_out unchanged
        if not self._isclose(new_P_loss3, self.results['P_loss']):
            self.logs.append(f"✗ Monotonicity: P_loss should be independent of P_rec")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: P_loss depends only on P_out, U2, R")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        success = True
        
        # Run individual tests
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Physical Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            if not test_func():
                success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        if success:
            self.logs.append("All tests passed ✓")
        else:
            self.logs.append("Some tests failed ✗")
        
        return success, self.logs