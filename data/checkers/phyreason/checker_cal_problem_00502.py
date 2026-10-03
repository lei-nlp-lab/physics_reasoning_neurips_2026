import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract required values from results with proper defaults
        # Use explicit float conversion to ensure numeric values
        self.U = float(results.get('U', 0.0))
        self.E = float(results.get('E', 0.0))
        
        # Extract constants from results with explicit defaults
        # If not found in results, use the known physical constants
        self.q = float(results.get('q', 1.60e-19))  # proton charge (C)
        self.m = float(results.get('m', 1.67e-27))  # proton mass (kg)
        self.v = float(results.get('v_final', 8.0e6))  # final velocity (m/s)
        self.d = float(results.get('d', 4.0))       # acceleration distance (m)
        
        # Expected values from steps analysis
        self.U_expected = 3.34e5   # V
        self.E_expected = 8.35e4   # V/m

    def test_invariants(self):
        logs = []
        
        # Invariant 1: Work-energy theorem (qU = 1/2 mv²)
        try:
            lhs = self.q * self.U
            rhs = 0.5 * self.m * self.v**2
            
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append({
                    'type': 'error',
                    'test': 'Work-energy theorem invariant',
                    'message': f'qU = {lhs:.3e}, (1/2)mv² = {rhs:.3e}',
                    'deviation': f'{abs(lhs-rhs)/rhs:.2%}'
                })
            else:
                logs.append({
                    'type': 'success',
                    'test': 'Work-energy theorem invariant',
                    'message': 'Work-energy theorem satisfied: qU = (1/2)mv²'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'Work-energy theorem invariant',
                'message': f'Exception: {str(e)}',
                'traceback': traceback.format_exc()
            })

        # Invariant 2: U = Ed relationship for uniform field
        try:
            if not math.isclose(self.U, self.E * self.d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append({
                    'type': 'error',
                    'test': 'U = Ed invariant',
                    'message': f'U = {self.U:.3e}, E*d = {self.E * self.d:.3e}',
                    'deviation': f'{abs(self.U - self.E*self.d)/self.U:.2%}'
                })
            else:
                logs.append({
                    'type': 'success',
                    'test': 'U = Ed invariant',
                    'message': 'Uniform field relationship satisfied: U = Ed'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'U = Ed invariant',
                'message': f'Exception: {str(e)}',
                'traceback': traceback.format_exc()
            })

        return logs

    def test_trace_consistency(self):
        logs = []
        
        # Check U consistency with expected value from steps
        try:
            if not math.isclose(self.U, self.U_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append({
                    'type': 'error',
                    'test': 'U value consistency',
                    'message': f'U = {self.U:.3e}, expected {self.U_expected:.3e}',
                    'deviation': f'{abs(self.U - self.U_expected)/self.U_expected:.2%}'
                })
            else:
                logs.append({
                    'type': 'success',
                    'test': 'U value consistency',
                    'message': f'U matches expected value: {self.U:.3e} V'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'U value consistency',
                'message': f'Exception: {str(e)}'
            })

        # Check E consistency with expected value from steps
        try:
            if not math.isclose(self.E, self.E_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append({
                    'type': 'error',
                    'test': 'E value consistency',
                    'message': f'E = {self.E:.3e}, expected {self.E_expected:.3e}',
                    'deviation': f'{abs(self.E - self.E_expected)/self.E_expected:.2%}'
                })
            else:
                logs.append({
                    'type': 'success',
                    'test': 'E value consistency',
                    'message': f'E matches expected value: {self.E:.3e} V/m'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'E value consistency',
                'message': f'Exception: {str(e)}'
            })

        # Check intermediate consistency: U calculated from E and d should match
        try:
            U_from_E = self.E * self.d
            if not math.isclose(self.U, U_from_E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append({
                    'type': 'warning',
                    'test': 'Intermediate consistency',
                    'message': f'U from step 1 ({self.U:.3e}) vs U from E*d ({U_from_E:.3e})',
                    'note': 'Small discrepancies may arise from rounding'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'Intermediate consistency',
                'message': f'Exception: {str(e)}'
            })

        return logs

    def test_monotonicity(self):
        logs = []
        
        # Save original values
        v_original = self.v
        d_original = self.d
        U_original = self.U
        E_original = self.E
        
        # Test 1: Perturb velocity (+5%) and check U trend
        try:
            # Ensure q is not zero to avoid division by zero
            if self.q == 0:
                logs.append({
                    'type': 'error',
                    'test': 'U monotonicity with v',
                    'message': f'q = {self.q}, cannot compute U_perturbed (division by zero)'
                })
            else:
                v_perturbed = v_original * 1.05
                U_perturbed = 0.5 * self.m * v_perturbed**2 / self.q
                
                if U_perturbed <= U_original:
                    logs.append({
                        'type': 'error',
                        'test': 'U monotonicity with v',
                        'message': f'U({v_original:.2e}) = {U_original:.3e}, U({v_perturbed:.2e}) = {U_perturbed:.3e}',
                        'issue': 'U should increase with increasing v'
                    })
                else:
                    logs.append({
                        'type': 'success',
                        'test': 'U monotonicity with v',
                        'message': 'U increases with increasing v (positive work)'
                    })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'U monotonicity with v',
                'message': f'Exception: {str(e)}'
            })
        
        # Test 2: Perturb distance (+10%) and check E trend (with fixed U)
        try:
            d_perturbed = d_original * 1.10
            E_perturbed = U_original / d_perturbed  # E = U/d
            
            if E_perturbed >= E_original:
                logs.append({
                    'type': 'error',
                    'test': 'E monotonicity with d',
                    'message': f'E({d_original:.1f}) = {E_original:.3e}, E({d_perturbed:.1f}) = {E_perturbed:.3e}',
                    'issue': 'E should decrease with increasing d (for fixed U)'
                })
            else:
                logs.append({
                    'type': 'success',
                    'test': 'E monotonicity with d',
                    'message': 'E decreases with increasing d (inverse relationship)'
                })
        except Exception as e:
            logs.append({
                'type': 'error',
                'test': 'E monotonicity with d',
                'message': f'Exception: {str(e)}'
            })
        
        # Restore original values
        self.v = v_original
        self.d = d_original
        
        return logs

    def run_all_tests(self):
        all_logs = []
        
        # Run all test suites
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log['type'] == 'error':
                success = False
                break
        
        return success, all_logs