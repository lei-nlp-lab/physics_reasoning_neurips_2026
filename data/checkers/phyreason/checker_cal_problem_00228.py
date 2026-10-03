import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results without direct attribute access
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Input parameters (with default fallbacks)
        self.inputs['S'] = getattr(solution_instance, 'S', 0.01)  # 100 cm² = 0.01 m²
        self.inputs['m'] = getattr(solution_instance, 'm', 1.0)   # kg
        self.inputs['T_A'] = getattr(solution_instance, 'T_A', 300.0)  # K
        self.inputs['h0'] = getattr(solution_instance, 'h0', 0.3)  # 30 cm = 0.3 m
        self.inputs['d'] = getattr(solution_instance, 'd', 0.03)   # 3 cm = 0.03 m
        self.inputs['T_C'] = getattr(solution_instance, 'T_C', 363.0)  # K
        self.inputs['ΔU'] = getattr(solution_instance, 'ΔU', 158.0)  # J
        self.inputs['p0'] = getattr(solution_instance, 'p0', 0.99e5)  # Pa
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)    # m/s²
        
        # Results from solution (with safe access)
        self.results['T_B'] = getattr(solution_instance, 'T_B', 
                                     getattr(solution_instance, 'results', {}).get('T_B', 0.0))
        self.results['p_B'] = getattr(solution_instance, 'p_B',
                                     getattr(solution_instance, 'results', {}).get('p_B', 0.0))
        self.results['p_C'] = getattr(solution_instance, 'p_C',
                                     getattr(solution_instance, 'results', {}).get('p_C', 0.0))
        self.results['W_0'] = getattr(solution_instance, 'W_0',
                                     getattr(solution_instance, 'results', {}).get('W_0', 0.0))
        self.results['Q'] = getattr(solution_instance, 'Q',
                                   getattr(solution_instance, 'results', {}).get('Q', 0.0))
        
        # Target answers for verification
        self.targets = {
            'T_B': 330.0,
            'p_C': 1.1e5,
            'Q': 188.0
        }
        
        self.logs = []
        
    def log(self, message, error=False):
        """Helper for logging test results"""
        if error:
            self.logs.append(f"❌ ERROR: {message}")
        else:
            self.logs.append(f"✅ {message}")
    
    def test_invariants(self):
        """Test physical invariants: Force equilibrium and First Law of Thermodynamics"""
        try:
            # 1. Force equilibrium at state B (piston stationary before stop)
            p_B_expected = self.inputs['p0'] + (self.inputs['m'] * self.inputs['g']) / self.inputs['S']
            if not math.isclose(self.results['p_B'], p_B_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Force equilibrium violated: p_B={self.results['p_B']:.1f}, expected={p_B_expected:.1f}")
                return False
            self.log(f"Force equilibrium verified: p_B={self.results['p_B']:.1f} Pa")
            
            # 2. First Law of Thermodynamics: ΔU = Q - W from A to C
            # W_total = W_0 (only work done in A->B, B->C is isochoric)
            ΔU_calc = self.results['Q'] - self.results['W_0']
            if not math.isclose(self.inputs['ΔU'], ΔU_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"First Law violation: ΔU={self.inputs['ΔU']:.1f}, Q-W={ΔU_calc:.1f}")
                return False
            self.log(f"First Law verified: ΔU={self.inputs['ΔU']:.1f} J = Q-W={ΔU_calc:.1f} J")
            
            # 3. Process invariants
            # A->B: Isobaric expansion
            V_A = self.inputs['S'] * self.inputs['h0']
            V_B = self.inputs['S'] * (self.inputs['h0'] + self.inputs['d'])
            T_B_expected = (V_B / V_A) * self.inputs['T_A']
            if not math.isclose(self.results['T_B'], T_B_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Isobaric process violation: T_B={self.results['T_B']:.1f}, expected={T_B_expected:.1f}")
                return False
            self.log(f"Isobaric process A->B verified: T_B={self.results['T_B']:.1f} K")
            
            # B->C: Isochoric heating
            p_C_expected = (self.inputs['T_C'] / self.results['T_B']) * self.results['p_B']
            if not math.isclose(self.results['p_C'], p_C_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Isochoric process violation: p_C={self.results['p_C']:.1f}, expected={p_C_expected:.1f}")
                return False
            self.log(f"Isochoric process B->C verified: p_C={self.results['p_C']:.1f} Pa")
            
            # 4. Work calculation verification
            W_expected = self.results['p_B'] * self.inputs['S'] * self.inputs['d']
            if not math.isclose(self.results['W_0'], W_expected, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Work calculation error: W_0={self.results['W_0']:.1f}, expected={W_expected:.1f}")
                return False
            self.log(f"Work calculation verified: W_0={self.results['W_0']:.1f} J")
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", error=True)
            return False
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers and input consistency"""
        try:
            # Check against target answers with lenient tolerance
            for key, target in self.targets.items():
                if key in self.results:
                    if not math.isclose(self.results[key], target, rel_tol=0.02, abs_tol=0.02):
                        self.log(f"Trace inconsistency: {key}={self.results[key]:.1f}, target={target:.1f}")
                        return False
                    self.log(f"Trace consistency: {key}={self.results[key]:.1f} matches target")
            
            # Verify that T_B is within reasonable bounds (between T_A and T_C)
            if not (self.inputs['T_A'] <= self.results['T_B'] <= self.inputs['T_C']):
                self.log(f"Temperature monotonicity violated: T_A={self.inputs['T_A']}, T_B={self.results['T_B']}, T_C={self.inputs['T_C']}")
                return False
            self.log(f"Temperature monotonicity verified: T_A ≤ T_B ≤ T_C")
            
            # Verify pressure monotonicity: p_A = p_B (isobaric) < p_C (isochoric heating)
            if not (self.results['p_B'] < self.results['p_C']):
                self.log(f"Pressure monotonicity violated: p_B={self.results['p_B']}, p_C={self.results['p_C']}")
                return False
            self.log(f"Pressure monotonicity verified: p_B < p_C")
            
            # Check volume ratio consistency
            V_ratio = (self.inputs['h0'] + self.inputs['d']) / self.inputs['h0']
            T_ratio = self.results['T_B'] / self.inputs['T_A']
            if not math.isclose(V_ratio, T_ratio, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Volume-temperature ratio mismatch: V_ratio={V_ratio:.3f}, T_ratio={T_ratio:.3f}")
                return False
            self.log(f"Volume-temperature ratio consistency verified")
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", error=True)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check expected trends"""
        original_m = self.inputs['m']
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate using the same formulas as solution
            # 1. New pressure at state B
            p_B_perturbed = self.inputs['p0'] + (perturbed_m * self.inputs['g']) / self.inputs['S']
            
            # 2. New temperature at state B (isobaric process, same volume ratio)
            V_ratio = (self.inputs['h0'] + self.inputs['d']) / self.inputs['h0']
            T_B_perturbed = V_ratio * self.inputs['T_A']
            
            # 3. New pressure at state C (isochoric heating)
            p_C_perturbed = (self.inputs['T_C'] / T_B_perturbed) * p_B_perturbed
            
            # 4. New work done
            W_perturbed = p_B_perturbed * self.inputs['S'] * self.inputs['d']
            
            # 5. New heat absorbed (using same ΔU)
            Q_perturbed = self.inputs['ΔU'] + W_perturbed
            
            # Check expected trends:
            # - Increasing mass increases pressure (p_B, p_C)
            if not (p_B_perturbed > self.results['p_B'] and p_C_perturbed > self.results['p_C']):
                self.log(f"Pressure monotonicity trend failed: p_B from {self.results['p_B']:.1f} to {p_B_perturbed:.1f}")
                return False
            
            # - Temperature at B remains same (depends only on geometry and T_A, not mass)
            if not math.isclose(T_B_perturbed, self.results['T_B'], rel_tol=0.02, abs_tol=0.02):
                self.log(f"Temperature T_B should be invariant to mass change")
                return False
            
            # - Increasing pressure increases work (W_0) since expansion at higher pressure
            if not (W_perturbed > self.results['W_0']):
                self.log(f"Work monotonicity trend failed: W_0 from {self.results['W_0']:.1f} to {W_perturbed:.1f}")
                return False
            
            # - Increasing work increases heat absorbed (Q) for fixed ΔU
            if not (Q_perturbed > self.results['Q']):
                self.log(f"Heat monotonicity trend failed: Q from {self.results['Q']:.1f} to {Q_perturbed:.1f}")
                return False
            
            self.log(f"Monotonicity test passed: All trends consistent with +10% mass perturbation")
            return True
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", error=True)
            return False
            
        finally:
            # Restore original mass value
            self.inputs['m'] = original_m
    
    def run_all_tests(self):
        """Execute all tests and return success boolean with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"\n--- Running {test_name} ---")
                if not test_func():
                    success = False
                    self.logs.append(f"❌ {test_name} FAILED")
                else:
                    self.logs.append(f"✅ {test_name} PASSED")
            except Exception as e:
                success = False
                self.logs.append(f"❌ {test_name} CRASHED: {str(e)}")
                self.logs.append(traceback.format_exc())
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs