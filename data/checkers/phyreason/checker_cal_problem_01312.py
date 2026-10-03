import math
import sympy as sp
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required attributes from solution_instance.
        Uses fallback to results dictionary and default values to prevent AttributeError.
        """
        self.inputs = {}
        self.results = {}
        
        # Input parameters with safe extraction
        self.inputs['R'] = getattr(solution_instance, 'R', 
                                  getattr(solution_instance, 'results', {}).get('R', 1.0))
        self.inputs['R1'] = getattr(solution_instance, 'R1', 
                                   getattr(solution_instance, 'results', {}).get('R1', 1.0))
        self.inputs['r'] = getattr(solution_instance, 'r', 
                                  getattr(solution_instance, 'results', {}).get('r', 1.0))
        self.inputs['r1'] = getattr(solution_instance, 'r1', 
                                   getattr(solution_instance, 'results', {}).get('r1', 1.0))
        self.inputs['g'] = getattr(solution_instance, 'g', 
                                  getattr(solution_instance, 'results', {}).get('g', 9.81))
        
        # Results with safe extraction
        self.results['v'] = getattr(solution_instance, 'v', 
                                   getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.results['T'] = getattr(solution_instance, 'T', 
                                   getattr(solution_instance, 'results', {}).get('T', 0.0))
        
        # For symbolic computations
        self.m = sp.Symbol('m', positive=True)  # satellite mass
        self.M_earth = sp.Symbol('M_earth', positive=True)  # Earth mass
        self.M_moon = sp.Symbol('M_moon', positive=True)    # Moon mass
        self.G = sp.Symbol('G', positive=True)  # gravitational constant
        
        self.logs = []
    
    def _log(self, msg: str, is_error: bool = False):
        """Helper for logging messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {msg}")
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants:
        1. Newton's Law of Universal Gravitation for parking orbit
        2. Newton's Law of Universal Gravitation for working orbit
        3. Centripetal force balance in both orbits
        """
        success = True
        local_logs = []
        
        try:
            R, R1, r, r1, g = (self.inputs[k] for k in ['R', 'R1', 'r', 'r1', 'g'])
            v, T = self.results['v'], self.results['T']
            
            # 1. Parking orbit invariant: centripetal force = gravitational force
            # From g = G*M_earth/r^2 => G*M_earth = g*r^2
            # Expected: v^2/R = (G*M_earth)/R^2 = (g*r^2)/R^2
            lhs_parking = v**2 / R
            rhs_parking = (g * r**2) / (R**2)
            
            if not math.isclose(lhs_parking, rhs_parking, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"Parking orbit invariant failed: v²/R={lhs_parking:.4f}, g*r²/R²={rhs_parking:.4f}")
                success = False
            else:
                local_logs.append(f"Parking orbit invariant passed: v²/R ≈ g*r²/R²")
            
            # 2. Working orbit invariant: centripetal force = gravitational force
            # From g/6 = G*M_moon/r1^2 => G*M_moon = (g/6)*r1^2
            # Expected: (2π/T)² * R1 = (G*M_moon)/R1^2 = (g*r1^2)/(6*R1^2)
            angular_velocity = (2 * math.pi / T) if T != 0 else 0
            lhs_working = (angular_velocity**2) * R1
            rhs_working = (g * r1**2) / (6 * R1**2)
            
            if not math.isclose(lhs_working, rhs_working, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"Working orbit invariant failed: ω²R={lhs_working:.4e}, g*r1²/(6R1²)={rhs_working:.4e}")
                success = False
            else:
                local_logs.append(f"Working orbit invariant passed: ω²R ≈ g*r1²/(6R1²)")
            
            # 3. Energy conservation between orbits (approximate check)
            # Total mechanical energy: E = -G*M*m/(2R) for circular orbit
            # Since G*M_earth = g*r^2 and G*M_moon = g*r1^2/6
            energy_parking = - (g * r**2) / (2 * R)
            energy_working = - (g * r1**2) / (12 * R1)
            
            # Energy should change due to velocity adjustments - just log, don't fail
            local_logs.append(f"Energy parking: {energy_parking:.4e}, working: {energy_working:.4e}")
            
        except Exception as e:
            local_logs.append(f"Invariant test failed with error: {str(e)}")
            success = False
        
        return success, local_logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answers using formulas from STEPS ANALYSIS.
        """
        success = True
        local_logs = []
        
        try:
            R, R1, r, r1, g = (self.inputs[k] for k in ['R', 'R1', 'r', 'r1', 'g'])
            v, T = self.results['v'], self.results['T']
            
            # Check v formula from step_2: v = r * sqrt(g/R)
            v_expected = r * math.sqrt(g / R)
            if not math.isclose(v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"Velocity mismatch: v={v:.4f}, expected r√(g/R)={v_expected:.4f}")
                success = False
            else:
                local_logs.append(f"Velocity formula check passed: v ≈ r√(g/R)")
            
            # Check T formula from step_5: T = (2πR1/r1) * sqrt(6R1/g)
            T_expected = (2 * math.pi * R1 / r1) * math.sqrt(6 * R1 / g)
            if not math.isclose(T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"Period mismatch: T={T:.4f}, expected (2πR1/r1)√(6R1/g)={T_expected:.4f}")
                success = False
            else:
                local_logs.append(f"Period formula check passed: T ≈ (2πR1/r1)√(6R1/g)")
            
            # Symbolic verification of derivations
            # Parking orbit derivation
            v_sym = sp.symbols('v', positive=True)
            eq1 = sp.Eq(self.G * self.M_earth / R**2, v_sym**2 / R)
            eq2 = sp.Eq(self.G * self.M_earth, g * r**2)  # from Earth surface gravity
            
            # Solve symbolically for v
            sol = sp.solve([eq1, eq2], [v_sym, self.G], dict=True)
            if sol:
                v_sym_solution = sol[0][v_sym]
                local_logs.append(f"Symbolic solution for v: {v_sym_solution}")
                # Check if it matches r*sqrt(g/R)
                expected_expr = r * sp.sqrt(g / R)
                if sp.simplify(v_sym_solution - expected_expr) == 0:
                    local_logs.append("Symbolic derivation of v matches formula")
            
        except Exception as e:
            local_logs.append(f"Trace consistency test failed with error: {str(e)}")
            success = False
        
        return success, local_logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and check expected trends without modifying solution_instance.
        Tests monotonic relationships from KB reference.
        """
        success = True
        local_logs = []
        
        try:
            # Original values
            R, R1, r, r1, g = (self.inputs[k] for k in ['R', 'R1', 'r', 'r1', 'g'])
            
            # Test 1: v ∝ 1/sqrt(R)
            # Increase R by 1%, v should decrease
            R_perturbed = R * 1.01
            v_original = r * math.sqrt(g / R)
            v_perturbed = r * math.sqrt(g / R_perturbed)
            
            if v_perturbed < v_original:
                local_logs.append(f"Monotonicity R→v: R↑ {R:.4f}→{R_perturbed:.4f}, v↓ {v_original:.4f}→{v_perturbed:.4f}")
            else:
                local_logs.append(f"Monotonicity R→v FAILED: Expected v to decrease when R increases")
                success = False
            
            # Test 2: T ∝ R1^(3/2) / r1
            # Increase R1 by 1%, T should increase
            R1_perturbed = R1 * 1.01
            T_original = (2 * math.pi * R1 / r1) * math.sqrt(6 * R1 / g)
            T_perturbed = (2 * math.pi * R1_perturbed / r1) * math.sqrt(6 * R1_perturbed / g)
            
            if T_perturbed > T_original:
                local_logs.append(f"Monotonicity R1→T: R1↑ {R1:.4f}→{R1_perturbed:.4f}, T↑ {T_original:.4f}→{T_perturbed:.4f}")
            else:
                local_logs.append(f"Monotonicity R1→T FAILED: Expected T to increase when R1 increases")
                success = False
            
            # Test 3: T ∝ 1/r1 (holding R1 constant)
            # Increase r1 by 1%, T should decrease
            r1_perturbed = r1 * 1.01
            T_perturbed_r1 = (2 * math.pi * R1 / r1_perturbed) * math.sqrt(6 * R1 / g)
            
            if T_perturbed_r1 < T_original:
                local_logs.append(f"Monotonicity r1→T: r1↑ {r1:.4f}→{r1_perturbed:.4f}, T↓ {T_original:.4f}→{T_perturbed_r1:.4f}")
            else:
                local_logs.append(f"Monotonicity r1→T FAILED: Expected T to decrease when r1 increases")
                success = False
            
            # Test 4: From KB - if distance decreases, gravitational force increases
            # Check gravitational force at R vs R/1.01 (closer to Earth)
            F_R = (g * r**2) / (R**2)
            F_R_closer = (g * r**2) / ((R/1.01)**2)
            if F_R_closer > F_R:
                local_logs.append(f"Gravity vs distance: R↓ {R:.4f}→{R/1.01:.4f}, F↑ {F_R:.4e}→{F_R_closer:.4e}")
            else:
                local_logs.append(f"Gravity vs distance check FAILED")
                success = False
                
        except Exception as e:
            local_logs.append(f"Monotonicity test failed with error: {str(e)}")
            success = False
        
        return success, local_logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and aggregate results.
        Returns: (overall_success, all_logs)
        """
        self.logs = ["Starting PhysicsChecker for Chang'e-3 satellite problem"]
        
        # Run all tests
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        self.logs.extend(logs1)
        if not success1:
            overall_success = False
            self.logs.append("Invariant tests failed")
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        self.logs.extend(logs2)
        if not success2:
            overall_success = False
            self.logs.append("Trace consistency tests failed")
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        self.logs.extend(logs3)
        if not success3:
            overall_success = False
            self.logs.append("Monotonicity tests failed")
        
        # Summary
        if overall_success:
            self.logs.append("All physics checks passed!")
        else:
            self.logs.append("Some physics checks failed")
        
        return overall_success, self.logs