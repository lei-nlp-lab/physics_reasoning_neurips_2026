import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = {
            'd': getattr(solution_instance, 'd', 
                        getattr(solution_instance, 'results', {}).get('d', None)),
            'x': getattr(solution_instance, 'x', 
                        getattr(solution_instance, 'results', {}).get('x', None)),
            'overlinev': getattr(solution_instance, 'overlinev', 
                                getattr(solution_instance, 'results', {}).get('overlinev', None))
        }
        
        # Inputs from problem context
        self.leg1 = 6.0  # north leg (m)
        self.leg2 = 8.0  # west leg (m)
        self.time = 5.0  # total time (s)
        
        self.logs = []
        
    def _log(self, message: str):
        self.logs.append(message)
        
    def test_invariants(self) -> bool:
        """Test physical invariants using KB theorems."""
        success = True
        
        # 1. Check displacement magnitude via Pythagorean theorem
        expected_displacement = math.sqrt(self.leg1**2 + self.leg2**2)
        if self.results['x'] is not None:
            if not math.isclose(self.results['x'], expected_displacement, 
                               rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Displacement invariant failed: {self.results['x']} vs expected {expected_displacement}")
                success = False
            else:
                self._log(f"Displacement invariant passed: {self.results['x']} ≈ {expected_displacement}")
        else:
            self._log("Displacement result missing")
            success = False
            
        # 2. Check total distance traveled
        expected_distance = self.leg1 + self.leg2
        if self.results['d'] is not None:
            if not math.isclose(self.results['d'], expected_distance,
                               rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance invariant failed: {self.results['d']} vs expected {expected_distance}")
                success = False
            else:
                self._log(f"Distance invariant passed: {self.results['d']} ≈ {expected_distance}")
        else:
            self._log("Distance result missing")
            success = False
            
        # 3. Check average velocity definition
        if self.results['x'] is not None and self.results['overlinev'] is not None:
            expected_velocity = self.results['x'] / self.time
            if not math.isclose(self.results['overlinev'], expected_velocity,
                               rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average velocity invariant failed: {self.results['overlinev']} vs expected {expected_velocity}")
                success = False
            else:
                self._log(f"Average velocity invariant passed: {self.results['overlinev']} ≈ {expected_velocity}")
        else:
            self._log("Missing results for average velocity check")
            success = False
            
        return success
        
    def test_trace_consistency(self) -> bool:
        """Check intermediate steps lead to final results."""
        success = True
        
        # Symbolic consistency check using sympy
        try:
            # Define symbols
            leg1, leg2, t = sp.symbols('leg1 leg2 t', positive=True)
            
            # From steps analysis:
            # step_3: displacement = sqrt(leg1^2 + leg2^2)
            # step_4: distance = leg1 + leg2
            # step_5: avg_velocity = displacement / t
            
            displacement_expr = sp.sqrt(leg1**2 + leg2**2)
            distance_expr = leg1 + leg2
            velocity_expr = displacement_expr / t
            
            # Numeric substitution
            numeric_subs = {leg1: self.leg1, leg2: self.leg2, t: self.time}
            
            # Calculate expected values
            expected_displacement = float(displacement_expr.subs(numeric_subs))
            expected_distance = float(distance_expr.subs(numeric_subs))
            expected_velocity = float(velocity_expr.subs(numeric_subs))
            
            # Compare with results
            if self.results['x'] is not None:
                if not math.isclose(self.results['x'], expected_displacement,
                                   rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency failed for displacement: {self.results['x']} vs {expected_displacement}")
                    success = False
            
            if self.results['d'] is not None:
                if not math.isclose(self.results['d'], expected_distance,
                                   rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency failed for distance: {self.results['d']} vs {expected_distance}")
                    success = False
            
            if self.results['overlinev'] is not None:
                if not math.isclose(self.results['overlinev'], expected_velocity,
                                   rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency failed for velocity: {self.results['overlinev']} vs {expected_velocity}")
                    success = False
                    
            if success:
                self._log("Trace consistency passed: All intermediate steps lead to final results")
                
        except Exception as e:
            self._log(f"Symbolic trace check failed with error: {str(e)}")
            success = False
            
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb inputs and check trends."""
        success = True
        
        try:
            # Test 1: Increase time -> average velocity should decrease
            perturbed_time = self.time * 1.1  # 10% increase
            original_velocity = self.results['overlinev']
            
            if original_velocity is not None and self.results['x'] is not None:
                perturbed_velocity = self.results['x'] / perturbed_time
                
                if not (perturbed_velocity < original_velocity or 
                       math.isclose(perturbed_velocity, original_velocity, rel_tol=1e-2)):
                    self._log(f"Monotonicity failed: velocity should decrease when time increases")
                    self._log(f"Original: {original_velocity}, After +10% time: {perturbed_velocity}")
                    success = False
                else:
                    self._log(f"Monotonicity passed: velocity decreases with increased time")
            else:
                self._log("Missing results for time perturbation test")
                
            # Test 2: Increase a leg -> both distance and displacement should increase
            perturbed_leg1 = self.leg1 * 1.1  # 10% increase
            original_displacement = self.results['x']
            original_distance = self.results['d']
            
            if original_displacement is not None and original_distance is not None:
                new_displacement = math.sqrt(perturbed_leg1**2 + self.leg2**2)
                new_distance = perturbed_leg1 + self.leg2
                
                # Check distance monotonicity
                if not (new_distance > original_distance or 
                       math.isclose(new_distance, original_distance, rel_tol=1e-2)):
                    self._log(f"Monotonicity failed: distance should increase when leg increases")
                    self._log(f"Original distance: {original_distance}, After +10% leg1: {new_distance}")
                    success = False
                    
                # Check displacement monotonicity  
                if not (new_displacement > original_displacement or 
                       math.isclose(new_displacement, original_displacement, rel_tol=1e-2)):
                    self._log(f"Monotonicity failed: displacement should increase when leg increases")
                    self._log(f"Original displacement: {original_displacement}, After +10% leg1: {new_displacement}")
                    success = False
                    
                if success:
                    self._log("Monotonicity passed: Both distance and displacement increase with leg length")
            else:
                self._log("Missing results for leg perturbation test")
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}")
            success = False
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs.clear()
        self._log("Starting Physics Audit...")
        
        results = []
        
        # Test 1: Invariants
        self._log("\n--- Testing Physical Invariants ---")
        inv_success = self.test_invariants()
        results.append(("Invariants", inv_success))
        
        # Test 2: Trace Consistency
        self._log("\n--- Testing Trace Consistency ---")
        trace_success = self.test_trace_consistency()
        results.append(("Trace Consistency", trace_success))
        
        # Test 3: Monotonicity
        self._log("\n--- Testing Monotonicity ---")
        mono_success = self.test_monotonicity()
        results.append(("Monotonicity", mono_success))
        
        # Final verdict
        all_success = all(s for _, s in results)
        
        self._log("\n" + "="*50)
        self._log("AUDIT SUMMARY:")
        for test_name, test_success in results:
            status = "PASS" if test_success else "FAIL"
            self._log(f"{test_name}: {status}")
        self._log(f"\nOverall: {'PASS' if all_success else 'FAIL'}")
        
        return all_success, self.logs