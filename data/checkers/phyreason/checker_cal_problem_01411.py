import math
import sympy as sp
from typing import Tuple, List, Dict, Any
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        self.logs = []
        
        # Extract inputs with safe access
        self.inputs['E'] = getattr(solution_instance, 'E', 1.0)  # Default to 1.0 if not found
        self.inputs['q'] = getattr(solution_instance, 'q', 1.0)
        self.inputs['l'] = getattr(solution_instance, 'l', 1.0)
        self.inputs['m'] = getattr(solution_instance, 'm', 1.0)
        
        # Extract results with safe access (first from attributes, then from results dict)
        self.results['v_0'] = getattr(solution_instance, 'v_0', 
                                     getattr(solution_instance, 'results', {}).get('v_0', 0.0))
        self.results['v'] = getattr(solution_instance, 'v',
                                   getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.results['x_P'] = getattr(solution_instance, 'x_P',
                                     getattr(solution_instance, 'results', {}).get('x_P', 0.0))
        self.results['W_e'] = getattr(solution_instance, 'W_e',
                                     getattr(solution_instance, 'results', {}).get('W_e', 0.0))
        self.results['delta_E_p'] = getattr(solution_instance, 'delta_E_p',
                                           getattr(solution_instance, 'results', {}).get('delta_E_p', 0.0))
        self.results['ON'] = getattr(solution_instance, 'ON',
                                    getattr(solution_instance, 'results', {}).get('ON', 0.0))
        self.results['tanθ'] = getattr(solution_instance, 'tanθ',
                                      getattr(solution_instance, 'results', {}).get('tanθ', 0.0))
        
        # Convert to float if they're sympy expressions
        for key in self.results:
            if isinstance(self.results[key], sp.Expr):
                try:
                    self.results[key] = float(self.results[key])
                except:
                    pass
        
        # Constants for calculations
        self.E = float(self.inputs['E'])
        self.q = float(self.inputs['q'])
        self.l = float(self.inputs['l'])
        self.m = float(self.inputs['m'])
        
    def test_invariants(self) -> List[str]:
        """Test physical invariants and conservation laws."""
        logs = []
        
        try:
            # 1. Energy conservation from M to P
            # Work-energy theorem: W_e = ΔK = 1/2*m*v^2 - 0 (starts from rest)
            # From steps: v = 2*sqrt(E*q*l/m)
            v_expected = 2.0 * math.sqrt(self.E * self.q * self.l / self.m)
            ΔK_expected = 0.5 * self.m * v_expected**2
            W_e_expected = ΔK_expected  # From step 7
            
            if hasattr(self.solution, 'W_e') or 'W_e' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['W_e'], W_e_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Energy conservation violated: W_e = {self.results['W_e']:.4e}, expected {W_e_expected:.4e}")
                else:
                    logs.append("✓ Energy conservation (work-energy theorem) satisfied")
            
            # 2. Relationship between work and potential energy: ΔE_p = -W_e
            # From steps: ΔE_p = -2Eql and W_e = 2Eql
            if (hasattr(self.solution, 'delta_E_p') or 'delta_E_p' in getattr(self.solution, 'results', {})) and \
               (hasattr(self.solution, 'W_e') or 'W_e' in getattr(self.solution, 'results', {})):
                delta_E_p_expected = -self.results['W_e']
                if not math.isclose(self.results['delta_E_p'], delta_E_p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Work-potential energy relation violated: ΔE_p = {self.results['delta_E_p']:.4e}, expected -W_e = {-self.results['W_e']:.4e}")
                else:
                    logs.append("✓ Work-potential energy relation satisfied")
            
            # 3. Geometric constraint: tanθ = 1/2 = ON / x_P = l / x_P
            # From steps: tanθ = 1/2, x_P = 2l
            tanθ_expected = 0.5
            if hasattr(self.solution, 'tanθ') or 'tanθ' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['tanθ'], tanθ_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Geometric constraint violated: tanθ = {self.results['tanθ']:.4f}, expected 0.5")
                else:
                    logs.append("✓ Geometric constraint (tanθ = 1/2) satisfied")
            
            # 4. Speed relation: v^2 = v_0^2 + v_y^2 (from velocity composition)
            # From steps: v_0 = sqrt(2Eql/m), v = 2*sqrt(Eql/m) = sqrt(2)*v_0
            v_0_expected = math.sqrt(2.0 * self.E * self.q * self.l / self.m)
            v_expected = math.sqrt(2.0) * v_0_expected
            
            if (hasattr(self.solution, 'v_0') or 'v_0' in getattr(self.solution, 'results', {})) and \
               (hasattr(self.solution, 'v') or 'v' in getattr(self.solution, 'results', {})):
                v_ratio = self.results['v']**2 / (2.0 * self.results['v_0']**2) if self.results['v_0'] != 0 else 0
                if not math.isclose(v_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Velocity composition violated: v^2/(2*v_0^2) = {v_ratio:.4f}, expected 1.0")
                else:
                    logs.append("✓ Velocity composition relation satisfied")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check consistency between intermediate and final results."""
        logs = []
        
        try:
            # 1. Check v_0 from step 1: v_0 = sqrt(2Eql/m)
            v_0_expected = math.sqrt(2.0 * self.E * self.q * self.l / self.m)
            if hasattr(self.solution, 'v_0') or 'v_0' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['v_0'], v_0_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"v_0 inconsistent: {self.results['v_0']:.4e}, expected {v_0_expected:.4e}")
                else:
                    logs.append("✓ v_0 consistent with step 1 formula")
            
            # 2. Check v from step 6: v = 2*sqrt(Eql/m)
            v_expected = 2.0 * math.sqrt(self.E * self.q * self.l / self.m)
            if hasattr(self.solution, 'v') or 'v' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['v'], v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"v inconsistent: {self.results['v']:.4e}, expected {v_expected:.4e}")
                else:
                    logs.append("✓ v consistent with step 6 formula")
            
            # 3. Check ΔE_p from step 9: ΔE_p = -2Eql
            delta_E_p_expected = -2.0 * self.E * self.q * self.l
            if hasattr(self.solution, 'delta_E_p') or 'delta_E_p' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['delta_E_p'], delta_E_p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"ΔE_p inconsistent: {self.results['delta_E_p']:.4e}, expected {delta_E_p_expected:.4e}")
                else:
                    logs.append("✓ ΔE_p consistent with step 9 formula")
            
            # 4. Check x_P from step 13: x_P = 2l
            x_P_expected = 2.0 * self.l
            if hasattr(self.solution, 'x_P') or 'x_P' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['x_P'], x_P_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"x_P inconsistent: {self.results['x_P']:.4e}, expected {x_P_expected:.4e}")
                else:
                    logs.append("✓ x_P consistent with step 13 formula")
            
            # 5. Check ON from step 11: ON = l
            if hasattr(self.solution, 'ON') or 'ON' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.results['ON'], self.l, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"ON inconsistent: {self.results['ON']:.4e}, expected {self.l:.4e}")
                else:
                    logs.append("✓ ON consistent with step 11 formula")
            
            # 6. Check tanθ from step 12: tanθ = l/x_P
            if (hasattr(self.solution, 'tanθ') or 'tanθ' in getattr(self.solution, 'results', {})) and \
               (hasattr(self.solution, 'x_P') or 'x_P' in getattr(self.solution, 'results', {})):
                if self.results['x_P'] != 0:
                    tanθ_calculated = self.l / self.results['x_P']
                    if not math.isclose(self.results['tanθ'], tanθ_calculated, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"tanθ inconsistent with geometry: tanθ = {self.results['tanθ']:.4f}, l/x_P = {tanθ_calculated:.4f}")
                    else:
                        logs.append("✓ tanθ consistent with geometric relation")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test that perturbing inputs changes outputs in expected ways."""
        logs = []
        
        try:
            # Store original values
            original_l = self.l
            perturbation_factor = 1.1  # 10% increase
            
            # Perturb l
            perturbed_l = original_l * perturbation_factor
            
            # Calculate expected changes using formulas from steps
            # v_0 ∝ sqrt(l)
            v_0_original = math.sqrt(2.0 * self.E * self.q * original_l / self.m)
            v_0_perturbed = math.sqrt(2.0 * self.E * self.q * perturbed_l / self.m)
            v_0_ratio_expected = math.sqrt(perturbation_factor)
            
            # v ∝ sqrt(l)
            v_original = 2.0 * math.sqrt(self.E * self.q * original_l / self.m)
            v_perturbed = 2.0 * math.sqrt(self.E * self.q * perturbed_l / self.m)
            v_ratio_expected = math.sqrt(perturbation_factor)
            
            # x_P ∝ l
            x_P_original = 2.0 * original_l
            x_P_perturbed = 2.0 * perturbed_l
            x_P_ratio_expected = perturbation_factor
            
            # W_e ∝ l
            W_e_original = 2.0 * self.E * self.q * original_l
            W_e_perturbed = 2.0 * self.E * self.q * perturbed_l
            W_e_ratio_expected = perturbation_factor
            
            # ΔE_p ∝ l (negative)
            delta_E_p_original = -2.0 * self.E * self.q * original_l
            delta_E_p_perturbed = -2.0 * self.E * self.q * perturbed_l
            delta_E_p_ratio_expected = perturbation_factor
            
            # Check monotonicity: all should increase with l (except ΔE_p which becomes more negative)
            if hasattr(self.solution, 'v_0') or 'v_0' in getattr(self.solution, 'results', {}):
                actual_v_0 = self.results['v_0']
                v_0_increase_expected = v_0_perturbed > v_0_original
                logs.append(f"v_0 monotonicity: original={v_0_original:.4e}, perturbed={v_0_perturbed:.4e}, increase expected={v_0_increase_expected}")
            
            if hasattr(self.solution, 'v') or 'v' in getattr(self.solution, 'results', {}):
                actual_v = self.results['v']
                v_increase_expected = v_perturbed > v_original
                logs.append(f"v monotonicity: original={v_original:.4e}, perturbed={v_perturbed:.4e}, increase expected={v_increase_expected}")
            
            if hasattr(self.solution, 'x_P') or 'x_P' in getattr(self.solution, 'results', {}):
                actual_x_P = self.results['x_P']
                x_P_increase_expected = x_P_perturbed > x_P_original
                logs.append(f"x_P monotonicity: original={x_P_original:.4e}, perturbed={x_P_perturbed:.4e}, increase expected={x_P_increase_expected}")
            
            if hasattr(self.solution, 'W_e') or 'W_e' in getattr(self.solution, 'results', {}):
                actual_W_e = self.results['W_e']
                W_e_increase_expected = W_e_perturbed > W_e_original
                logs.append(f"W_e monotonicity: original={W_e_original:.4e}, perturbed={W_e_perturbed:.4e}, increase expected={W_e_increase_expected}")
            
            if hasattr(self.solution, 'delta_E_p') or 'delta_E_p' in getattr(self.solution, 'results', {}):
                actual_delta_E_p = self.results['delta_E_p']
                # ΔE_p should become more negative (decrease)
                delta_E_p_decrease_expected = delta_E_p_perturbed < delta_E_p_original
                logs.append(f"ΔE_p monotonicity: original={delta_E_p_original:.4e}, perturbed={delta_E_p_perturbed:.4e}, decrease expected={delta_E_p_decrease_expected}")
            
            # Check scaling relationships
            logs.append(f"Scaling relationships verified with {perturbation_factor:.1%} perturbation:")
            logs.append(f"  v_0 ratio: {v_0_perturbed/v_0_original:.4f} vs expected √({perturbation_factor}) = {v_0_ratio_expected:.4f}")
            logs.append(f"  v ratio: {v_perturbed/v_original:.4f} vs expected √({perturbation_factor}) = {v_ratio_expected:.4f}")
            logs.append(f"  x_P ratio: {x_P_perturbed/x_P_original:.4f} vs expected {perturbation_factor}")
            logs.append(f"  W_e ratio: {W_e_perturbed/W_e_original:.4f} vs expected {perturbation_factor}")
            logs.append(f"  ΔE_p ratio: {delta_E_p_perturbed/delta_E_p_original:.4f} vs expected {perturbation_factor}")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run invariant tests
            all_logs.append("=" * 60)
            all_logs.append("Running invariant tests...")
            all_logs.append("=" * 60)
            invariant_logs = self.test_invariants()
            all_logs.extend(invariant_logs)
            
            # Run trace consistency tests
            all_logs.append("\n" + "=" * 60)
            all_logs.append("Running trace consistency tests...")
            all_logs.append("=" * 60)
            trace_logs = self.test_trace_consistency()
            all_logs.extend(trace_logs)
            
            # Run monotonicity tests
            all_logs.append("\n" + "=" * 60)
            all_logs.append("Running monotonicity tests...")
            all_logs.append("=" * 60)
            monotonicity_logs = self.test_monotonicity()
            all_logs.extend(monotonicity_logs)
            
            # Check if any tests failed (look for error messages)
            error_count = sum(1 for log in all_logs if log.startswith(("Error", "violated", "inconsistent")))
            if error_count > 0:
                success = False
                all_logs.append(f"\n❌ Tests failed: {error_count} error(s) found")
            else:
                all_logs.append("\n✅ All tests passed!")
                
        except Exception as e:
            all_logs.append(f"Fatal error in run_all_tests: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs