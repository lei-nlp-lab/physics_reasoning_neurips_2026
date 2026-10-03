import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safely extract inputs from solution instance
        self.E_emf = getattr(solution_instance, 'E_emf', 60.0)  # Battery EMF
        self.r = getattr(solution_instance, 'r', 2.0)
        self.R1 = getattr(solution_instance, 'R1', 8.0)
        self.R2 = getattr(solution_instance, 'R2', 15.0)
        self.d = getattr(solution_instance, 'd', 0.04)  # 4cm in meters
        self.Q = getattr(solution_instance, 'Q', 6e-8)
        self.d_AC = getattr(solution_instance, 'd_AC', 0.03)  # 3cm from plate A to C
        
        # Safely extract results from solution instance
        results = getattr(solution_instance, 'results', {})
        self.I = results.get('I', 0.0)
        self.U = results.get('U', 0.0)  # Voltage across capacitor
        self.C = results.get('C', 0.0)
        self.E_field = results.get('E', 0.0)  # Electric field strength
        self.U_AC = results.get('U_AC', 0.0)
        self.varphi_A = results.get('varphi_A', 0.0)
        self.varphi_C = results.get('varphi_C', 0.0)
        
        # Constants
        self.epsilon0 = 8.854e-12 if hasattr(solution_instance, 'epsilon0') else 8.854e-12
        
        # For monotonicity perturbation
        self.original_R1 = self.R1
        
    def fuzzy_equal(self, a: float, b: float) -> bool:
        """Check if two numbers are equal within 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> List[str]:
        """Test physical invariants using KB theorems."""
        logs = []
        
        # 1. Ohm's Law for total circuit
        try:
            I_expected = self.E_emf / (self.R1 + self.R2 + self.r)
            if not self.fuzzy_equal(self.I, I_expected):
                logs.append(f"FAIL: Ohm's Law violation. I={self.I:.3f}, expected {I_expected:.3f}")
            else:
                logs.append(f"PASS: Ohm's Law for total circuit (I = E/(R1+R2+r))")
        except ZeroDivisionError:
            logs.append("ERROR: Division by zero in Ohm's Law")
        
        # 2. Voltage across R2 (same as capacitor voltage)
        U_expected = self.I * self.R2
        if not self.fuzzy_equal(self.U, U_expected):
            logs.append(f"FAIL: Voltage across R2. U={self.U:.3f}, expected {U_expected:.3f}")
        else:
            logs.append(f"PASS: Voltage across R2 (U = I*R2)")
        
        # 3. Capacitance definition
        if self.U != 0:
            C_expected = self.Q / self.U
            if not self.fuzzy_equal(self.C, C_expected):
                logs.append(f"FAIL: Capacitance definition. C={self.C:.3e}, expected {C_expected:.3e}")
            else:
                logs.append(f"PASS: Capacitance definition (C = Q/U)")
        
        # 4. Electric field between plates
        if self.d != 0:
            E_expected = self.U / self.d
            if not self.fuzzy_equal(self.E_field, E_expected):
                logs.append(f"FAIL: Electric field. E={self.E_field:.3f}, expected {E_expected:.3f}")
            else:
                logs.append(f"PASS: Electric field (E = U/d)")
        
        # 5. Potential difference from electric field
        U_AC_expected = self.E_field * self.d_AC
        if not self.fuzzy_equal(self.U_AC, abs(U_AC_expected)):
            logs.append(f"FAIL: U_AC from electric field. U_AC={self.U_AC:.3f}, expected {U_AC_expected:.3f}")
        else:
            logs.append(f"PASS: U_AC from electric field (U_AC = E*d_AC)")
        
        # 6. Potential difference relation (U_AC = φ_A - φ_C)
        U_AC_from_potentials = self.varphi_A - self.varphi_C
        if not self.fuzzy_equal(self.U_AC, abs(U_AC_from_potentials)):
            logs.append(f"FAIL: U_AC from potentials. U_AC={self.U_AC:.3f}, φ_A-φ_C={U_AC_from_potentials:.3f}")
        else:
            logs.append(f"PASS: U_AC from potentials (U_AC = φ_A - φ_C)")
        
        # 7. Plate A is grounded
        if not self.fuzzy_equal(self.varphi_A, 0.0):
            logs.append(f"FAIL: Plate A not grounded. φ_A={self.varphi_A:.3f}, expected 0")
        else:
            logs.append(f"PASS: Plate A is grounded (φ_A=0)")
        
        # 8. Check φ_C sign consistency with electric field direction
        # Electric field points from higher to lower potential
        # Since U = φ_A - φ_B and U > 0, φ_A > φ_B
        # For point C between A and B, φ_A > φ_C > φ_B if field is uniform
        # But given φ_A=0 and U_AC positive, we have 0 - φ_C = positive → φ_C negative
        if self.U_AC > 0 and self.varphi_C >= 0:
            logs.append(f"WARNING: Potential sign inconsistency. U_AC positive suggests φ_C should be negative")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate steps with final answers."""
        logs = []
        
        # Check all steps from the steps analysis
        steps_expected = {
            'I': 2.4,
            'U': 36.0,
            'C': 1.67e-9,
            'E_field': 900.0,
            'U_AC': 27.0,
            'varphi_C': -27.0
        }
        
        for key, expected in steps_expected.items():
            actual = getattr(self, key, None)
            if actual is not None:
                if key == 'C' and not self.fuzzy_equal(actual, expected):
                    # Special handling for capacitance due to rounding
                    C_alt = self.Q / self.U
                    if self.fuzzy_equal(actual, C_alt):
                        logs.append(f"PASS: {key} consistent (actual={actual:.3e}, expected={expected:.3e}, alt={C_alt:.3e})")
                    else:
                        logs.append(f"FAIL: {key} mismatch. actual={actual:.3e}, expected={expected:.3e}")
                elif not self.fuzzy_equal(actual, expected):
                    logs.append(f"FAIL: {key} mismatch. actual={actual:.3f}, expected={expected:.3f}")
                else:
                    logs.append(f"PASS: {key} consistent with steps analysis")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test monotonic behavior by perturbing R1."""
        logs = []
        
        # Save original state
        original_R1 = self.R1
        
        try:
            # Perturb R1 upward by 10%
            perturbed_R1 = original_R1 * 1.1
            self.R1 = perturbed_R1
            
            # Recalculate using formulas (not calling solution)
            I_perturbed = self.E_emf / (self.R1 + self.R2 + self.r)
            
            # Check trend: if R1 increases, I should decrease
            if I_perturbed < self.I:
                logs.append(f"PASS: Monotonicity - Increasing R1 decreases I ({I_perturbed:.3f} < {self.I:.3f})")
            else:
                logs.append(f"FAIL: Monotonicity violation - I didn't decrease when R1 increased")
            
            # Check other quantities that should also decrease
            U_perturbed = I_perturbed * self.R2
            if U_perturbed < self.U:
                logs.append(f"PASS: Monotonicity - U also decreases when R1 increases")
            else:
                logs.append(f"WARNING: U didn't decrease as expected")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
        finally:
            # Restore original R1
            self.R1 = original_R1
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                test_logs = test_func()
                # Check if any FAIL messages in logs
                test_success = not any(log.startswith("FAIL") for log in test_logs)
                if not test_success:
                    success = False
                all_logs.extend(test_logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall success: {success}")
        
        return success, all_logs

# Optional: Helper function for symbolic checks
def symbolic_checks():
    """Demonstrate symbolic physics laws from KB."""
    I, R, V, P, delta_t = sp.symbols('I R V P delta_t')
    E_avg, delta_x, V_1, V_0, sigma_net, epsilon0 = sp.symbols('E_avg delta_x V_1 V_0 sigma_net epsilon0')
    
    # Ohm's Law formulas
    ohm_laws = [
        sp.Eq(V, I * R),
        sp.Eq(P, I**2 * R),
        sp.Eq(P, V**2 / R),
        sp.Eq(P, V * I),  # Also valid
    ]
    
    # Electric field and potential
    field_potential = [
        sp.Eq(V_1 - V_0, -E_avg * delta_x),  # Note: delta_V = -E_avg*delta_x
        sp.Eq(E_avg, (V_0 - V_1) / delta_x),  # Alternative form
    ]
    
    # Parallel plate formulas
    parallel_plate = [
        sp.Eq(E_avg, (V_0 - V_1) / delta_x),
        sp.Eq(E_avg, sigma_net / (2 * epsilon0)),
    ]
    
    return ohm_laws, field_potential, parallel_plate

if __name__ == "__main__":
    # Example usage with a dummy solution instance
    class DummySolution:
        def __init__(self):
            self.E_emf = 60.0
            self.r = 2.0
            self.R1 = 8.0
            self.R2 = 15.0
            self.d = 0.04
            self.Q = 6e-8
            self.d_AC = 0.03
            self.results = {
                'I': 2.4,
                'U': 36.0,
                'C': 1.67e-9,
                'E': 900.0,
                'U_AC': 27.0,
                'varphi_A': 0.0,
                'varphi_C': -27.0
            }
    
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)