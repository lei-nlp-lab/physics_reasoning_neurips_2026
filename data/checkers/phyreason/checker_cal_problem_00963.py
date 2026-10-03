import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults from problem statement
        self.B = getattr(solution_instance, 'B', 0.5)
        self.L = getattr(solution_instance, 'L', 0.2)
        self.v = getattr(solution_instance, 'v', 8.0)
        self.R = getattr(solution_instance, 'R', 0.1)
        
        # Safe extraction of results with fallback to dictionary
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.Q = getattr(solution_instance, 'Q', getattr(solution_instance, 'results', {}).get('Q', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'results', {}).get('E', 0.0))
        
        # Store original values for restoration after perturbations
        self._original_F = self.F
        self._original_Q = self.Q
        self._original_I = self.I
        self._original_E = self.E
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # 1. Check Motional EMF: E = B * L * v
        expected_E = self.B * self.L * self.v
        if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"EMF invariant fails: expected {expected_E:.4f}, got {self.E:.4f}")
            success = False
        
        # 2. Check Ohm's Law: I = E / R
        expected_I = self.E / self.R
        if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Current invariant fails: expected {expected_I:.4f}, got {self.I:.4f}")
            success = False
        
        # 3. Check Magnetic Force: F = B * I * L
        expected_F = self.B * self.I * self.L
        if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Force invariant fails: expected {expected_F:.4f}, got {self.F:.4f}")
            success = False
        
        # 4. Check Time: t = 2L / v
        expected_t = 2 * self.L / self.v
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Time invariant fails: expected {expected_t:.4f}, got {self.t:.4f}")
            success = False
        
        # 5. Check Joule's Law: Q = I² * R * t
        expected_Q = self.I**2 * self.R * self.t
        if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Heat invariant fails: expected {expected_Q:.4f}, got {self.Q:.4f}")
            success = False
        
        # 6. Check Energy Conservation: Work done by force = Heat generated
        # Work = F * distance (force acts over 2L as coil enters and exits)
        work_done = self.F * (2 * self.L)
        if not math.isclose(work_done, self.Q, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy conservation fails: work {work_done:.4f} ≠ heat {self.Q:.4f}")
            success = False
        
        # 7. Check Force Balance (Newton's 1st Law): Net force = 0 for constant velocity
        # The pulling force F must equal the magnetic drag force B*I*L
        net_force = self.F - self.B * self.I * self.L
        if not math.isclose(net_force, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Force balance fails: net force = {net_force:.4f} (should be 0)")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # Verify intermediate calculations from the steps analysis
        # Step 1: E = 0.8 V (already checked in invariants)
        # Step 2 & 3: I = E/R = 0.8/0.1 = 8 A (already checked)
        # Step 4: F = B*I*L = 0.5 * 8 * 0.2 = 0.8 N (already checked)
        # Step 5: t = 2L/v = 0.4/8 = 0.05 s (already checked)
        # Step 6: Q = I²*R*t = 64 * 0.1 * 0.05 = 0.32 J (already checked)
        
        # Cross-check final answers against target values
        target_values = [0.8, 0.8, 0.32]  # E, F, Q in V, N, J
        computed_values = [self.E, self.F, self.Q]
        
        for i, (target, computed) in enumerate(zip(target_values, computed_values)):
            if not math.isclose(computed, target, rel_tol=1e-2, abs_tol=1e-2):
                quantity = ["EMF", "Force", "Heat"][i]
                logs.append(f"Trace consistency: {quantity} mismatch: target {target}, computed {computed}")
                success = False
        
        # Verify all quantities are positive (physics sanity)
        for name, val in [("EMF", self.E), ("Current", self.I), ("Force", self.F), 
                          ("Time", self.t), ("Heat", self.Q)]:
            if val < -1e-6:  # Allow tiny negative due to rounding
                logs.append(f"Negative {name}: {val}")
                success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True
        
        # Perturb velocity by +10% and check trends
        perturbed_v = self.v * 1.1
        
        # Recalculate using formulas (symbolic then numeric fallback)
        try:
            # Symbolic relationships
            E_sym = self.B * self.L * perturbed_v
            I_sym = E_sym / self.R
            F_sym = self.B * I_sym * self.L
            t_sym = 2 * self.L / perturbed_v
            Q_sym = I_sym**2 * self.R * t_sym
            
            # Convert to float
            E_new = float(E_sym)
            I_new = float(I_sym)
            F_new = float(F_sym)
            Q_new = float(Q_sym)
        except:
            # Numerical fallback
            E_new = self.B * self.L * perturbed_v
            I_new = E_new / self.R
            F_new = self.B * I_new * self.L
            t_new = 2 * self.L / perturbed_v
            Q_new = I_new**2 * self.R * t_new
        
        # Check monotonicity: increasing v should increase E, I, F, Q
        if not (E_new > self.E):
            logs.append(f"EMF monotonicity fails: v↑ should increase E (was {self.E:.4f}, now {E_new:.4f})")
            success = False
        
        if not (I_new > self.I):
            logs.append(f"Current monotonicity fails: v↑ should increase I (was {self.I:.4f}, now {I_new:.4f})")
            success = False
        
        if not (F_new > self.F):
            logs.append(f"Force monotonicity fails: v↑ should increase F (was {self.F:.4f}, now {F_new:.4f})")
            success = False
        
        if not (Q_new > self.Q):
            logs.append(f"Heat monotonicity fails: v↑ should increase Q (was {self.Q:.4f}, now {Q_new:.4f})")
            success = False
        
        # Check time monotonicity: increasing v should decrease t
        t_new = 2 * self.L / perturbed_v
        if not (t_new < self.t):
            logs.append(f"Time monotonicity fails: v↑ should decrease t (was {self.t:.4f}, now {t_new:.4f})")
            success = False
        
        # Restore original values (perturbation was only in local variables)
        # No need to restore as we didn't modify instance attributes
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        overall_success = True
        
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, test_logs = test_func()
                logs.append(f"=== {test_name} Test ===")
                if test_logs:
                    logs.extend(test_logs)
                else:
                    logs.append("All checks passed.")
                logs.append(f"Result: {'PASS' if success else 'FAIL'}")
                overall_success = overall_success and success
            except Exception as e:
                logs.append(f"=== {test_name} Test ===")
                logs.append(f"ERROR: {str(e)}")
                logs.append("Result: FAIL")
                overall_success = False
        
        return overall_success, logs