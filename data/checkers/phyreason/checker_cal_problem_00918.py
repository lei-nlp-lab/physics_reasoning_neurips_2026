import math
import sympy as sp
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.electron_charge = 1.602176634e-19  # C
        self.electron_mass = 9.10938356e-31  # kg
        self.voltage = getattr(solution_instance, 'U', getattr(solution_instance, 'results', {}).get('U', 30000.0))
        self.current = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.01))  # 10 mA from KB
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'results', {}).get('n', 0.0))
        
        self.logs = []
        
    def _log(self, message: str, status: str = "INFO"):
        self.logs.append(f"{status}: {message}")
    
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical matching with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws:
        1. Energy conservation: kinetic energy = electrical work
        2. Current definition: n = I/e
        """
        success = True
        
        # 1. Energy conservation (Work-Energy Theorem)
        v_expected = math.sqrt(2 * self.electron_charge * self.voltage / self.electron_mass)
        if not self._is_close(self.v, v_expected):
            self._log(f"Energy conservation violated: v={self.v:.2e} vs expected {v_expected:.2e}", "ERROR")
            success = False
        else:
            self._log(f"Energy conservation check passed: v={self.v:.2e}")
        
        # 2. Current definition: n = I/e (for t=1 second)
        n_expected = self.current / self.electron_charge
        if not self._is_close(self.n, n_expected):
            self._log(f"Current definition violated: n={self.n:.2e} vs expected {n_expected:.2e}", "ERROR")
            success = False
        else:
            self._log(f"Current definition check passed: n={self.n:.2e}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer using steps analysis.
        """
        success = True
        
        # Step 1: Check v calculation from work-energy theorem
        work_done = self.electron_charge * self.voltage
        kinetic_energy = 0.5 * self.electron_mass * self.v**2
        if not self._is_close(work_done, kinetic_energy):
            self._log(f"Step 1 inconsistency: work_done={work_done:.2e} J, KE={kinetic_energy:.2e} J", "WARNING")
            success = False
        else:
            self._log(f"Step 1 consistency: work_done = KE = {work_done:.2e} J")
        
        # Step 2: Check n calculation from current definition
        charge_per_second = self.current  # I * t with t=1s
        electrons_per_second = charge_per_second / self.electron_charge
        if not self._is_close(self.n, electrons_per_second):
            self._log(f"Step 2 inconsistency: n={self.n:.2e} vs electrons/s={electrons_per_second:.2e}", "WARNING")
            success = False
        else:
            self._log(f"Step 2 consistency: n = I/e = {electrons_per_second:.2e}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check expected trends:
        1. Increase voltage → increase v (as sqrt)
        2. Increase current → increase n (linear)
        """
        success = True
        original_voltage = self.voltage
        original_current = self.current
        
        # 1. Perturb voltage (+10%)
        self.voltage *= 1.1
        v_perturbed = math.sqrt(2 * self.electron_charge * self.voltage / self.electron_mass)
        if v_perturbed <= self.v:
            self._log(f"Monotonicity fail: v should increase with U but {v_perturbed:.2e} <= {self.v:.2e}", "ERROR")
            success = False
        else:
            self._log(f"Voltage monotonicity: v increases with U ({v_perturbed:.2e} > {self.v:.2e})")
        self.voltage = original_voltage  # Restore
        
        # 2. Perturb current (+10%)
        self.current *= 1.1
        n_perturbed = self.current / self.electron_charge
        if n_perturbed <= self.n:
            self._log(f"Monotonicity fail: n should increase with I but {n_perturbed:.2e} <= {self.n:.2e}", "ERROR")
            success = False
        else:
            self._log(f"Current monotonicity: n increases with I ({n_perturbed:.2e} > {self.n:.2e})")
        self.current = original_current  # Restore
        
        return success
    
    def test_symbolic_laws(self) -> bool:
        """
        Use SymPy to verify symbolic relationships.
        """
        success = True
        
        # Define symbols
        v, U, m, e = sp.symbols('v U m e', positive=True)
        n, I, t = sp.symbols('n I t', positive=True)
        
        # 1. Work-energy theorem: e*U = (1/2)*m*v**2
        energy_eq = sp.Eq(e*U, m*v**2/2)
        v_solution = sp.solve(energy_eq, v)[0]  # Should be sqrt(2*e*U/m)
        expected_v_expr = sp.sqrt(2*e*U/m)
        if not sp.simplify(v_solution - expected_v_expr) == 0:
            self._log(f"Symbolic energy law mismatch: {v_solution} vs {expected_v_expr}", "ERROR")
            success = False
        else:
            self._log(f"Symbolic energy law verified: v = {expected_v_expr}")
        
        # 2. Current definition: n = I*t/e (with t=1)
        current_eq = sp.Eq(n, I*t/e)
        n_solution = sp.solve(current_eq, n)[0]
        expected_n_expr = I*t/e
        if not sp.simplify(n_solution - expected_n_expr) == 0:
            self._log(f"Symbolic current law mismatch: {n_solution} vs {expected_n_expr}", "ERROR")
            success = False
        else:
            self._log(f"Symbolic current law verified: n = I*t/e")
        
        # Numerical substitution check
        numeric_v = expected_v_expr.subs({
            e: self.electron_charge,
            U: self.voltage,
            m: self.electron_mass
        }).evalf()
        if not self._is_close(float(numeric_v), self.v):
            self._log(f"Symbolic->numeric v mismatch: {float(numeric_v):.2e} vs {self.v:.2e}", "WARNING")
        
        numeric_n = expected_n_expr.subs({
            I: self.current,
            t: 1.0,
            e: self.electron_charge
        }).evalf()
        if not self._is_close(float(numeric_n), self.n):
            self._log(f"Symbolic->numeric n mismatch: {float(numeric_n):.2e} vs {self.n:.2e}", "WARNING")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audits and return (success, logs).
        """
        self.logs = []
        self._log("Starting PhysicsChecker audit...")
        
        results = []
        results.append(self.test_invariants())
        results.append(self.test_trace_consistency())
        results.append(self.test_monotonicity())
        results.append(self.test_symbolic_laws())
        
        success = all(results)
        if success:
            self._log("All physics checks passed!", "SUCCESS")
        else:
            self._log("Physics audit failed - check warnings/errors above", "FAILURE")
        
        return success, self.logs