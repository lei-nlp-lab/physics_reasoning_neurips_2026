import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # Input parameters (with defaults 0 or 1 to avoid division by zero)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.k = getattr(solution_instance, 'k', 1.0)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.v = getattr(solution_instance, 'v', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        
        # Results - use safe access with fallback to 0.0
        self.results = getattr(solution_instance, 'results', {})
        
        # Primary answers
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        self.W = getattr(solution_instance, 'W', self.results.get('W', 0.0))
        
        # Intermediate results (if available)
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.R_parallel = getattr(solution_instance, 'R_parallel', self.results.get('R_parallel', 0.0))
        self.I = getattr(solution_instance, 'I', self.results.get('I', 0.0))
        self.I_PQ = getattr(solution_instance, 'I_PQ', self.results.get('I_PQ', 0.0))
        self.F_ampere = getattr(solution_instance, 'F_ampere', self.results.get('F_ampere', 0.0))
        self.overlineE = getattr(solution_instance, 'overlineE', self.results.get('overlineE', 0.0))
        self.overlineI = getattr(solution_instance, 'overlineI', self.results.get('overlineI', 0.0))
        self.Deltaphi = getattr(solution_instance, 'Deltaphi', self.results.get('Deltaphi', 0.0))
        
        # Logs for reporting
        self.logs = []
        
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
        
    def test_invariants(self) -> bool:
        """
        Check fundamental physics invariants:
        1. Force balance: F = F_ampere (from equilibrium)
        2. Ohm's Law and circuit consistency
        3. Work-energy consistency
        """
        passed = True
        
        # 1. Check that external force equals Ampere force (from equilibrium)
        # Recompute F_ampere from derived chain if available
        if self.I_PQ != 0.0 and self.B != 0.0 and self.l != 0.0:
            F_ampere_calc = self.B * self.I_PQ * self.l
            if not math.isclose(self.F, F_ampere_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Force balance: F={self.F:.3g} != F_ampere={F_ampere_calc:.3g}",
                    False
                )
            else:
                passed &= self._log("Force balance F = F_ampere holds", True)
        
        # 2. Check Ohm's Law consistency in the circuit
        # From steps: E = k, R_parallel = R/2, I = E/(R_parallel + R)
        if self.E != 0.0 and self.R != 0.0:
            R_parallel_calc = self.R / 2
            I_calc = self.E / (R_parallel_calc + self.R)
            if not math.isclose(self.I, I_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Ohm's Law: I={self.I:.3g} != E/(R_parallel+R)={I_calc:.3g}",
                    False
                )
            else:
                passed &= self._log("Ohm's Law for circuit holds", True)
        
        # 3. Check work-energy consistency: W = 0.5*m*v^2 - (2/3)*k*q
        W_calc = 0.5 * self.m * self.v**2 - (2/3) * self.k * self.q
        if not math.isclose(self.W, W_calc, rel_tol=1e-2, abs_tol=1e-2):
            passed &= self._log(
                f"Work-energy: W={self.W:.3g} != 0.5*m*v^2 - (2/3)*k*q={W_calc:.3g}",
                False
            )
        else:
            passed &= self._log("Work-energy consistency holds", True)
            
        return passed
    
    def test_trace_consistency(self) -> bool:
        """
        Verify intermediate results are consistent with final answers.
        """
        passed = True
        
        # Check the full derivation chain for F
        # Step 1: E = k
        if not math.isclose(self.E, self.k, rel_tol=1e-2, abs_tol=1e-2):
            passed &= self._log(f"Step1: E={self.E:.3g} != k={self.k:.3g}", False)
        
        # Step 2: R_parallel = R/2
        R_parallel_calc = self.R / 2
        if not math.isclose(self.R_parallel, R_parallel_calc, rel_tol=1e-2, abs_tol=1e-2):
            passed &= self._log(
                f"Step2: R_parallel={self.R_parallel:.3g} != R/2={R_parallel_calc:.3g}",
                False
            )
        
        # Step 3: I = E/(R_parallel + R)
        if self.R_parallel != 0.0 or self.R != 0.0:
            I_calc = self.E / (self.R_parallel + self.R)
            if not math.isclose(self.I, I_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Step3: I={self.I:.3g} != E/(R_parallel+R)={I_calc:.3g}",
                    False
                )
        
        # Step 4: I_PQ = I/2
        if self.I != 0.0:
            I_PQ_calc = self.I / 2
            if not math.isclose(self.I_PQ, I_PQ_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Step4: I_PQ={self.I_PQ:.3g} != I/2={I_PQ_calc:.3g}",
                    False
                )
        
        # Step 5: F_ampere = B * I_PQ * l
        if self.I_PQ != 0.0 and self.B != 0.0 and self.l != 0.0:
            F_ampere_calc = self.B * self.I_PQ * self.l
            if not math.isclose(self.F_ampere, F_ampere_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Step5: F_ampere={self.F_ampere:.3g} != B*I_PQ*l={F_ampere_calc:.3g}",
                    False
                )
        
        # Step 6-7: F = F_ampere = (B*k*l)/(3*R)
        F_calc = (self.B * self.k * self.l) / (3 * self.R)
        if not math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-2):
            passed &= self._log(
                f"Step7: F={self.F:.3g} != (B*k*l)/(3*R)={F_calc:.3g}",
                False
            )
        
        # Check the derivation chain for W (if intermediate results available)
        if self.overlineE != 0.0 and self.Deltaphi != 0.0:
            # Step 8-9: overlineE = Δφ/Δt, Δφ = B*l*x
            # We don't have x, so skip quantitative check
            pass
            
        # Step 10: overlineI = overlineE/(2R)
        if self.overlineE != 0.0 and self.R != 0.0:
            overlineI_calc = self.overlineE / (2 * self.R)
            if not math.isclose(self.overlineI, overlineI_calc, rel_tol=1e-2, abs_tol=1e-2):
                passed &= self._log(
                    f"Step10: overlineI={self.overlineI:.3g} != overlineE/(2R)={overlineI_calc:.3g}",
                    False
                )
        
        if passed:
            self._log("All trace consistency checks passed", True)
        return passed
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check output trends are physically reasonable.
        Restores original values after checks.
        """
        passed = True
        
        # Store original values
        orig_B = self.B
        orig_k = self.k
        orig_l = self.l
        orig_R = self.R
        orig_m = self.m
        orig_v = self.v
        orig_q = self.q
        
        # Test 1: F should increase with B
        self.B *= 1.1  # +10%
        F_new = (self.B * self.k * self.l) / (3 * self.R)
        if not (F_new > self.F):
            passed &= self._log(
                f"Monotonicity: F does not increase when B increases (F_old={self.F:.3g}, F_new={F_new:.3g})",
                False
            )
        self.B = orig_B  # restore
        
        # Test 2: F should increase with k
        self.k *= 1.1
        F_new = (self.B * self.k * self.l) / (3 * self.R)
        if not (F_new > self.F):
            passed &= self._log(
                f"Monotonicity: F does not increase when k increases (F_old={self.F:.3g}, F_new={F_new:.3g})",
                False
            )
        self.k = orig_k
        
        # Test 3: F should increase with l
        self.l *= 1.1
        F_new = (self.B * self.k * self.l) / (3 * self.R)
        if not (F_new > self.F):
            passed &= self._log(
                f"Monotonicity: F does not increase when l increases (F_old={self.F:.3g}, F_new={F_new:.3g})",
                False
            )
        self.l = orig_l
        
        # Test 4: F should decrease with R
        self.R *= 1.1
        F_new = (self.B * self.k * self.l) / (3 * self.R)
        if not (F_new < self.F):
            passed &= self._log(
                f"Monotonicity: F does not decrease when R increases (F_old={self.F:.3g}, F_new={F_new:.3g})",
                False
            )
        self.R = orig_R
        
        # Test 5: W should increase with m
        self.m *= 1.1
        W_new = 0.5 * self.m * self.v**2 - (2/3) * self.k * self.q
        if not (W_new > self.W):
            passed &= self._log(
                f"Monotonicity: W does not increase when m increases (W_old={self.W:.3g}, W_new={W_new:.3g})",
                False
            )
        self.m = orig_m
        
        # Test 6: W should increase with v
        self.v *= 1.1
        W_new = 0.5 * self.m * self.v**2 - (2/3) * self.k * self.q
        if not (W_new > self.W):
            passed &= self._log(
                f"Monotonicity: W does not increase when v increases (W_old={self.W:.3g}, W_new={W_new:.3g})",
                False
            )
        self.v = orig_v
        
        # Test 7: W should decrease with k*q product
        self.k *= 1.1
        W_new = 0.5 * self.m * self.v**2 - (2/3) * self.k * self.q
        if not (W_new < self.W):
            passed &= self._log(
                f"Monotonicity: W does not decrease when k increases (W_old={self.W:.3g}, W_new={W_new:.3g})",
                False
            )
        self.k = orig_k
        
        if passed:
            self._log("All monotonicity checks passed", True)
        return passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall = True
        for name, test_func in tests:
            self.logs.append(f"--- {name} ---")
            try:
                overall &= test_func()
            except Exception as e:
                self.logs.append(f"[ERROR] {name} test crashed: {e}")
                overall = False
        
        # Summary
        self.logs.append("=" * 40)
        if overall:
            self.logs.append("All physics checks PASSED")
        else:
            self.logs.append("Some physics checks FAILED")
        
        return overall, self.logs