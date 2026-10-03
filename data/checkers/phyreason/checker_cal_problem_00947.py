import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results without direct attribute access.
        """
        # Known constants from problem statement
        self.R = 10.0  # Ω (each resistor)
        self.U2_given = 20.0  # V (voltmeter reading without diode)
        
        # Extract results using safe getattr with fallback
        results = getattr(solution_instance, 'results', {})
        
        # Primary RMS voltage (from waveform analysis)
        self.U1 = getattr(solution_instance, 'U_1', results.get('U_1', 0.0))
        
        # Turns ratio n1/n2
        self.n1_n2 = getattr(solution_instance, 'fracn_1n_2', results.get('fracn_1n_2', 0.0))
        
        # Total resistance in secondary
        self.R_total = getattr(solution_instance, 'R_total', results.get('R_total', 0.0))
        
        # Powers
        self.P2 = getattr(solution_instance, 'P_2', results.get('P_2', 0.0))
        self.P1 = getattr(solution_instance, 'P_1', results.get('P_1', 0.0))
        
        # Voltmeter reading with diode
        self.U_diode = getattr(solution_instance, 'U', results.get('U', 0.0))
        
        # Expected values from steps analysis
        self.U1_expected = 200.0  # V
        self.n1_n2_expected = 10.0  # 10/1
        self.R_total_expected = 5.0  # Ω (R/2)
        self.P2_expected = 80.0  # W
        self.P1_expected = 80.0  # W
        self.U_diode_expected = 10.0 * math.sqrt(2)  # ≈ 14.142 V
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Test logs
        self.logs = []

    def _log(self, msg: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {msg}")
        return success

    def test_invariants(self) -> bool:
        """Verify fundamental physics laws."""
        all_pass = True
        
        # 1. Transformer voltage ratio invariant
        if math.isclose(self.n1_n2, self.U1 / self.U2_given, 
                        rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Transformer ratio: n1/n2 = {self.n1_n2} ≈ U1/U2 = {self.U1}/{self.U2_given}")
        else:
            all_pass &= self._log(f"Transformer ratio mismatch: n1/n2={self.n1_n2}, U1/U2={self.U1/self.U2_given}", False)
        
        # 2. Power conservation in ideal transformer
        if math.isclose(self.P1, self.P2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Power conservation: P1 = {self.P1} ≈ P2 = {self.P2}")
        else:
            all_pass &= self._log(f"Power not conserved: P1={self.P1}, P2={self.P2}", False)
        
        # 3. Secondary power formula (P = V²/R)
        P2_calc = self.U2_given**2 / self.R_total
        if math.isclose(self.P2, P2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Secondary power: P2 = {self.P2} ≈ U2²/R_total = {P2_calc}")
        else:
            all_pass &= self._log(f"Secondary power mismatch: P2={self.P2}, U2²/R_total={P2_calc}", False)
        
        # 4. Diode voltage relationship (energy equivalence)
        # Without diode: P_full = U2²/R * T
        # With diode (half-wave): P_half = (U_diode²/R) * T
        # Energy equivalence: (U2²/R) * (T/2) = (U_diode²/R) * T => U_diode = U2/√2
        U_diode_calc = self.U2_given / math.sqrt(2)
        if math.isclose(self.U_diode, U_diode_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Diode voltage: U = {self.U_diode} ≈ U2/√2 = {U_diode_calc}")
        else:
            all_pass &= self._log(f"Diode voltage mismatch: U={self.U_diode}, U2/√2={U_diode_calc}", False)
        
        return all_pass

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        all_pass = True
        
        # Check Step 1: Primary voltage calculation
        if math.isclose(self.U1, self.U1_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Primary voltage U1 = {self.U1} ≈ expected {self.U1_expected} V")
        else:
            all_pass &= self._log(f"Primary voltage mismatch: {self.U1} vs {self.U1_expected} V", False)
        
        # Check Step 1: Turns ratio
        if math.isclose(self.n1_n2, self.n1_n2_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Turns ratio n1/n2 = {self.n1_n2} ≈ expected {self.n1_n2_expected}")
        else:
            all_pass &= self._log(f"Turns ratio mismatch: {self.n1_n2} vs {self.n1_n2_expected}", False)
        
        # Check Step 2: Total resistance (parallel resistors)
        if math.isclose(self.R_total, self.R_total_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Total resistance R_total = {self.R_total} ≈ R/2 = {self.R_total_expected} Ω")
        else:
            all_pass &= self._log(f"Total resistance mismatch: {self.R_total} vs {self.R_total_expected} Ω", False)
        
        # Check Step 2: Powers
        if math.isclose(self.P2, self.P2_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Secondary power P2 = {self.P2} ≈ expected {self.P2_expected} W")
        else:
            all_pass &= self._log(f"Secondary power mismatch: {self.P2} vs {self.P2_expected} W", False)
        
        if math.isclose(self.P1, self.P1_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Primary power P1 = {self.P1} ≈ expected {self.P1_expected} W")
        else:
            all_pass &= self._log(f"Primary power mismatch: {self.P1} vs {self.P1_expected} W", False)
        
        # Check Step 3: Diode voltage
        if math.isclose(self.U_diode, self.U_diode_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            all_pass &= self._log(f"Diode voltage U = {self.U_diode} ≈ expected {self.U_diode_expected} V")
        else:
            all_pass &= self._log(f"Diode voltage mismatch: {self.U_diode} vs {self.U_diode_expected} V", False)
        
        return all_pass

    def test_monotonicity(self) -> bool:
        """Perturb inputs and check physical trends."""
        all_pass = True
        
        # Test 1: Increase secondary resistance -> power should decrease
        R_perturbed = self.R * 1.1  # 10% increase
        R_total_perturbed = R_perturbed / 2.0  # Parallel combination
        P2_perturbed = self.U2_given**2 / R_total_perturbed
        
        if P2_perturbed < self.P2:
            all_pass &= self._log(f"Monotonicity (resistance): R↑ {self.R}→{R_perturbed}Ω → P2↓ {self.P2:.1f}→{P2_perturbed:.1f}W")
        else:
            all_pass &= self._log(f"Monotonicity violation: R↑ but P2 not decreased", False)
        
        # Test 2: Increase primary voltage -> secondary voltage and power should increase proportionally
        U1_perturbed = self.U1 * 1.1  # 10% increase
        U2_perturbed = U1_perturbed / self.n1_n2  # From transformer ratio
        P2_U_perturbed = U2_perturbed**2 / self.R_total_expected
        
        if U2_perturbed > self.U2_given and P2_U_perturbed > self.P2:
            all_pass &= self._log(f"Monotonicity (voltage): U1↑ {self.U1}→{U1_perturbed}V → U2↑ {self.U2_given}→{U2_perturbed:.1f}V → P2↑ {self.P2}→{P2_U_perturbed:.1f}W")
        else:
            all_pass &= self._log(f"Monotonicity violation: U1↑ but U2 or P2 not increased", False)
        
        # Test 3: Diode voltage should scale with U2
        U_diode_perturbed = U2_perturbed / math.sqrt(2)
        if U_diode_perturbed > self.U_diode:
            all_pass &= self._log(f"Monotonicity (diode): U2↑ → U_diode↑ {self.U_diode:.2f}→{U_diode_perturbed:.2f}V")
        else:
            all_pass &= self._log(f"Monotonicity violation: U2↑ but U_diode not increased", False)
        
        return all_pass

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all checks and return overall success with logs."""
        self.logs = ["Physics Audit Started"]
        
        success = True
        success &= self._log("=== Invariant Tests ===")
        success &= self.test_invariants()
        
        success &= self._log("\n=== Trace Consistency Tests ===")
        success &= self.test_trace_consistency()
        
        success &= self._log("\n=== Monotonicity Tests ===")
        success &= self.test_monotonicity()
        
        self.logs.append(f"\nOverall: {'PASS' if success else 'FAIL'}")
        return success, self.logs

# Example usage demonstration
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'U_1': 200.0,
                'fracn_1n_2': 10.0,
                'R_total': 5.0,
                'P_2': 80.0,
                'P_1': 80.0,
                'U': 10.0 * math.sqrt(2)
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)