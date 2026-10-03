import math
import sympy as sp
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access.
        solution_instance should have a .results dictionary with keys:
        'U1', '$fracDeltaPhiDeltat$', 'U', '$fracI_2I_1$'
        """
        self.solution = solution_instance
        
        # Safe extraction from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract numerical values with fallbacks
        self.U1 = self._safe_get('U1', 0.0)
        self.dPhi_dt = self._safe_get('$fracDeltaPhiDeltat$', 0.0)  # ΔΦ/Δt
        self.U = self._safe_get('U', 0.0)  # Bulb normal voltage
        self.I2_I1_ratio = self._safe_get('$fracI_2I_1$', 0.0)  # I₂/I₁
        
        # Known constants from problem
        self.N_primary = 220  # Turns in primary coil
        self.source_voltage = 220  # Voltage across MN (from step_2 equation)
        
        # For symbolic computations
        self.symbols = self._init_symbols()
        
        # Store logs for reporting
        self.logs = []
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get a value from results or attributes."""
        # Try results first
        val = self.results.get(key, default)
        if val == default:
            # Try direct attribute as fallback
            val = getattr(self.solution, key, default)
        # Ensure it's a float
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    def _init_symbols(self) -> Dict:
        """Initialize symbolic variables."""
        U, U1, I1, I2, N1, N2, dPhi_dt, t = sp.symbols('U U1 I1 I2 N1 N2 dPhi_dt t')
        return {
            'U': U, 'U1': U1, 'I1': I1, 'I2': I2,
            'N1': N1, 'N2': N2, 'dPhi_dt': dPhi_dt, 't': t
        }
    
    def _log(self, message: str, success: bool = True):
        """Add a message to logs."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify physics invariants:
        1. Ideal transformer power conservation: U1*I1 = U2*I2
        2. Series circuit voltage sums
        3. Faraday's Law consistency
        """
        all_pass = True
        
        # 1. Power conservation for ideal transformer (assuming no losses)
        # For bulbs L3 and L4 in secondary: they are identical and in series
        # So U_secondary = 2*U (each bulb has voltage U when normal)
        # Since transformer is ideal: U1/U2 = N1/N2 and I1/I2 = N2/N1
        # Power: U1*I1 should equal U2*I2
        U2 = 2 * self.U  # Voltage across secondary (two bulbs in series)
        # From step_1: I2/I1 = 1 (given as 1/1 in KB)
        if not math.isclose(self.I2_I1_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Current ratio I2/I1 = {self.I2_I1_ratio} ≠ 1", False)
            all_pass = False
        else:
            # Check power: U1*I1 vs U2*I2 (I2 = I1)
            power_primary = self.U1  # Proportional to U1*I1 (with I1=1)
            power_secondary = U2  # Proportional to U2*I2 (with I2=1)
            if not math.isclose(power_primary, power_secondary, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Power conservation: U1={self.U1} ≠ U2={U2}", False)
                all_pass = False
            else:
                self._log(f"Power conservation: U1={self.U1:.1f} ≈ U2={U2:.1f}")
        
        # 2. Series circuit voltage sum (from step_2)
        # Source voltage = voltage across (L1 + L2) + voltage across primary
        # 220 = 2U + 2U  => 220 = 4U
        expected_U = self.source_voltage / 4
        if not math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Bulb voltage U: {self.U} ≠ {expected_U} (from 220=4U)", False)
            all_pass = False
        else:
            self._log(f"Bulb voltage U={self.U:.1f} matches 220=4U")
        
        # 3. Faraday's Law: U1 = N1 * (dΦ/dt) for instantaneous AC
        # The step_4 gives maximum rate: (ΔΦ/Δt)_max = U1_max / N1
        # U1_max = √2 * U1 (RMS to peak for sinusoidal AC)
        U1_max = self.U1 * math.sqrt(2)
        expected_dPhi_dt = U1_max / self.N_primary
        
        if not math.isclose(self.dPhi_dt, expected_dPhi_dt, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Faraday's Law: ΔΦ/Δt={self.dPhi_dt:.3f} ≠ {expected_dPhi_dt:.3f} (U1_max/N1)", False)
            all_pass = False
        else:
            self._log(f"Faraday's Law: ΔΦ/Δt={self.dPhi_dt:.3f} matches U1_max/N1={expected_dPhi_dt:.3f}")
        
        return all_pass
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Steps from KB:
        step_1: I2/I1 = 1
        step_2: 220 - 2U = 2U => U = 55
        step_3: U1 = 2U = 110
        step_4: ΔΦ/Δt = U1_max/N1 = (110√2)/220 = 0.5√2
        """
        all_pass = True
        
        # Check step_1 consistency
        if not math.isclose(self.I2_I1_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_1 failed: I2/I1={self.I2_I1_ratio} ≠ 1", False)
            all_pass = False
        
        # Check step_2: U should be 55
        if not math.isclose(self.U, 55.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_2 failed: U={self.U} ≠ 55", False)
            all_pass = False
        
        # Check step_3: U1 = 2U
        if not math.isclose(self.U1, 2*self.U, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_3 failed: U1={self.U1} ≠ 2U={2*self.U}", False)
            all_pass = False
        
        # Check step_4: ΔΦ/Δt = 0.5√2 ≈ 0.7071
        expected_dPhi_dt = 0.5 * math.sqrt(2)
        if not math.isclose(self.dPhi_dt, expected_dPhi_dt, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_4 failed: ΔΦ/Δt={self.dPhi_dt:.3f} ≠ 0.5√2={expected_dPhi_dt:.3f}", False)
            all_pass = False
        
        if all_pass:
            self._log("All trace steps consistent")
        
        return all_pass
    
    def test_monotonicity(self) -> bool:
        """
        Perturb source voltage and check trends.
        If source voltage increases, U, U1, and ΔΦ/Δt should increase proportionally.
        """
        original_source = self.source_voltage
        
        # Perturb by +10%
        perturbed_source = original_source * 1.1
        
        # Recalculate using formulas from steps (keeping N1 fixed)
        perturbed_U = perturbed_source / 4
        perturbed_U1 = 2 * perturbed_U
        perturbed_U1_max = perturbed_U1 * math.sqrt(2)
        perturbed_dPhi_dt = perturbed_U1_max / self.N_primary
        
        # Calculate expected ratios (should all be 1.1)
        ratio_U = perturbed_U / self.U
        ratio_U1 = perturbed_U1 / self.U1
        ratio_dPhi_dt = perturbed_dPhi_dt / self.dPhi_dt
        
        expected_ratio = 1.1
        tolerance = 0.02  # 2% tolerance for numerical effects
        
        all_pass = True
        
        if not math.isclose(ratio_U, expected_ratio, rel_tol=tolerance, abs_tol=tolerance):
            self._log(f"Monotonicity U: ratio={ratio_U:.3f} ≠ {expected_ratio}", False)
            all_pass = False
        else:
            self._log(f"Monotonicity U: ratio={ratio_U:.3f} ≈ {expected_ratio}")
        
        if not math.isclose(ratio_U1, expected_ratio, rel_tol=tolerance, abs_tol=tolerance):
            self._log(f"Monotonicity U1: ratio={ratio_U1:.3f} ≠ {expected_ratio}", False)
            all_pass = False
        else:
            self._log(f"Monotonicity U1: ratio={ratio_U1:.3f} ≈ {expected_ratio}")
        
        if not math.isclose(ratio_dPhi_dt, expected_ratio, rel_tol=tolerance, abs_tol=tolerance):
            self._log(f"Monotonicity ΔΦ/Δt: ratio={ratio_dPhi_dt:.3f} ≠ {expected_ratio}", False)
            all_pass = False
        else:
            self._log(f"Monotonicity ΔΦ/Δt: ratio={ratio_dPhi_dt:.3f} ≈ {expected_ratio}")
        
        # Restore original values (they weren't actually changed, just computed)
        return all_pass
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        self.logs = []  # Clear previous logs
        
        # Start test sequence
        self._log("=== Physics Checker Started ===")
        
        results = []
        results.append(("Invariants", self.test_invariants()))
        results.append(("Trace Consistency", self.test_trace_consistency()))
        results.append(("Monotonicity", self.test_monotonicity()))
        
        # Summary
        all_passed = all(r[1] for r in results)
        self._log(f"=== Summary: {'ALL PASS' if all_passed else 'SOME FAILED'} ===")
        for name, passed in results:
            self._log(f"  {name}: {'PASS' if passed else 'FAIL'}")
        
        return all_passed, self.logs