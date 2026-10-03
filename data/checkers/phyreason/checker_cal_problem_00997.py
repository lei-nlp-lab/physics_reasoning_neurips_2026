import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using ONLY the results dictionary."""
        # Get the results dictionary - this is the ONLY source of values
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract numerical results from results dictionary
        # Handle both Unicode and ASCII keys for rho
        self.V = results_dict.get('V', 0.0)
        self.d = results_dict.get('d', 0.0)
        self.S = results_dict.get('S', 0.0)
        
        # Try to get rho with Unicode key, fall back to ASCII if needed
        self.rho = results_dict.get('ρ', 0.0)  # Unicode rho
        if self.rho == 0.0:
            self.rho = results_dict.get('rho', 0.0)  # ASCII fallback
            
        self.V_0 = results_dict.get('V_0', 0.0)
        self.N_grids = results_dict.get('N_grids', 0.0)
        
        # Constants from problem statement
        self.N_drops = 50  # 50 drops make 1 mL
        self.concentration_decimal = 0.6 / 1000  # 0.6 mL oleic acid per 1000 mL solution
        
        # Side length of each grid square (calculated from given area)
        if self.N_grids > 0 and self.S > 0:
            self.grid_side = math.sqrt(self.S / self.N_grids)
        else:
            self.grid_side = 0.0
        
        # Physical constants (for reference, though not all used directly)
        self.k_B = 1.380649e-23  # J/K
        self.pi = math.pi
        
        self.logs = []

    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Check fundamental physical relationships from the problem."""
        success = True
        
        # 1. Concentration check: ρ should be 0.06% (0.0006 as decimal)
        expected_rho = 0.0006
        if not math.isclose(self.rho, expected_rho, rel_tol=1e-2, abs_tol=1e-6):
            success = self._log(f"Concentration mismatch: ρ={self.rho} vs expected {expected_rho}", False)
        else:
            self._log(f"Concentration check passed: ρ={self.rho}", True)
        
        # 2. Volume relationship: V = 1 mL = 1e-6 m³ (for 50 drops)
        expected_V = 1e-6
        if not math.isclose(self.V, expected_V, rel_tol=1e-2, abs_tol=1e-9):
            success = self._log(f"Volume mismatch: V={self.V} vs expected {expected_V}", False)
        else:
            self._log(f"Volume check passed: V={self.V}", True)
        
        # 3. Volume of pure oleic acid per drop: V_0 = (concentration * V) / N_drops
        # Handle potential division by zero
        if self.N_drops == 0:
            success = self._log("N_drops is zero, cannot calculate V_0", False)
        else:
            calculated_V0 = (self.concentration_decimal * self.V) / self.N_drops
            # Use more tolerant comparison for very small numbers
            if abs(self.V_0) < 1e-20 and abs(calculated_V0) < 1e-20:
                # Both are essentially zero
                self._log(f"V_0 is near zero: {self.V_0}", True)
            elif not math.isclose(self.V_0, calculated_V0, rel_tol=1e-1, abs_tol=1e-14):
                success = self._log(
                    f"V_0 mismatch: {self.V_0} vs calculated {calculated_V0}", False)
            else:
                self._log(f"V_0 calculation passed: {self.V_0}", True)
        
        # 4. Molecular diameter: d = V_0 / S
        if self.S == 0:
            success = self._log("S is zero, cannot calculate d", False)
        else:
            calculated_d = self.V_0 / self.S
            # Use more tolerant comparison for very small numbers
            if abs(self.d) < 1e-20 and abs(calculated_d) < 1e-20:
                # Both are essentially zero
                self._log(f"d is near zero: {self.d}", True)
            elif not math.isclose(self.d, calculated_d, rel_tol=1e-1, abs_tol=1e-12):
                success = self._log(
                    f"Diameter mismatch: d={self.d} vs calculated {calculated_d}", False)
            else:
                self._log(f"Diameter calculation passed: d={self.d}", True)
        
        # 5. Grid area consistency: S = N_grids * (grid_side)^2
        if self.N_grids > 0 and self.grid_side > 0:
            calculated_S = self.N_grids * (self.grid_side ** 2)
            if not math.isclose(self.S, calculated_S, rel_tol=1e-2, abs_tol=1e-4):
                success = self._log(
                    f"Area mismatch: S={self.S} vs calculated {calculated_S}", False)
            else:
                self._log(f"Grid area consistency passed", True)
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final results."""
        success = True
        
        # Check each step from the STEPS ANALYSIS
        # Step 1: N_grids should be 85
        expected_grids = 85
        if not math.isclose(self.N_grids, expected_grids, rel_tol=1e-2, abs_tol=0.5):
            success = self._log(
                f"Grid count mismatch: N_grids={self.N_grids} vs expected {expected_grids}", False)
        else:
            self._log(f"Grid count check passed: N_grids={self.N_grids}", True)
        
        # Step 2: S should be 7.65e-2 m²
        expected_S = 7.65e-2
        if not math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-4):
            success = self._log(
                f"Area mismatch: S={self.S} vs expected {expected_S}", False)
        else:
            self._log(f"Area check passed: S={self.S}", True)
        
        # Step 3: Concentration and volume already checked in invariants
        
        # Step 4: V_0 should be 1.2e-11 m³
        expected_V0 = 1.2e-11
        # Use more tolerant comparison for very small numbers
        if abs(self.V_0) < 1e-20 and abs(expected_V0) < 1e-20:
            self._log(f"V_0 is near zero (expected near zero)", True)
        elif not math.isclose(self.V_0, expected_V0, rel_tol=1e-1, abs_tol=1e-13):
            success = self._log(
                f"Volume per drop mismatch: V_0={self.V_0} vs expected {expected_V0}", False)
        else:
            self._log(f"Volume per drop check passed: V_0={self.V_0}", True)
        
        # Step 5: d should be 1.57e-10 m
        expected_d = 1.57e-10
        # Use more tolerant comparison for very small numbers
        if abs(self.d) < 1e-20 and abs(expected_d) < 1e-20:
            self._log(f"d is near zero (expected near zero)", True)
        elif not math.isclose(self.d, expected_d, rel_tol=1e-1, abs_tol=1e-12):
            success = self._log(
                f"Diameter mismatch: d={self.d} vs expected {expected_d}", False)
        else:
            self._log(f"Diameter check passed: d={self.d}", True)
        
        return success

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        success = True
        
        # Skip monotonicity tests if values are too small
        if abs(self.d) < 1e-20 or abs(self.V_0) < 1e-20 or abs(self.S) < 1e-20:
            self._log("Skipping monotonicity tests due to near-zero values", True)
            return True
        
        # Perturb concentration: if concentration increases, d should increase
        # since V_0 ∝ concentration and d ∝ V_0
        perturbed_rho = self.concentration_decimal * 1.1  # +10%
        perturbed_V0 = (perturbed_rho * self.V) / self.N_drops
        perturbed_d = perturbed_V0 / self.S
        
        if perturbed_d > self.d:
            self._log(f"Monotonicity 1 (concentration): d increases with concentration ✓", True)
        else:
            success = self._log(
                f"Monotonicity 1 (concentration): d should increase with concentration", False)
        
        # Perturb volume: if V increases, d should increase
        perturbed_V = self.V * 1.1  # +10%
        perturbed_V0_2 = (self.concentration_decimal * perturbed_V) / self.N_drops
        perturbed_d_2 = perturbed_V0_2 / self.S
        
        if perturbed_d_2 > self.d:
            self._log(f"Monotonicity 2 (volume): d increases with volume ✓", True)
        else:
            success = self._log(
                f"Monotonicity 2 (volume): d should increase with volume", False)
        
        # Perturb area: if S increases, d should decrease (inverse relationship)
        perturbed_S = self.S * 1.1  # +10%
        perturbed_d_3 = self.V_0 / perturbed_S
        
        if perturbed_d_3 < self.d:
            self._log(f"Monotonicity 3 (area): d decreases with area ✓", True)
        else:
            success = self._log(
                f"Monotonicity 3 (area): d should decrease with area", False)
        
        return success

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"--- {test_name} ---")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        if overall_success:
            self.logs.append("✓ All physics checks passed")
        else:
            self.logs.append("✗ Some physics checks failed")
        
        return overall_success, self.logs