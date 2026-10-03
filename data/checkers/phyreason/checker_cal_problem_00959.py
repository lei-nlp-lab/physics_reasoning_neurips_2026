import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses getattr and .get() to avoid AttributeError in sandbox.
        """
        # Extract results from solution_instance.results (safe access)
        results = getattr(solution_instance, 'results', {})
        
        # Extract all needed values from results dictionary with defaults
        self.N = results.get('N', 1000.0)
        self.S = results.get('S', 20.0 * 1e-4)  # m^2
        self.r = results.get('r', 1.0)
        self.k = results.get('k', 100.0)
        self.R = results.get('R', 1.5)
        self.L = results.get('L', 0.10)  # m
        self.delta_x = results.get('delta_x', 0.005)  # m
        
        # Target variables from contract - try multiple possible key variations
        self.B2 = results.get('B₂', results.get('B2', 0.0))
        # Try multiple variations for the rate of change key
        self.delta_B1_dt = results.get('ΔB₁/Δt', results.get('delta_B₁/delta_t', 0.0))
        self.I = results.get('I', 0.0)
        self.F_spring = results.get('F_spring', 0.0)
        self.F_Ampere = results.get('F_Ampere', 0.0)
        self.E = results.get('E', 0.0)

        # Store intermediate values from steps analysis for verification
        self.step2_delta_B1_dt = 5.0  # T/s (from step 2)
        self.step3_E = 10.0  # V (from step 3)
        self.step4_I = 2.5  # A (from step 4)
        self.step7_B2 = 4.0  # T (from step 7)

        # Constants
        self.g = 9.8  # m/s^2 (though not used directly, kept for completeness)

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Faraday's law: E = N * (ΔB₁/Δt) * S
        2. Ohm's law: I = E / (r + 2R)
        3. Ampere's force: F_Ampere = B₂ * I * L
        4. Hooke's law (two springs): F_spring_total = 2 * k * Δx
        5. Force balance: F_Ampere = F_spring_total (equilibrium)
        Uses fuzzy matching with 1-2% tolerance.
        """
        logs = []
        success = True

        # 1. Faraday's law
        E_calc = self.N * self.delta_B1_dt * self.S
        if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL Faraday's law: E={self.E}, calculated {E_calc}")
            success = False
        else:
            logs.append(f"PASS Faraday's law: E={self.E} ≈ N*(ΔB₁/Δt)*S = {E_calc}")

        # 2. Ohm's law
        total_resistance = self.r + 2 * self.R
        # Avoid division by zero
        if total_resistance == 0:
            logs.append("FAIL Ohm's law: Total resistance is zero")
            success = False
        else:
            I_calc = self.E / total_resistance
            if not math.isclose(self.I, I_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL Ohm's law: I={self.I}, calculated {I_calc}")
                success = False
            else:
                logs.append(f"PASS Ohm's law: I={self.I} ≈ E/(r+2R) = {I_calc}")

        # 3. Ampere's force
        F_Ampere_calc = self.B2 * self.I * self.L
        if not math.isclose(self.F_Ampere, F_Ampere_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL Ampere's force: F_Ampere={self.F_Ampere}, calculated {F_Ampere_calc}")
            success = False
        else:
            logs.append(f"PASS Ampere's force: F_Ampere={self.F_Ampere} ≈ B₂*I*L = {F_Ampere_calc}")

        # 4. Hooke's law (two springs)
        F_spring_total_calc = 2 * self.k * self.delta_x
        # Compare total spring force (2 * F_spring) with calculated total
        if not math.isclose(2 * self.F_spring, F_spring_total_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL Hooke's law: 2*F_spring={2*self.F_spring}, calculated total {F_spring_total_calc}")
            success = False
        else:
            logs.append(f"PASS Hooke's law: 2*F_spring={2*self.F_spring} ≈ 2*k*Δx = {F_spring_total_calc}")

        # 5. Force balance: F_Ampere = total spring force
        if not math.isclose(self.F_Ampere, F_spring_total_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL Force balance: F_Ampere={self.F_Ampere}, total spring force={F_spring_total_calc}")
            success = False
        else:
            logs.append(f"PASS Force balance: F_Ampere={self.F_Ampere} ≈ 2*k*Δx = {F_spring_total_calc}")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results from the STEPS ANALYSIS with final computed values.
        """
        logs = []
        success = True

        # Check ΔB₁/Δt from step 2
        if not math.isclose(self.delta_B1_dt, self.step2_delta_B1_dt, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL ΔB₁/Δt consistency: {self.delta_B1_dt} vs expected {self.step2_delta_B1_dt}")
            success = False
        else:
            logs.append(f"PASS ΔB₁/Δt consistent: {self.delta_B1_dt}")

        # Check E from step 3
        if not math.isclose(self.E, self.step3_E, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL E consistency: {self.E} vs expected {self.step3_E}")
            success = False
        else:
            logs.append(f"PASS E consistent: {self.E}")

        # Check I from step 4
        if not math.isclose(self.I, self.step4_I, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL I consistency: {self.I} vs expected {self.step4_I}")
            success = False
        else:
            logs.append(f"PASS I consistent: {self.I}")

        # Check B₂ from step 7
        if not math.isclose(self.B2, self.step7_B2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL B₂ consistency: {self.B2} vs expected {self.step7_B2}")
            success = False
        else:
            logs.append(f"PASS B₂ consistent: {self.B2}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (Δx) and verify the trend in B₂.
        Since B₂ = (2 * k * Δx) / (I * L) and I = E / (r + 2R) with E independent of Δx,
        increasing Δx should increase B₂.
        We perturb by +1%, compute new B₂, check increase, then restore.
        """
        logs = []
        original_delta_x = self.delta_x
        perturbation = 0.01  # 1% increase

        try:
            # Perturb Δx
            self.delta_x = original_delta_x * (1 + perturbation)
            # Recompute I (unchanged since E and resistances are fixed)
            # Avoid division by zero
            total_resistance = self.r + 2 * self.R
            if total_resistance == 0:
                logs.append("FAIL Monotonicity: Total resistance is zero")
                self.delta_x = original_delta_x
                return False, logs
                
            I_new = self.E / total_resistance
            
            # Avoid division by zero in B₂ calculation
            if I_new * self.L == 0:
                logs.append("FAIL Monotonicity: I*L is zero")
                self.delta_x = original_delta_x
                return False, logs
                
            # Recompute B₂ from force balance: B₂_new = (2 * k * Δx_new) / (I_new * L)
            B2_new = (2 * self.k * self.delta_x) / (I_new * self.L)

            # Check monotonicity: B2_new should be > original B2
            if B2_new > self.B2:
                logs.append(f"PASS Monotonicity: Δx ↑ {perturbation*100}% → B₂ ↑ ({self.B2} → {B2_new})")
            else:
                logs.append(f"FAIL Monotonicity: Δx ↑ {perturbation*100}% but B₂ did not increase ({self.B2} → {B2_new})")
                # Restore before returning
                self.delta_x = original_delta_x
                return False, logs

            # Restore original value
            self.delta_x = original_delta_x
            return True, logs
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            # Restore on error
            self.delta_x = original_delta_x
            return False, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and aggregated logs.
        """
        all_logs = []
        overall_success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs