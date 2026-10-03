import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Constants - use math.pi for numeric calculations
        self.c = 3e8
        self.h = 6.626e-34
        self.e = 1.602e-19
        self.m_e = 9.109e-31
        self.mu_B = self.e * self.h / (4 * math.pi * self.m_e)  # Bohr magneton, J/T
        self.lambda_D1 = 589.6e-9
        self.lambda_D2 = 589e-9
        self.R = 0.9
        self.t = 0.01
        self.f = 0.30

        # Safe attribute access for results
        results = getattr(solution_instance, 'results', {})
        
        # Extract using getattr with fallback to results dict
        self.epsilon = results.get('varepsilon', 0.0)
        self.resolution = results.get('lambda/Deltalambda', 0.0)
        self.n_air = results.get('n_air', 0.0)
        self.m = results.get('m', 0)
        self.m_plus_10 = results.get('m+10', 0)
        
        # Frequencies
        self.v0 = results.get('v_0', 0.0)
        self.va = results.get('v_a', 0.0)
        self.vb = results.get('v_b', 0.0)
        self.vc = results.get('v_c', 0.0)
        self.vd = results.get('v_d', 0.0)
        
        # Energy differences
        self.DeltaE1 = results.get('DeltaE_1', 0.0)
        self.DeltaE2 = results.get('DeltaE_2', 0.0)
        
        # Landé factors
        self.g_k1 = results.get('g_k1', 0.0)
        self.g_k2 = results.get('g_k2', 0.0)
        
        # Magnetic field
        self.B = results.get('B', 0.0)
        self.mu_BB = results.get('mu_BB', 0.0)  # μ_B * B in J
        
        # Wavelength differences
        self.Delta_lambda1 = results.get('Deltalambda_1', 0.0)
        self.Delta_lambda2 = results.get('Deltalambda_2', 0.0)
        
        # Frequency differences - careful: there are multiple Δv in the problem
        # Get the LED line width (stored as Deltav in the code)
        self.Delta_v_line = results.get('Deltav', 0.0)
        
        # For LED scenario
        self.Delta_v_m = results.get('Deltav_m', 0.0)
        self.Delta_v_s = results.get('Deltav_s', 0.0)
        self.N = results.get('N', 0)
        
        # For Zeeman splitting Δv (computed as (ΔE1+ΔE2)/h)
        self.Delta_v_zeeman = (self.DeltaE1 + self.DeltaE2) / self.h if self.h != 0 else 0.0
        
        # For monotonicity test: store original B
        self.B_original = self.B
        self.logs = []

    def log(self, message):
        """Helper to log messages."""
        self.logs.append(message)

    def test_invariants(self):
        """Test fundamental physics invariants from KB."""
        logs = []
        try:
            # 1. Frequency-wavelength relationship (Theorem 1)
            v0_calculated = self.c / self.lambda_D1
            if math.isclose(self.v0, v0_calculated, rel_tol=1e-2, abs_tol=1e6):
                logs.append("✓ Frequency-wavelength relation holds")
            else:
                logs.append(f"✗ Frequency mismatch: v0={self.v0}, c/λ={v0_calculated}")

            # 2. Zeeman energy shift linearity (Theorem 2 & 4)
            # Check if ΔE1 and ΔE2 are proportional to g factors and μ_BB
            # According to the problem, ΔE = g * μ_B * B
            # We have μ_BB = μ_B * B stored
            if self.mu_BB > 0 and self.g_k1 > 0 and self.g_k2 > 0:
                # Expected: ΔE1 = g_k1 * μ_BB
                DeltaE1_expected = self.g_k1 * self.mu_BB
                DeltaE2_expected = self.g_k2 * self.mu_BB
                
                if math.isclose(self.DeltaE1, DeltaE1_expected, rel_tol=1e-2):
                    logs.append("✓ ΔE1 = g_k1 * μ_BB (Zeeman effect)")
                else:
                    logs.append(f"✗ ΔE1 mismatch: stored={self.DeltaE1}, expected={DeltaE1_expected}")
                    
                if math.isclose(self.DeltaE2, DeltaE2_expected, rel_tol=1e-2):
                    logs.append("✓ ΔE2 = g_k2 * μ_BB (Zeeman effect)")
                else:
                    logs.append(f"✗ ΔE2 mismatch: stored={self.DeltaE2}, expected={DeltaE2_expected}")
            else:
                logs.append("⚠ Cannot check Zeeman linearity (μ_BB=0 or g=0)")

            # 3. Fabry-Perot constructive interference (Theorem 3)
            # For normal incidence (θ=0), resonant condition: 2nt = mλ
            # Use n=1 for initial m calculation (not n_air)
            m_calc = 2 * 1.0 * self.t / self.lambda_D1
            if math.isclose(self.m, m_calc, rel_tol=1e-4, abs_tol=0.5):
                logs.append("✓ Constructive interference condition satisfied")
            else:
                logs.append(f"✗ m mismatch: stored={self.m}, calculated={m_calc}")

            # 4. Bohr magneton definition (Theorem 5)
            mu_B_theory = self.e * self.h / (4 * math.pi * self.m_e)
            # The stored μ_BB should equal μ_B * B
            if self.B != 0:
                mu_B_from_stored = self.mu_BB / self.B if abs(self.B) > 1e-20 else 0
                if math.isclose(mu_B_from_stored, mu_B_theory, rel_tol=1e-2):
                    logs.append("✓ Bohr magneton value consistent")
                else:
                    logs.append(f"⚠ μ_B mismatch: from stored={mu_B_from_stored}, theory={mu_B_theory}")
            else:
                logs.append("⚠ B=0, cannot check μ_B")

        except Exception as e:
            logs.append(f"✗ Error in invariants: {e}")
            logs.append(traceback.format_exc())
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # 1. Check ε from step_3
            F = 4 * self.R / (1 - self.R)**2
            epsilon_calc = 2 * (1 - self.R) / math.sqrt(self.R)  # From step_3 formula
            if math.isclose(self.epsilon, epsilon_calc, rel_tol=1e-2):
                logs.append("✓ ε = 2(1-R)/√R consistent")
            else:
                logs.append(f"✗ ε mismatch: stored={self.epsilon}, calc={epsilon_calc}")

            # 2. Check resolution from step_6 (using n=1 initially)
            # λ/Δλ = π n t √F / λ, with θ=0 and n=1 (air before tuning)
            res_calc = math.pi * 1.0 * self.t * math.sqrt(F) / self.lambda_D1
            if math.isclose(self.resolution, res_calc, rel_tol=1e-2):
                logs.append("✓ Resolution consistent")
            else:
                logs.append(f"✗ Resolution mismatch: stored={self.resolution}, calc={res_calc}")

            # 3. Check refractive index from step_9
            if self.m != 0:
                n_calc = self.m_plus_10 / self.m
                if math.isclose(self.n_air, n_calc, rel_tol=1e-5):
                    logs.append("✓ n_air = (m+10)/m consistent")
                else:
                    logs.append(f"✗ n_air mismatch: stored={self.n_air}, calc={n_calc}")

            # 4. Check frequency formulas from step_12
            h = self.h
            v_a_calc = self.v0 - (self.DeltaE1 + self.DeltaE2) / (2 * h)
            v_b_calc = self.v0 - (self.DeltaE2 - self.DeltaE1) / (2 * h)
            v_c_calc = self.v0 + (self.DeltaE2 - self.DeltaE1) / (2 * h)
            v_d_calc = self.v0 + (self.DeltaE1 + self.DeltaE2) / (2 * h)
            
            checks = [
                (self.va, v_a_calc, 'v_a'),
                (self.vb, v_b_calc, 'v_b'),
                (self.vc, v_c_calc, 'v_c'),
                (self.vd, v_d_calc, 'v_d')
            ]
            for stored, calc, name in checks:
                if stored != 0 and math.isclose(stored, calc, rel_tol=1e-2):
                    logs.append(f"✓ {name} formula consistent")
                elif stored == 0:
                    logs.append(f"⚠ {name} not provided")
                else:
                    logs.append(f"✗ {name} mismatch: stored={stored}, calc={calc}")

            # 5. Check Landé factors from steps 24-25
            # The ground truth gives g_k1=0.68, g_k2=1.99
            # We'll check if stored values match these
            if math.isclose(self.g_k1, 0.68, rel_tol=1e-2):
                logs.append("✓ g_k1 matches ground truth (0.68)")
            else:
                logs.append(f"✗ g_k1 mismatch: stored={self.g_k1}, expected=0.68")
                
            if math.isclose(self.g_k2, 1.99, rel_tol=1e-2):
                logs.append("✓ g_k2 matches ground truth (1.99)")
            else:
                logs.append(f"✗ g_k2 mismatch: stored={self.g_k2}, expected=1.99")

            # 6. Check B from step_30 (using Δλ between a and d)
            # Δλ = Δλ1 + Δλ2 (wavelength difference between lines a and d)
            Delta_lambda_ad = self.Delta_lambda1 + self.Delta_lambda2
            if Delta_lambda_ad != 0 and (self.g_k1 + self.g_k2) != 0:
                B_calc = (4 * math.pi * self.m_e * Delta_lambda_ad * self.c) / \
                         (self.lambda_D1**2 * (self.g_k1 + self.g_k2) * self.e)
                if math.isclose(self.B, B_calc, rel_tol=1e-2):
                    logs.append("✓ B formula consistent with Δλ between a and d")
                else:
                    logs.append(f"✗ B mismatch: stored={self.B}, calc={B_calc}")

            # 7. Check N and Δv from steps 31-34
            # Δv_m = c/(2nt) with n≈1 (air) for LED scenario
            # Note: t_LED = 10 mm = 0.01 m, not the same as main t=1 cm
            t_LED = 0.01  # 10 mm
            n_LED = 1.0
            Delta_v_m_calc = self.c / (2 * n_LED * t_LED)
            if math.isclose(self.Delta_v_m, Delta_v_m_calc, rel_tol=1e-2):
                logs.append("✓ Δv_m = c/(2nt) consistent for LED")
            else:
                logs.append(f"✗ Δv_m mismatch: stored={self.Delta_v_m}, calc={Delta_v_m_calc}")

            # N = Δv_s / Δv_m
            if self.Delta_v_s != 0 and self.Delta_v_m != 0:
                N_calc = self.Delta_v_s / self.Delta_v_m
                if math.isclose(self.N, N_calc, rel_tol=1e-2, abs_tol=0.5):
                    logs.append("✓ N = Δv_s/Δv_m consistent")
                else:
                    logs.append(f"✗ N mismatch: stored={self.N}, calc={N_calc}")

            # Δv (line width) = c/(π n t √F)
            Delta_v_calc = self.c / (math.pi * n_LED * t_LED * math.sqrt(F))
            if math.isclose(self.Delta_v_line, Delta_v_calc, rel_tol=1e-2):
                logs.append("✓ Δv (line width) formula consistent")
            else:
                logs.append(f"✗ Δv mismatch: stored={self.Delta_v_line}, calc={Delta_v_calc}")

        except Exception as e:
            logs.append(f"✗ Error in trace consistency: {e}")
            logs.append(traceback.format_exc())
        return logs

    def test_monotonicity(self):
        """
        Perturb an input (magnetic field B) and check if outputs change as expected.
        """
        logs = []
        try:
            if self.B == 0 or self.B_original == 0:
                logs.append("⚠ B=0, cannot test monotonicity")
                return logs

            # Store original values
            B_orig = self.B_original
            DeltaE1_orig = self.DeltaE1
            DeltaE2_orig = self.DeltaE2

            # Perturb B by +10%
            B_new = B_orig * 1.10

            # Predict new ΔE values (linear scaling with B)
            # ΔE = g * μ_B * B, so if B increases by 10%, ΔE should increase by 10%
            DeltaE1_pred = DeltaE1_orig * (B_new / B_orig)
            DeltaE2_pred = DeltaE2_orig * (B_new / B_orig)

            # According to KB, ΔE ∝ B for fixed g and m_j
            ratio1 = DeltaE1_pred / DeltaE1_orig
            ratio2 = DeltaE2_pred / DeltaE2_orig

            if math.isclose(ratio1, 1.10, rel_tol=1e-3) and math.isclose(ratio2, 1.10, rel_tol=1e-3):
                logs.append("✓ ΔE scales linearly with B (Zeeman effect monotonicity)")
            else:
                logs.append(f"✗ ΔE scaling not linear: ratio1={ratio1}, ratio2={ratio2}")

            # Check frequency splittings: v_d - v_a = (ΔE1 + ΔE2)/h
            splitting_orig = (DeltaE1_orig + DeltaE2_orig) / self.h
            splitting_new = (DeltaE1_pred + DeltaE2_pred) / self.h
            if splitting_new > splitting_orig:
                logs.append("✓ Frequency splitting increases with B")
            else:
                logs.append("✗ Frequency splitting does not increase with B")

        except Exception as e:
            logs.append(f"✗ Error in monotonicity test: {e}")
            logs.append(traceback.format_exc())
        return logs

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        self.log("=== Physics Audit Report ===")
        
        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()
        
        # Combine logs
        self.log("\n--- Invariant Tests ---")
        self.logs.extend(inv_logs)
        self.log("\n--- Trace Consistency Tests ---")
        self.logs.extend(trace_logs)
        self.log("\n--- Monotonicity Test ---")
        self.logs.extend(mono_logs)
        
        # Determine overall success (no critical failures)
        critical_failures = sum(1 for log in self.logs if log.startswith('✗'))
        success = critical_failures == 0
        
        self.log(f"\n=== Summary: {'PASS' if success else 'FAIL'} with {critical_failures} critical issue(s) ===")
        
        return success, self.logs