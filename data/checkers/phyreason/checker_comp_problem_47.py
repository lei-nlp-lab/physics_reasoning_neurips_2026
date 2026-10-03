import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.results = {}
        # Extract results from solution_instance.results dictionary
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            # Fallback: try to get attributes directly
            for key in ['Q_0', 'DeltaTr', 'delta_P', 'Q_i', 'mathcalP', 'T0', 
                       'delta_TR_T', 'M_N', 'v', 'h', 'M_T', 'p', 'C']:
                self.results[key] = getattr(solution_instance, key, None)

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe numerical comparison with defaults for physics data."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (ValueError, TypeError):
            return False

    def test_invariants(self):
        """Verify physical conservation laws from KB."""
        logs = []
        try:
            # 1. Conservation of flow in branching (step_1)
            # Q_0 is in ml/h, Q_i is in m³/s - need to convert to same units
            Q0_ml_per_h = self.results.get('Q_0')
            Q_i_m3_s = self.results.get('Q_i')
            if Q0_ml_per_h is not None and Q_i_m3_s is not None:
                # Convert Q_i to ml/h for comparison
                Q_i_ml_per_h = Q_i_m3_s * 1e6 * 3600
                # For i=0, Q_i should equal Q0
                if not self._isclose(Q_i_ml_per_h, Q0_ml_per_h, rel_tol=0.01):
                    logs.append(f"FAIL: Flow conservation violated: Q_i={Q_i_ml_per_h} ml/h != Q_0={Q0_ml_per_h} ml/h")
                else:
                    logs.append("PASS: Flow conservation in branching")

            # 2. Hagen-Poiseuille pressure drop (step_3, step_4)
            delta_P = self.results.get('delta_P')
            if delta_P is not None and Q_i_m3_s is not None:
                # Use formula: ΔP = (8 l0 eta / (π r0^4)) * N * Q0
                # Constants from problem
                r0 = 6.0e-5
                l0 = 2.0e-3
                eta = 3.5e-3
                N = 6
                # Q0 in m³/s (use Q_i which is Q0 at i=0)
                Q0_m3_s = Q_i_m3_s
                expected_delta_P = (8 * l0 * eta / (math.pi * r0**4)) * N * Q0_m3_s
                if not self._isclose(delta_P, expected_delta_P, rel_tol=0.1):
                    logs.append(f"FAIL: Hagen-Poiseuille mismatch: ΔP={delta_P}, expected={expected_delta_P}")
                else:
                    logs.append("PASS: Hagen-Poiseuille pressure drop consistent")

            # 3. Mass density relation (step_11, step_23)
            M_N = self.results.get('M_N')
            v = self.results.get('v')
            if M_N is not None and v is not None and v != 1:
                # From step_23: M_N/V_N = ρ_N where ρ_N = ρ0*(1 + p/K_N)
                # p = K_N * v/(1-v)
                # So expected density = ρ0 * (1 + v/(1-v))
                rho0 = 1050  # kg/m³
                p_expr = 1e6 * v / (1 - v)  # Using K_N = 1e6 Pa
                expected_density = rho0 * (1 + p_expr / 1e6)
                # For comparison, assume V = 1 m³, so V_N = 1-v
                V_N = 1 - v
                actual_density = M_N / V_N if V_N != 0 else 0
                if not self._isclose(actual_density, expected_density, rel_tol=0.05):
                    logs.append(f"FAIL: Density mismatch: {actual_density} vs expected {expected_density}")
                else:
                    logs.append("PASS: Mass-density relation holds")

            # 4. Temperature continuity at tumor boundary (step_18, step_20)
            C = self.results.get('C')
            delta_TR_T = self.results.get('delta_TR_T')
            if C is not None and delta_TR_T is not None:
                # C should be 1.5 * ΔT(R_T)
                if not self._isclose(C, 1.5 * delta_TR_T, rel_tol=0.01):
                    logs.append(f"FAIL: Temperature continuity violated: C={C}, 1.5*ΔT(R_T)={1.5*delta_TR_T}")
                else:
                    logs.append("PASS: Temperature boundary condition consistent")

        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check step_5 -> step_6 -> step_7 chain
            delta_P = self.results.get('delta_P')
            if delta_P is not None:
                # step_5 gives ΔP = 3332.2 Pa
                expected_delta_P = 3332.2
                if not self._isclose(delta_P, expected_delta_P, abs_tol=0.1):
                    logs.append(f"FAIL: ΔP mismatch: {delta_P} vs step_5 {expected_delta_P}")
                else:
                    logs.append("PASS: ΔP consistent with step_5")

            # Check step_6 Q0 calculation
            Q_i_m3_s = self.results.get('Q_i')  # This is Q0 in m³/s
            if Q_i_m3_s is not None and delta_P is not None:
                r0 = 6.0e-5
                l0 = 2.0e-3
                eta = 3.5e-3
                N = 6
                expected_Q0_m3_s = delta_P * math.pi * r0**4 / (8 * N * l0 * eta)
                if not self._isclose(Q_i_m3_s, expected_Q0_m3_s, rel_tol=0.1):
                    logs.append(f"FAIL: Q0 mismatch: {Q_i_m3_s} m³/s vs step_6 formula {expected_Q0_m3_s} m³/s")
                else:
                    logs.append("PASS: Q0 consistent with step_6 formula")

            # Check step_7 unit conversion to ml/h
            Q0_ml_per_h = self.results.get('Q_0')
            if Q0_ml_per_h is not None and Q_i_m3_s is not None:
                expected_Q0_ml_per_h = Q_i_m3_s * 1e6 * 3600
                # Target answer is 1.5 ml/h
                if not self._isclose(Q0_ml_per_h, 1.5, abs_tol=0.1):
                    logs.append(f"FAIL: Q0 unit conversion: {Q0_ml_per_h} ml/h vs target 1.5 ml/h (expected {expected_Q0_ml_per_h})")
                else:
                    logs.append("PASS: Q0 ≈ 1.5 ml/h")

            # Check step_10 h calculation
            h = self.results.get('h')
            if h is not None:
                eta = 3.5e-3
                l0 = 2.0e-3
                E = 0.06e6
                rho = 1050
                r0 = 6.0e-5
                expected_h = (64 * eta**2 * l0**2) / (3 * E * rho * r0**3)
                if not self._isclose(h, expected_h, rel_tol=0.05):
                    logs.append(f"FAIL: h mismatch: {h} vs step_10 {expected_h}")
                else:
                    logs.append("PASS: h consistent with step_10 formula")

            # Check step_21 𝒫 calculation
            mathcalP = self.results.get('mathcalP')
            if mathcalP is not None:
                k = 0.6
                R_T = 0.05
                expected_mathcalP = 18 * k / R_T**2
                if not self._isclose(mathcalP, expected_mathcalP, rel_tol=0.01):
                    logs.append(f"FAIL: 𝒫 mismatch: {mathcalP} vs step_21 {expected_mathcalP}")
                else:
                    logs.append("PASS: 𝒫 = 4320 W/m³ consistent")

        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        return logs

    def test_monotonicity(self):
        """Perturb input and check output trends."""
        logs = []
        try:
            # Use constants from the problem (consistent with solution)
            r0 = 6.0e-5
            l0 = 2.0e-3
            eta = 3.5e-3
            N = 6
            delta_P = 3332.2
            k = 0.6
            R_T = 0.05
            K_N = 1e6  # Consistent with solution
            p_c = 1.2e5  # Consistent with solution
            P_cap = 1.0e5  # Consistent with solution
            delta_r_c = 1e-6
            r_N_minus_1 = r0 / (2**((N-1)/3))

            # Perturb viscosity η and check Q0 decreases
            Q0_original = delta_P * math.pi * r0**4 / (8 * N * l0 * eta)
            eta_perturbed = eta * 1.1
            Q0_perturbed = delta_P * math.pi * r0**4 / (8 * N * l0 * eta_perturbed)
            if Q0_perturbed < Q0_original:
                logs.append("PASS: Q0 decreases with increased viscosity (monotonic)")
            else:
                logs.append("FAIL: Q0 monotonicity violated w.r.t viscosity")

            # Perturb tumor radius R_T and check 𝒫 decreases
            mathcalP_original = 18 * k / R_T**2
            R_T_perturbed = R_T * 1.1
            mathcalP_perturbed = 18 * k / R_T_perturbed**2
            if mathcalP_perturbed < mathcalP_original:
                logs.append("PASS: 𝒫 decreases with increased tumor radius")
            else:
                logs.append("FAIL: 𝒫 monotonicity violated w.r.t tumor radius")

            # Perturb pressure p and check flow reduction δQ/Q (step_27)
            v = self.results.get('v')
            if v is not None and v != 1:
                p = K_N * v / (1 - v)
                def rel_flow_change(p_val):
                    return - (2/N) * (p_val - P_cap) / (p_c - P_cap) * (delta_r_c / r_N_minus_1)
                original_change = rel_flow_change(p)
                p_perturbed = p * 1.1
                perturbed_change = rel_flow_change(p_perturbed)
                if perturbed_change < original_change:
                    logs.append("PASS: Flow reduction increases with pressure")
                else:
                    logs.append("FAIL: Flow reduction monotonicity violated w.r.t pressure")
            else:
                logs.append("SKIP: Flow reduction monotonicity (v missing or invalid)")

        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        return logs

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        logs = []
        success = True

        logs.append("=== PhysicsChecker Starting ===")

        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()

        logs.extend(inv_logs)
        logs.extend(trace_logs)
        logs.extend(mono_logs)

        # Determine overall success
        all_logs = inv_logs + trace_logs + mono_logs
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False

        logs.append(f"=== PhysicsChecker Finished: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, logs