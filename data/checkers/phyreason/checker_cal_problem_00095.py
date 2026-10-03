import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        self.extract_attributes()

    def extract_attributes(self):
        """Extract all inputs and results with safe fallbacks."""
        # Input parameters (from problem statement)
        self.inputs['m_A'] = 1.0  # kg
        self.inputs['m_B'] = 4.0  # kg
        self.inputs['l'] = 1.0    # m
        self.inputs['E_k'] = 10.0 # J
        self.inputs['mu'] = 0.20
        self.inputs['g'] = 10.0   # m/s^2

        # Results mapping (using safe getattr)
        result_keys = [
            'delta_x', 'x_A', 't_2', 's', 'v_Aprimeprime',
            'x_Aprime', 'sprime', 'v_A', 'v_Aprime', 'v_B',
            't_1', 'v_B0', 'x_B', 'v_A0', 'x_Bprime'
        ]
        
        for key in result_keys:
            self.results[key] = getattr(self.solution, key, None)
            if self.results[key] is None:
                # Fallback to solution.results dict if available
                self.results[key] = getattr(self.solution, 'results', {}).get(key, None)

    def fuzzy_eq(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy equality check for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Test fundamental physics invariants."""
        logs = []
        success = True

        # 1. Conservation of momentum after spring release
        try:
            v_A = self.results.get('v_A')
            v_B = self.results.get('v_B')
            if v_A is not None and v_B is not None:
                # v_A is right (+), v_B magnitude but B moves left (-)
                total_momentum = self.inputs['m_A'] * v_A + self.inputs['m_B'] * (-v_B)
                if not self.fuzzy_eq(total_momentum, 0.0):
                    logs.append(f"FAIL: Momentum not conserved after spring release. Total p = {total_momentum}")
                    success = False
                else:
                    logs.append("PASS: Momentum conserved after spring release")
        except Exception as e:
            logs.append(f"ERROR in momentum check: {e}")
            success = False

        # 2. Conservation of energy after spring release
        try:
            v_A = self.results.get('v_A')
            v_B = self.results.get('v_B')
            if v_A is not None and v_B is not None:
                KE = 0.5 * self.inputs['m_A'] * v_A**2 + 0.5 * self.inputs['m_B'] * v_B**2
                if not self.fuzzy_eq(KE, self.inputs['E_k']):
                    logs.append(f"FAIL: KE mismatch after spring release. Calculated {KE} vs given {self.inputs['E_k']}")
                    success = False
                else:
                    logs.append("PASS: Kinetic energy matches given value after spring release")
        except Exception as e:
            logs.append(f"ERROR in energy check: {e}")
            success = False

        # 3. Newton's 2nd Law for friction (check signs)
        try:
            # For block A: friction force should oppose motion (negative acceleration)
            v_A = self.results.get('v_A')
            t_1 = self.results.get('t_1')
            if v_A is not None and t_1 is not None and t_1 > 0:
                expected_acc_A = -self.inputs['mu'] * self.inputs['g']
                actual_acc_A = -v_A / t_1  # From v_f = v_0 + a*t with v_f=0
                if not self.fuzzy_eq(actual_acc_A, expected_acc_A):
                    logs.append(f"FAIL: Newton's 2nd law for A: a_A={actual_acc_A} vs expected {expected_acc_A}")
                    success = False
                else:
                    logs.append("PASS: Newton's 2nd law holds for block A friction")
        except Exception as e:
            logs.append(f"ERROR in Newton's 2nd law check for A: {e}")

        # 4. Work-energy theorem for displacements
        try:
            # Check displacement of B (step 7)
            v_B = self.results.get('v_B')
            x_B = self.results.get('x_B')
            if v_B is not None and x_B is not None:
                work_friction = -self.inputs['mu'] * self.inputs['m_B'] * self.inputs['g'] * x_B
                delta_KE = 0 - 0.5 * self.inputs['m_B'] * v_B**2
                if not self.fuzzy_eq(work_friction, delta_KE):
                    logs.append(f"FAIL: Work-energy for B: W={work_friction} vs ΔKE={delta_KE}")
                    success = False
                else:
                    logs.append("PASS: Work-energy theorem holds for block B")
        except Exception as e:
            logs.append(f"ERROR in work-energy check: {e}")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True

        try:
            # Check step 12: s = x_B + Δx
            x_B = self.results.get('x_B')
            delta_x = self.results.get('delta_x')
            s = self.results.get('s')
            if None not in (x_B, delta_x, s):
                if not self.fuzzy_eq(s, x_B + delta_x):
                    logs.append(f"FAIL: s ({s}) != x_B ({x_B}) + Δx ({delta_x}) = {x_B+delta_x}")
                    success = False
                else:
                    logs.append("PASS: Distance s consistent with components")
        except Exception as e:
            logs.append(f"ERROR in s consistency: {e}")

        try:
            # Check step 19: s' = x_A' + x_B'
            x_Aprime = self.results.get('x_Aprime')
            x_Bprime = self.results.get('x_Bprime')
            sprime = self.results.get('sprime')
            if None not in (x_Aprime, x_Bprime, sprime):
                if not self.fuzzy_eq(sprime, x_Aprime + x_Bprime):
                    logs.append(f"FAIL: s' ({sprime}) != x_A' ({x_Aprime}) + x_B' ({x_Bprime}) = {x_Aprime+x_Bprime}")
                    success = False
                else:
                    logs.append("PASS: Final distance s' consistent")
        except Exception as e:
            logs.append(f"ERROR in s' consistency: {e}")

        try:
            # Check collision velocities (step 16) satisfy momentum conservation
            v_A0 = self.results.get('v_A0')
            v_B0 = self.results.get('v_B0')
            v_Aprimeprime = self.results.get('v_Aprimeprime')
            if None not in (v_A0, v_B0, v_Aprimeprime):
                # Before collision: A moves left with velocity -v_Aprimeprime, B is at rest (0)
                # v_Aprimeprime is stored as positive magnitude
                # After collision: A moves right with v_A0, B moves left with -v_B0
                p_before = self.inputs['m_A'] * (-v_Aprimeprime) + self.inputs['m_B'] * 0
                p_after = self.inputs['m_A'] * v_A0 + self.inputs['m_B'] * (-v_B0)
                if not self.fuzzy_eq(p_before, p_after):
                    logs.append(f"FAIL: Momentum in collision: before={p_before}, after={p_after}")
                    success = False
                else:
                    logs.append("PASS: Momentum conserved in collision")
        except Exception as e:
            logs.append(f"ERROR in collision momentum: {e}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input parameter and check if results change in expected direction.
        Use only numeric calculations, no SymPy symbols.
        """
        logs = []
        success = True

        original_mA = self.inputs['m_A']
        perturbed_mA = original_mA * 1.1  # Increase by 10%

        try:
            # Get original velocities from results
            original_vA = self.results.get('v_A')
            original_vB = self.results.get('v_B')
            
            if original_vA is None or original_vB is None:
                logs.append("SKIP: Missing v_A or v_B for monotonicity test")
                return True, logs

            # Numerically compute what velocities would be with perturbed mass
            # Using conservation laws:
            # 1) m_A*v_A - m_B*v_B = 0 (A right +v_A, B left -v_B, v_B positive magnitude)
            # 2) 0.5*m_A*v_A^2 + 0.5*m_B*v_B^2 = E_k
            
            # From momentum: v_B = (m_A/m_B)*v_A
            # Substitute into energy: 0.5*m_A*v_A^2 + 0.5*m_B*(m_A^2/m_B^2)*v_A^2 = E_k
            # Simplify: 0.5*m_A*v_A^2 + 0.5*(m_A^2/m_B)*v_A^2 = E_k
            # Factor: 0.5*m_A*v_A^2*(1 + m_A/m_B) = E_k
            
            # For original mass (already known from results)
            # For perturbed mass:
            m_B = self.inputs['m_B']
            E_k = self.inputs['E_k']
            
            # Calculate v_A for perturbed mass
            v_A_pert = math.sqrt(2 * E_k / (perturbed_mA * (1 + perturbed_mA / m_B)))
            # Calculate v_B for perturbed mass using momentum: v_B = (m_A/m_B)*v_A
            v_B_pert = (perturbed_mA / m_B) * v_A_pert
            
            # Check monotonicity: if m_A increases, v_A should decrease, v_B should increase
            if not (v_A_pert < original_vA and v_B_pert > original_vB):
                logs.append(f"FAIL: Monotonicity violated. v_A: {v_A_pert} vs {original_vA}, v_B: {v_B_pert} vs {original_vB}")
                success = False
            else:
                logs.append("PASS: Monotonicity holds for mass perturbation")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False
        finally:
            # Restore original values
            self.inputs['m_A'] = original_mA

        return success, logs

    def run_all_tests(self):
        """Run all tests and return final status."""
        all_logs = ["=== Physics Checker Started ==="]
        overall_success = True

        # Run each test suite
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n--- Running {test_name} ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"CRASH in {test_name}: {traceback.format_exc()}")
                overall_success = False

        all_logs.append(f"\n=== Final Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs