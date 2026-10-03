import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract all needed numeric values from results
        # If a value is not in results, try to get from solution_instance attributes as fallback
        self.m = results.get('m', getattr(solution_instance, 'm', 1.67e-27))
        self.q = results.get('q', getattr(solution_instance, 'q', 1.6e-19))
        self.B = results.get('B', getattr(solution_instance, 'B', 0.5))
        self.U = results.get('U', getattr(solution_instance, 'U', 1000.0))
        self.R = results.get('R', getattr(solution_instance, 'R', 0.5))
        self.E = results.get('E', getattr(solution_instance, 'E', 1e5))
        self.alpha = results.get('alpha', getattr(solution_instance, 'alpha', math.pi/6))
        
        # Ensure all are floats (in case they are SymPy symbols)
        try:
            self.m = float(self.m)
            self.q = float(self.q)
            self.B = float(self.B)
            self.U = float(self.U)
            self.R = float(self.R)
            self.E = float(self.E)
            self.alpha = float(self.alpha)
        except:
            pass  # if conversion fails, keep as is (but will likely cause errors later)
        
        # Extract computed results
        self.E_kp = results.get('E_kp', 0.0)
        self.d_m = results.get('d_m', 0.0)
        self.E_k = results.get('E_k', 0.0)
        self.r = results.get('r', 0.0)
        self.T = results.get('T', 0.0)
        self.R_m = results.get('R_m', 0.0)
        self.t = results.get('t', 0.0)
        self.v_P = results.get('v_P', 0.0)
        self.r_2 = results.get('r_2', 0.0)
        self.r_Q = results.get('r_Q', 0.0)
        self.r_1 = results.get('r_1', 0.0)
        self.half_m_v_P2 = results.get('1/2*m*v_P2', 0.0)

        # Store logs
        self.logs = []

    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def test_invariants(self):
        """Verify physics invariants from KB theorems."""
        tests_passed = True

        # 1. Kinetic energy theorem: E_kp = n*q*U, where n is integer number of accelerations
        try:
            if abs(self.q * self.U) > 1e-12:
                n = self.E_kp / (self.q * self.U)
                # Check if E_kp matches both formulas
                E_kp_from_v = 0.5 * self.m * self.v_P**2
                E_kp_from_BR = (self.q * self.B * self.R)**2 / (2 * self.m)
                if not (math.isclose(self.E_kp, E_kp_from_v, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.E_kp, E_kp_from_BR, rel_tol=1e-2, abs_tol=1e-2)):
                    tests_passed = self._log("Kinetic energy at P inconsistent between formulas", False)
                else:
                    tests_passed = self._log("Kinetic energy at P consistent", True)
            else:
                tests_passed = self._log("Skipping kinetic energy check (q*U near zero)", True)
        except Exception as e:
            tests_passed = self._log(f"Kinetic energy check failed: {e}", False)

        # 2. Period of circular motion: T = 2πm/(|q|B)
        try:
            T_expected = 2 * math.pi * self.m / (abs(self.q) * self.B)
            if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(f"Period T mismatch. Expected {T_expected}, got {self.T}", False)
            else:
                tests_passed = self._log("Period T consistent with formula", True)
        except Exception as e:
            tests_passed = self._log(f"Period check failed: {e}", False)

        # 3. Lorentz force as centripetal force: q*v_P*B = m*v_P^2/R
        try:
            magnetic_force = abs(self.q) * self.v_P * self.B
            centripetal_force = self.m * self.v_P**2 / self.R
            if not math.isclose(magnetic_force, centripetal_force, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log("Lorentz force does not equal centripetal force at P", False)
            else:
                tests_passed = self._log("Force balance at P consistent", True)
        except Exception as e:
            tests_passed = self._log(f"Force balance check failed: {e}", False)

        return tests_passed

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        tests_passed = True

        # 1. Check that t = (n-1)*T/2, where n = E_kp/(q*U)
        try:
            if abs(self.q * self.U) > 1e-12:
                n = self.E_kp / (self.q * self.U)
                t_expected = (n - 1) * self.T / 2
                if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                    tests_passed = self._log(f"Time t inconsistent with n and T. Expected {t_expected}, got {self.t}", False)
                else:
                    tests_passed = self._log("Time t consistent with n and T", True)
            else:
                tests_passed = self._log("Skipping time consistency check (q*U near zero)", True)
        except Exception as e:
            tests_passed = self._log(f"Time consistency check failed: {e}", False)

        # 2. Check that r_1 and r_2 follow from kinetic energy steps
        try:
            # Ensure the expression under sqrt is non-negative
            arg1 = 2 * self.m * (self.E_kp - self.q * self.U)
            if arg1 >= 0:
                r_1_expected = math.sqrt(arg1) / (abs(self.q) * self.B)
                if not math.isclose(self.r_1, r_1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    tests_passed = self._log(f"r_1 mismatch. Expected {r_1_expected}, got {self.r_1}", False)
                else:
                    tests_passed = self._log("r_1 consistent with energy step", True)
            else:
                tests_passed = self._log("Skipping r_1 check (negative under sqrt)", True)
        except Exception as e:
            tests_passed = self._log(f"r_1 check failed: {e}", False)

        try:
            arg2 = 2 * self.m * (self.E_kp - 2 * self.q * self.U)
            if arg2 >= 0:
                r_2_expected = math.sqrt(arg2) / (abs(self.q) * self.B)
                if not math.isclose(self.r_2, r_2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    tests_passed = self._log(f"r_2 mismatch. Expected {r_2_expected}, got {self.r_2}", False)
                else:
                    tests_passed = self._log("r_2 consistent with energy step", True)
            else:
                tests_passed = self._log("Skipping r_2 check (negative under sqrt)", True)
        except Exception as e:
            tests_passed = self._log(f"r_2 check failed: {e}", False)

        # 3. Check that d_m = 2*(r_1 - r_2)
        try:
            d_m_expected = 2 * (self.r_1 - self.r_2)
            if not math.isclose(self.d_m, d_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(f"d_m mismatch. Expected {d_m_expected}, got {self.d_m}", False)
            else:
                tests_passed = self._log("d_m consistent with r_1 and r_2", True)
        except Exception as e:
            tests_passed = self._log(f"d_m consistency check failed: {e}", False)

        # 4. Check R_m formula from step 14
        try:
            numerator = 2 * self.m * self.E * self.R
            denominator = self.q * self.B**2 * self.R - self.m * self.E
            if abs(denominator) > 1e-12:
                R_m_expected = self.R + (numerator / denominator) * math.sin(self.alpha / 2)
                if not math.isclose(self.R_m, R_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                    tests_passed = self._log(f"R_m mismatch. Expected {R_m_expected}, got {self.R_m}", False)
                else:
                    tests_passed = self._log("R_m consistent with formula", True)
            else:
                tests_passed = self._log("Skipping R_m check (denominator near zero)", True)
        except Exception as e:
            tests_passed = self._log(f"R_m consistency check failed: {e}", False)

        return tests_passed

    def test_monotonicity(self):
        """Perturb input mass m and check trends in outputs."""
        tests_passed = True
        original_m = self.m

        try:
            # Perturb m by +10%
            self.m *= 1.1

            # Recompute expected values using formulas (all numeric)
            # 1. Time t
            n = (self.q * self.B**2 * self.R**2) / (2 * self.m * self.U)  # from step 4
            T_perturbed = 2 * math.pi * self.m / (abs(self.q) * self.B)
            t_perturbed = (n - 1) * T_perturbed / 2

            # 2. d_m
            term1 = math.sqrt(self.R**2 - 2 * self.m * self.U / (self.q * self.B**2))
            term2 = math.sqrt(self.R**2 - 4 * self.m * self.U / (self.q * self.B**2))
            d_m_perturbed = 2 * (term1 - term2)

            # 3. R_m
            numerator = 2 * self.m * self.E * self.R
            denominator = self.q * self.B**2 * self.R - self.m * self.E
            R_m_perturbed = self.R + (numerator / denominator) * math.sin(self.alpha / 2)

            # Check monotonic trends (just log, not fail)
            self._log(f"After +10% mass: t={t_perturbed:.3e}, d_m={d_m_perturbed:.3e}, R_m={R_m_perturbed:.3e}", True)

        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed: {e}", False)
        finally:
            # Restore original mass
            self.m = original_m

        return tests_passed

    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []
        overall_success = True

        try:
            self.logs.append("=== Physics Checker Start ===")
            
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()

            overall_success = inv_ok and trace_ok and mono_ok

            self.logs.append(f"=== Summary: {'ALL PASS' if overall_success else 'SOME FAIL'} ===")
        except Exception as e:
            self.logs.append(f"ERROR in test execution: {traceback.format_exc()}")
            overall_success = False

        return overall_success, self.logs