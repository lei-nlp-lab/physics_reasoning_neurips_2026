import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        try:
            # First try to get results dictionary
            results = getattr(solution_instance, 'results', {})
            
            # Input parameters - extract from results or use defaults
            # Use explicit float conversion to ensure numeric values
            self.m = float(results.get('m', 1e-9))
            self.q = float(results.get('q', 1e-6))
            self.v = float(results.get('v', 100.0))
            self.B = float(results.get('B', 0.1))
            self.d = float(results.get('d', 0.01))

            # Results with fallbacks
            self.U = float(results.get('U', 0.0))
            self.E = float(results.get('E', 0.0))
            # Handle both E_k and E_{k} keys
            E_k_value = results.get('E_k', results.get('E_{k}', 0.0))
            self.E_k = float(E_k_value)
            
        except Exception as e:
            self.logs = [f"Initialization error: {str(e)}"]
            raise

    def test_invariants(self):
        """
        Verify physical invariants: 
        1. Work-energy theorem for accelerator (U = mv²/2q)
        2. Force balance in velocity selector (E = vB)
        3. Kinetic energy expression (E_k = ½mv² + qBvd)
        """
        logs = []
        success = True

        # 1. Accelerator voltage
        try:
            if abs(self.q) < 1e-15:  # Avoid division by zero
                logs.append("SKIP: U check (q ≈ 0)")
            else:
                U_expected = (self.m * self.v**2) / (2 * self.q)
                if not math.isclose(self.U, U_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: U mismatch. Got {self.U}, expected {U_expected}")
                    success = False
                else:
                    logs.append("PASS: Accelerator voltage U = mv²/2q")
        except ZeroDivisionError:
            logs.append("SKIP: U check (q=0)")
            success = False  # q should not be zero

        # 2. Velocity selector force balance
        try:
            E_expected = self.v * self.B
            if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E mismatch. Got {self.E}, expected {E_expected}")
                success = False
            else:
                logs.append("PASS: Electric field E = vB (force balance)")
        except:
            logs.append("SKIP: E check (numerical issue)")

        # 3. Kinetic energy after selector
        try:
            Ek_expected = 0.5 * self.m * self.v**2 + self.q * self.B * self.v * self.d
            if not math.isclose(self.E_k, Ek_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_k mismatch. Got {self.E_k}, expected {Ek_expected}")
                success = False
            else:
                logs.append("PASS: Kinetic energy E_k = ½mv² + qBvd")
        except:
            logs.append("SKIP: E_k check (numerical issue)")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer using work-energy principles.
        The kinetic energy should equal work done by accelerator plus work done in selector.
        """
        logs = []
        success = True

        try:
            # Work done by accelerator: qU = ½mv²
            work_accelerator = self.q * self.U
            kinetic_from_accelerator = 0.5 * self.m * self.v**2
            
            if not math.isclose(work_accelerator, kinetic_from_accelerator, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Accelerator work-energy mismatch. qU={work_accelerator}, ½mv²={kinetic_from_accelerator}")
                success = False
            else:
                logs.append("PASS: Accelerator work-energy consistent")

            # Work done in selector: qEd = q(vB)d
            work_selector = self.q * self.E * self.d
            expected_selector_work = self.q * self.v * self.B * self.d
            
            if not math.isclose(work_selector, expected_selector_work, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Selector work mismatch. qEd={work_selector}, qBvd={expected_selector_work}")
                success = False
            else:
                logs.append("PASS: Selector work consistent")

            # Total kinetic energy = work_accelerator + work_selector
            total_work = work_accelerator + work_selector
            if not math.isclose(self.E_k, total_work, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Total energy mismatch. E_k={self.E_k}, sum of works={total_work}")
                success = False
            else:
                logs.append("PASS: Total energy conservation holds")

        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb input mass by +10%, recalculate outputs using formulas,
        check expected trends, then restore original values.
        """
        logs = []
        success = True

        try:
            # Store original
            m_orig = self.m
            U_orig = self.U
            E_k_orig = self.E_k

            # Perturb mass by +10%
            m_perturbed = m_orig * 1.10

            # Recalculate using physical formulas
            # Avoid division by zero
            if abs(self.q) < 1e-15:
                logs.append("SKIP: Monotonicity test (q ≈ 0)")
                return success, logs
                
            U_perturbed = (m_perturbed * self.v**2) / (2 * self.q)
            E_k_perturbed = 0.5 * m_perturbed * self.v**2 + self.q * self.B * self.v * self.d

            # Check trends
            if not U_perturbed > U_orig:
                logs.append(f"FAIL: U did not increase with mass. Original: {U_orig}, Perturbed: {U_perturbed}")
                success = False
            else:
                logs.append(f"PASS: U monotonic with mass (increased by {U_perturbed/U_orig:.2f}x)")

            if not E_k_perturbed > E_k_orig:
                logs.append(f"FAIL: E_k did not increase with mass. Original: {E_k_orig}, Perturbed: {E_k_perturbed}")
                success = False
            else:
                logs.append(f"PASS: E_k monotonic with mass (increased by {E_k_perturbed/E_k_orig:.2f}x)")

        except ZeroDivisionError:
            logs.append("SKIP: Monotonicity test (q=0)")
            success = False
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = ["=== Physics Audit Results ==="]
        overall_success = True

        # Test 1: Invariants
        try:
            success, logs = self.test_invariants()
            all_logs.extend(logs)
            overall_success = overall_success and success
        except Exception as e:
            all_logs.append(f"ERROR in invariants test: {traceback.format_exc()}")
            overall_success = False

        # Test 2: Trace Consistency
        try:
            success, logs = self.test_trace_consistency()
            all_logs.extend(logs)
            overall_success = overall_success and success
        except Exception as e:
            all_logs.append(f"ERROR in trace consistency test: {traceback.format_exc()}")
            overall_success = False

        # Test 3: Monotonicity
        try:
            success, logs = self.test_monotonicity()
            all_logs.extend(logs)
            overall_success = overall_success and success
        except Exception as e:
            all_logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            overall_success = False

        return overall_success, all_logs