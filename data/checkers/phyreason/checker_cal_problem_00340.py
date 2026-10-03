import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Given problem inputs
        self.q = -3e-6  # C
        self.W_AB_ext = 6e-4  # J (work against field from A→B)
        self.W_BC_elec = -9e-4  # J (work by field from B→C)
        
        # Safe extraction of results (fallback to 0.0 if missing)
        results = getattr(solution_instance, 'results', {})
        
        # Potential differences
        self.U_AB = getattr(solution_instance, 'U_AB', results.get('U_AB', results.get('U_{AB}', 0.0)))
        self.U_BC = getattr(solution_instance, 'U_BC', results.get('U_BC', results.get('U_{BC}', 0.0)))
        self.U_AC = getattr(solution_instance, 'U_AC', results.get('U_AC', results.get('U_{AC}', 0.0)))
        
        # Electric potentials
        self.phi_A = getattr(solution_instance, 'phi_A', results.get('phi_A', results.get('varphi_A', results.get('varphi_{A}', 0.0))))
        self.phi_B = 0.0  # Reference point from steps
        self.phi_C = getattr(solution_instance, 'phi_C', results.get('phi_C', results.get('varphi_C', results.get('varphi_{C}', 0.0))))
        
        # Potential energies
        self.E_PA = getattr(solution_instance, 'E_PA', results.get('E_PA', results.get('E_{PA}', 0.0)))
        self.E_PC = getattr(solution_instance, 'E_PC', results.get('E_PC', results.get('E_{PC}', 0.0)))
        
        # Store original for monotonicity test
        self._original_results = {
            'U_AB': self.U_AB, 'U_BC': self.U_BC, 'U_AC': self.U_AC,
            'phi_A': self.phi_A, 'phi_C': self.phi_C,
            'E_PA': self.E_PA, 'E_PC': self.E_PC
        }
        
        # Expected values from steps analysis (for trace consistency)
        self.expected = {
            'U_AB': 200.0,
            'U_BC': 300.0,
            'U_AC': 500.0,
            'phi_A': 200.0,
            'phi_C': -300.0,
            'E_PA': -6e-4,
            'E_PC': 9e-4
        }

    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        if a == 0 and b == 0:
            return True
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)

    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        success = True
        
        try:
            # 1. Work-Energy Theorem: W_ext = -W_field = ΔU
            W_AB_field = -self.W_AB_ext  # Field does negative work when charge is moved against it
            delta_U_AB_calc = -W_AB_field  # ΔU = -W_field
            
            # From results: ΔU_AB = E_PB - E_PA = 0 - E_PA = -E_PA
            delta_U_AB_from_results = -self.E_PA
            
            if not self._is_close(delta_U_AB_calc, delta_U_AB_from_results):
                logs.append(f"Work-Energy Theorem violation for A→B: ΔU_AB calculated {delta_U_AB_calc} ≠ {delta_U_AB_from_results} from results")
                success = False
            
            # 2. Potential difference definition: U_AB = φ_A - φ_B
            U_AB_from_potentials = self.phi_A - self.phi_B
            if not self._is_close(self.U_AB, U_AB_from_potentials):
                logs.append(f"Potential difference definition violation: U_AB = φ_A - φ_B gives {U_AB_from_potentials} ≠ {self.U_AB}")
                success = False
            
            # 3. Work from field: W_BC_elec = q * (φ_B - φ_C) = q * (-U_BC)
            W_BC_calc = self.q * (self.phi_B - self.phi_C)
            if not self._is_close(self.W_BC_elec, W_BC_calc):
                logs.append(f"Electric work mismatch for B→C: calculated {W_BC_calc} ≠ given {self.W_BC_elec}")
                success = False
            
            # 4. Additivity of potential differences
            U_AC_calc = self.U_AB + self.U_BC
            if not self._is_close(self.U_AC, U_AC_calc):
                logs.append(f"Additivity violation: U_AB + U_BC = {U_AC_calc} ≠ U_AC = {self.U_AC}")
                success = False
            
            # 5. Potential energy definition: E_P = q * φ
            E_PA_calc = self.q * self.phi_A
            E_PC_calc = self.q * self.phi_C
            if not self._is_close(self.E_PA, E_PA_calc):
                logs.append(f"Potential energy definition violation at A: q*φ_A = {E_PA_calc} ≠ {self.E_PA}")
                success = False
            if not self._is_close(self.E_PC, E_PC_calc):
                logs.append(f"Potential energy definition violation at C: q*φ_C = {E_PC_calc} ≠ {self.E_PC}")
                success = False
                
        except Exception as e:
            logs.append(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with steps analysis."""
        logs = []
        success = True
        
        try:
            # Check each result against expected values from steps
            for key, expected in self.expected.items():
                actual = getattr(self, key)
                if not self._is_close(actual, expected):
                    logs.append(f"Trace inconsistency: {key} = {actual} ≠ expected {expected}")
                    success = False
            
            # Additional consistency: U_AB from work definition
            U_AB_from_work = -self.W_AB_ext / self.q  # W_ext = -q*U_AB → U_AB = -W_ext/q
            if not self._is_close(self.U_AB, U_AB_from_work):
                logs.append(f"U_AB inconsistency: from work definition {U_AB_from_work} ≠ {self.U_AB}")
                success = False
            
            # U_BC from work definition (W_field = q*U_BC)
            U_BC_from_work = self.W_BC_elec / self.q
            if not self._is_close(self.U_BC, U_BC_from_work):
                logs.append(f"U_BC inconsistency: from work definition {U_BC_from_work} ≠ {self.U_BC}")
                success = False
                
        except Exception as e:
            logs.append(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Perturb charge magnitude and verify proportional trends."""
        logs = []
        success = True
        
        try:
            # Perturb: double the charge magnitude (more negative)
            q_perturbed = self.q * 2.0  # -6e-6 C
            
            # Recalculate using formulas from KB theorems
            # 1. Potential differences should remain SAME (independent of test charge)
            U_AB_perturbed = -self.W_AB_ext / q_perturbed
            U_BC_perturbed = self.W_BC_elec / q_perturbed
            
            # Check if U_AB scales inversely with charge
            expected_U_AB_ratio = self.q / q_perturbed
            actual_U_AB_ratio = U_AB_perturbed / self.U_AB if self.U_AB != 0 else 0
            if not self._is_close(expected_U_AB_ratio, actual_U_AB_ratio):
                logs.append(f"Monotonicity violation: U_AB doesn't scale inversely with q. Ratio expected {expected_U_AB_ratio}, got {actual_U_AB_ratio}")
                success = False
            
            # 2. Potential energies should scale linearly with charge
            # Using φ values from original (potentials are field properties, independent of test charge)
            E_PA_perturbed = q_perturbed * self.phi_A
            expected_E_ratio = q_perturbed / self.q
            actual_E_ratio = E_PA_perturbed / self.E_PA if self.E_PA != 0 else 0
            if not self._is_close(expected_E_ratio, actual_E_ratio):
                logs.append(f"Monotonicity violation: E_PA doesn't scale linearly with q. Ratio expected {expected_E_ratio}, got {actual_E_ratio}")
                success = False
            
            # 3. Verify work scaling: W_ext ∝ |q| for fixed potential difference
            W_AB_ext_perturbed = -q_perturbed * self.U_AB
            expected_W_ratio = q_perturbed / self.q
            actual_W_ratio = W_AB_ext_perturbed / self.W_AB_ext if self.W_AB_ext != 0 else 0
            if not self._is_close(expected_W_ratio, actual_W_ratio):
                logs.append(f"Monotonicity violation: External work doesn't scale with q. Ratio expected {expected_W_ratio}, got {actual_W_ratio}")
                success = False
                
        except Exception as e:
            logs.append(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                if not test_success:
                    all_logs.append(f"FAILED: {test_name}")
                    all_logs.extend(test_logs)
                    overall_success = False
                else:
                    all_logs.append(f"PASSED: {test_name}")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        return overall_success, all_logs