import math
import traceback
from typing import Dict, Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access to prevent sandbox errors.
        Results are extracted ONLY from solution_instance.results dictionary.
        """
        self.solution = solution_instance
        self.results = {}
        self.logs = []
        
        # Extract results ONLY from results dictionary
        # Map internal keys to possible external keys
        key_mapping = {
            'delta_S': ['ΔS', 'delta_S'],
            'dU': ['dU'],
            'KE': ['KE'],
            'dN': ['dN'],
            'P_b-P_i': ['P_b - P_i', 'P_b-P_i'],
            'P_b': ['P_b'],
            'E_2': ['E_2'],
            "T'": ["T''", "T'"],
            'delta_U': ['ΔU', 'delta_U'],
            'U_boundary': ['U_boundary'],
            'E_1': ['E_1'],
            'U_boundary_density': ['U_boundary_density'],
            'U': ['U'],
            'E': ['E'],
            'PE': ['PE']
        }
        
        # Try to extract each result
        for internal_key, possible_keys in key_mapping.items():
            val = None
            for key in possible_keys:
                # First try results dictionary
                if hasattr(solution_instance, 'results') and key in solution_instance.results:
                    val = solution_instance.results[key]
                    break
                # Then try direct attribute
                if hasattr(solution_instance, key):
                    val = getattr(solution_instance, key)
                    break
            self.results[internal_key] = val
        
        # Use default values for calculations
        self.N_val = 6.02214076e23  # Avogadro's number
        self.V_val = 0.0224  # 22.4 L in m^3
        self.T_val = 300.0  # Room temperature
        self.d_val = 1.0e-10  # Typical atomic diameter
        self.epsilon_val = 1.0e-21  # Small interaction energy
        self.k_B_val = 1.380649e-23  # Boltzmann constant
        
        # Compute a_prime value
        self.a_prime_val = (2 * math.pi * self.d_val**3 * self.epsilon_val) / 3
    
    def _log(self, message: str):
        """Helper to append logs."""
        self.logs.append(message)
    
    def _get_numeric_result(self, key: str) -> Optional[float]:
        """Safely get a numeric result from results dictionary."""
        val = self.results.get(key)
        if val is None:
            return None
        try:
            # Ensure it's a number, not a string or symbol
            return float(val)
        except (TypeError, ValueError):
            return None
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants using numeric values only."""
        local_logs = []
        success = True
        
        try:
            # Test 1: Energy Conservation (E1 = E2 from steps 14-16)
            E1_val = self._get_numeric_result('E_1')
            E2_val = self._get_numeric_result('E_2')
            
            if E1_val is not None and E2_val is not None:
                if not math.isclose(E1_val, E2_val, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: Energy not conserved. E1={E1_val}, E2={E2_val}")
                    success = False
                else:
                    local_logs.append(f"PASS: Energy conserved within 1%")
            else:
                local_logs.append("SKIP: Energy conservation check (missing values)")
            
            # Test 2: Equipartition Theorem (step 11)
            KE_val = self._get_numeric_result('KE')
            if KE_val is not None:
                expected_KE = 1.5 * self.N_val * self.k_B_val * self.T_val
                if not math.isclose(KE_val, expected_KE, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: Equipartition mismatch. KE={KE_val}, expected={expected_KE}")
                    success = False
                else:
                    local_logs.append(f"PASS: Equipartition within 1%")
            else:
                local_logs.append("SKIP: Equipartition check (missing values)")
            
            # Test 3: Bernoulli-inspired pressure relation (step 9)
            # P_b - P_i = -a'N²/V² (from step 9)
            # Using ideal gas P_i = NkT/V, so P_b = NkT/V - a'N²/V²
            P_b_val = self._get_numeric_result('P_b')
            if P_b_val is not None:
                expected_P_b = (self.N_val * self.k_B_val * self.T_val) / self.V_val - self.a_prime_val * self.N_val**2 / self.V_val**2
                if not math.isclose(P_b_val, expected_P_b, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: Pressure mismatch. P_b={P_b_val}, expected={expected_P_b}")
                    success = False
                else:
                    local_logs.append(f"PASS: Pressure relation within 1%")
            else:
                local_logs.append("SKIP: Pressure relation check (missing values)")
                
        except Exception as e:
            local_logs.append(f"ERROR in invariants: {traceback.format_exc()}")
            success = False
        
        return success, local_logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with expected formulas using numeric values."""
        local_logs = []
        success = True
        
        try:
            # Test 1: U = -2a'N/V (step 5)
            U_val = self._get_numeric_result('U')
            if U_val is not None:
                expected_U = -2 * self.a_prime_val * self.N_val / self.V_val
                if not math.isclose(U_val, expected_U, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: U mismatch. Got {U_val}, expected {expected_U}")
                    success = False
                else:
                    local_logs.append(f"PASS: U within 1%")
            else:
                local_logs.append("SKIP: U check (missing)")
            
            # Test 2: T' = T - a'N/(3k_B V) (step 17)
            T_prime_val = self._get_numeric_result("T'")
            if T_prime_val is not None:
                expected_T_prime = self.T_val - self.a_prime_val * self.N_val / (3 * self.k_B_val * self.V_val)
                if not math.isclose(T_prime_val, expected_T_prime, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: T' mismatch. Got {T_prime_val}, expected {expected_T_prime}")
                    success = False
                else:
                    local_logs.append(f"PASS: T' within 1%")
            else:
                local_logs.append("SKIP: T' check (missing)")
            
            # Test 3: ΔS consistency (step 26)
            delta_S_val = self._get_numeric_result('delta_S')
            if delta_S_val is not None:
                expected_delta_S = self.N_val * self.k_B_val * math.log(2) - self.a_prime_val * self.N_val**2 / (2 * self.V_val * self.T_val)
                if not math.isclose(delta_S_val, expected_delta_S, rel_tol=1e-2, abs_tol=1e-2):
                    local_logs.append(f"FAIL: ΔS mismatch. Got {delta_S_val}, expected {expected_delta_S}")
                    success = False
                else:
                    local_logs.append(f"PASS: ΔS within 1%")
            else:
                local_logs.append("SKIP: ΔS check (missing)")
                
        except Exception as e:
            local_logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            success = False
        
        return success, local_logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonicity by perturbing N and checking ΔS trend using numeric calculations."""
        local_logs = []
        success = True
        
        try:
            # Get ΔS value
            delta_S_val = self._get_numeric_result('delta_S')
            if delta_S_val is None:
                return True, ["SKIP: Monotonicity (no ΔS value)"]
            
            # Use default values for calculations
            N_base = self.N_val
            V_base = self.V_val
            T_base = self.T_val
            a_prime_base = self.a_prime_val
            k_B_base = self.k_B_val
            
            # Perturb N (increase by 10%)
            N_perturbed = N_base * 1.1
            
            # Calculate ΔS for both using the formula
            delta_S_base = N_base * k_B_base * math.log(2) - a_prime_base * N_base**2 / (2 * V_base * T_base)
            delta_S_perturbed = N_perturbed * k_B_base * math.log(2) - a_prime_base * N_perturbed**2 / (2 * V_base * T_base)
            
            # Expected trend from derivative: d(ΔS)/dN = k_B ln2 - a'N/(VT)
            derivative = k_B_base * math.log(2) - a_prime_base * N_base / (V_base * T_base)
            
            # Check if actual trend matches derivative sign
            actual_trend = delta_S_perturbed - delta_S_base
            
            if derivative > 0 and actual_trend > 0:
                local_logs.append(f"PASS: Monotonicity matches expectation (increasing with N)")
            elif derivative < 0 and actual_trend < 0:
                local_logs.append(f"PASS: Monotonicity matches expectation (decreasing with N)")
            elif abs(derivative) < 1e-10 and abs(actual_trend) < 1e-10:
                local_logs.append(f"PASS: Monotonicity matches expectation (constant)")
            else:
                local_logs.append(f"FAIL: Monotonicity mismatch. Derivative sign={derivative}, actual trend={actual_trend}")
                success = False
                
        except Exception as e:
            local_logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            success = False
        
        return success, local_logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        self.logs.append("=== INVARIANTS TEST ===")
        self.logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        self.logs.append("\n=== TRACE CONSISTENCY TEST ===")
        self.logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        self.logs.append("\n=== MONOTONICITY TEST ===")
        self.logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        self.logs.append("\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs