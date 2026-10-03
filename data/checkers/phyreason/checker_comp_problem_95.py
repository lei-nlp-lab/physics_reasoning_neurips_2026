import math
import traceback
from typing import Tuple, List, Any, Dict, Optional

class PhysicsChecker:
    """Physics auditor for proton-photon collision problem."""
    
    def __init__(self, solution_instance: Any) -> None:
        """Initialize checker with safe attribute access."""
        self.solution = solution_instance
        self.logs: List[str] = []
        
        # Safe extraction of inputs and results
        self.c = 3.00e8  # m/s (from constants)
        
        # Extract masses (in eV/c² units from constants)
        self.m_p_eV = 938e6  # MeV to eV
        self.m_delta_eV = 1232e6  # MeV to eV
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Energy and momentum quantities - use results dict directly
        self.E_b = results.get('E', 0.00117)  # eV
        self.E_p = results.get('E_p', 0.0)
        self.p_p = 0.0  # Will compute if needed
        
        # Rest frame photon energy
        self.E_gamma_rest = results.get('E_γ', 0.0)
        
        # Delta particle quantities
        self.E_delta = results.get('E_delta_', 0.0)
        self.p_delta = results.get('p_delta_', 0.0)
        self.E_delta_sq = results.get('E_delta_2', 0.0)
        self.p_delta_sq = results.get('p_delta_2', 0.0)
        
        # Frame transformation parameters
        self.alpha = results.get('alpha', 0.0)
        self.beta = results.get('β', 0.0)
        self.gamma = results.get('γ', 0.0)
        
        # Derived quantity
        self.E_p_plus_p_p = results.get('E_p+p_p', 0.0)
        
        # Convert energies to consistent units (eV)
        if isinstance(self.E_p, str) and '×' in self.E_p:
            # Handle scientific notation strings like "1.4×10^20"
            self.E_p = float(self.E_p.replace('×10^', 'e').replace('×', 'e'))
        
        # Log extracted values
        self.logs.append(f"Extracted: m_p = {self.m_p_eV:.3e} eV/c², m_Δ = {self.m_delta_eV:.3e} eV/c²")
        self.logs.append(f"Extracted: E_b = {self.E_b:.3e} eV, E_p = {self.E_p:.3e} eV")
        self.logs.append(f"Extracted: E_γ(rest) = {self.E_gamma_rest:.3e}, α = {self.alpha:.3e}, β = {self.beta:.3e}, γ = {self.gamma:.3e}")
    
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching with physics tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test conservation laws and relativistic invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check rest frame photon energy formula (step 4)
            E_gamma_expected = (self.m_delta_eV**2 - self.m_p_eV**2) / (2 * self.m_p_eV)
            if not self._isclose(self.E_gamma_rest, E_gamma_expected, rel_tol=1e-2):
                logs.append(f"FAIL: E_γ(rest) mismatch. Got {self.E_gamma_rest:.3e}, expected {E_gamma_expected:.3e}")
                success = False
            else:
                logs.append(f"OK: E_γ(rest) = (m_Δ² - m_p²)/(2m_p) = {E_gamma_expected:.3e}")
            
            # 2. Check alpha definition (step 5)
            if self.E_gamma_rest > 0:
                alpha_expected = self.E_b / self.E_gamma_rest
                if not self._isclose(self.alpha, alpha_expected, rel_tol=1e-2):
                    logs.append(f"FAIL: α = E_b/E_γ mismatch. Got {self.alpha:.3e}, expected {alpha_expected:.3e}")
                    success = False
                else:
                    logs.append(f"OK: α = E_b/E_γ = {alpha_expected:.3e}")
            
            # 3. Check beta from alpha (step 6)
            if self.alpha > 0:
                beta_expected = (1 - self.alpha**2) / (1 + self.alpha**2)
                if not self._isclose(self.beta, beta_expected, rel_tol=1e-2):
                    logs.append(f"FAIL: β = (1-α²)/(1+α²) mismatch. Got {self.beta:.3e}, expected {beta_expected:.3e}")
                    success = False
                else:
                    logs.append(f"OK: β = (1-α²)/(1+α²) = {beta_expected:.3e}")
            
            # 4. Check gamma from beta (step 7)
            if self.beta < 1:
                gamma_expected = 1 / math.sqrt(1 - self.beta**2)
                if not self._isclose(self.gamma, gamma_expected, rel_tol=1e-2):
                    logs.append(f"FAIL: γ = 1/√(1-β²) mismatch. Got {self.gamma:.3e}, expected {gamma_expected:.3e}")
                    success = False
                else:
                    logs.append(f"OK: γ = 1/√(1-β²) = {gamma_expected:.3e}")
            
            # 5. Check energy-momentum relation for delta
            if self.E_delta_sq > 0 and self.p_delta_sq > 0:
                expected = self.p_delta_sq + self.m_delta_eV**2
                if not self._isclose(self.E_delta_sq, expected, rel_tol=1e-2):
                    logs.append(f"FAIL: E_Δ² ≠ p_Δ² + m_Δ². Got {self.E_delta_sq:.3e}, expected {expected:.3e}")
                    success = False
                else:
                    logs.append(f"OK: E_Δ² = p_Δ² + m_Δ² = {expected:.3e}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate steps and final answer."""
        logs = []
        success = True
        
        try:
            # 1. Check step 12: E_p + p_p = (m_Δ² - m_p²)/(2E_b)
            expected_sum = (self.m_delta_eV**2 - self.m_p_eV**2) / (2 * self.E_b)
            if not self._isclose(self.E_p_plus_p_p, expected_sum, rel_tol=1e-2):
                logs.append(f"FAIL: E_p + p_p mismatch. Got {self.E_p_plus_p_p:.3e}, expected {expected_sum:.3e}")
                success = False
            else:
                logs.append(f"OK: E_p + p_p = (m_Δ² - m_p²)/(2E_b) = {expected_sum:.3e}")
            
            # 2. Check step 9/13 approximation: E_p ≈ (m_Δ² - m_p²)/(4E_b)
            expected_approx = (self.m_delta_eV**2 - self.m_p_eV**2) / (4 * self.E_b)
            if not self._isclose(self.E_p, expected_approx, rel_tol=1e-2):
                logs.append(f"WARNING: E_p ≈ (m_Δ² - m_p²)/(4E_b) off by >1%. Got {self.E_p:.3e}, expected {expected_approx:.3e}")
                # Not a failure - it's an approximation
            else:
                logs.append(f"OK: E_p ≈ (m_Δ² - m_p²)/(4E_b) = {expected_approx:.3e}")
            
            # 3. Check step 15 numerical computation - use actual E_b from results
            expected_E_p_num = (1232e6**2 - 938e6**2) / (4 * self.E_b)
            if not self._isclose(self.E_p, expected_E_p_num, rel_tol=1e-2):
                logs.append(f"FAIL: Numerical E_p mismatch. Got {self.E_p:.3e}, expected {expected_E_p_num:.3e}")
                success = False
            else:
                logs.append(f"OK: Numerical E_p = (1232² - 938²)/(4×{self.E_b:.3e}) = {expected_E_p_num:.3e} eV")
            
            # 4. Check photon energy conversion (step 14)
            E_b_J = self.E_b * 1.602e-19  # Convert eV to J
            # Verify it's consistent with h*c/lambda relationship
            # (we don't have lambda, so we can't check this directly)
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test monotonic relationship: increasing E_b should decrease E_p."""
        logs = []
        success = True
        
        try:
            # Save original state
            E_b_original = self.E_b
            E_p_original = self.E_p
            
            # Perturb E_b by +10%
            E_b_perturbed = E_b_original * 1.1
            
            # Compute expected E_p using formula from step 9
            E_p_perturbed = (self.m_delta_eV**2 - self.m_p_eV**2) / (4 * E_b_perturbed)
            
            # Check monotonicity: larger E_b → smaller E_p
            if E_p_perturbed >= E_p_original:
                logs.append(f"FAIL: Monotonicity violated. E_b↑ from {E_b_original:.3e} to {E_b_perturbed:.3e} should decrease E_p")
                logs.append(f"  Got: E_p = {E_p_perturbed:.3e}, original = {E_p_original:.3e}")
                success = False
            else:
                logs.append(f"OK: Monotonicity holds. E_b↑ {E_b_original:.3e}→{E_b_perturbed:.3e} decreases E_p {E_p_original:.3e}→{E_p_perturbed:.3e}")
            
            # Check inverse proportionality roughly holds
            expected_ratio = E_b_original / E_b_perturbed
            actual_ratio = E_p_original / E_p_perturbed
            
            if not self._isclose(actual_ratio, expected_ratio, rel_tol=5e-2):  # 5% tolerance for approximation
                logs.append(f"WARNING: Inverse proportionality approximate. Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
            
            # Restore original values
            self.E_b = E_b_original
            self.E_p = E_p_original
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Restore on error too
            self.E_b = E_b_original
            self.E_p = E_p_original
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics checks and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: {test_name} FAILED")
                else:
                    all_logs.append(f"Result: {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall physics check: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"Final computed proton energy: E_p = {self.E_p:.3e} eV")
        all_logs.append(f"Expected order of magnitude: ~1.4×10²⁰ eV")
        
        # Check final answer against expected
        expected_min = 1.0e20  # Order of magnitude check
        expected_max = 2.0e20
        
        if self.E_p < expected_min or self.E_p > expected_max:
            all_logs.append(f"WARNING: Final E_p outside expected range [{expected_min:.1e}, {expected_max:.1e}]")
        
        return overall_success, all_logs