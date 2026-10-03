import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract all inputs and results without direct attribute access.
        """
        # Input parameters
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 0.0))
        self.beta = getattr(solution_instance, 'beta', solution_instance.results.get('beta', 0.0))
        self.c = getattr(solution_instance, 'c', 3.0e8)  # default speed of light
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 1.0e9))  # default frequency
        
        # Results from solution (using mapping provided)
        self.v_x_bob = getattr(solution_instance, 'v_x_bob', 
                              solution_instance.results.get('v_x_bob', 0.0))
        self.v_y_bob = getattr(solution_instance, 'v_y_bob', 
                              solution_instance.results.get('v_y_bob', 0.0))
        self.delta_t = getattr(solution_instance, 'delta_t', 
                              solution_instance.results.get('delta_t', 0.0))
        self.delta_t_total = getattr(solution_instance, 'delta_t_total', 
                                    solution_instance.results.get('delta_t_total', 0.0))
        self.costheta = getattr(solution_instance, 'costheta', 
                               solution_instance.results.get('costheta', 0.0))
        self.theta = getattr(solution_instance, 'theta', 
                            solution_instance.results.get('theta', 0.0))  # in degrees
        self.f_prime = getattr(solution_instance, 'f_prime', 
                              solution_instance.results.get('f_prime', 0.0))
        
        # Store logs
        self.logs = []

    def _log(self, message: str, success: bool) -> None:
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """
        Verify physics invariants:
        1. Speed of light invariance: muon speed in Bob's frame should be c
        2. Lorentz factor monotonicity with beta
        3. Check that delta_t_total formula matches derived expression
        """
        success = True
        
        # 1. Speed of light invariance
        v_bob = math.sqrt(self.v_x_bob**2 + self.v_y_bob**2)
        if not self._is_close(v_bob, self.c):
            self._log(f"Muon speed in Bob's frame = {v_bob:.2e} ≠ c = {self.c:.2e}", False)
            success = False
        else:
            self._log(f"Muon speed in Bob's frame = {v_bob:.2e} ≈ c", True)
        
        # 2. Lorentz factor monotonicity
        gamma = 1.0 / math.sqrt(1 - self.beta**2)
        if self.beta >= 0 and self.beta < 1:
            # gamma should be > 1 for beta > 0
            if self.beta > 0 and gamma <= 1:
                self._log(f"Gamma = {gamma:.3f} should be > 1 for beta = {self.beta:.3f}", False)
                success = False
            else:
                self._log(f"Gamma = {gamma:.3f} > 1 for beta = {self.beta:.3f}", True)
        
        # 3. Check delta_t_total formula consistency (not time dilation direction)
        # The formula is: Δt_total = (h/c) * (1+β cosθ)/√(1-β²)
        expected_total = (self.h / self.c) * (1 + self.beta * self.costheta) / math.sqrt(1 - self.beta**2)
        if not self._is_close(self.delta_t_total, expected_total):
            self._log(f"Δt_total = {self.delta_t_total:.2e} ≠ formula = {expected_total:.2e}", False)
            success = False
        else:
            self._log(f"Δt_total = {self.delta_t_total:.2e} matches formula", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Based on steps analysis.
        """
        success = True
        
        # Step 1: v_x_bob = βc
        expected_vx = self.beta * self.c
        if not self._is_close(self.v_x_bob, expected_vx):
            self._log(f"v_x_bob = {self.v_x_bob:.2e} ≠ βc = {expected_vx:.2e}", False)
            success = False
        else:
            self._log(f"v_x_bob = {self.v_x_bob:.2e} = βc", True)
        
        # Step 3: v_y_bob = c√(1-β²)
        expected_vy = self.c * math.sqrt(1 - self.beta**2)
        if not self._is_close(self.v_y_bob, expected_vy):
            self._log(f"v_y_bob = {self.v_y_bob:.2e} ≠ c√(1-β²) = {expected_vy:.2e}", False)
            success = False
        else:
            self._log(f"v_y_bob = {self.v_y_bob:.2e} = c√(1-β²)", True)
        
        # Step 5: Δt = h/(c√(1-β²))
        expected_dt = self.h / (self.c * math.sqrt(1 - self.beta**2))
        if not self._is_close(self.delta_t, expected_dt):
            self._log(f"Δt = {self.delta_t:.2e} ≠ h/(c√(1-β²)) = {expected_dt:.2e}", False)
            success = False
        else:
            self._log(f"Δt = {self.delta_t:.2e} = h/(c√(1-β²))", True)
        
        # Step 7: Δt_total = (h/c) * (1+β cosθ)/√(1-β²)
        expected_total = (self.h / self.c) * (1 + self.beta * self.costheta) / math.sqrt(1 - self.beta**2)
        if not self._is_close(self.delta_t_total, expected_total):
            self._log(f"Δt_total = {self.delta_t_total:.2e} ≠ (h/c)(1+β cosθ)/√(1-β²) = {expected_total:.2e}", False)
            success = False
        else:
            self._log(f"Δt_total = {self.delta_t_total:.2e} matches relativistic expression", True)
        
        # Step 10: cosθ = (√(1-β²)-1)/β
        if abs(self.beta) > 1e-9:  # avoid division by zero
            expected_costheta = (math.sqrt(1 - self.beta**2) - 1) / self.beta
            if not self._is_close(self.costheta, expected_costheta):
                self._log(f"cosθ = {self.costheta:.4f} ≠ (√(1-β²)-1)/β = {expected_costheta:.4f}", False)
                success = False
            else:
                self._log(f"cosθ = {self.costheta:.4f} matches derived formula", True)
        
        # Step 11: For β=1/2, θ ≈ 105.5°
        if self._is_close(self.beta, 0.5, abs_tol=1e-3):
            if not self._is_close(self.theta, 105.5, rel_tol=1e-2):
                self._log(f"θ = {self.theta:.1f}° ≠ 105.5° for β=0.5", False)
                success = False
            else:
                self._log(f"θ = {self.theta:.1f}° ≈ 105.5° for β=0.5", True)
        
        # Step 13: f' = f * √(1-β²)/(1+β cosθ)
        expected_fprime = self.f * math.sqrt(1 - self.beta**2) / (1 + self.beta * self.costheta)
        if not self._is_close(self.f_prime, expected_fprime, rel_tol=1e-2):
            self._log(f"f' = {self.f_prime:.2e} ≠ f√(1-β²)/(1+β cosθ) = {expected_fprime:.2e}", False)
            success = False
        else:
            self._log(f"f' = {self.f_prime:.2e} matches Doppler formula", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb β and check monotonic trends:
        1. As β increases, Lorentz factor gamma increases
        2. As β increases, cosθ decreases (for β>0)
        3. As β increases, Doppler factor changes appropriately
        """
        success = True
        
        # Store original values
        original_beta = self.beta
        perturbed_beta = original_beta * 1.1  # 10% increase
        
        # Calculate perturbed quantities using formulas
        # 1. Lorentz factor monotonicity
        gamma_original = 1.0 / math.sqrt(1 - original_beta**2)
        gamma_perturbed = 1.0 / math.sqrt(1 - perturbed_beta**2)
        
        if gamma_perturbed <= gamma_original:
            self._log(f"Gamma non-monotonic: γ({perturbed_beta:.3f}) = {gamma_perturbed:.3f} ≤ γ({original_beta:.3f}) = {gamma_original:.3f}", False)
            success = False
        else:
            self._log(f"Gamma monotonic: γ({perturbed_beta:.3f}) = {gamma_perturbed:.3f} > γ({original_beta:.3f}) = {gamma_original:.3f}", True)
        
        # 2. Angle θ (only if β > 0)
        if original_beta > 1e-9:
            costheta_original = (math.sqrt(1 - original_beta**2) - 1) / original_beta
            costheta_perturbed = (math.sqrt(1 - perturbed_beta**2) - 1) / perturbed_beta
            
            # Since cosθ decreases as β increases, θ should increase
            if costheta_perturbed >= costheta_original:
                self._log(f"cosθ non-monotonic: cosθ({perturbed_beta:.3f}) = {costheta_perturbed:.4f} ≥ cosθ({original_beta:.3f}) = {costheta_original:.4f}", False)
                success = False
            else:
                self._log(f"cosθ monotonic: cosθ({perturbed_beta:.3f}) = {costheta_perturbed:.4f} < cosθ({original_beta:.3f}) = {costheta_original:.4f}", True)
        
        # 3. Doppler shift (for fixed f and fixed cosθ from original solution)
        if abs(original_beta) > 1e-9:
            # Use the original cosθ value
            fprime_original = math.sqrt(1 - original_beta**2) / (1 + original_beta * self.costheta)
            fprime_perturbed = math.sqrt(1 - perturbed_beta**2) / (1 + perturbed_beta * self.costheta)
            
            # For source moving away (cosθ negative), f' should decrease as β increases
            if fprime_perturbed >= fprime_original:
                self._log(f"Doppler shift non-monotonic: f'/f({perturbed_beta:.3f}) = {fprime_perturbed:.4f} ≥ f'/f({original_beta:.3f}) = {fprime_original:.4f}", False)
                success = False
            else:
                self._log(f"Doppler shift monotonic: f'/f({perturbed_beta:.3f}) = {fprime_perturbed:.4f} < f'/f({original_beta:.3f}) = {fprime_original:.4f}", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics audits and return success flag with logs.
        """
        self.logs = ["=== Physics Audit Report ==="]
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Trends", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs