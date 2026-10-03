import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize checker with safe attribute access from solution instance."""
        self.solution = solution_instance
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem
        self.a = 2.7e-4  # atm^-1
        self.p = 10.0  # GeV/c (momentum)
        
        # K values from ground truth
        self.K_p = 0.094
        self.K_k = 0.05
        self.K_pi = 0.014
        
        # Convert angles from degrees to radians
        self.deg_to_rad = math.pi / 180
        
    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Check that K is positive and reasonable
            if 'K' in self.results:
                K = self.results['K']
                if 0.04 < K < 0.06:  # Should be around 0.05 for kaons
                    logs.append("✓ K value is reasonable (0.05 for kaons)")
                else:
                    logs.append(f"✗ K value {K:.3f} outside expected range 0.04-0.06")
            
            # 2. Check T_min calculation for n=1.33
            if 'T_min' in self.results:
                n_water = 1.33
                beta_min = 1 / n_water
                gamma_min = 1 / math.sqrt(1 - beta_min**2)
                T_min_calc = gamma_min - 1  # in units of mc²
                
                if math.isclose(self.results['T_min'], T_min_calc, rel_tol=0.01, abs_tol=0.01):
                    logs.append("✓ T_min calculation correct for n=1.33")
                else:
                    logs.append(f"✗ T_min mismatch: {self.results['T_min']:.3f} vs {T_min_calc:.3f}")
            
            # 3. Check n is close to 1.0016
            if 'n' in self.results:
                n = self.results['n']
                if 1.001 < n < 1.002:
                    logs.append("✓ n value is reasonable (~1.0016)")
                else:
                    logs.append(f"✗ n value {n:.5f} outside expected range 1.001-1.002")
            
            # 4. Check delta_p is positive and around 0.3
            if 'delta_p' in self.results:
                delta_p = self.results['delta_p']
                if 0.2 < delta_p < 0.4:
                    logs.append("✓ delta_p is reasonable (~0.3 GeV/c)")
                else:
                    logs.append(f"✗ delta_p value {delta_p:.3f} outside expected range 0.2-0.4")
            
            # 5. Check pressure formula consistency
            if all(key in self.results for key in ['K', 'n']):
                K = self.results['K']
                # Compute ζ_1/2 from K values
                zeta_calc = (4 * self.K_k**2 - self.K_pi**2) / 6
                n_calc = 1 + zeta_calc
                
                if math.isclose(self.results['n'], n_calc, rel_tol=0.01, abs_tol=0.01):
                    logs.append("✓ n consistent with K values through ζ_1/2 formula")
                else:
                    logs.append(f"✗ n mismatch: {self.results['n']:.5f} vs calculated {n_calc:.5f}")
            
        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check consistency between intermediate and final results."""
        logs = []
        
        try:
            # 1. Check ζ_1/2 calculation
            zeta_calc = (4 * self.K_k**2 - self.K_pi**2) / 6
            logs.append(f"ζ_1/2 from formula: {zeta_calc:.6f}")
            
            # 2. Check P_1/2 = ζ_1/2 / a
            P_half_calc = zeta_calc / self.a
            logs.append(f"P_1/2 calculated: {P_half_calc:.2f} atm")
            
            # 3. Check n = 1 + ζ_1/2
            n_calc = 1 + zeta_calc
            logs.append(f"n calculated: {n_calc:.5f}")
            
            # 4. Check angles would be approximately correct
            # Compute β from K: β = 1/√(1+K²)
            beta_k = 1 / math.sqrt(1 + self.K_k**2)
            beta_pi = 1 / math.sqrt(1 + self.K_pi**2)
            
            # Compute θ = acos(1/(β n))
            if 'n' in self.results:
                n_val = self.results['n']
                theta_k = math.acos(1/(beta_k * n_val))
                theta_pi = math.acos(1/(beta_pi * n_val))
                theta_k_deg = theta_k * 180 / math.pi
                theta_pi_deg = theta_pi * 180 / math.pi
                
                logs.append(f"θ_κ from n={n_val:.5f}: {theta_k_deg:.2f}°")
                logs.append(f"θ_π from n={n_val:.5f}: {theta_pi_deg:.2f}°")
                logs.append(f"Ratio θ_π/θ_κ: {theta_pi_deg/theta_k_deg:.2f}")
            
            # 5. Check Δθ/Δp calculations
            if 'n' in self.results:
                n_val = self.results['n']
                theta_k = math.acos(1/(beta_k * n_val))
                theta_pi = math.acos(1/(beta_pi * n_val))
                
                delta_theta_k_delta_p = self.K_k**2 / (theta_k * self.p)
                delta_theta_pi_delta_p = self.K_pi**2 / (theta_pi * self.p)
                
                delta_theta_k_delta_p_deg = delta_theta_k_delta_p * 180 / math.pi
                delta_theta_pi_delta_p_deg = delta_theta_pi_delta_p * 180 / math.pi
                
                logs.append(f"Δθ_κ/Δp: {delta_theta_k_delta_p_deg:.3f} deg/(GeV/c)")
                logs.append(f"Δθ_π/Δp: {delta_theta_pi_delta_p_deg:.3f} deg/(GeV/c)")
                
                # Total
                total = delta_theta_k_delta_p_deg + delta_theta_pi_delta_p_deg
                logs.append(f"Total Δθ/Δp: {total:.3f} deg/(GeV/c)")
            
            # 6. Check Δp calculation
            if 'n' in self.results:
                n_val = self.results['n']
                theta_k = math.acos(1/(beta_k * n_val))
                theta_pi = math.acos(1/(beta_pi * n_val))
                theta_k_deg = theta_k * 180 / math.pi
                theta_pi_deg = theta_pi * 180 / math.pi
                
                delta_theta_max = (theta_pi_deg - theta_k_deg) / 10
                
                delta_theta_k_delta_p = self.K_k**2 / (theta_k * self.p)
                delta_theta_pi_delta_p = self.K_pi**2 / (theta_pi * self.p)
                delta_theta_total_delta_p = (delta_theta_k_delta_p + delta_theta_pi_delta_p) * 180 / math.pi
                
                if delta_theta_total_delta_p != 0:
                    delta_p_calc = delta_theta_max / delta_theta_total_delta_p
                    logs.append(f"Δp calculated: {delta_p_calc:.3f} GeV/c")
                    
                    if 'delta_p' in self.results:
                        if math.isclose(self.results['delta_p'], delta_p_calc, rel_tol=0.1, abs_tol=0.1):
                            logs.append("✓ delta_p consistent with calculation")
                        else:
                            logs.append(f"✗ delta_p mismatch: {self.results['delta_p']:.3f} vs {delta_p_calc:.3f}")
            
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test monotonic trends by perturbing parameters - CORRECTED VERSION."""
        logs = []
        
        try:
            # 1. Test that T_min actually decreases when n increases (correct physics)
            if 'T_min' in self.results:
                # Test with n values around 1.33
                n_test1 = 1.33
                n_test2 = 1.5
                
                beta_min1 = 1 / n_test1
                beta_min2 = 1 / n_test2
                
                gamma_min1 = 1 / math.sqrt(1 - beta_min1**2)
                gamma_min2 = 1 / math.sqrt(1 - beta_min2**2)
                
                T_min1 = gamma_min1 - 1
                T_min2 = gamma_min2 - 1
                
                # T_min should DECREASE when n increases from 1.33 to 1.5
                if T_min2 < T_min1:
                    logs.append("✓ T_min decreases when n increases (correct physics)")
                else:
                    logs.append(f"✗ T_min should decrease when n increases, but T_min1={T_min1:.3f}, T_min2={T_min2:.3f}")
            
            # 2. Test pressure vs refractive index (should increase with n)
            if 'n' in self.results:
                n_val = self.results['n']
                P_half = (n_val - 1) / self.a
                
                # Perturb n by +1%
                n_perturbed = n_val * 1.01
                P_half_perturbed = (n_perturbed - 1) / self.a
                
                if P_half_perturbed > P_half:
                    logs.append("✓ Pressure increases with refractive index (P_1/2 ∝ n-1)")
                else:
                    logs.append(f"✗ P_1/2 should increase with n")
            
            # 3. Test Cherenkov angle vs momentum - CORRECTED: θ should INCREASE with p
            if 'n' in self.results:
                n_val = self.results['n']
                K_k = self.K_k
                
                # Original momentum p = 10 GeV/c gives β = 1/√(1+K²)
                beta_original = 1 / math.sqrt(1 + K_k**2)
                theta_original = math.acos(1/(beta_original * n_val))
                
                # Increase momentum by 10% (K decreases proportionally)
                p_perturbed = self.p * 1.1
                K_perturbed = self.K_k * (self.p / p_perturbed)  # K = mc/p
                beta_perturbed = 1 / math.sqrt(1 + K_perturbed**2)
                
                if beta_perturbed * n_val > 1:  # Cherenkov condition
                    theta_perturbed = math.acos(1/(beta_perturbed * n_val))
                    
                    # CORRECTED: θ should INCREASE with p (faster particles have larger angles)
                    if theta_perturbed > theta_original:
                        logs.append("✓ Cherenkov angle increases with increasing momentum (correct physics)")
                    else:
                        logs.append(f"✗ θ should increase when p increases, but θ_original={theta_original:.4f}, θ_perturbed={theta_perturbed:.4f}")
                else:
                    logs.append("✓ Cherenkov condition no longer satisfied with increased p")
            
        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        # Run each test suite
        for test_name, test_func in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for failures (lines starting with ✗)
            if any(log.startswith('✗') for log in test_logs):
                success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if success:
            all_logs.append("✓ All physics checks passed within tolerances")
        else:
            all_logs.append("✗ Some physics checks failed - review discrepancies")
        
        return success, all_logs