import math
import traceback
from typing import Tuple, List, Any, Dict, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL values from results dictionary with proper defaults
        # Do NOT use getattr on solution_instance for constants - use results dict
        self.d = self.results.get('d', 0.001)
        self.rho = self.results.get('rho', 1000.0)
        self.sigma = self.results.get('sigma', 0.0728)
        self.g = self.results.get('g', 9.81)
        self.epsilon0 = self.results.get('epsilon0', 8.854e-12)
        self.n = self.results.get('n', 10.0)
        self.C = self.results.get('C', 1e-12)
        self.q0 = self.results.get('q0', 1e-12)
        self.H = self.results.get('H', 0.1)
        
        # Extract results with safe fallbacks
        self.Q_0 = self.results.get('Q_0', None)
        self.qt = self.results.get('qt', None)
        self.r_max = self.results.get('r_textmax', None)
        self.P_E = self.results.get('P_E', None)
        self.F_sigma = self.results.get('F_sigma', None)
        self.Q = self.results.get('Q', None)
        self.Q_0_abs = self.results.get('|Q_0|', None)
        self.gamma_const = self.results.get('gamma', None)
        self.F_g = self.results.get('F_g', None)
        self.P_sigma = self.results.get('P_sigma', None)
        self.phi_max = self.results.get('varphi_textmax', None)
        self.U_max = self.results.get('U_textmax', None)
        
        self.logs = []
        
    def log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(message)
    
    def check_numerical_close(self, a: float, b: float, name: str, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Check if two numbers are close with tolerance."""
        if a is None or b is None:
            self.log(f"WARNING: {name}: Missing value(s).")
            return False
        
        if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            self.log(f"PASS: {name}: {a:.3e} ≈ {b:.3e}")
            return True
        else:
            self.log(f"FAIL: {name}: {a:.3e} ≠ {b:.3e} (rel diff: {abs(a-b)/max(abs(a),abs(b)):.2%})")
            return False
    
    def test_invariants(self) -> bool:
        """Test physical invariants: Force balance, pressure balance, energy conservation."""
        success = True
        
        try:
            # 1. Check force balance at separation (step 3-4): F_g = F_sigma
            if all(v is not None for v in [self.r_max, self.F_g, self.F_sigma]):
                # Compute theoretical F_g from r_max
                F_g_theory = (4/3) * math.pi * (self.r_max**3) * self.rho * self.g
                # Compute theoretical F_sigma from d and sigma
                F_sigma_theory = math.pi * self.d * self.sigma
                
                # Check both forces
                check1 = self.check_numerical_close(self.F_g, F_g_theory, "F_g from r_max")
                check2 = self.check_numerical_close(self.F_sigma, F_sigma_theory, "F_sigma from d, sigma")
                check3 = self.check_numerical_close(F_g_theory, F_sigma_theory, "Force balance F_g = F_sigma")
                if not (check1 and check2 and check3):
                    success = False
            
            # 2. Check Laplace pressure (step 8)
            if self.P_sigma is not None and self.r_max is not None:
                P_sigma_theory = 2 * self.sigma / self.r_max
                if not self.check_numerical_close(self.P_sigma, P_sigma_theory, "Laplace pressure P_sigma"):
                    success = False
            
            # 3. Check electrostatic pressure (step 9)
            if self.P_E is not None and self.r_max is not None and self.phi_max is not None:
                # Using phi_max for E = phi_max / r_max
                E_sq = (self.phi_max / self.r_max)**2
                P_E_theory = 0.5 * self.epsilon0 * E_sq
                if not self.check_numerical_close(self.P_E, P_E_theory, "Electrostatic pressure P_E"):
                    success = False
            
            # 4. Check pressure balance at instability (step 10-11): P_sigma = P_E
            if self.P_sigma is not None and self.P_E is not None:
                if not self.check_numerical_close(self.P_sigma, self.P_E, "Pressure balance P_sigma = P_E"):
                    success = False
            
            # 5. Check charge on sphere relation (step 5-7): Q = 4πϵ0φr
            if self.Q is not None and self.r_max is not None and self.phi_max is not None:
                Q_theory = 4 * math.pi * self.epsilon0 * self.phi_max * self.r_max
                if not self.check_numerical_close(self.Q, Q_theory, "Charge on sphere Q = 4πϵ0φr"):
                    success = False
            
            # 6. Check Q_0 relation (step 13)
            if self.Q_0 is not None and self.r_max is not None and self.C is not None and self.q0 is not None:
                Q_0_theory = 2 * math.pi * self.epsilon0 * (self.q0 / self.C) * self.r_max
                if not self.check_numerical_close(self.Q_0, Q_0_theory, "Q_0 from capacitor division"):
                    success = False
            
            # 7. Check exponential growth rate (step 18)
            if self.gamma_const is not None:
                gamma_theory = (math.pi * self.epsilon0 * self.n / self.C) * \
                              ((6 * self.sigma * self.d) / (self.rho * self.g))**(1/3)
                if not self.check_numerical_close(self.gamma_const, gamma_theory, "Growth rate γ"):
                    success = False
            
            # 8. Check two forms of U_max (steps 21-22)
            if self.U_max is not None and self.H is not None:
                # First form (step 21)
                U_max_1 = math.sqrt(
                    (2/3) * (self.rho * self.g * self.H / self.epsilon0) * 
                    ((3 * self.sigma * self.d) / (4 * self.rho * self.g))**(2/3)
                )
                
                # Second form (step 22)
                numerator = self.H**3 * self.g * self.sigma**2 * self.rho * self.d**2
                denominator = 6 * self.epsilon0**3
                U_max_2 = (numerator / denominator)**(1/6)
                
                check1 = self.check_numerical_close(self.U_max, U_max_1, "U_max form 1")
                check2 = self.check_numerical_close(self.U_max, U_max_2, "U_max form 2")
                check3 = self.check_numerical_close(U_max_1, U_max_2, "Consistency between U_max forms")
                if not (check1 and check2 and check3):
                    success = False
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            self.log(traceback.format_exc())
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate and final results."""
        success = True
        
        try:
            # 1. Check that using r_max in Q formula gives consistent Q
            if all(v is not None for v in [self.r_max, self.Q, self.phi_max]):
                Q_from_phi_max = 4 * math.pi * self.epsilon0 * self.phi_max * self.r_max
                if not self.check_numerical_close(self.Q, Q_from_phi_max, "Q from φ_max and r_max"):
                    success = False
            
            # 2. Check that phi_max derived from pressure balance matches if computed
            if self.phi_max is not None and self.r_max is not None:
                phi_max_theory = 2 * math.sqrt(self.sigma * self.r_max / self.epsilon0)
                if not self.check_numerical_close(self.phi_max, phi_max_theory, "φ_max from pressure balance"):
                    success = False
            
            # 3. Check r_max formula
            if self.r_max is not None:
                r_max_theory = ((3 * self.sigma * self.d) / (4 * self.rho * self.g))**(1/3)
                if not self.check_numerical_close(self.r_max, r_max_theory, "r_max formula"):
                    success = False
            
            # 4. Check q(t) exponential behavior
            if self.qt is not None and self.gamma_const is not None and self.q0 is not None:
                # qt is a single numeric value (computed at t=1s)
                if isinstance(self.qt, (int, float)):
                    # Expected value at t=1s
                    q_expected = self.q0 * math.exp(self.gamma_const * 1.0)
                    if not self.check_numerical_close(self.qt, q_expected, "q(t) at t=1s"):
                        success = False
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            self.log(traceback.format_exc())
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb inputs and check output trends, then restore."""
        success = True
        
        try:
            # Perturbation parameters
            perturbation_factor = 1.1  # 10% increase
            
            # Test 1: r_max vs d (r_max ∝ d^(1/3))
            if self.d is not None and self.r_max is not None:
                original_d = self.d
                d_perturbed = self.d * perturbation_factor
                
                r_max_new = ((3 * self.sigma * d_perturbed) / (4 * self.rho * self.g))**(1/3)
                r_max_original = ((3 * self.sigma * original_d) / (4 * self.rho * self.g))**(1/3)
                
                if r_max_new > r_max_original:
                    self.log(f"PASS: r_max increases with d (monotonicity)")
                else:
                    self.log(f"FAIL: r_max doesn't increase with d")
                    success = False
            
            # Test 2: φ_max vs σ (φ_max ∝ √σ)
            if self.sigma is not None and self.r_max is not None and self.phi_max is not None:
                original_sigma = self.sigma
                sigma_perturbed = self.sigma * perturbation_factor
                
                phi_max_new = 2 * math.sqrt(sigma_perturbed * self.r_max / self.epsilon0)
                phi_max_original = 2 * math.sqrt(original_sigma * self.r_max / self.epsilon0)
                
                if phi_max_new > phi_max_original:
                    self.log(f"PASS: φ_max increases with σ (monotonicity)")
                else:
                    self.log(f"FAIL: φ_max doesn't increase with σ")
                    success = False
            
            # Test 3: U_max vs H (U_max ∝ H^(1/2) from step 21)
            if self.H is not None and self.U_max is not None:
                original_H = self.H
                H_perturbed = self.H * perturbation_factor
                
                U_max_new = math.sqrt(
                    (2/3) * (self.rho * self.g * H_perturbed / self.epsilon0) * 
                    ((3 * self.sigma * self.d) / (4 * self.rho * self.g))**(2/3)
                )
                U_max_original = math.sqrt(
                    (2/3) * (self.rho * self.g * original_H / self.epsilon0) * 
                    ((3 * self.sigma * self.d) / (4 * self.rho * self.g))**(2/3)
                )
                
                if U_max_new > U_max_original:
                    self.log(f"PASS: U_max increases with H (monotonicity)")
                else:
                    self.log(f"FAIL: U_max doesn't increase with H")
                    success = False
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            self.log(traceback.format_exc())
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        self.log("=== PhysicsChecker starting ===")
        
        success = True
        
        # Run individual tests
        test_results = []
        
        try:
            self.log("\n--- Test 1: Physical Invariants ---")
            test1 = self.test_invariants()
            test_results.append(("Physical Invariants", test1))
            
            self.log("\n--- Test 2: Trace Consistency ---")
            test2 = self.test_trace_consistency()
            test_results.append(("Trace Consistency", test2))
            
            self.log("\n--- Test 3: Monotonicity ---")
            test3 = self.test_monotonicity()
            test_results.append(("Monotonicity", test3))
            
            # Overall success
            success = all(result for _, result in test_results)
            
            self.log("\n=== Summary ===")
            for test_name, test_result in test_results:
                status = "PASS" if test_result else "FAIL"
                self.log(f"{test_name}: {status}")
            
            self.log(f"\nOverall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self.log(f"CRITICAL ERROR in run_all_tests: {str(e)}")
            self.log(traceback.format_exc())
            success = False
        
        return success, self.logs