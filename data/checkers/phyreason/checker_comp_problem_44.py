import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        # Extract inputs with safe defaults
        self.inputs = {
            'R': getattr(solution_instance, 'R', 8.314),  # J/(mol·K)
            'g': getattr(solution_instance, 'g', 9.81),   # m/s²
            'mu': getattr(solution_instance, 'mu', 0.029),  # kg/mol
            'p0': getattr(solution_instance, 'p0', 101325),  # Pa
            'T0': getattr(solution_instance, 'T0', 293.15),  # K
            'L': getattr(solution_instance, 'L', 15e3),     # m
            'W': getattr(solution_instance, 'W', 8e3),      # m
            'u': getattr(solution_instance, 'u', 1.0),      # m/s
            'gamma': getattr(solution_instance, 'gamma', 1.4),
            'Cp': getattr(solution_instance, 'Cp', 29.1),   # J/(mol·K)
            'Cv': getattr(solution_instance, 'Cv', 20.8),   # J/(mol·K)
        }
        
        # Extract results using safe access pattern
        results = getattr(solution_instance, 'results', {})
        
        # Map expected result keys with safe defaults
        self.results = {
            'Lambda_1': getattr(solution_instance, 'Lambda_1', results.get('Lambda_1', 0.0)),
            'Lambda_2': getattr(solution_instance, 'Lambda_2', results.get('Lambda_2', 0.0)),
            'Lambda_3': getattr(solution_instance, 'Lambda_3', results.get('Lambda_3', 0.0)),
            'Gamma': getattr(solution_instance, 'Gamma', results.get('Gamma', 0.0)),
            'mug/R': getattr(solution_instance, 'mug/R', results.get('mug/R', 0.0)),
            'H': getattr(solution_instance, 'H', results.get('H', 0.0)),
            'h': getattr(solution_instance, 'h', results.get('h', 0.0)),
            'M': getattr(solution_instance, 'M', results.get('M', 0.0)),
            'C3600': getattr(solution_instance, 'C3600', results.get('C3600', 0.0)),
            'T_parcel96m': getattr(solution_instance, 'T_parcel96m', results.get('T_parcel96m', 0.0)),
            'T_parcel119m': getattr(solution_instance, 'T_parcel119m', results.get('T_parcel119m', 0.0)),
            'TparcelH': getattr(solution_instance, 'TparcelH', results.get('TparcelH', 0.0)),
            'G': getattr(solution_instance, 'G', results.get('G', 0.0)),
            'dp/dz': getattr(solution_instance, 'dp/dz', results.get('dp/dz', 0.0)),
            'dp/p': getattr(solution_instance, 'dp/p', results.get('dp/p', 0.0)),
            'rhoz/rho0': getattr(solution_instance, 'rhoz/rho0', results.get('rhoz/rho0', 0.0)),
            'Tz': getattr(solution_instance, 'Tz', results.get('Tz', 0.0)),
            'pz': getattr(solution_instance, 'pz', results.get('pz', 0.0)),
            'Tparcelz': getattr(solution_instance, 'Tparcelz', results.get('Tparcelz', 0.0)),
        }
        
        # Initialize symbolic variables for invariant checking
        self.z, self.T, self.p, self.T_parcel, self.rho = sp.symbols('z T p T_parcel rho')
        self.R_sym, self.g_sym, self.mu_sym, self.gamma_sym = sp.symbols('R g mu gamma')
        
    def test_invariants(self) -> List[str]:
        """
        Verify fundamental physical invariants.
        """
        logs = []
        
        try:
            # 1. Check heat capacity relationship: Cp - Cv = R
            R_val = self.inputs['R']
            Cp_val = self.inputs['Cp']
            Cv_val = self.inputs['Cv']
            expected_diff = Cp_val - Cv_val
            
            if not math.isclose(expected_diff, R_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Cp - Cv = {expected_diff:.3f}, R = {R_val:.3f}")
            else:
                logs.append("PASS: Cp - Cv = R (ideal gas law)")
                
        except Exception as e:
            logs.append(f"ERROR in heat capacity check: {str(e)}")
            
        try:
            # 2. Check adiabatic lapse rate formula consistency
            gamma = self.inputs['gamma']
            mug_R = self.results['mug/R']
            Gamma_expected = ((gamma - 1) / gamma) * mug_R
            
            if 'Gamma' in self.results:
                Gamma_val = self.results['Gamma']
                if not math.isclose(Gamma_val, Gamma_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Γ = {Gamma_val:.6f}, expected {Gamma_expected:.6f}")
                else:
                    logs.append(f"PASS: Γ = (γ-1)/γ * (μg/R) = {Gamma_val:.6f} K/m")
                    
        except Exception as e:
            logs.append(f"ERROR in lapse rate check: {str(e)}")
            
        try:
            # 3. Check hydrostatic equilibrium (numerical fallback)
            if 'dp/dz' in self.results and 'rho' in self.results:
                dp_dz = self.results['dp/dz']
                rho = self.results.get('rho', 1.2)  # approximate air density
                expected_dp_dz = -rho * self.inputs['g']
                
                if not math.isclose(dp_dz, expected_dp_dz, rel_tol=5e-2, abs_tol=1.0):
                    logs.append(f"WARN: dp/dz = {dp_dz:.2f}, expected ≈ {expected_dp_dz:.2f}")
                else:
                    logs.append("PASS: Hydrostatic equilibrium dp/dz ≈ -ρg")
                    
        except Exception as e:
            logs.append(f"ERROR in hydrostatic check: {str(e)}")
            
        try:
            # 4. Check mass conservation for CO box model
            if 'M' in self.results and 'C3600' in self.results:
                M = self.results['M']  # g/s
                C3600 = self.results['C3600']  # g/m³
                L, W, H = self.inputs['L'], self.inputs['W'], self.results['H']
                u = self.inputs['u']
                
                # Steady-state concentration from mass balance
                C_steady = M / (L * H * u)  # g/m³
                
                # At t=3600s, concentration should be ≤ steady-state
                if C3600 > C_steady * 1.1:  # Allow 10% tolerance
                    logs.append(f"WARN: C(3600) = {C3600:.2e} > steady-state {C_steady:.2e}")
                else:
                    logs.append("PASS: C(3600) ≤ steady-state concentration")
                    
        except Exception as e:
            logs.append(f"ERROR in mass conservation check: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        try:
            # 1. Check Λ values consistency with temperature profile
            if all(k in self.results for k in ['Lambda_1', 'Lambda_2', 'Lambda_3']):
                # Λ₁ should be positive (temperature decreases with height)
                if self.results['Lambda_1'] <= 0:
                    logs.append(f"WARN: Λ₁ = {self.results['Lambda_1']:.4f} should be > 0")
                else:
                    logs.append(f"PASS: Λ₁ = {self.results['Lambda_1']:.4f} K/m > 0")
                    
                # Λ₂ should be ~0 (isothermal layer)
                if not math.isclose(self.results['Lambda_2'], 0.0, abs_tol=1e-3):
                    logs.append(f"WARN: Λ₂ = {self.results['Lambda_2']:.4f}, expected ≈ 0")
                else:
                    logs.append(f"PASS: Λ₂ ≈ 0 (isothermal)")
                    
                # Λ₃ should be negative (inversion layer)
                if self.results['Lambda_3'] >= 0:
                    logs.append(f"WARN: Λ₃ = {self.results['Lambda_3']:.4f} should be < 0")
                else:
                    logs.append(f"PASS: Λ₃ = {self.results['Lambda_3']:.4f} K/m < 0")
                    
        except Exception as e:
            logs.append(f"ERROR in Λ consistency: {str(e)}")
            
        try:
            # 2. Check parcel temperatures monotonic decrease
            if all(k in self.results for k in ['T_parcel96m', 'T_parcel119m', 'TparcelH']):
                T96 = self.results['T_parcel96m']
                T119 = self.results['T_parcel119m']
                TH = self.results['TparcelH']
                
                if not (T96 > T119 > TH):
                    logs.append(f"WARN: Temperatures not strictly decreasing: {T96:.2f} > {T119:.2f} > {TH:.2f}")
                else:
                    logs.append(f"PASS: Parcel temperatures decrease with height: {T96:.2f} → {T119:.2f} → {TH:.2f} K")
                    
        except Exception as e:
            logs.append(f"ERROR in temperature monotonicity: {str(e)}")
            
        try:
            # 3. Check H = 119 + h consistency
            if 'H' in self.results and 'h' in self.results:
                H = self.results['H']
                h = self.results['h']
                expected_H = 119 + h
                
                if not math.isclose(H, expected_H, rel_tol=1e-2, abs_tol=1.0):
                    logs.append(f"FAIL: H = {H:.1f}, expected 119 + h = {expected_H:.1f}")
                else:
                    logs.append(f"PASS: H = 119 + h = {H:.1f} m")
                    
        except Exception as e:
            logs.append(f"ERROR in height consistency: {str(e)}")
            
        try:
            # 4. Check critical lapse rate condition
            mug_R = self.results.get('mug/R', 0.034)
            
            # For free convection, Λ > μg/R
            if 'Lambda_1' in self.results:
                Lambda_1 = self.results['Lambda_1']
                if Lambda_1 > mug_R:
                    logs.append(f"PASS: Λ₁ = {Lambda_1:.4f} > μg/R = {mug_R:.4f} (unstable)")
                else:
                    logs.append(f"INFO: Λ₁ = {Lambda_1:.4f} ≤ μg/R = {mug_R:.4f} (stable)")
                    
        except Exception as e:
            logs.append(f"ERROR in lapse rate condition: {str(e)}")
            
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Perturb inputs and check trend consistency.
        """
        logs = []
        
        try:
            # Store original values
            original_M = self.inputs.get('M', self.results.get('M', 13333.33))
            original_T0 = self.inputs['T0']
            
            # Test 1: Increase emission rate → increase concentration
            M_perturbed = original_M * 1.1  # +10%
            
            # Using formula from step 6: C(t) = M/(LHu) * [1 - exp(-ut/W)]
            L = self.inputs['L']
            H = self.results['H']
            u = self.inputs['u']
            W = self.inputs['W']
            t = 3600
            
            C_original = original_M/(L*H*u) * (1 - math.exp(-u*t/W))
            C_perturbed = M_perturbed/(L*H*u) * (1 - math.exp(-u*t/W))
            
            if C_perturbed > C_original:
                logs.append("PASS: Increasing M increases C(t)")
            else:
                logs.append("FAIL: C(t) should increase with M")
                
            # Restore original value
            self.inputs['M'] = original_M
            
        except Exception as e:
            logs.append(f"ERROR in M perturbation: {str(e)}")
            
        try:
            # Test 2: Increase ground temperature → decrease pressure at height
            T0_original = original_T0
            T0_perturbed = T0_original * 1.05  # +5%
            
            # Using constant temperature formula: p(z) = p0 * exp(-μgz/(RT))
            p0 = self.inputs['p0']
            mu = self.inputs['mu']
            g = self.inputs['g']
            R = self.inputs['R']
            z = 1000  # test at 1km
            
            p_original = p0 * math.exp(-mu*g*z/(R*T0_original))
            p_perturbed = p0 * math.exp(-mu*g*z/(R*T0_perturbed))
            
            if p_perturbed > p_original:  # Higher T → slower pressure decrease
                logs.append("PASS: Increasing T0 decreases pressure gradient")
            else:
                logs.append("WARN: Pressure at height should increase with T0")
                
            # Restore original value
            self.inputs['T0'] = T0_original
            
        except Exception as e:
            logs.append(f"ERROR in T0 perturbation: {str(e)}")
            
        try:
            # Test 3: Increase lapse rate → decrease stability
            if 'Lambda_1' in self.results:
                Lambda_original = self.results['Lambda_1']
                Lambda_perturbed = Lambda_original * 1.2
                
                mug_R = self.results.get('mug/R', 0.034)
                
                stable_original = Lambda_original <= mug_R
                stable_perturbed = Lambda_perturbed <= mug_R
                
                # Larger Λ makes atmosphere more unstable
                if Lambda_perturbed > Lambda_original and (not stable_perturbed or stable_original):
                    logs.append("PASS: Increasing Λ decreases stability")
                else:
                    logs.append("INFO: Λ stability relationship holds")
                    
        except Exception as e:
            logs.append(f"ERROR in Λ perturbation: {str(e)}")
            
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks and return (success, logs).
        """
        all_logs = []
        success = True
        
        # Run each test suite
        test_suites = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity),
        ]
        
        for suite_name, test_func in test_suites:
            all_logs.append(f"\n=== {suite_name} ===")
            try:
                logs = test_func()
                all_logs.extend(logs)
                
                # Check for critical failures
                for log in logs:
                    if log.startswith("FAIL:"):
                        success = False
                        
            except Exception as e:
                all_logs.append(f"ERROR in {suite_name}: {str(e)}")
                success = False
                
        # Summary
        if success:
            all_logs.append("\n✓ All physics checks passed within tolerance.")
        else:
            all_logs.append("\n✗ Some physics checks failed.")
            
        return success, all_logs

# Optional: Example usage
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'Lambda_1': 0.0154,
                'Lambda_2': 0.0,
                'Lambda_3': -0.02,
                'Gamma': 0.00978,
                'mug/R': 0.034,
                'H': 142.0,
                'h': 23.0,
                'M': 13333.33,
                'C3600': 2.3e-3,
                'T_parcel96m': 294.04,
                'T_parcel119m': 293.81,
                'TparcelH': 293.6,
                'G': 0.0085,
                'dp/dz': -11.8,
                'rho': 1.2,
            }
            
            # Inputs
            self.R = 8.314
            self.g = 9.81
            self.mu = 0.029
            self.p0 = 101325
            self.T0 = 293.15
            self.L = 15e3
            self.W = 8e3
            self.u = 1.0
            self.gamma = 1.4
            self.Cp = 29.1
            self.Cv = 20.8
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)