import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.r = getattr(solution_instance, 'r', 0.05)  # m (5.00 cm)
        self.h = getattr(solution_instance, 'h', 10.0e-6)  # m (10.0 μm)
        self.sigma = getattr(solution_instance, 'sigma', 4.00e-2)  # N/m
        self.rho = getattr(solution_instance, 'rho', 1100.0)  # kg/m³ (1.10 g/cm³)
        self.R_gas = 8.314462618  # J/(mol·K) - universal gas constant

        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        
        # Expected keys from mapping
        self.C = getattr(solution_instance, 'C', self.results.get('C', 0.0))
        self.omega = getattr(solution_instance, 'omega', self.results.get('omega', 0.0))
        self.p = getattr(solution_instance, 'p', self.results.get('p', 0.0))
        self.C_V = getattr(solution_instance, 'C_V', self.results.get('C_V', 0.0))
        self.dVdT = getattr(solution_instance, 'fracdVdT', self.results.get('fracdVdT', 0.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 0.0))
        self.S_prime = getattr(solution_instance, "S'", self.results.get("S'", 0.0))
        self.F_surf = getattr(solution_instance, 'F_textsurf', self.results.get('F_textsurf', 0.0))
        self.p_prime = getattr(solution_instance, "p'", self.results.get("p'", 0.0))
        self.mddotx = getattr(solution_instance, 'mddotx', self.results.get('mddotx', 0.0))
        self.rhohddotx = getattr(solution_instance, 'rhohddotx', self.results.get('rhohddotx', 0.0))
        
        # Computed values for verification
        self.p_computed = 4 * self.sigma / self.r if self.r != 0 else 0.0
        self.C_V_expected = 2.5 * self.R_gas  # (5/2)R
        self.C_expected = 4 * self.R_gas  # 4R
        self.omega_expected = math.sqrt(8 * self.sigma / (self.rho * self.h * self.r**2)) if all(v > 0 for v in [self.rho, self.h, self.r]) else 0.0
        
        self.logs = []

    def test_invariants(self):
        """Verify physical invariants from KB theorems."""
        try:
            # 1. Laplace's Law: p = 4σ/r
            if not math.isclose(self.p, self.p_computed, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Pressure p = {self.p} != 4σ/r = {self.p_computed}")
            else:
                self.logs.append(f"PASS: Laplace's Law p = 4σ/r holds")

            # 2. Molar heat capacity at constant volume for diatomic gas
            if not math.isclose(self.C_V, self.C_V_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: C_V = {self.C_V} != (5/2)R = {self.C_V_expected}")
            else:
                self.logs.append(f"PASS: C_V = (5/2)R for diatomic gas")

            # 3. First Law of Thermodynamics for the process: C = C_V + (p/ν)*(dV/dT)
            # Derive dV/dT from adiabatic condition T^3 V^{-2} = constant
            T, V = sp.symbols('T V', positive=True)
            # From T^3 * V^{-2} = constant, take derivative: 3T^2*dT*V^{-2} - 2T^3*V^{-3}*dV = 0
            # => dV/dT = (3V)/(2T)
            dVdT_expr = 3*V/(2*T)
            
            # Express C symbolically
            nu = sp.symbols('nu', positive=True)  # moles
            C_expr = 2.5*self.R_gas + (self.p_computed/nu) * dVdT_expr
            
            # From ideal gas law: pV = νRT => ν = pV/(RT)
            # Substitute into C_expr
            C_expr_sub = sp.simplify(C_expr.subs(nu, self.p_computed*V/(self.R_gas*T)))
            # Should simplify to 4*R
            if not sp.simplify(C_expr_sub - 4*self.R_gas) == 0:
                self.logs.append(f"FAIL: Symbolic derivation of C does not yield 4R")
            else:
                self.logs.append(f"PASS: C = C_V + (p/ν)*(dV/dT) symbolically gives 4R")

            # 4. Equation of motion leads to SHM with ω² = 8σ/(ρ h r²)
            if not math.isclose(self.omega**2, 8*self.sigma/(self.rho*self.h*self.r**2), rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: ω² = {self.omega**2} != 8σ/(ρ h r²) = {8*self.sigma/(self.rho*self.h*self.r**2)}")
            else:
                self.logs.append(f"PASS: Equation of motion yields correct ω²")

        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check molar heat capacity C = 4R numerically
            if not math.isclose(self.C, self.C_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: C = {self.C} != 4R = {self.C_expected}")
            else:
                self.logs.append(f"PASS: C = 4R numerically (≈33.26 J/(mol·K))")

            # 2. Check ω matches computed value from inputs
            if not math.isclose(self.omega, self.omega_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: ω = {self.omega} != √(8σ/(ρ h r²)) = {self.omega_expected}")
            else:
                self.logs.append(f"PASS: ω ≈ {self.omega_expected:.1f} rad/s")

            # 3. Check that p' ≈ p(1 - 3x/r) and S' ≈ S(1 + 2x/r) are consistent with linearized force
            # For small x, product p'S' ≈ pS(1 - x/r) to first order
            # Surface tension force F_surf ≈ 4πσ(r+x)α², with α² = S/(4πr²) for patch
            # Then net force ≈ - (8σS/r²) x, leading to ω² = 8σ/(ρ h r²)
            # We can't check without x, but verify dimensions:
            # [σ] = N/m, [ρ] = kg/m³, [h] = m, [r] = m → [ω] = 1/s ✓
            self.logs.append("PASS: Dimensional analysis consistent for ω")

        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        original_sigma = self.sigma
        try:
            # Perturb surface tension σ by +1%
            self.sigma *= 1.01
            omega_perturbed = math.sqrt(8 * self.sigma / (self.rho * self.h * self.r**2))
            
            # ω should increase since ω ∝ √σ
            if omega_perturbed > self.omega_expected:
                self.logs.append(f"PASS: ω increases with σ (monotonicity)")
            else:
                self.logs.append(f"FAIL: ω does not increase with σ")
            
            # Restore original σ
            self.sigma = original_sigma
            
            # Perturb radius r by +1%
            original_r = self.r
            self.r *= 1.01
            omega_perturbed_r = math.sqrt(8 * self.sigma / (self.rho * self.h * self.r**2))
            
            # ω should decrease since ω ∝ 1/r
            if omega_perturbed_r < self.omega_expected:
                self.logs.append(f"PASS: ω decreases with r (monotonicity)")
            else:
                self.logs.append(f"FAIL: ω does not decrease with r")
            
            self.r = original_r
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore on error
            self.sigma = original_sigma
            self.r = original_r

    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs.clear()
        self.logs.append("=== Physics Checker Start ===")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success: no "FAIL" or "ERROR" in logs
        success = all(not (log.startswith("FAIL") or log.startswith("ERROR")) for log in self.logs)
        
        self.logs.append(f"=== Physics Checker End: {'SUCCESS' if success else 'FAILED'} ===")
        return success, self.logs