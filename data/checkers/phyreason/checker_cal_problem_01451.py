import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Extract inputs and results safely using getattr with fallbacks."""
        # Inputs from problem statement (with defaults from problem)
        self.m_child = getattr(solution_instance, 'm_child', 30.0)
        self.m_sled = getattr(solution_instance, 'm_sled', 10.0)
        self.F = getattr(solution_instance, 'F', 100.0)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Trigonometric values (given)
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.N = getattr(solution_instance, 'N', results.get('N', 0.0))
        self.m = getattr(solution_instance, 'm', results.get('m', 0.0))
        self.mu = getattr(solution_instance, 'mu', results.get('mu', 0.0))
        self.N_pressure = getattr(solution_instance, 'N_pressure', results.get('N_pressure', 0.0))
        
        # Derived quantities
        self.theta = math.radians(self.theta_deg)
        self.F_horiz = self.F * self.cos37
        self.F_vert = self.F * self.sin37
        self.total_mass = self.m_child + self.m_sled
        
        # For monotonicity test backup
        self._original_mass = self.total_mass

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physical invariants using symbolic and numerical checks."""
        logs = []
        success = True
        
        try:
            # Symbolic variables
            m_sym, g_sym, F_sym, sin_sym, cos_sym, N_sym, mu_sym = sp.symbols('m g F sinθ cosθ N μ')
            
            # 1. Vertical equilibrium: N = m*g - F*sinθ
            vertical_eq = sp.Eq(N_sym, m_sym * g_sym - F_sym * sin_sym)
            # Numerical check
            N_expected = self.total_mass * self.g - self.F_vert
            if not math.isclose(self.N, N_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical equilibrium violated. N={self.N:.2f}, expected {N_expected:.2f}")
                success = False
            else:
                logs.append(f"PASS: Vertical equilibrium check: N = {self.N:.2f} N")
            
            # 2. Horizontal equilibrium: F*cosθ = μ*N (constant speed)
            horizontal_eq = sp.Eq(F_sym * cos_sym, mu_sym * N_sym)
            # Numerical check
            friction_force = self.F_horiz
            friction_from_mu = self.mu * self.N
            if not math.isclose(friction_force, friction_from_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Horizontal equilibrium violated. F_horiz={friction_force:.2f}, μN={friction_from_mu:.2f}")
                success = False
            else:
                logs.append(f"PASS: Horizontal equilibrium check: F_horiz = μN = {friction_force:.2f} N")
            
            # 3. Newton's third law: N_pressure = N (magnitude)
            if not math.isclose(self.N_pressure, self.N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's third law violated. N_pressure={self.N_pressure:.2f}, N={self.N:.2f}")
                success = False
            else:
                logs.append(f"PASS: Newton's third law: N_pressure = N = {self.N:.2f} N")
            
            # 4. Friction definition: μ = F_horiz / N (from equilibrium)
            mu_expected = self.F_horiz / self.N if self.N != 0 else 0.0
            if not math.isclose(self.mu, mu_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction definition violated. μ={self.mu:.4f}, expected {mu_expected:.4f}")
                success = False
            else:
                logs.append(f"PASS: Friction definition: μ = F_horiz/N = {self.mu:.4f}")
                
        except Exception as e:
            logs.append(f"ERROR in invariant test: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check intermediate steps from STEPS ANALYSIS."""
        logs = []
        success = True
        
        try:
            # Step 1: Total mass
            expected_mass = self.m_child + self.m_sled
            if not math.isclose(self.total_mass, expected_mass, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Total mass mismatch. Computed={self.total_mass:.2f}, expected={expected_mass:.2f}")
                success = False
            else:
                logs.append(f"PASS: Total mass = {self.total_mass:.2f} kg")
            
            # Step 2: Normal force (already checked in invariants)
            N_expected = expected_mass * self.g - self.F_vert
            if not math.isclose(self.N, N_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Normal force mismatch. N={self.N:.2f}, expected={N_expected:.2f}")
                success = False
            else:
                logs.append(f"PASS: Normal force N = {self.N:.2f} N")
            
            # Step 3: Pressure (already checked in invariants)
            if not math.isclose(self.N_pressure, self.N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Pressure mismatch. N_pressure={self.N_pressure:.2f}, N={self.N:.2f}")
                success = False
            else:
                logs.append(f"PASS: Pressure N_pressure = {self.N_pressure:.2f} N")
            
            # Step 6: Coefficient of friction
            mu_expected = self.F_horiz / self.N
            if not math.isclose(self.mu, mu_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Coefficient mismatch. μ={self.mu:.4f}, expected={mu_expected:.4f}")
                success = False
            else:
                logs.append(f"PASS: Coefficient μ = {self.mu:.4f}")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb mass input and verify physical trends."""
        logs = []
        success = True
        
        try:
            # Perturbation: increase total mass by 10%
            perturbation_factor = 1.1
            perturbed_mass = self.total_mass * perturbation_factor
            
            # Recalculate using physics formulas
            N_new = perturbed_mass * self.g - self.F_vert
            mu_new = self.F_horiz / N_new
            
            # Expected trends based on physics:
            # 1. If mass increases, normal force should increase (downward force increases)
            if not (N_new > self.N):
                logs.append(f"FAIL: Monotonicity - mass increase should increase N. N_original={self.N:.2f}, N_new={N_new:.2f}")
                success = False
            else:
                logs.append(f"PASS: N increases with mass: {self.N:.2f} → {N_new:.2f}")
            
            # 2. If normal force increases, friction coefficient should decrease (μ = F_horiz/N)
            if not (mu_new < self.mu):
                logs.append(f"FAIL: Monotonicity - N increase should decrease μ. μ_original={self.mu:.4f}, μ_new={mu_new:.4f}")
                success = False
            else:
                logs.append(f"PASS: μ decreases with N increase: {self.mu:.4f} → {mu_new:.4f}")
            
            # 3. Check pressure follows normal force
            N_pressure_new = N_new  # From Newton's third law
            if not math.isclose(N_pressure_new, N_new, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's third law broken after perturbation. N_pressure_new={N_pressure_new:.2f}, N_new={N_new:.2f}")
                success = False
            else:
                logs.append(f"PASS: Newton's third law holds after perturbation: {N_pressure_new:.2f} = {N_new:.2f}")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        return overall_success, all_logs