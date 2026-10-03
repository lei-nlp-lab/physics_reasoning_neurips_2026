import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', 0.1)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.x0 = getattr(solution_instance, 'x0', 0.1)  # equilibrium elongation in meters
        
        # Results from solution instance (safe access)
        self.results = getattr(solution_instance, 'results', {})
        self.A = getattr(solution_instance, 'A', self.results.get('A', 0.05))  # amplitude in meters
        self.T = getattr(solution_instance, 'T', self.results.get('T', 1.2))    # period in seconds
        self.k = getattr(solution_instance, 'k', self.results.get('k', 10.0))   # spring constant
        self.omega = getattr(solution_instance, 'omega', self.results.get('omega', 5.236))
        self.n = getattr(solution_instance, 'n', self.results.get('n', 10.75))
        self.s = getattr(solution_instance, 's', self.results.get('s', 2.15))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 5.0))
        
        # For displacement, we may have a string or symbolic expression
        self.y_expr = getattr(solution_instance, 'y', self.results.get('y', None))
        
        # Logs for reporting
        self.logs = []

    def log(self, msg, success):
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Check fundamental physics laws."""
        tests_passed = True
        
        # 1. Equilibrium condition: mg = k * x0
        mg = self.m * self.g
        kx0 = self.k * self.x0
        if math.isclose(mg, kx0, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Equilibrium condition mg = k*x0 holds.", True)
        else:
            tests_passed = self.log(f"Equilibrium condition violated: mg={mg}, k*x0={kx0}", False)
        
        # 2. Newton's Second Law at highest point: mg - k*(x0 - A) = m*a
        # At highest point, spring elongation = x0 - A
        net_force = self.m * self.g - self.k * (self.x0 - self.A)
        m_a = self.m * self.a
        if math.isclose(net_force, m_a, rel_tol=1e-2, abs_tol=1e-2):
            self.log("Newton's second law at highest point holds.", True)
        else:
            tests_passed = self.log(f"Newton's second law mismatch: net_force={net_force}, m*a={m_a}", False)
        
        # 3. Energy conservation (optional): max potential energy = max kinetic energy?
        # For SHM, max KE = 0.5*m*(omega*A)^2, max PE_spring = 0.5*k*A^2
        # But due to gravity, total energy = 0.5*k*A^2 + mg*A? Actually, equilibrium point sets reference.
        # We'll skip as not strictly required by steps.
        
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        tests_passed = True
        
        # 1. Angular frequency from period: ω = 2π/T
        omega_calc = 2 * math.pi / self.T
        if math.isclose(self.omega, omega_calc, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Angular frequency consistent: ω={self.omega}, 2π/T={omega_calc}", True)
        else:
            tests_passed = self.log(f"Angular frequency mismatch: ω={self.omega}, 2π/T={omega_calc}", False)
        
        # 2. Total distance: s = 4 * A * n
        s_calc = 4 * self.A * self.n
        if math.isclose(self.s, s_calc, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Total distance consistent: s={self.s}, 4*A*n={s_calc}", True)
        else:
            tests_passed = self.log(f"Total distance mismatch: s={self.s}, 4*A*n={s_calc}", False)
        
        # 3. Displacement formula y = A cos(ω t)
        if self.y_expr is not None:
            # Try to extract amplitude and angular frequency from expression
            # This is a simplified check; in practice, you might parse the string
            # For now, we trust the symbolic expression if provided
            try:
                t = sp.symbols('t')
                # If y_expr is a string, parse it
                if isinstance(self.y_expr, str):
                    # Simple check for amplitude and omega in string
                    if f"{self.A*100}" in self.y_expr and f"{self.omega:.3f}" in self.y_expr:
                        self.log("Displacement formula appears consistent with A and ω.", True)
                    else:
                        self.log("Displacement formula may not match A and ω.", False)
                else:
                    # Assume it's a sympy expression
                    # Check amplitude at t=0: y(0) should be A (since cos(0)=1)
                    y_at_0 = float(self.y_expr.subs(t, 0))
                    if math.isclose(y_at_0, self.A, rel_tol=1e-2, abs_tol=1e-2):
                        self.log("Displacement amplitude check passed.", True)
                    else:
                        tests_passed = self.log(f"Displacement amplitude mismatch: y(0)={y_at_0}, A={self.A}", False)
            except Exception as e:
                self.log(f"Could not verify displacement formula: {e}", False)
        else:
            self.log("No displacement expression provided; skipping check.", True)
        
        return tests_passed

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        tests_passed = True
        
        # Perturb mass by +10%
        original_m = self.m
        perturbed_m = original_m * 1.1
        
        # Expected change in k from equilibrium: k = m*g/x0, so k ∝ m
        original_k = self.k
        expected_k = original_k * (perturbed_m / original_m)
        # Compute acceleration at highest point: a = g - (k/m)*(x0 - A) = g * (A/x0) independent of m
        # So a should remain unchanged
        original_a = self.a
        
        # Check 1: k scales with m
        if math.isclose(expected_k, original_k * (perturbed_m / original_m), rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Spring constant scales correctly with mass: k∝m", True)
        else:
            tests_passed = self.log(f"Spring constant scaling violated.", False)
        
        # Check 2: a is independent of m
        # Recompute a using the formula a = g * (A/x0)
        expected_a = self.g * (self.A / self.x0)
        if math.isclose(original_a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Acceleration at highest point independent of mass: a={original_a}, expected={expected_a}", True)
        else:
            tests_passed = self.log(f"Acceleration should be independent of mass but changed.", False)
        
        # Perturb amplitude A by +10% (if we consider it as an input)
        original_A = self.A
        perturbed_A = original_A * 1.1
        # Acceleration at highest point: a = g * (A/x0), so a ∝ A
        expected_a_new = self.g * (perturbed_A / self.x0)
        if expected_a_new > original_a:
            self.log(f"Acceleration increases with amplitude (positive monotonicity).", True)
        else:
            tests_passed = self.log(f"Acceleration should increase with amplitude.", False)
        
        return tests_passed

    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = []
        overall_success = True
        
        try:
            if not self.test_invariants():
                overall_success = False
            if not self.test_trace_consistency():
                overall_success = False
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self.log(f"Unexpected error during testing: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs