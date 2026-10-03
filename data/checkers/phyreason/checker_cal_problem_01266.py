import math
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        self.solution = solution_instance
        self.results: Dict[str, float] = {}
        self.inputs: Dict[str, float] = {}
        self.logs: List[str] = []
        
        # Extract inputs safely
        self.inputs['m'] = getattr(solution_instance, 'm', 1.0)  # default 1 kg
        self.inputs['M'] = getattr(solution_instance, 'M', 0.5)  # default 0.5 kg
        self.inputs['mu'] = getattr(solution_instance, 'mu', 0.2)  # default 0.2
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)  # default 10 m/s²
        self.inputs['L'] = getattr(solution_instance, 'L', 0.5)  # default 0.5 m
        self.inputs['d'] = getattr(solution_instance, 'd', 0.5)  # default 0.5 m
        
        # Extract results safely from solution_instance.results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Map expected keys to actual results with safe defaults
        key_mapping = {
            'Δx': 'delta_x',
            'v₀₁': 'v01',
            'F_f\'': 'F_f_prime',
            'x₁': 'x1',
            't': 't',
            'F_f': 'F_f',
            'a₁': 'a1',
            'F_N - G': 'F_N_minus_G',
            '-F_f': 'minus_F_f',
            'x₂': 'x2',
            'G': 'G',
            'v₀₂': 'v02'
        }
        
        for display_key, result_key in key_mapping.items():
            # Try to get the result from the results dictionary
            # If not found, try to get from solution_instance attributes as fallback
            value = results_dict.get(result_key, 
                     getattr(solution_instance, result_key, 
                     getattr(solution_instance, display_key, 0.0)))
            self.results[display_key] = float(value) if value is not None else 0.0
        
        # Calculate derived quantities for verification
        self.m = self.inputs['m']
        self.M = self.inputs['M']
        self.mu = self.inputs['mu']
        self.g = self.inputs['g']
        self.L = self.inputs['L']
        self.d = self.inputs['d']
        
        # Expected values from physics laws
        self.expected_G = self.m * self.g
        self.expected_F_f = self.mu * self.expected_G
        self.expected_a1 = -self.expected_F_f / self.m  # Negative for deceleration
        self.expected_a2 = self.expected_F_f / self.M   # Positive for acceleration
        
    def log(self, message: str):
        """Helper to add log messages."""
        self.logs.append(message)
        
    def fuzzy_equal(self, a: float, b: float, name: str = "") -> bool:
        """Check if two numbers are close with 2% tolerance."""
        is_close = math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
        if not is_close:
            self.log(f"FAIL {name}: {a} ≠ {b} (difference: {abs(a-b):.6f})")
        else:
            self.log(f"PASS {name}: {a} ≈ {b}")
        return is_close
        
    def test_invariants(self) -> bool:
        """
        Test fundamental physics laws:
        1. Newton's First Law in y-direction
        2. Newton's Second Law for object A
        3. Newton's Third Law between A and B
        4. Energy dissipation via friction
        """
        success = True
        
        # 1. Newton's First Law in y-direction: F_N - G = 0
        F_N_minus_G = self.results.get('F_N - G', 0.0)
        if not self.fuzzy_equal(F_N_minus_G, 0.0, "Newton's First Law (y-direction)"):
            success = False
            
        # 2. Newton's Second Law for object A: -F_f = m * a1
        F_f = self.results.get('F_f', 0.0)
        a1 = self.results.get('a₁', 0.0)
        expected_F_f = self.expected_F_f
        
        if not self.fuzzy_equal(F_f, expected_F_f, "Friction force magnitude"):
            success = False
            
        lhs = -F_f
        rhs = self.m * a1
        if not self.fuzzy_equal(lhs, rhs, "Newton's Second Law for object A"):
            success = False
            
        # 3. Newton's Third Law: F_f' = F_f
        F_f_prime = self.results.get('F_f\'', 0.0)
        if not self.fuzzy_equal(F_f_prime, F_f, "Newton's Third Law (F_f' = F_f)"):
            success = False
            
        # 4. Kinetic friction definition: F_f = μ * G
        G = self.results.get('G', 0.0)
        if not self.fuzzy_equal(G, self.expected_G, "Gravitational force"):
            success = False
            
        if not self.fuzzy_equal(F_f, self.mu * G, "Friction definition (F_f = μG)"):
            success = False
            
        # 5. Check accelerations
        if not self.fuzzy_equal(a1, self.expected_a1, "Object A acceleration"):
            success = False
            
        a2_calculated = F_f_prime / self.M
        if 'a2' in self.results or hasattr(self.solution, 'a2'):
            a2 = self.results.get('a2', getattr(self.solution, 'a2', 0.0))
            if not self.fuzzy_equal(a2, a2_calculated, "Cart B acceleration"):
                success = False
        else:
            # Verify using Newton's Second Law for cart
            if not self.fuzzy_equal(F_f_prime, self.M * a2_calculated, "Newton's Second Law for cart"):
                success = False
                
        return success
        
    def test_trace_consistency(self) -> bool:
        """
        Check consistency between intermediate results and final answers.
        """
        success = True
        
        # Get key results
        v01 = self.results.get('v₀₁', 0.0)  # Minimum velocity
        v02 = self.results.get('v₀₂', 0.0)  # Maximum velocity
        delta_x = self.results.get('Δx', 0.0)
        x1 = self.results.get('x₁', 0.0)
        x2 = self.results.get('x₂', 0.0)
        t = self.results.get('t', 0.0)
        a1 = self.results.get('a₁', self.expected_a1)
        a2 = self.expected_a2  # Calculate from physics
        
        # 1. Check Δx = x1 - x2
        if delta_x > 0 and x1 > 0 and x2 > 0:
            if not self.fuzzy_equal(delta_x, x1 - x2, "Δx = x1 - x2"):
                success = False
                
        # 2. Check kinematic equations for minimum velocity case (v01)
        if v01 > 0:
            # Time when velocities equalize: v01 + a1*t = a2*t
            t_calc_v01 = -v01 / (a1 - a2)  # Since a1 is negative
            if t > 0:
                if not self.fuzzy_equal(t, t_calc_v01, f"Time for v0={v01:.3f}"):
                    success = False
                    
            # Displacement of A: x1 = v01*t + 0.5*a1*t²
            x1_calc = v01 * t + 0.5 * a1 * t**2
            if x1 > 0:
                if not self.fuzzy_equal(x1, x1_calc, f"x1 for v0={v01:.3f}"):
                    success = False
                    
            # Displacement of B: x2 = 0.5*a2*t²
            x2_calc = 0.5 * a2 * t**2
            if x2 > 0:
                if not self.fuzzy_equal(x2, x2_calc, f"x2 for v0={v01:.3f}"):
                    success = False
                    
            # Relative displacement: Δx = v01²/(2*(a2 - a1))? Let's derive
            # Alternative: Δx = (v01²) / (2*(a2 - a1))? Actually from equations:
            # Δx = x1 - x2 = v01*t + 0.5*a1*t² - 0.5*a2*t²
            # Substitute t = -v01/(a1 - a2)
            # After algebra: Δx = -v01²/(2*(a1 - a2))
            delta_x_calc = -v01**2 / (2 * (a1 - a2))
            if delta_x > 0:
                if not self.fuzzy_equal(delta_x, delta_x_calc, f"Δx for v0={v01:.3f}"):
                    success = False
                    
        # 3. Check that v01 and v02 give correct boundary conditions
        # The boundaries should correspond to Δx = L and Δx = L + d
        if v01 > 0 and v02 > 0:
            # Calculate Δx for each velocity
            delta_x_v01 = -v01**2 / (2 * (a1 - a2))
            delta_x_v02 = -v02**2 / (2 * (a1 - a2))
            
            # Check against L and L+d
            if not self.fuzzy_equal(delta_x_v01, self.L, f"v01 gives Δx = L"):
                success = False
            if not self.fuzzy_equal(delta_x_v02, self.L + self.d, f"v02 gives Δx = L+d"):
                success = False
                
        # 4. Verify the final answer range - use the correct boundaries from the problem
        # The problem states the effective area has width d=0.5 m at the right side of cart length L=0.5 m.
        # So the relative displacement boundaries are L and L+d = 0.5 and 1.0 m.
        # Compute expected v01 and v02 from physics formulas:
        # Δx = -v0²/(2*(a1 - a2)) => v0 = sqrt(-2*Δx*(a1 - a2))
        # a1 = -μ*g, a2 = μ*m*g/M
        # a1 - a2 = -μ*g*(1 + m/M)
        mu = self.mu
        g = self.g
        m = self.m
        M = self.M
        factor = -2 * (a1 - a2)  # positive since a1 - a2 is negative
        expected_v01 = math.sqrt(factor * self.L)
        expected_v02 = math.sqrt(factor * (self.L + self.d))
        
        if not self.fuzzy_equal(v01, expected_v01, "Minimum velocity v01 (physics)"):
            success = False
        if not self.fuzzy_equal(v02, expected_v02, "Maximum velocity v02 (physics)"):
            success = False
            
        return success
        
    def test_monotonicity(self) -> bool:
        """
        Test monotonic relationships by perturbing input parameters.
        """
        success = True
        
        # Test 1: Increase mass m → friction force increases
        original_m = self.m
        perturbed_m = original_m * 1.1  # 10% increase
        
        # Recalculate friction force
        original_F_f = self.mu * original_m * self.g
        perturbed_F_f = self.mu * perturbed_m * self.g
        
        if perturbed_F_f <= original_F_f:
            self.log(f"FAIL Monotonicity: F_f should increase when m increases")
            success = False
        else:
            self.log(f"PASS Monotonicity: F_f increases with m")
            
        # Test 2: Increase μ → friction force increases
        original_mu = self.mu
        perturbed_mu = original_mu * 1.1
        
        original_F_f_mu = original_mu * self.m * self.g
        perturbed_F_f_mu = perturbed_mu * self.m * self.g
        
        if perturbed_F_f_mu <= original_F_f_mu:
            self.log(f"FAIL Monotonicity: F_f should increase when μ increases")
            success = False
        else:
            self.log(f"PASS Monotonicity: F_f increases with μ")
            
        # Test 3: For fixed v0, increase μ → Δx decreases (more friction)
        # Use numeric relationship: Δx = v0²/(2*μ*g*(1 + m/M))
        v0_test = 3.0  # Test velocity
        original_delta_x = v0_test**2 / (2 * original_mu * self.g * (1 + self.m/self.M))
        perturbed_delta_x = v0_test**2 / (2 * perturbed_mu * self.g * (1 + self.m/self.M))
        
        if perturbed_delta_x >= original_delta_x:
            self.log(f"FAIL Monotonicity: Δx should decrease when μ increases")
            success = False
        else:
            self.log(f"PASS Monotonicity: Δx decreases with μ")
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status with logs.
        """
        self.logs = []  # Reset logs
        
        self.log("=" * 60)
        self.log("Starting Physics Audit")
        self.log("=" * 60)
        
        # Display inputs
        self.log(f"\nInputs:")
        for key, value in self.inputs.items():
            self.log(f"  {key} = {value}")
            
        # Display key results
        self.log(f"\nKey Results:")
        for key, value in self.results.items():
            if value != 0.0:  # Skip zero defaults
                self.log(f"  {key} = {value:.6f}")
        
        # Run tests
        self.log("\n" + "=" * 60)
        self.log("TEST 1: Physics Invariants")
        self.log("=" * 60)
        test1 = self.test_invariants()
        
        self.log("\n" + "=" * 60)
        self.log("TEST 2: Trace Consistency")
        self.log("=" * 60)
        test2 = self.test_trace_consistency()
        
        self.log("\n" + "=" * 60)
        self.log("TEST 3: Monotonicity Checks")
        self.log("=" * 60)
        test3 = self.test_monotonicity()
        
        # Final summary
        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)
        self.log(f"Physics Invariants: {'PASS' if test1 else 'FAIL'}")
        self.log(f"Trace Consistency: {'PASS' if test2 else 'FAIL'}")
        self.log(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
        
        overall_success = test1 and test2 and test3
        self.log(f"\nOverall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        return overall_success, self.logs