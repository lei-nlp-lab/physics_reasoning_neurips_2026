import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract required results with safe defaults
        self.v_CDprime = self.results.get('v_CDprime', 0.0)  # v_{CD}'
        self.v_ABprime = self.results.get('v_ABprime', 0.0)  # v_{AB}'
        self.v_all = self.results.get('v_all', 0.0)          # common velocity
        self.x = self.results.get('x', 0.0)                  # displacement
        self.v_CD = self.results.get('v_CD', 0.0)            # v_{CD} (intermediate)
        self.v_AB = self.results.get('v_AB', 0.0)            # v_{AB} (intermediate)
        
        # Physical constants from problem
        self.v0 = 10.0          # m/s
        self.m_C = 1.0          # kg
        self.m_D = 1.0          # kg
        self.m_A = 1.0          # kg
        self.m_B = 2.0          # kg
        self.mu = 0.1           # friction coefficient
        self.g = 10.0           # m/s²
        
        # Combined masses
        self.m_prime = self.m_C + self.m_D  # 2 kg (new block)
        self.M_prime = self.m_A + self.m_B  # 3 kg (new skateboard)
        
        # Parameter k (unknown, may be symbolic or numeric)
        self.k = getattr(solution_instance, 'k', None)
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe fuzzy comparison with SymPy fallback."""
        try:
            if isinstance(a, sp.Basic) or isinstance(b, sp.Basic):
                # Symbolic comparison
                diff = sp.simplify(a - b)
                return diff == 0 or abs(float(diff)) < abs_tol
            else:
                return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            # Fallback to numeric with tolerance
            return abs(float(a) - float(b)) < abs_tol
        
    def test_invariants(self):
        """Test conservation laws from KB."""
        try:
            # 1. Momentum conservation for C-D collision (Step 1)
            k = self.k if self.k is not None else sp.symbols('k')
            initial_p_CD = self.m_C * self.v0 + self.m_D * (-k * self.v0)  # D moves left
            final_p_CD = self.m_prime * self.v_CD
            
            if not self.safe_isclose(initial_p_CD, final_p_CD):
                self.log(f"FAIL: C-D momentum conservation: {initial_p_CD} ≠ {final_p_CD}")
                return False
            self.log("PASS: C-D momentum conservation")
            
            # 2. Momentum conservation for A-B collision (Step 3)
            initial_p_AB = self.m_A * self.v0 + self.m_B * (-k * self.v0)  # B moves left
            final_p_AB = self.M_prime * self.v_AB
            
            if not self.safe_isclose(initial_p_AB, final_p_AB):
                self.log(f"FAIL: A-B momentum conservation: {initial_p_AB} ≠ {final_p_AB}")
                return False
            self.log("PASS: A-B momentum conservation")
            
            # 3. Energy conservation during relative motion (Step 8)
            # Work-energy theorem: friction work = ΔKE
            work_friction = self.mu * self.m_prime * self.g * self.x
            ke_initial = 0.5 * self.m_prime * self.v_CDprime**2
            ke_final = 0.5 * (self.m_prime + self.M_prime) * self.v_all**2
            delta_ke = ke_initial - ke_final
            
            if not self.safe_isclose(work_friction, delta_ke):
                self.log(f"FAIL: Energy balance: friction work {work_friction} ≠ ΔKE {delta_ke}")
                return False
            self.log("PASS: Work-energy theorem")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check v_CD' formula (Step 1)
            expected_v_CDprime = 5 * (1 - self.k) if self.k is not None else sp.sympify('5*(1-k)')
            if not self.safe_isclose(self.v_CDprime, expected_v_CDprime):
                self.log(f"FAIL: v_CD' mismatch: {self.v_CDprime} ≠ {expected_v_CDprime}")
                return False
            self.log("PASS: v_CD' formula correct")
            
            # 2. Check v_AB' formula (Step 3)
            expected_v_ABprime = (10 - 20*self.k)/3 if self.k is not None else sp.sympify('(10-20*k)/3')
            if not self.safe_isclose(self.v_ABprime, expected_v_ABprime):
                self.log(f"FAIL: v_AB' mismatch: {self.v_ABprime} ≠ {expected_v_ABprime}")
                return False
            self.log("PASS: v_AB' formula correct")
            
            # 3. Check v_all from momentum (Step 7)
            expected_v_all = self.m_prime * self.v_CDprime / (self.m_prime + self.M_prime)
            if not self.safe_isclose(self.v_all, expected_v_all):
                self.log(f"FAIL: v_all mismatch: {self.v_all} ≠ {expected_v_all}")
                return False
            self.log("PASS: v_all consistent with momentum")
            
            # 4. Check x from energy (Step 8)
            expected_x = (0.5*self.m_prime*self.v_CDprime**2 - 0.5*(self.m_prime+self.M_prime)*self.v_all**2) / (self.mu*self.m_prime*self.g)
            if not self.safe_isclose(self.x, expected_x):
                self.log(f"FAIL: x mismatch: {self.x} ≠ {expected_x}")
                return False
            self.log("PASS: x consistent with energy equation")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
            
    def test_monotonicity(self):
        """Test that increasing k decreases displacement x."""
        try:
            if self.k is None:
                # Symbolic check: dx/dk should be negative for k ∈ [0,1]
                k_sym = sp.symbols('k')
                x_expr = 0.5*self.m_prime*(5*(1-k_sym))**2 - 0.5*(self.m_prime+self.M_prime)*(self.m_prime*5*(1-k_sym)/(self.m_prime+self.M_prime))**2
                x_expr = x_expr / (self.mu*self.m_prime*self.g)
                dx_dk = sp.diff(x_expr, k_sym)
                
                # Evaluate at sample points in [0,1]
                test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
                for point in test_points:
                    val = dx_dk.subs(k_sym, point)
                    if val >= 0:
                        self.log(f"FAIL: dx/dk at k={point} = {val} ≥ 0")
                        return False
                self.log("PASS: x decreases with k (symbolic)")
                
            else:
                # Numerical perturbation check
                delta = 0.01
                k_perturbed = self.k + delta
                
                # Recalculate using formulas from steps
                v_CDprime_pert = 5 * (1 - k_perturbed)
                v_all_pert = self.m_prime * v_CDprime_pert / (self.m_prime + self.M_prime)
                x_pert = (0.5*self.m_prime*v_CDprime_pert**2 - 0.5*(self.m_prime+self.M_prime)*v_all_pert**2) / (self.mu*self.m_prime*self.g)
                
                if x_pert >= self.x:
                    self.log(f"FAIL: x(k+{delta}) = {x_pert} ≥ x(k) = {self.x}")
                    return False
                self.log("PASS: x decreases with increasing k")
                
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
            
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs.clear()
        success = True
        
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            self.log(f"\n=== Running {test_name} Test ===")
            if not test_func():
                success = False
                
        return success, self.logs