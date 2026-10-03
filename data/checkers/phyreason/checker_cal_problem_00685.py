import math
import sympy as sp
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safely extract all inputs from solution instance
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'emf', 3.0))
        self.R = getattr(solution_instance, 'R', 4.0)
        
        # Bulb specifications
        self.U_L_rated = 2.0  # Rated voltage
        self.P_L_rated = 0.4  # Rated power
        
        # Safely extract results using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key quantities with safe defaults
        self.r = getattr(solution_instance, 'r', self.results.get('r', 0.0))
        self.U_M = getattr(solution_instance, 'U_M', self.results.get('U_M', 0.0))
        self.U_L = getattr(solution_instance, 'U_L', self.results.get('U_L', 0.0))
        self.I_L = getattr(solution_instance, 'I_L', self.results.get('I_L', 0.0))
        self.eta = getattr(solution_instance, 'eta', self.results.get('eta', 0.0))
        
        # For eta, also try the symbol '\\eta' if 'eta' not found
        if not hasattr(solution_instance, 'eta') and 'eta' not in self.results:
            self.eta = getattr(solution_instance, '\\eta', self.results.get('\\eta', 0.0))
        
        # Initialize logs
        self.logs = []
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log(self, message: str, success: bool):
        """Add message to logs with success indicator."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        success = True
        
        # 1. Check bulb normal operation (Step 1)
        expected_I_L = self.P_L_rated / self.U_L_rated  # 0.2 A
        if not self._is_close(self.I_L, expected_I_L):
            self._log(f"Bulb current mismatch: {self.I_L} vs expected {expected_I_L}", False)
            success = False
        else:
            self._log(f"Bulb current correct: {self.I_L} ≈ {expected_I_L}", True)
        
        # Check bulb power: P = U_L * I_L should equal rated power
        P_L_calc = self.U_L * self.I_L
        if not self._is_close(P_L_calc, self.P_L_rated):
            self._log(f"Bulb power mismatch: {P_L_calc} vs rated {self.P_L_rated}", False)
            success = False
        else:
            self._log(f"Bulb power correct: {P_L_calc} ≈ {self.P_L_rated}", True)
        
        # 2. Ohm's law for circuit with rheostat (Step 2)
        # E = U_L + I_L * (r + R)
        lhs1 = self.E
        rhs1 = self.U_L + self.I_L * (self.r + self.R)
        if not self._is_close(lhs1, rhs1):
            self._log(f"Ohm's law failure (switch 1): {lhs1} ≠ {rhs1}", False)
            success = False
        else:
            self._log(f"Ohm's law holds (switch 1): {lhs1} ≈ {rhs1}", True)
        
        # 3. Ohm's law for circuit with motor (Step 3)
        # E = U_L + U_M + I_L * r
        lhs2 = self.E
        rhs2 = self.U_L + self.U_M + self.I_L * self.r
        if not self._is_close(lhs2, rhs2):
            self._log(f"Ohm's law failure (switch 2): {lhs2} ≠ {rhs2}", False)
            success = False
        else:
            self._log(f"Ohm's law holds (switch 2): {lhs2} ≈ {rhs2}", True)
        
        # 4. Power efficiency bounds (Step 4)
        if self.eta < 0 or self.eta > 100:
            self._log(f"Efficiency out of bounds: η = {self.eta}%", False)
            success = False
        else:
            self._log(f"Efficiency within bounds: η = {self.eta}%", True)
        
        # 5. Power input vs output
        P_in = self.E * self.I_L
        P_out = (self.U_L + self.U_M) * self.I_L
        eta_calc = (P_out / P_in) * 100 if P_in != 0 else 0
        
        if not self._is_close(self.eta, eta_calc):
            self._log(f"Efficiency calc mismatch: {self.eta}% vs {eta_calc}%", False)
            success = False
        else:
            self._log(f"Efficiency consistent: {self.eta}% ≈ {eta_calc}%", True)
        
        # 6. Physical plausibility checks
        if self.r < 0:
            self._log(f"Negative internal resistance: r = {self.r}Ω", False)
            success = False
        else:
            self._log(f"Internal resistance positive: r = {self.r}Ω", True)
        
        if self.U_M < 0:
            self._log(f"Negative motor voltage: U_M = {self.U_M}V", False)
            success = False
        else:
            self._log(f"Motor voltage positive: U_M = {self.U_M}V", True)
        
        if self.U_M > self.E:
            self._log(f"Motor voltage exceeds EMF: {self.U_M}V > {self.E}V", False)
            success = False
        else:
            self._log(f"Motor voltage less than EMF: {self.U_M}V < {self.E}V", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate steps with final results."""
        success = True
        
        # Step 2 consistency: Calculate r from formula
        r_calc = (self.E - self.U_L) / self.I_L - self.R
        if not self._is_close(self.r, r_calc):
            self._log(f"Internal resistance mismatch: {self.r}Ω vs calculated {r_calc}Ω", False)
            success = False
        else:
            self._log(f"Internal resistance consistent: {self.r}Ω ≈ {r_calc}Ω", True)
        
        # Step 3 consistency: Calculate U_M from formula
        U_M_calc = self.E - self.U_L - self.I_L * self.r
        if not self._is_close(self.U_M, U_M_calc):
            self._log(f"Motor voltage mismatch: {self.U_M}V vs calculated {U_M_calc}V", False)
            success = False
        else:
            self._log(f"Motor voltage consistent: {self.U_M}V ≈ {U_M_calc}V", True)
        
        # Step 4 consistency: Calculate eta from formula
        eta_calc = (self.U_L + self.U_M) / self.E * 100
        if not self._is_close(self.eta, eta_calc):
            self._log(f"Efficiency mismatch: {self.eta}% vs calculated {eta_calc}%", False)
            success = False
        else:
            self._log(f"Efficiency consistent: {self.eta}% ≈ {eta_calc}%", True)
        
        # Check target answers
        target_r = 1.0
        target_U_M = 0.8
        target_eta = 93.3
        
        if not self._is_close(self.r, target_r):
            self._log(f"r not matching target: {self.r}Ω vs {target_r}Ω", False)
            success = False
        else:
            self._log(f"r matches target: {self.r}Ω", True)
        
        if not self._is_close(self.U_M, target_U_M):
            self._log(f"U_M not matching target: {self.U_M}V vs {target_U_M}V", False)
            success = False
        else:
            self._log(f"U_M matches target: {self.U_M}V", True)
        
        if not self._is_close(self.eta, target_eta):
            self._log(f"η not matching target: {self.eta}% vs {target_eta}%", False)
            success = False
        else:
            self._log(f"η matches target: {self.eta}%", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check expected trends.
        We'll perturb the rheostat resistance R and check its effect on r.
        """
        success = True
        
        # Store original values
        R_original = self.R
        r_original = self.r
        U_M_original = self.U_M
        eta_original = self.eta
        
        # Perturb R by 10% increase
        R_perturbed = R_original * 1.1
        
        # Recalculate using formulas from steps analysis
        # Step 2: r = (E - U_L)/I_L - R
        r_perturbed = (self.E - self.U_L) / self.I_L - R_perturbed
        
        # Step 3: U_M = E - U_L - I_L * r
        U_M_perturbed = self.E - self.U_L - self.I_L * r_perturbed
        
        # Step 4: η = (U_L + U_M)/E * 100
        eta_perturbed = (self.U_L + U_M_perturbed) / self.E * 100
        
        # Expected trends based on physics:
        # 1. If R increases, r should decrease (since r = (E-U_L)/I_L - R)
        # 2. If r decreases, U_M should increase (U_M = E - U_L - I_L*r)
        # 3. If U_M increases, η should increase (η ∝ U_L + U_M)
        
        # Check trend 1: r decreases when R increases
        if not (r_perturbed < r_original):
            self._log(f"Monotonicity fail: r didn't decrease when R increased (from {r_original} to {r_perturbed})", False)
            success = False
        else:
            self._log(f"r decreases as expected when R increases: {r_original} → {r_perturbed}", True)
        
        # Check trend 2: U_M increases when r decreases
        if not (U_M_perturbed > U_M_original):
            self._log(f"Monotonicity fail: U_M didn't increase when r decreased (from {U_M_original} to {U_M_perturbed})", False)
            success = False
        else:
            self._log(f"U_M increases as expected when r decreases: {U_M_original} → {U_M_perturbed}", True)
        
        # Check trend 3: η increases when U_M increases
        if not (eta_perturbed > eta_original):
            self._log(f"Monotonicity fail: η didn't increase when U_M increased (from {eta_original} to {eta_perturbed})", False)
            success = False
        else:
            self._log(f"η increases as expected when U_M increases: {eta_original} → {eta_perturbed}", True)
        
        # Restore original values
        self.R = R_original
        self.r = r_original
        self.U_M = U_M_original
        self.eta = eta_original
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status with logs."""
        self.logs = []
        overall_success = True
        
        self.logs.append("=== Physics Checker Starting ===")
        
        # Run invariant tests
        self.logs.append("\n--- Testing Physics Invariants ---")
        if not self.test_invariants():
            overall_success = False
        
        # Run trace consistency tests
        self.logs.append("\n--- Testing Trace Consistency ---")
        if not self.test_trace_consistency():
            overall_success = False
        
        # Run monotonicity tests
        self.logs.append("\n--- Testing Monotonicity ---")
        if not self.test_monotonicity():
            overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs

# For symbolic verification (optional, can be called separately)
def symbolic_verification():
    """Create symbolic equations to verify the relationships."""
    E, U_L, I_L, r, R, U_M, eta = sp.symbols('E U_L I_L r R U_M eta')
    
    # Equation from Step 2
    eq1 = sp.Eq(E, U_L + I_L * (r + R))
    
    # Equation from Step 3
    eq2 = sp.Eq(E, U_L + U_M + I_L * r)
    
    # Equation from Step 4
    eq3 = sp.Eq(eta, (U_L + U_M) / E * 100)
    
    # Power relationship for bulb
    P_L = sp.symbols('P_L')
    eq4 = sp.Eq(P_L, U_L * I_L)
    
    return [eq1, eq2, eq3, eq4]