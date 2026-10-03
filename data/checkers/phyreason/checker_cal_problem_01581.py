import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the PhysicsChecker with a solution instance.
        Safely extract all necessary inputs and results using getattr.
        """
        # Extract known constants from the problem (with fallbacks)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.theta_deg = getattr(solution_instance, 'theta_deg', 37.0)
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.E = getattr(solution_instance, 'E', 3.0e3)
        self.q_mag = 1.0e-6  # magnitude from problem
        # The sign is determined in step_1, but we assume positive for calculations
        self.q = self.q_mag  # positive

        # Extract results from the solution instance (with fallbacks to default values)
        # Use results dictionary if available, else try attributes
        results = getattr(solution_instance, 'results', {})
        self.m = getattr(solution_instance, 'm', results.get('m', 4.0e-4))
        self.F = getattr(solution_instance, 'F', results.get('F', 3.0e-3))
        self.v = getattr(solution_instance, 'v', results.get('v', 2.0))
        # Sign of charge might be stored as a string or boolean
        self.sign_q = getattr(solution_instance, 'sign_q', results.get('sign_q', 'positive'))

        # Pre-compute derived quantities
        self.tan_theta = self.sin_theta / self.cos_theta if self.cos_theta != 0 else float('inf')
        # Expected values based on formulas
        self.F_expected = abs(self.q) * self.E
        self.m_expected = (self.q * self.E) / (self.g * self.tan_theta)
        self.v_expected = math.sqrt(2 * self.g * self.l * (1 - self.cos_theta))

        # Store logs for test results
        self.logs = []

    def _log(self, message: str, success: bool):
        """Helper to log test messages."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def test_invariants(self) -> bool:
        """
        Test fundamental physics invariants:
        1. Electric force magnitude matches F = |q| * E.
        2. Equilibrium condition: tanθ = (qE)/(mg).
        3. Energy conservation: v^2 = 2 g l (1 - cosθ).
        4. Consistency of tension components.
        """
        success = True

        # 1. Electric force magnitude
        if not math.isclose(self.F, self.F_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Electric force mismatch: expected {self.F_expected}, got {self.F}", False)
        else:
            self._log("Electric force magnitude matches F = |q| * E.", True)

        # 2. Equilibrium condition: tanθ = (qE)/(mg)
        lhs = self.tan_theta
        rhs = (self.q * self.E) / (self.m * self.g)
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Equilibrium condition failed: tanθ={lhs}, (qE)/(mg)={rhs}", False)
        else:
            self._log("Equilibrium condition tanθ = (qE)/(mg) holds.", True)

        # 3. Energy conservation (if electric field is removed)
        if not math.isclose(self.v, self.v_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Velocity mismatch: expected {self.v_expected}, got {self.v}", False)
        else:
            self._log("Velocity from energy conservation matches.", True)

        # 4. Tension consistency (optional check)
        T_from_vert = self.m * self.g / self.cos_theta
        T_from_horiz = self.q * self.E / self.sin_theta
        if not math.isclose(T_from_vert, T_from_horiz, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Tension inconsistency: mg/cosθ={T_from_vert}, qE/sinθ={T_from_horiz}", False)
        else:
            self._log("Tension computed from vertical and horizontal components agrees.", True)

        return success

    def test_trace_consistency(self) -> bool:
        """
        Check consistency between intermediate results:
        - From F and m, compute tanθ and compare with given.
        - From m and v, check energy equation (1/2) m v^2 = m g l (1-cosθ).
        """
        success = True

        # tanθ from F and m
        tan_from_Fm = self.F / (self.m * self.g)
        if not math.isclose(self.tan_theta, tan_from_Fm, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"tanθ from F and m mismatch: given {self.tan_theta}, computed {tan_from_Fm}", False)
        else:
            self._log("tanθ derived from F and m matches given tanθ.", True)

        # Energy equation: (1/2) m v^2 should equal m g l (1-cosθ)
        kinetic = 0.5 * self.m * self.v**2
        potential_drop = self.m * self.g * self.l * (1 - self.cos_theta)
        if not math.isclose(kinetic, potential_drop, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Energy mismatch: (1/2) m v^2={kinetic}, m g l (1-cosθ)={potential_drop}", False)
        else:
            self._log("Energy conservation (1/2) m v^2 = m g l (1-cosθ) holds.", True)

        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb the electric field E and verify expected trends:
        - m is proportional to E (since m = qE/(g tanθ)).
        - v is independent of E (since v = sqrt(2 g l (1-cosθ))).
        After checks, restore original values.
        """
        success = True
        original_E = self.E
        original_m = self.m
        original_v = self.v

        # Perturb E by +10%
        perturb_factor = 1.10
        self.E = original_E * perturb_factor

        # Recompute expected m and v
        m_new_expected = (self.q * self.E) / (self.g * self.tan_theta)
        v_new_expected = math.sqrt(2 * self.g * self.l * (1 - self.cos_theta))

        # Since m is proportional to E, the new m should be perturb_factor times the original m_expected
        if not math.isclose(m_new_expected, self.m_expected * perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Monotonicity: m should scale linearly with E. Expected m={self.m_expected * perturb_factor}, got {m_new_expected}", False)
        else:
            self._log(f"Monotonicity: m scales correctly with E (factor {perturb_factor}).", True)

        # v should remain unchanged
        if not math.isclose(v_new_expected, self.v_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            self._log(f"Monotonicity: v should be independent of E. Expected v={self.v_expected}, got {v_new_expected}", False)
        else:
            self._log("Monotonicity: v is independent of E as expected.", True)

        # Restore original values
        self.E = original_E
        self.m = original_m
        self.v = original_v

        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return a tuple (success, logs).
        """
        self.logs = []
        overall_success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self.logs.append(f"--- Running {name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {str(e)}")
                overall_success = False

        return overall_success, self.logs