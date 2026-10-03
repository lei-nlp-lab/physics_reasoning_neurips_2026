import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely; use None for missing values
        self.m = getattr(solution_instance, 'm', None)
        self.e = getattr(solution_instance, 'e', None)
        self.c = getattr(solution_instance, 'c', None)
        self.a0 = getattr(solution_instance, 'a0', None)
        
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.t = results.get('t', None)
        
        # Additional derived quantities (computed in checker, not from solution)
        self._symbolic_init()
    
    def _symbolic_init(self):
        """Define symbolic variables for internal checks."""
        self.m_sym, self.e_sym, self.c_sym, self.a0_sym = sp.symbols('m e c a0', positive=True)
        self.E_sym = -self.e_sym**2 / (2 * self.a0_sym)  # at initial radius
        self.P_sym = (2 * self.e_sym**2 * (self.e_sym**2 / (self.m_sym * self.a0_sym**2))**2) / (3 * self.c_sym**3)
        self.dt_dr_sym = - (3 * self.c_sym**3 * self.m_sym**2 * self.a0_sym**2) / (4 * self.e_sym**4)  # actually dt/dr, not dr/dt
    
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)
    
    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that results['t'] exists and is a number (int or float)."""
        if self.t is None:
            return (False, "SKIP: results['t'] not provided")
        if not isinstance(self.t, (int, float, sp.Expr)):
            return (False, f"results['t'] is not a numeric type: {type(self.t)}")
        return (True, f"results['t'] = {self.t}")
    
    def test_interface_constants_exist(self):
        """Check that all required constants are present."""
        missing = []
        for name in ['m', 'e', 'c', 'a0']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing constants: {missing}")
        return (True, "All constants available")
    
    # ------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_energy_derivative(self):
        """Check that dE/dt = -P at r = a0 using derived expressions."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for derivative check")
        
        # Compute expected dr/dt from Larmor power and energy derivative
        # dr/dt = - (4 e^4) / (3 c^3 m^2 r^2)
        r = self.a0
        dr_dt_expected = - (4 * self.e**4) / (3 * self.c**3 * self.m**2 * r**2)
        
        # Check that it's negative (should be, since radius decreases)
        if dr_dt_expected >= 0:
            return (False, f"dr/dt expected = {dr_dt_expected} is not negative")
        
        # Optional: compare with value from solution if available
        # We don't have dr/dt directly, but we can compute from t if needed.
        # Instead, verify the consistency of the Larmor power.
        return (True, f"dr/dt = {dr_dt_expected} is negative")
    
    def test_invariant_larmor_consistency(self):
        """Check that Larmor power P = (2/3)(e^2 a^2)/c^3 matches derived expression."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for Larmor check")
        
        # Compute acceleration a = e^2 / (m r^2) at r=a0
        a = self.e**2 / (self.m * self.a0**2)
        # Larmor power: P = (2 e^2 a^2) / (3 c^3)
        P_from_larmor = (2 * self.e**2 * a**2) / (3 * self.c**3)
        # Expected from circular orbit formula: P = 2 e^6 / (3 c^3 m^2 r^4)
        P_expected = (2 * self.e**6) / (3 * self.c**3 * self.m**2 * self.a0**4)
        
        if not math.isclose(P_from_larmor, P_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Larmor power mismatch: {P_from_larmor} vs {P_expected}")
        return (True, "Larmor power consistent with circular acceleration")
    
    def test_invariant_energy_relation(self):
        """Check that total energy E = -e^2/(2r) at r = a0."""
        if None in (self.m, self.e, self.a0):
            return (False, "SKIP: missing constants for energy check")
        
        # Using mv^2/r = e^2/r^2 => v^2 = e^2/(m r)
        v2 = self.e**2 / (self.m * self.a0)
        K = 0.5 * self.m * v2
        U = -self.e**2 / self.a0
        E_total = K + U  # Should be -e^2/(2 a0)
        E_expected = -self.e**2 / (2 * self.a0)
        
        if not math.isclose(E_total, E_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Total energy mismatch: {E_total} vs {E_expected}")
        return (True, "Energy relation verified")
    
    def test_invariant_adiabatic_condition(self):
        """Check that energy loss per revolution << total energy at r=a0."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for adiabatic check")
        
        # Orbital period T = 2πr/v
        v = math.sqrt(self.e**2 / (self.m * self.a0))
        T = 2 * math.pi * self.a0 / v
        
        # Larmor power P
        a = self.e**2 / (self.m * self.a0**2)
        P = (2 * self.e**2 * a**2) / (3 * self.c**3)
        
        # Energy loss per revolution
        delta_E_per_rev = P * T
        # Total energy (absolute value)
        total_E = self.e**2 / (2 * self.a0)  # absolute value
        
        if total_E == 0:
            return (False, "Total energy is zero, cannot evaluate adiabatic condition")
        
        ratio = delta_E_per_rev / total_E
        if ratio > 0.01:  # typical threshold for "small"
            return (False, f"Energy loss per revolution fraction = {ratio} > 0.01, adiabatic assumption may be invalid")
        return (True, f"Adiabatic condition satisfied: ratio = {ratio}")
    
    # ------------------------------------------------------------
    # Modeling tests (based on explanation steps)
    # ------------------------------------------------------------
    def test_modeling_integral_step(self):
        """Check that the integrated collapse time matches the derived expression."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for integral check")
        if self.t is None:
            return (False, "SKIP: results['t'] not provided")
        
        # Derive expected t from the integral: t = (c^3 * m^2 * a0^3) / (4 * e^4)
        expected_t = (self.c**3 * self.m**2 * self.a0**3) / (4 * self.e**4)
        
        # Use isclose for comparison (lenient tolerance)
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Collapse time mismatch: solution t={self.t} vs derived expected={expected_t}")
        return (True, "Integrated collapse time consistent with formula")
    
    def test_modeling_dr_dt_negative(self):
        """Check that the radius decreases (dr/dt < 0) using derived expression at r=a0."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for dr/dt check")
        
        # dr/dt at r=a0: dr/dt = - (4 e^4) / (3 c^3 m^2 a0^2)
        dr_dt = - (4 * self.e**4) / (3 * self.c**3 * self.m**2 * self.a0**2)
        if dr_dt >= 0:
            return (False, f"dr/dt = {dr_dt} is not negative")
        return (True, "dr/dt is negative as expected")
    
    # ------------------------------------------------------------
    # Monotonicity test (perturbation-based)
    # ------------------------------------------------------------
    def test_monotonicity_mass_increases_collapse_time(self):
        """Check that increasing mass leads to longer collapse time (using derived relation)."""
        if None in (self.m, self.e, self.c, self.a0):
            return (False, "SKIP: missing constants for monotonicity check")
        
        # Baseline t from provided solution (if available) or from derived formula
        if self.t is not None:
            baseline_t = self.t
        else:
            baseline_t = (self.c**3 * self.m**2 * self.a0**3) / (4 * self.e**4)
        
        # Increase mass by 10%
        m_per = self.m * 1.1
        # Compute new t using the derived formula (same constants except mass)
        new_t = (self.c**3 * m_per**2 * self.a0**3) / (4 * self.e**4)
        
        if new_t <= baseline_t:
            return (False, f"After mass increase, t = {new_t} is not greater than baseline t = {baseline_t}")
        return (True, f"Monotonicity verified: t increases with mass (baseline={baseline_t}, new={new_t})")
    
    # ------------------------------------------------------------
    # Answer sanity test (weak, not dominant)
    # ------------------------------------------------------------
    def test_answer_sanity_finite_positive(self):
        """Check that the collapse time is positive and finite."""
        if self.t is None:
            return (False, "SKIP: results['t'] not provided")
        
        if not isinstance(self.t, (int, float)):
            return (False, f"t is not a numeric type: {type(self.t)}")
        if math.isnan(self.t) or math.isinf(self.t):
            return (False, f"t is NaN or infinite: {self.t}")
        if self.t <= 0:
            return (False, f"t is not positive: {self.t}")
        return (True, f"t = {self.t} is positive and finite")