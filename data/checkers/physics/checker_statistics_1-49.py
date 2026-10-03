import math
import traceback

class PhysicsChecker:
    """
    Audits a thermodynamics solution for a helium gas undergoing isobaric and adiabatic processes.
    Checks physical invariants, interface completeness, modeling consistency, monotonicity, and provides a weak answer sanity check.
    """

    def __init__(self, solution_instance):
        """
        Initialize checker with solution_instance. Extract all relevant data safely using None defaults.
        """
        self.solution = solution_instance
        self.results = {}
        # Try to get results dict from solution_instance
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif hasattr(solution_instance, 'get') and callable(solution_instance.get):
            # Some instances might have a get method (like dict-like)
            self.results = solution_instance.get('results', {})

        # Physical constants and givens
        self.R = self._get('R', None)  # Gas constant (float)
        self.n = self._get('n', None)  # amount in mol (float)
        self.T_A = self._get('T_A', None)  # Initial temperature in K
        self.P_A = self._get('P_A', None)  # Initial pressure in Pa
        # Volumes (may be given or computed, but expect float)
        self.V_A = self._get('V_A', None)
        self.V_B = self._get('V_B', None)
        self.V_C = self._get('V_C', None)
        # Heat capacities (can be computed from R if not provided)
        self.Cv = self._get('Cv', None)
        self.Cp = self._get('Cp', None)
        self.gamma = self._get('gamma', None)
        # Explanation steps (optional)
        self.explanation_steps = solution_instance.get('explanation_steps', []) if hasattr(solution_instance, 'get') else getattr(solution_instance, 'explanation_steps', None)

        # Result quantities to be checked
        self.W_AB = self._get('W_AB', None)
        self.delta_U_AB = self._get('delta_U_AB', None)
        self.Q_AB = self._get('Q_AB', None)
        self.P_C = self._get('P_C', None)
        self.delta_S_BC = self._get('delta_S_BC', None)

        # If heat capacities not given, compute from R (if R available)
        if self.R is not None and self.Cv is None:
            self.Cv = 1.5 * self.R
        if self.R is not None and self.Cp is None:
            self.Cp = 2.5 * self.R
        if self.Cp is not None and self.Cv is not None and self.Cv != 0:
            self.gamma = self.Cp / self.Cv

    def _get(self, key, default=None):
        """
        Try to get value from solution_instance: first from results dict, then from direct attribute.
        """
        # Try results dict
        if key in self.results:
            return self.results[key]
        # Try direct attribute
        if hasattr(self.solution, key):
            return getattr(self.solution, key)
        # If solution is a dict-like object with get method
        if hasattr(self.solution, 'get') and callable(self.solution.get):
            return self.solution.get(key, default)
        return default

    # ------- Interface Tests -------

    def test_interface_required_keys(self):
        """Check that all required result keys exist and are not None."""
        required_keys = ['W_AB', 'delta_U_AB', 'Q_AB', 'P_C', 'delta_S_BC']
        missing = [k for k in required_keys if self._get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_givens_type(self):
        """Check that numeric givens are valid numbers."""
        givens = [self.n, self.T_A, self.P_A, self.R, self.V_A, self.V_B, self.V_C]
        if any(g is None for g in givens):
            return (True, "SKIP: Some givens missing, cannot check types.")
        # All are numbers if present
        if not all(isinstance(g, (int, float)) for g in givens):
            return (False, "Some givens are not numeric.")
        return (True, "All numeric givens have valid type.")

    # ------- Invariant / Theorem Tests -------

    def test_invariant_ideal_gas_A(self):
        """Check that state A satisfies PV = nRT."""
        if any(x is None for x in [self.P_A, self.V_A, self.n, self.R, self.T_A]):
            return (True, "SKIP: Missing data for ideal gas law at A.")
        lhs = self.P_A * self.V_A
        rhs = self.n * self.R * self.T_A
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ideal gas law violated at A: P_A*V_A={lhs} vs nRT_A={rhs}")
        return (True, "Ideal gas law satisfied at A.")

    def test_invariant_isobaric_work(self):
        """Check that W_AB = P_A * (V_B - V_A)."""
        if any(x is None for x in [self.W_AB, self.P_A, self.V_A, self.V_B]):
            return (True, "SKIP: Missing data for isobaric work check.")
        expected = self.P_A * (self.V_B - self.V_A)
        if not math.isclose(self.W_AB, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W_AB mismatch: computed {self.W_AB}, expected {expected}")
        return (True, f"Isobaric work consistent; W_AB = {self.W_AB}")

    def test_invariant_internal_energy(self):
        """Check ΔU_AB = n Cv (T_B - T_A) where T_B = P_A V_B / (n R)."""
        if any(x is None for x in [self.delta_U_AB, self.n, self.Cv, self.P_A, self.V_B, self.R, self.T_A]):
            return (True, "SKIP: Missing data for internal energy check.")
        T_B = self.P_A * self.V_B / (self.n * self.R)
        expected = self.n * self.Cv * (T_B - self.T_A)
        if not math.isclose(self.delta_U_AB, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ΔU_AB mismatch: computed {self.delta_U_AB}, expected {expected}")
        return (True, "Internal energy change consistent.")

    def test_invariant_first_law_AB(self):
        """Check Q_AB = ΔU_AB + W_AB (First Law)."""
        if any(x is None for x in [self.Q_AB, self.delta_U_AB, self.W_AB]):
            return (True, "SKIP: Missing data for First Law check.")
        expected = self.delta_U_AB + self.W_AB
        if not math.isclose(self.Q_AB, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_AB mismatch: computed {self.Q_AB}, expected {expected}")
        return (True, "First law satisfied for process A→B.")

    def test_invariant_adiabatic_entropy(self):
        """Check ΔS_BC = 0 for reversible adiabatic process."""
        if self.delta_S_BC is None:
            return (True, "SKIP: Missing entropy change value.")
        # Accept both 0 and equivalent string or number
        try:
            val = float(self.delta_S_BC)
        except (TypeError, ValueError):
            return (False, f"Entropy change '{self.delta_S_BC}' cannot be converted to float.")
        if not math.isclose(val, 0.0, abs_tol=1e-6):
            return (False, f"ΔS_BC should be zero, got {val}.")
        return (True, "Entropy change for adiabatic process is zero.")

    def test_invariant_adiabatic_pressure(self):
        """Check P_C = P_A * (V_B / V_C)^gamma for reversible adiabatic B→C."""
        if any(x is None for x in [self.P_C, self.P_A, self.V_B, self.V_C, self.gamma]):
            return (True, "SKIP: Missing data for adiabatic pressure check.")
        expected = self.P_A * (self.V_B / self.V_C) ** self.gamma
        if not math.isclose(self.P_C, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_C mismatch: computed {self.P_C}, expected {expected}")
        return (True, "Adiabatic pressure relation holds.")

    # ------- Modeling / Trace Tests -------

    def test_modeling_heat_capacity_ratio(self):
        """Check gamma = Cp/Cv = 5/3 for monatomic ideal gas."""
        if self.gamma is None:
            return (True, "SKIP: gamma not available.")
        if not math.isclose(self.gamma, 5.0/3.0, rel_tol=1e-3):
            return (False, f"Gamma should be 5/3, got {self.gamma}.")
        return (True, "Heat capacity ratio consistent with monatomic gas.")

    def test_trace_work_positive(self):
        """Work done by gas in expansion (W_AB) should be positive."""
        if self.W_AB is None:
            return (True, "SKIP: W_AB not available.")
        if self.W_AB <= 0:
            return (False, f"W_AB should be positive for expansion, got {self.W_AB}.")
        return (True, "W_AB is positive.")

    def test_trace_temperature_increase(self):
        """In isobaric expansion, temperature should increase (T_B > T_A)."""
        if any(x is None for x in [self.T_A, self.P_A, self.V_B, self.n, self.R]):
            return (True, "SKIP: Cannot compute T_B.")
        T_B = self.P_A * self.V_B / (self.n * self.R)
        if T_B <= self.T_A:
            return (False, f"T_B={T_B} should be > T_A={self.T_A} for expansion.")
        return (True, f"Temperature increases: T_B={T_B} > T_A={self.T_A}.")

    def test_trace_adiabatic_pressure_decrease(self):
        """During adiabatic expansion, pressure decreases: P_C < P_A (since P_B = P_A)."""
        if any(x is None for x in [self.P_C, self.P_A]):
            return (True, "SKIP: Missing pressures for comparison.")
        if self.P_C >= self.P_A:
            return (False, f"P_C={self.P_C} should be less than P_A={self.P_A} for expansion.")
        return (True, "Pressure decreases in adiabatic expansion.")

    # ------- Monotonicity Tests -------

    def test_monotonicity_work_increases_with_volume(self):
        """
        Check that W_AB increases with V_B. Compute baseline and perturb V_B upward by 1%.
        """
        if any(x is None for x in [self.P_A, self.V_A, self.V_B, self.W_AB]):
            return (True, "SKIP: Missing data for monotonicity of work.")
        # Baseline work from formula using given V_B
        W_baseline = self.P_A * (self.V_B - self.V_A)
        # Perturb V_B: increase by 1%
        V_B_pert = self.V_B * 1.01
        W_pert = self.P_A * (V_B_pert - self.V_A)
        # The monotonicity: W_pert > W_baseline
        if W_pert <= W_baseline:
            return (False, f"Work should increase with V_B; baseline {W_baseline}, perturbed {W_pert}.")
        # Also ensure the stored W_AB is consistent with baseline
        if not math.isclose(self.W_AB, W_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Stored W_AB {self.W_AB} does not match computed baseline {W_baseline}.")
        return (True, "Monotonicity of work with volume holds.")

    def test_monotonicity_heat_increases_with_volume(self):
        """
        Check that Q_AB increases with V_B (since Q = nCp(T_B-T_A) and T_B increases with V_B).
        Perturb V_B upward by 1% and compute the expected Q.
        """
        if any(x is None for x in [self.n, self.Cp, self.P_A, self.V_A, self.V_B, self.R, self.T_A, self.Q_AB]):
            return (True, "SKIP: Missing data for monotonicity of heat.")
        # Baseline T_B and Q
        T_B_baseline = self.P_A * self.V_B / (self.n * self.R)
        Q_baseline = self.n * self.Cp * (T_B_baseline - self.T_A)
        # Perturb V_B
        V_B_pert = self.V_B * 1.01
        T_B_pert = self.P_A * V_B_pert / (self.n * self.R)
        Q_pert = self.n * self.Cp * (T_B_pert - self.T_A)
        if Q_pert <= Q_baseline:
            return (False, f"Heat should increase with V_B; baseline {Q_baseline}, perturbed {Q_pert}.")
        # Check stored Q_AB consistent with baseline
        if not math.isclose(self.Q_AB, Q_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Stored Q_AB {self.Q_AB} does not match computed baseline {Q_baseline}.")
        return (True, "Monotonicity of heat with volume holds.")

    # ------- Weak Answer Sanity Check -------

    def test_answer_sanity_magnitudes(self):
        """
        Weak check: All energy quantities (W_AB, ΔU_AB, Q_AB) should be of order 1e5 J,
        P_C should be of order 1e5 Pa, and ΔS_BC should be zero.
        No isclose, no re-derivation.
        """
        failures = []
        # W_AB
        if self.W_AB is not None:
            if not (1e4 <= self.W_AB <= 1e6):
                failures.append(f"W_AB magnitude suspicious: {self.W_AB}")
        # ΔU_AB
        if self.delta_U_AB is not None:
            if not (1e4 <= self.delta_U_AB <= 1e6):
                failures.append(f"ΔU_AB magnitude suspicious: {self.delta_U_AB}")
        # Q_AB
        if self.Q_AB is not None:
            if not (1e4 <= self.Q_AB <= 1e6):
                failures.append(f"Q_AB magnitude suspicious: {self.Q_AB}")
        # P_C
        if self.P_C is not None:
            if not (1e4 <= self.P_C <= 1e6):
                failures.append(f"P_C magnitude suspicious: {self.P_C}")
        # ΔS_BC
        if self.delta_S_BC is not None:
            try:
                val = float(self.delta_S_BC)
                if not (-0.1 <= val <= 0.1):
                    failures.append(f"ΔS_BC should be near zero, got {val}")
            except (TypeError, ValueError):
                failures.append(f"ΔS_BC not numeric: {self.delta_S_BC}")
        if failures:
            return (False, "; ".join(failures))
        return (True, "Answer magnitude ranges plausible.")

    # ------- Main runner -------

    def run_all_tests(self):
        """
        Discover and run all test_ methods, collect logs.
        Returns (all_pass, logs_list).
        """
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