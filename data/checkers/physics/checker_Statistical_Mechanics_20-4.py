import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens with None defaults (NO numeric fallbacks)
        self.n = getattr(solution_instance, 'n', None)
        self.R = getattr(solution_instance, 'R', None)          # universal constant, but still None if missing
        self.T1 = getattr(solution_instance, 'T1', None)       # initial temperature (K)
        self.V1 = getattr(solution_instance, 'V1', None)       # initial volume (m^3)
        self.m_car = getattr(solution_instance, 'm_car', None)
        self.v_car = getattr(solution_instance, 'v_car', None) # speed in m/s
        self.T_env = getattr(solution_instance, 'T_env', None)
        self.m_Al = getattr(solution_instance, 'm_Al', None)
        self.c_Al = getattr(solution_instance, 'c_Al', None)
        self.T_c = getattr(solution_instance, 'T_c', None)
        self.m_w = getattr(solution_instance, 'm_w', None)
        self.c_w = getattr(solution_instance, 'c_w', None)
        self.T_w = getattr(solution_instance, 'T_w', None)

        # Derived constants – only gamma is R‑independent
        self.gamma = 7.0 / 5.0   # diatomic, fixed

        # Results from solution_instance.results dict
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.delta_U_total = results.get('delta_U_total', None)
        self.W_total = results.get('W_total', None)
        self.Q_total = results.get('Q_total', None)
        self.final_volume = results.get('final_volume', None)
        self.delta_S_cars = results.get('delta_S_cars', None)
        self.delta_S_total = results.get('delta_S_total', None)

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_required_results_present(self):
        """All required result keys must be present and not None."""
        missing = []
        if self.delta_U_total is None:
            missing.append('delta_U_total')
        if self.W_total is None:
            missing.append('W_total')
        if self.Q_total is None:
            missing.append('Q_total')
        if self.final_volume is None:
            missing.append('final_volume')
        if self.delta_S_cars is None:
            missing.append('delta_S_cars')
        if self.delta_S_total is None:
            missing.append('delta_S_total')
        if missing:
            return False, f"Missing result keys: {', '.join(missing)}"
        return True, "All required results present."

    def test_interface_givens_available(self):
        """Check that essential givens are not None (otherwise cannot verify)."""
        missing = []
        if self.n is None:
            missing.append('n')
        if self.T1 is None:
            missing.append('T1')
        if self.V1 is None:
            missing.append('V1')
        if self.m_car is None:
            missing.append('m_car')
        if self.v_car is None:
            missing.append('v_car')
        if self.T_env is None:
            missing.append('T_env')
        if self.m_Al is None:
            missing.append('m_Al')
        if self.c_Al is None:
            missing.append('c_Al')
        if self.T_c is None:
            missing.append('T_c')
        if self.m_w is None:
            missing.append('m_w')
        if self.c_w is None:
            missing.append('c_w')
        if self.T_w is None:
            missing.append('T_w')
        if missing:
            return False, f"Missing given variables: {', '.join(missing)}"
        return True, "All essential givens available."

    # ----------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ----------------------------------------------------------------
    def test_invariant_delta_U_total_zero(self):
        """Overall change in internal energy must be zero because T returns to initial."""
        if self.delta_U_total is None:
            return False, "SKIP: delta_U_total not available"
        if abs(self.delta_U_total) > 1e-9:
            return False, f"delta_U_total should be 0, got {self.delta_U_total}"
        return True, "delta_U_total is exactly zero."

    def test_invariant_first_law_cycle(self):
        """For the cycle, Q_total - W_total = delta_U_total = 0."""
        if None in (self.Q_total, self.W_total, self.delta_U_total):
            return False, "SKIP: one of the total quantities missing"
        if not math.isclose(self.Q_total - self.W_total, self.delta_U_total, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Q - W = {self.Q_total - self.W_total} but delta_U = {self.delta_U_total}"
        return True, "First law holds for the entire cycle."

    def test_invariant_adiabatic_final_volume_given_inputs(self):
        """Re-derive final volume from given T1, V1, gamma and compare with result."""
        if self.final_volume is None:
            return False, "SKIP: final_volume not available"
        if None in (self.T1, self.V1):
            return False, "SKIP: T1 or V1 missing"
        # V2 = 2*V1, T2 = 2*T1 (isobaric doubling)
        V2 = 2.0 * self.V1
        T2 = 2.0 * self.T1
        exponent = 1.0 / (self.gamma - 1.0)
        expected_V3 = V2 * (T2 / self.T1) ** exponent   # = V2 * 2^(1/(γ-1))
        if not math.isclose(self.final_volume, expected_V3, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Final volume {self.final_volume} not close to expected {expected_V3}"
        return True, "Final volume consistent with adiabatic relation."

    def test_invariant_car_entropy_via_KE(self):
        """Entropy change for cars should equal total KE / T_env."""
        if None in (self.delta_S_cars, self.m_car, self.v_car, self.T_env):
            return False, "SKIP: required variables missing"
        expected = (self.m_car * self.v_car ** 2) / self.T_env   # 2 * 0.5 * m * v^2 = m * v^2
        if not math.isclose(self.delta_S_cars, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"delta_S_cars {self.delta_S_cars} not close to expected {expected}"
        return True, "Car entropy change consistent with KE dissipation."

    def test_invariant_cup_water_entropy_via_Tf(self):
        """Total entropy change from cup+water must satisfy equilibrium formula."""
        if None in (self.delta_S_total, self.m_Al, self.c_Al, self.T_c, self.m_w, self.c_w, self.T_w):
            return False, "SKIP: cup/water variables missing"
        # Equilibrium temperature
        num = self.m_Al * self.c_Al * self.T_c + self.m_w * self.c_w * self.T_w
        den = self.m_Al * self.c_Al + self.m_w * self.c_w
        if den == 0:
            return False, "Denominator zero in Tf calculation"
        Tf = num / den
        expected = (self.m_Al * self.c_Al * math.log(Tf / self.T_c) +
                    self.m_w * self.c_w * math.log(Tf / self.T_w))
        if not math.isclose(self.delta_S_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"delta_S_total {self.delta_S_total} not close to expected {expected}"
        return True, "Cup+water entropy change consistent."

    # ----------------------------------------------------------------
    # Modeling/trace tests
    # ----------------------------------------------------------------
    def test_modeling_Q_total_equals_W_total(self):
        """Since overall delta_U=0, Q_total must equal W_total (within tolerance)."""
        if None in (self.Q_total, self.W_total):
            return False, "SKIP: one of Q_total or W_total missing"
        if not math.isclose(self.Q_total, self.W_total, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Q_total {self.Q_total} not close to W_total {self.W_total}"
        return True, "Q_total = W_total as required by first law."

    def test_modeling_Q_positive_finite(self):
        """Heat supplied should be positive and finite (order 10^4 J)."""
        if self.Q_total is None:
            return False, "SKIP: Q_total missing"
        if self.Q_total <= 0:
            return False, f"Q_total should be positive, got {self.Q_total}"
        if not math.isfinite(self.Q_total):
            return False, "Q_total is infinite or NaN"
        if self.Q_total < 1e3 or self.Q_total > 1e5:
            return False, f"Q_total magnitude {self.Q_total} outside expected range (1e3-1e5)"
        return True, "Q_total positive and within reasonable magnitude."

    def test_modeling_final_volume_positive(self):
        """Final volume must be positive and greater than initial volume."""
        if self.final_volume is None:
            return False, "SKIP: final_volume missing"
        if self.final_volume <= 0:
            return False, f"final_volume <= 0: {self.final_volume}"
        if self.V1 is not None and self.final_volume <= self.V1:
            return False, f"Final volume {self.final_volume} not > initial {self.V1}"
        return True, "final_volume positive and > V1."

    def test_modeling_delta_S_cars_positive_finite(self):
        """Car entropy change should be positive and order kJ/K."""
        if self.delta_S_cars is None:
            return False, "SKIP: delta_S_cars missing"
        if self.delta_S_cars <= 0:
            return False, f"delta_S_cars should be positive, got {self.delta_S_cars}"
        if not math.isfinite(self.delta_S_cars):
            return False, "delta_S_cars is infinite or NaN"
        if self.delta_S_cars < 1e2 or self.delta_S_cars > 1e4:
            return False, f"delta_S_cars magnitude {self.delta_S_cars} outside expected range (1e2-1e4)"
        return True, "delta_S_cars positive and reasonable magnitude."

    def test_modeling_delta_S_total_positive(self):
        """Cup+water entropy change should be positive and order J/K."""
        if self.delta_S_total is None:
            return False, "SKIP: delta_S_total missing"
        if self.delta_S_total <= 0:
            return False, f"delta_S_total should be positive, got {self.delta_S_total}"
        if not math.isfinite(self.delta_S_total):
            return False, "delta_S_total is infinite or NaN"
        if self.delta_S_total < 0.1 or self.delta_S_total > 10:
            return False, f"delta_S_total magnitude {self.delta_S_total} outside expected range (0.1-10)"
        return True, "delta_S_total positive and within reasonable magnitude."

    # ----------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input)
    # ----------------------------------------------------------------
    def test_monotonicity_work_increases_with_n(self):
        """Increasing amount of gas (n) by 10% should increase total work (theoretically)."""
        if None in (self.n, self.W_total, self.R, self.T1):
            return False, "SKIP: required variables (n, R, T1) missing"
        if self.R == 0:
            return False, "R is zero, cannot compute baseline"
        gamma = self.gamma
        coeff = gamma / (gamma - 1.0)
        base_W = coeff * self.n * self.R * self.T1   # theoretical total work
        # First verify that solution's W_total matches the theoretical baseline
        if not math.isclose(self.W_total, base_W, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"W_total {self.W_total} does not match theoretical base {base_W}"
        # Now perturb n by +10%
        n_pert = self.n * 1.1
        perturbed_W = coeff * n_pert * self.R * self.T1
        if not (perturbed_W > base_W):
            return False, "Perturbed work is not larger than baseline (monotonicity violation)"
        return True, "Work increases with n (theoretical monotonicity confirmed)."

    def test_monotonicity_car_entropy_increases_with_mass(self):
        """Increasing car mass by 10% should increase entropy change (theoretically)."""
        if None in (self.delta_S_cars, self.m_car, self.v_car, self.T_env):
            return False, "SKIP: required variables missing"
        if self.T_env == 0:
            return False, "T_env is zero, cannot compute baseline"
        base_S = (self.m_car * self.v_car ** 2) / self.T_env
        # Verify solution's delta_S_cars matches theoretical
        if not math.isclose(self.delta_S_cars, base_S, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"delta_S_cars {self.delta_S_cars} does not match theoretical base {base_S}"
        m_pert = self.m_car * 1.1
        perturbed_S = (m_pert * self.v_car ** 2) / self.T_env
        if not (perturbed_S > base_S):
            return False, "Perturbed entropy not larger than baseline (monotonicity violation)"
        return True, "Car entropy increases with mass (theoretical monotonicity confirmed)."

    # ----------------------------------------------------------------
    # Weak answer sanity (only one, no hardcoded target values)
    # ----------------------------------------------------------------
    def test_answer_sanity_signs_and_finite(self):
        """Weak sanity: all result quantities have correct signs and are finite."""
        issues = []
        if self.delta_U_total is not None and abs(self.delta_U_total) > 1e-6:
            issues.append(f"delta_U_total not zero: {self.delta_U_total}")
        if self.W_total is not None and (self.W_total <= 0 or not math.isfinite(self.W_total)):
            issues.append("W_total not positive/finite")
        if self.Q_total is not None and (self.Q_total <= 0 or not math.isfinite(self.Q_total)):
            issues.append("Q_total not positive/finite")
        if self.final_volume is not None and (self.final_volume <= 0 or not math.isfinite(self.final_volume)):
            issues.append("final_volume not positive/finite")
        if self.delta_S_cars is not None and (self.delta_S_cars <= 0 or not math.isfinite(self.delta_S_cars)):
            issues.append("delta_S_cars not positive/finite")
        if self.delta_S_total is not None and (self.delta_S_total <= 0 or not math.isfinite(self.delta_S_total)):
            issues.append("delta_S_total not positive/finite")
        if issues:
            return False, "; ".join(issues)
        return True, "All signs and finiteness checks passed."

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
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