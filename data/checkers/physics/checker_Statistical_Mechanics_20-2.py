import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (defaults to None if missing)
        self.C = getattr(solution_instance, 'C', None)
        self.tau_h = getattr(solution_instance, 'tau_h', None)
        self.tau_l = getattr(solution_instance, 'tau_l', None)
        # Extract results from the results dictionary
        results = getattr(solution_instance, 'results', {})
        self.T_f = results.get('T_f', None)
        self.W_max = results.get('W_max', None)

    # ----------------------------------------------------------------------
    # Interface tests (required keys & types)
    # ----------------------------------------------------------------------
    def test_interface_required_givens(self):
        if self.C is None:
            return (False, "Missing given: C")
        if self.tau_h is None:
            return (False, "Missing given: tau_h")
        if self.tau_l is None:
            return (False, "Missing given: tau_l")
        if self.tau_h <= self.tau_l:
            return (False, "tau_h must be greater than tau_l")
        if self.C <= 0:
            return (False, "C must be positive")
        return (True, "All givens present and consistent")

    def test_interface_required_results(self):
        if self.T_f is None:
            return (False, "Missing result: T_f")
        if self.W_max is None:
            return (False, "Missing result: W_max")
        return (True, "Results T_f and W_max present")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_entropy_zero(self):
        if any(v is None for v in (self.C, self.tau_h, self.tau_l, self.T_f)):
            return (False, "SKIP: missing required data for entropy check")
        delta_S = self.C * math.log(self.T_f / self.tau_h) + self.C * math.log(self.T_f / self.tau_l)
        if math.isclose(delta_S, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Total entropy change is zero")
        else:
            return (False, f"Total entropy change is {delta_S}, not zero")

    def test_invariant_final_temperature_range(self):
        if any(v is None for v in (self.tau_l, self.tau_h, self.T_f)):
            return (False, "SKIP: missing required data for range check")
        if self.tau_l < self.T_f < self.tau_h:
            return (True, f"T_f={self.T_f} is between tau_l={self.tau_l} and tau_h={self.tau_h}")
        else:
            return (False, f"T_f={self.T_f} is not between reservoirs' initial temperatures")

    def test_invariant_work_formula(self):
        if any(v is None for v in (self.C, self.tau_h, self.tau_l, self.W_max)):
            return (False, "SKIP: missing required data for work formula check")
        expected_W = self.C * (math.sqrt(self.tau_h) - math.sqrt(self.tau_l))**2
        if math.isclose(self.W_max, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "W_max matches the derived formula")
        else:
            return (False, f"W_max={self.W_max} does not match expected {expected_W}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (one relation at a time)
    # ----------------------------------------------------------------------
    def test_modeling_work_via_heat(self):
        if any(v is None for v in (self.C, self.tau_h, self.tau_l, self.T_f, self.W_max)):
            return (False, "SKIP: missing required data for work via heat check")
        Q_h = self.C * (self.tau_h - self.T_f)
        Q_l = self.C * (self.T_f - self.tau_l)
        W_calc = Q_h - Q_l
        if math.isclose(self.W_max, W_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "W_max equals Q_h - Q_l")
        else:
            return (False, f"W_max={self.W_max} != Q_h-Q_l={W_calc}")

    def test_modeling_heat_positive(self):
        if any(v is None for v in (self.tau_h, self.tau_l, self.T_f)):
            return (False, "SKIP: missing required data for heat sign check")
        if self.tau_h > self.T_f and self.T_f > self.tau_l:
            return (True, "Heat transfer directions are correct (Q_h>0, Q_l>0)")
        else:
            return (False, "Temperature ordering does not allow positive heat transfers")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb a parameter and compare)
    # ----------------------------------------------------------------------
    def test_monotonicity_work_increases_with_tau_h(self):
        if any(v is None for v in (self.C, self.tau_h, self.tau_l, self.W_max)):
            return (False, "SKIP: missing required data for monotonicity check")
        epsilon = 0.01 * self.tau_h
        tau_h_pert = self.tau_h + epsilon
        # Use the derived formula for the perturbed value
        W_pert = self.C * (math.sqrt(tau_h_pert) - math.sqrt(self.tau_l))**2
        if W_pert > self.W_max:
            return (True, "Work increases with tau_h (as expected)")
        else:
            return (False, f"Perturbed work {W_pert} not greater than baseline {self.W_max}")

    def test_monotonicity_work_decreases_with_tau_l(self):
        if any(v is None for v in (self.C, self.tau_h, self.tau_l, self.W_max)):
            return (False, "SKIP: missing required data for monotonicity check")
        epsilon = 0.01 * self.tau_l
        tau_l_pert = self.tau_l + epsilon
        W_pert = self.C * (math.sqrt(self.tau_h) - math.sqrt(tau_l_pert))**2
        if W_pert < self.W_max:
            return (True, "Work decreases with tau_l (as expected)")
        else:
            return (False, f"Perturbed work {W_pert} not less than baseline {self.W_max}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (positive, finite – NO math.isclose, NO re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        logs = []
        all_ok = True
        if self.W_max is not None:
            if self.W_max <= 0:
                all_ok = False
                logs.append("W_max non-positive")
            elif not math.isfinite(self.W_max):
                all_ok = False
                logs.append("W_max not finite")
        else:
            logs.append("W_max missing")
            all_ok = False
        if self.T_f is not None:
            if self.T_f <= 0:
                all_ok = False
                logs.append("T_f non-positive")
            elif not math.isfinite(self.T_f):
                all_ok = False
                logs.append("T_f not finite")
        else:
            logs.append("T_f missing")
            all_ok = False
        if all_ok:
            return (True, "Answers are positive and finite")
        else:
            return (False, "; ".join(logs))

    # ----------------------------------------------------------------------
    # Test harness – discover and run all test_* methods
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)