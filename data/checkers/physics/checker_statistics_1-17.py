import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the thermodynamic problem involving ideal gas:
    (a) isobaric heat, (b) internal energy change, (c) work done, (d) isochoric heat.
    """
    def __init__(self, solution_instance):
        # Extract results from solution_instance
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.Q_p = results.get('Q_p', None)
        self.delta_U = results.get('delta_U', None)
        self.W = results.get('W', None)
        self.Q_v = results.get('Q_v', None)

        # Givens: extract from solution_instance or results; default to None
        self.mass = getattr(solution_instance, 'mass', results.get('mass', None))
        self.M_N2 = getattr(solution_instance, 'M_N2', results.get('M_N2', None))
        self.delta_T = getattr(solution_instance, 'delta_T', results.get('delta_T', None))
        self.c_v = getattr(solution_instance, 'c_v', results.get('c_v', None))
        self.R = getattr(solution_instance, 'R', results.get('R', None))

        # Derived quantities
        if self.c_v is not None and self.R is not None:
            self.c_p = self.c_v + self.R
        else:
            self.c_p = None

        if self.mass is not None and self.M_N2 is not None and self.M_N2 != 0:
            self.n = self.mass / self.M_N2
        else:
            self.n = None

    def _skip_message(self, prereq_name):
        return f"SKIP: {prereq_name} not available"

    # ---------------- Interface tests ----------------
    def test_interface_Q_p_exists(self):
        if self.Q_p is None:
            return (False, self._skip_message("Q_p"))
        if not isinstance(self.Q_p, (int, float)):
            return (False, "Q_p is not numeric")
        return (True, "Q_p present and numeric")

    def test_interface_delta_U_exists(self):
        if self.delta_U is None:
            return (False, self._skip_message("delta_U"))
        if not isinstance(self.delta_U, (int, float)):
            return (False, "delta_U is not numeric")
        return (True, "delta_U present and numeric")

    def test_interface_W_exists(self):
        if self.W is None:
            return (False, self._skip_message("W"))
        if not isinstance(self.W, (int, float)):
            return (False, "W is not numeric")
        return (True, "W present and numeric")

    def test_interface_Q_v_exists(self):
        if self.Q_v is None:
            return (False, self._skip_message("Q_v"))
        if not isinstance(self.Q_v, (int, float)):
            return (False, "Q_v is not numeric")
        return (True, "Q_v present and numeric")

    # ---------------- Invariant tests (use math.isclose) ----------------
    def test_invariant_first_law(self):
        # First law: Q_p = ΔU + W (isobaric, work out positive)
        if any(x is None for x in (self.Q_p, self.delta_U, self.W)):
            return (False, self._skip_message("Q_p, delta_U, or W"))
        total = self.delta_U + self.W
        if math.isclose(self.Q_p, total, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_p ≈ ΔU + W: {self.Q_p} ≈ {total}")
        else:
            return (False, f"First law violated: Q_p={self.Q_p}, ΔU+W={total}")

    def test_invariant_isochoric_heat(self):
        # At constant volume, Q_v = ΔU
        if any(x is None for x in (self.Q_v, self.delta_U)):
            return (False, self._skip_message("Q_v or delta_U"))
        if math.isclose(self.Q_v, self.delta_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_v ≈ ΔU: {self.Q_v} ≈ {self.delta_U}")
        else:
            return (False, f"Isochoric heat differs from ΔU: Q_v={self.Q_v}, ΔU={self.delta_U}")

    def test_invariant_work_via_nRdeltaT(self):
        # Work for isobaric ideal gas: W = n R ΔT
        if any(x is None for x in (self.n, self.R, self.delta_T, self.W)):
            return (False, self._skip_message("n, R, delta_T, or W"))
        expected_W = self.n * self.R * self.delta_T
        if math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W ≈ n R ΔT: {self.W} ≈ {expected_W}")
        else:
            return (False, f"Work not consistent with nRΔT: W={self.W}, expected={expected_W}")

    # ---------------- Modeling tests (use math.isclose) ----------------
    def test_modeling_deltaU_cv(self):
        # ΔU = n c_v ΔT
        if any(x is None for x in (self.n, self.c_v, self.delta_T, self.delta_U)):
            return (False, self._skip_message("n, c_v, delta_T, or delta_U"))
        expected = self.n * self.c_v * self.delta_T
        if math.isclose(self.delta_U, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔU ≈ n c_v ΔT: {self.delta_U} ≈ {expected}")
        else:
            return (False, f"ΔU deviates: ΔU={self.delta_U}, expected n c_v ΔT={expected}")

    def test_modeling_Qp_cp(self):
        # Q_p = n c_p ΔT, with c_p = c_v + R
        if any(x is None for x in (self.n, self.c_p, self.delta_T, self.Q_p)):
            return (False, self._skip_message("n, c_p, delta_T, or Q_p"))
        expected = self.n * self.c_p * self.delta_T
        if math.isclose(self.Q_p, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_p ≈ n c_p ΔT: {self.Q_p} ≈ {expected}")
        else:
            return (False, f"Q_p deviates: Q_p={self.Q_p}, expected n c_p ΔT={expected}")

    def test_modeling_Qv_via_ncv(self):
        # Q_v = n c_v ΔT
        if any(x is None for x in (self.n, self.c_v, self.delta_T, self.Q_v)):
            return (False, self._skip_message("n, c_v, delta_T, or Q_v"))
        expected = self.n * self.c_v * self.delta_T
        if math.isclose(self.Q_v, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_v ≈ n c_v ΔT: {self.Q_v} ≈ {expected}")
        else:
            return (False, f"Q_v deviates: Q_v={self.Q_v}, expected n c_v ΔT={expected}")

    # ---------------- Monotonicity tests (perturbation) ----------------
    def test_monotonicity_mass_increase(self):
        # If mass increases, all heat/energy/work should increase proportionally.
        if any(x is None for x in (self.mass, self.M_N2, self.c_p, self.c_v, self.R, self.delta_T, self.Q_p, self.delta_U, self.W, self.Q_v)):
            return (False, self._skip_message("mass, M_N2, c_p, c_v, R, delta_T, or results"))

        # Perturb mass by +10%
        pert_mass = self.mass * 1.1
        pert_n = pert_mass / self.M_N2
        pert_Q_p = pert_n * self.c_p * self.delta_T
        pert_delta_U = pert_n * self.c_v * self.delta_T
        pert_W = pert_n * self.R * self.delta_T
        pert_Q_v = pert_n * self.c_v * self.delta_T

        checks = []
        if pert_Q_p <= self.Q_p:
            checks.append("Q_p should increase with mass")
        if pert_delta_U <= self.delta_U:
            checks.append("delta_U should increase with mass")
        if pert_W <= self.W:
            checks.append("W should increase with mass")
        if pert_Q_v <= self.Q_v:
            checks.append("Q_v should increase with mass")

        if checks:
            return (False, "; ".join(checks))
        return (True, "All perturbed quantities > baseline, monotonicity with mass satisfied")

    def test_monotonicity_temperature_increase(self):
        # If ΔT increases, all heat/energy/work should increase.
        if any(x is None for x in (self.n, self.c_p, self.c_v, self.R, self.delta_T, self.Q_p, self.delta_U, self.W, self.Q_v)):
            return (False, self._skip_message("n, c_p, c_v, R, delta_T, or results"))

        pert_delta_T = self.delta_T + 1  # increase by 1°C
        pert_Q_p = self.n * self.c_p * pert_delta_T
        pert_delta_U = self.n * self.c_v * pert_delta_T
        pert_W = self.n * self.R * pert_delta_T
        pert_Q_v = self.n * self.c_v * pert_delta_T

        checks = []
        if pert_Q_p <= self.Q_p:
            checks.append("Q_p should increase with ΔT")
        if pert_delta_U <= self.delta_U:
            checks.append("delta_U should increase with ΔT")
        if pert_W <= self.W:
            checks.append("W should increase with ΔT")
        if pert_Q_v <= self.Q_v:
            checks.append("Q_v should increase with ΔT")

        if checks:
            return (False, "; ".join(checks))
        return (True, "All perturbed quantities > baseline, monotonicity with ΔT satisfied")

    # ---------------- One weak answer sanity test ----------------
    def test_answer_sanity_broad_range(self):
        # All results should be finite, positive, and within a physically reasonable range.
        # Do NOT use isclose or re-derive values.
        results_list = [self.Q_p, self.delta_U, self.W, self.Q_v]
        for i, val in enumerate(results_list):
            if val is None:
                return (False, f"SKIP: result at index {i} missing")
            if not isinstance(val, (int, float)):
                return (False, f"result at index {i} not numeric")
            if not math.isfinite(val):
                return (False, f"result at index {i} is infinite or NaN")
            if val <= 0:
                return (False, f"result at index {i} is not positive")
        # Broad magnitude check: all should be between 1000 and 100000 cal (reasonable for this problem)
        for val in results_list:
            if not (1000 <= val <= 100000):
                return (False, f"result {val} out of reasonable range [1000, 100000]")
        return (True, "All results finite, positive, and in broad range")

    # ---------------- Runner ----------------
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)