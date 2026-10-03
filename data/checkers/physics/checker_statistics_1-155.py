import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for ice thickness growth problem.
    Extracts required givens and results safely; performs unit tests,
    invariant checks, modeling consistency, and monotonicity tests.
    """
    def __init__(self, solution_instance):
        # -- Givens (extracted safely, default None) --
        self.kappa = getattr(solution_instance, 'kappa', None)
        self.deltaT = getattr(solution_instance, 'deltaT', None)
        if self.deltaT is None:
            T_m = getattr(solution_instance, 'T_m', None)
            T_s = getattr(solution_instance, 'T_s', None)
            if T_m is not None and T_s is not None:
                self.deltaT = T_m - T_s
        self.rho = getattr(solution_instance, 'rho', None)
        self.latent_heat = getattr(solution_instance, 'L', None)      # latent heat per mass
        self.z0 = getattr(solution_instance, 'z0', None)              # initial thickness

        # -- Results (extracted from solution_instance.results dict or as attribute) --
        results = getattr(solution_instance, 'results', {})
        self.rate_formula = results.get('rate_formula', None) or getattr(solution_instance, 'rate_formula', None)
        self.time_sec = results.get('time_sec', None) or getattr(solution_instance, 'time_sec', None)
        self.time_min = results.get('time_min', None) or getattr(solution_instance, 'time_min', None)

        # Helper flag to know if all required givens are present
        self.have_givens = all([
            self.kappa is not None,
            self.deltaT is not None,
            self.rho is not None,
            self.latent_heat is not None,
            self.z0 is not None
        ])

    # ----------------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_has_rate_formula(self):
        if self.rate_formula is None:
            return (False, "Missing 'rate_formula' in results")
        return (True, "rate_formula present")

    def test_interface_has_time_sec(self):
        if self.time_sec is None:
            return (False, "Missing 'time_sec' in results")
        return (True, "time_sec present")

    def test_interface_has_time_min(self):
        if self.time_min is None:
            return (False, "Missing 'time_min' in results")
        return (True, "time_min present")

    def test_interface_rate_formula_is_string(self):
        if self.rate_formula is None:
            return (False, "SKIP: rate_formula missing")
        if not isinstance(self.rate_formula, str):
            return (False, f"rate_formula is not a string, got {type(self.rate_formula)}")
        return (True, "rate_formula is string")

    def test_interface_time_sec_is_numeric(self):
        if self.time_sec is None:
            return (False, "SKIP: time_sec missing")
        if not isinstance(self.time_sec, (int, float)):
            return (False, f"time_sec is not numeric, got {type(self.time_sec)}")
        return (True, "time_sec is numeric")

    def test_interface_time_min_is_numeric(self):
        if self.time_min is None:
            return (False, "SKIP: time_min missing")
        if not isinstance(self.time_min, (int, float)):
            return (False, f"time_min is not numeric, got {type(self.time_min)}")
        return (True, "time_min is numeric")

    # ----------------------------------------------------------------------
    # Invariant tests – theorem‐based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_rate_formula_structure(self):
        """Check that the rate formula contains essential physical symbols."""
        if self.rate_formula is None:
            return (False, "SKIP: rate_formula missing")
        text = self.rate_formula.lower()
        # Accept either 'kappa' or '\kappa'
        has_kappa = 'kappa' in text or '\\kappa' in text
        has_rho = 'rho' in text or '\\rho' in text
        has_L = ' L ' in text or 'L' in text  # crude but acceptable
        if not (has_kappa and has_rho and has_L):
            detail = []
            if not has_kappa: detail.append("missing kappa/\\kappa")
            if not has_rho: detail.append("missing rho/\\rho")
            if not has_L: detail.append("missing L")
            return (False, f"rate_formula lacks essential symbols: {', '.join(detail)}")
        return (True, "rate_formula contains kappa, rho, L")

    def test_invariant_growth_rate_positive(self):
        """At initial thickness, the growth rate must be positive."""
        if not self.have_givens:
            return (False, "SKIP: givens incomplete")
        try:
            rate = self.kappa * self.deltaT / (self.rho * self.latent_heat * self.z0)
            if rate <= 0:
                return (False, f"Computed growth rate {rate:.6g} is not positive")
            return (True, f"Growth rate at z0: {rate:.6g} cm/s > 0")
        except Exception as e:
            return (False, f"test_invariant_growth_rate_positive crashed: {e}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests – check consistency with derived physics
    # ----------------------------------------------------------------------
    def test_modeling_time_sec_derived(self):
        """Re‑compute the doubling time from givens and compare with time_sec."""
        if not self.have_givens or self.time_sec is None:
            return (False, "SKIP: missing givens or time_sec")
        try:
            # t = ρ L ( (2z0)² - z0² ) / (2 κ ΔT)
            expected = (
                self.rho * self.latent_heat *
                ( (2*self.z0)**2 - self.z0**2 ) /
                (2 * self.kappa * self.deltaT)
            )
            if math.isclose(self.time_sec, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"time_sec {self.time_sec} ≈ expected {expected:.4g}")
            else:
                return (False, f"time_sec {self.time_sec} differs from expected {expected:.4g}")
        except Exception as e:
            return (False, f"test_modeling_time_sec_derived crashed: {e}")

    def test_modeling_time_min_derived(self):
        """Re‑compute the doubling time in minutes and compare with time_min."""
        if not self.have_givens or self.time_min is None:
            return (False, "SKIP: missing givens or time_min")
        try:
            expected_sec = (
                self.rho * self.latent_heat *
                ( (2*self.z0)**2 - self.z0**2 ) /
                (2 * self.kappa * self.deltaT)
            )
            expected_min = expected_sec / 60.0
            if math.isclose(self.time_min, expected_min, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"time_min {self.time_min} ≈ expected {expected_min:.4g}")
            else:
                return (False, f"time_min {self.time_min} differs from expected {expected_min:.4g}")
        except Exception as e:
            return (False, f"test_modeling_time_min_derived crashed: {e}")

    def test_modeling_time_seconds_minutes_conversion(self):
        """Check that time_sec and time_min satisfy the conversion factor."""
        if self.time_sec is None or self.time_min is None:
            return (False, "SKIP: missing time_sec or time_min")
        try:
            converted = self.time_sec / 60.0
            if math.isclose(self.time_min, converted, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"time_min {self.time_min} ≈ time_sec/60 = {converted:.4g}")
            else:
                return (False, f"time_min {self.time_min} not close to time_sec/60 = {converted:.4g}")
        except Exception as e:
            return (False, f"test_modeling_time_conversion crashed: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity tests – directional behavior via perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_time_decreases_with_deltaT(self):
        """
        Increasing the temperature difference ΔT should shorten the doubling time.
        Baseline: use given ΔT; perturb: ΔT' = 1.1 * ΔT.
        """
        if not self.have_givens:
            return (False, "SKIP: givens incomplete")
        try:
            # Baseline time (for doubling from z0 to 2z0)
            t_base = (
                self.rho * self.latent_heat *
                ( (2*self.z0)**2 - self.z0**2 ) /
                (2 * self.kappa * self.deltaT)
            )
            deltaT_pert = self.deltaT * 1.1   # 10% increase
            t_pert = (
                self.rho * self.latent_heat *
                ( (2*self.z0)**2 - self.z0**2 ) /
                (2 * self.kappa * deltaT_pert)
            )
            if t_pert < t_base:
                return (True, f"time decreases with ΔT: {t_pert:.4g} < {t_base:.4g}")
            else:
                return (False, f"time did not decrease: perturbed {t_pert:.4g} vs baseline {t_base:.4g}")
        except Exception as e:
            return (False, f"test_monotonicity_time_decreases_with_deltaT crashed: {e}")

    def test_monotonicity_time_increases_with_initial_thickness(self):
        """
        Doubling time should increase when the initial thickness is larger.
        Baseline: z0; perturb: z0' = 1.1 * z0.
        """
        if not self.have_givens:
            return (False, "SKIP: givens incomplete")
        try:
            # Time for doubling from z0 to 2z0
            t_base = (
                self.rho * self.latent_heat *
                ( (2*self.z0)**2 - self.z0**2 ) /
                (2 * self.kappa * self.deltaT)
            )
            z0_pert = self.z0 * 1.1   # 10% larger initial thickness
            t_pert = (
                self.rho * self.latent_heat *
                ( (2*z0_pert)**2 - z0_pert**2 ) /
                (2 * self.kappa * self.deltaT)
            )
            if t_pert > t_base:
                return (True, f"time increases with initial thickness: {t_pert:.4g} > {t_base:.4g}")
            else:
                return (False, f"time did not increase: perturbed {t_pert:.4g} vs baseline {t_base:.4g}")
        except Exception as e:
            return (False, f"test_monotonicity_time_increases_with_initial_thickness crashed: {e}")

    # ----------------------------------------------------------------------
    # Main runner – discovers and executes all test_* methods
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)