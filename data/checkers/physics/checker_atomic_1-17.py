import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract givens safely – ALL defaults are None
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}
        self.I = givens.get('I', None)
        self.p = givens.get('p', None)
        self.m_p = givens.get('m_p', None)
        self.Z = givens.get('Z', None)
        self.z = givens.get('z', None)
        self.t = givens.get('t', None)
        self.rho = givens.get('rho', None)
        self.X0 = givens.get('X0', None)
        self.r = givens.get('r', None)
        self.L = givens.get('L', None)
        self.theta0_deg = givens.get('theta0_deg', None)
        self.theta_deg = givens.get('theta_deg', None)
        self.e2 = givens.get('e2', None)           # formerly 1.44
        self.NA = givens.get('NA', None)           # formerly 6.022e23
        self.A = givens.get('A', None)             # formerly 27
        self.const_multiple = givens.get('const_multiple', None)  # formerly 15.0

        # Extract results safely – ALL defaults are None
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.dsigma_formula = results.get('dsigma_formula', None)
        self.dsigma_30 = results.get('dsigma_30', None)
        self.N3 = results.get('N3', None)
        self.sigma_gt5 = results.get('sigma_gt5', None)
        self.N5 = results.get('N5', None)
        self.theta_rms = results.get('theta_rms', None)

        # Derived quantities – only compute if all needed givens are present
        self.E = None
        self.beta = None
        self.n_target = None
        if all(v is not None for v in [self.p, self.m_p]):
            self.E = math.sqrt(self.p**2 + self.m_p**2) - self.m_p
            self.beta = self.p / math.sqrt(self.p**2 + self.m_p**2)
        if all(v is not None for v in [self.rho, self.t, self.NA, self.A]):
            self.n_target = self.rho * self.t * self.NA / self.A

        # Convert angles to radians if given in degrees
        self.theta_rad = math.radians(self.theta_deg) if self.theta_deg is not None else None
        self.theta0_rad = math.radians(self.theta0_deg) if self.theta0_deg is not None else None

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_keys_exist(self) -> Tuple[bool, str]:
        """Check that all required keys are present in results."""
        required = ['dsigma_formula', 'dsigma_30', 'N3', 'sigma_gt5', 'N5', 'theta_rms']
        missing = [key for key in required if getattr(self, key) is None]
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_dsigma_formula_string(self) -> Tuple[bool, str]:
        """Check dsigma_formula is a non-empty string containing 'sin'."""
        if self.dsigma_formula is None:
            return (False, "SKIP: dsigma_formula missing")
        if not isinstance(self.dsigma_formula, str):
            return (False, "dsigma_formula is not a string")
        if len(self.dsigma_formula) == 0:
            return (False, "dsigma_formula is empty")
        if 'sin' not in self.dsigma_formula and '\\sin' not in self.dsigma_formula:
            return (False, "dsigma_formula does not contain sin (maybe not Rutherford)")
        return (True, "dsigma_formula appears valid")

    def test_interface_numeric_values(self) -> Tuple[bool, str]:
        """Check that final numeric results are positive and finite."""
        numeric_keys = ['dsigma_30', 'N3', 'sigma_gt5', 'N5', 'theta_rms']
        for key in numeric_keys:
            val = getattr(self, key, None)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{key} is not numeric")
            if not math.isfinite(val):
                return (False, f"{key} is not finite")
            if val <= 0:
                return (False, f"{key} is not positive")
        return (True, "Numeric values are positive and finite")

    # -----------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # -----------------------------------------------------------------
    def test_invariant_dsigma_positive(self) -> Tuple[bool, str]:
        if self.dsigma_30 is None:
            return (False, "SKIP: dsigma_30 missing")
        if self.dsigma_30 > 0:
            return (True, "dsigma_30 is positive")
        return (False, "dsigma_30 should be positive")

    def test_invariant_N3_less_than_beam(self) -> Tuple[bool, str]:
        if self.N3 is None or self.I is None:
            return (False, "SKIP: N3 or I missing")
        if self.N3 < self.I:
            return (True, f"N3={self.N3:.3e} < I={self.I:.3e}")
        return (False, f"N3 should be less than beam intensity I={self.I:.3e}")

    def test_invariant_N5_less_than_beam(self) -> Tuple[bool, str]:
        if self.N5 is None or self.I is None:
            return (False, "SKIP: N5 or I missing")
        if self.N5 < self.I:
            return (True, f"N5={self.N5:.3e} < I={self.I:.3e}")
        return (False, f"N5 should be less than beam intensity I={self.I:.3e}")

    def test_invariant_sigma_gt5_greater_than_dsigma_30(self) -> Tuple[bool, str]:
        if self.sigma_gt5 is None or self.dsigma_30 is None:
            return (False, "SKIP: sigma_gt5 or dsigma_30 missing")
        if self.sigma_gt5 > self.dsigma_30:
            return (True, f"sigma_gt5={self.sigma_gt5:.3e} > dsigma_30={self.dsigma_30:.3e}")
        return (False, "sigma_gt5 should be larger than dsigma_30 (integrated > differential at fixed angle)")

    def test_invariant_N5_greater_than_N3(self) -> Tuple[bool, str]:
        if self.N5 is None or self.N3 is None:
            return (False, "SKIP: N5 or N3 missing")
        if self.N5 > self.N3:
            return (True, f"N5={self.N5:.3e} > N3={self.N3:.3e}")
        return (False, "N5 should be larger than N3 because integrated over larger solid angle")

    def test_invariant_theta_rms_reasonable(self) -> Tuple[bool, str]:
        if self.theta_rms is None:
            return (False, "SKIP: theta_rms missing")
        if 0 < self.theta_rms < 1.0:
            return (True, f"theta_rms={self.theta_rms:.5f} rad is reasonable for thin foil")
        return (False, f"theta_rms={self.theta_rms:.5f} rad seems unreasonable (should be small)")

    # -----------------------------------------------------------------
    # Modeling/trace consistency tests
    # -----------------------------------------------------------------
    def test_modeling_N3_from_givens(self) -> Tuple[bool, str]:
        """Recompute N3 from givens and compare with provided N3."""
        if self.N3 is None:
            return (False, "SKIP: N3 missing")
        if any(v is None for v in [self.dsigma_30, self.I, self.n_target, self.r, self.L]):
            return (False, "SKIP: missing givens for N3 recomputation")
        dOmega = math.pi * self.r**2 / self.L**2
        N3_calc = self.I * self.n_target * self.dsigma_30 * dOmega
        if math.isclose(N3_calc, self.N3, rel_tol=0.05, abs_tol=1):
            return (True, f"N3 computed {N3_calc:.3e} matches provided {self.N3:.3e}")
        return (False, f"N3 mismatch: computed {N3_calc:.3e}, provided {self.N3:.3e}")

    def test_modeling_sigma_gt5_from_givens(self) -> Tuple[bool, str]:
        """Recompute integrated cross section for theta > theta0 and compare."""
        if self.sigma_gt5 is None:
            return (False, "SKIP: sigma_gt5 missing")
        if any(v is None for v in [self.z, self.Z, self.e2, self.E, self.theta0_rad]):
            return (False, "SKIP: missing givens for sigma_gt5 recomputation")
        A_val = (self.z * self.Z * self.e2 / (4.0 * self.E))**2  # fm^2
        sin_half = math.sin(self.theta0_rad / 2.0)
        sigma_calc_fm2 = 4.0 * math.pi * A_val * (1.0 / (sin_half**2) - 1.0)
        sigma_calc_cm2 = sigma_calc_fm2 * 1e-26  # convert fm^2 to cm^2
        if math.isclose(sigma_calc_cm2, self.sigma_gt5, rel_tol=0.10, abs_tol=1e-26):
            return (True, f"sigma_gt5 computed {sigma_calc_cm2:.3e} matches provided {self.sigma_gt5:.3e}")
        return (False, f"sigma_gt5 mismatch: computed {sigma_calc_cm2:.3e}, provided {self.sigma_gt5:.3e}")

    def test_modeling_N5_from_givens(self) -> Tuple[bool, str]:
        """Recompute N5 from givens and compare."""
        if self.N5 is None:
            return (False, "SKIP: N5 missing")
        if any(v is None for v in [self.I, self.n_target, self.sigma_gt5]):
            return (False, "SKIP: missing givens for N5 recomputation")
        N5_calc = self.I * self.n_target * self.sigma_gt5
        if math.isclose(N5_calc, self.N5, rel_tol=0.10, abs_tol=1):
            return (True, f"N5 computed {N5_calc:.3e} matches provided {self.N5:.3e}")
        return (False, f"N5 mismatch: computed {N5_calc:.3e}, provided {self.N5:.3e}")

    def test_modeling_theta_rms_from_givens(self) -> Tuple[bool, str]:
        """Recompute multiple scattering angle and compare."""
        if self.theta_rms is None:
            return (False, "SKIP: theta_rms missing")
        if any(v is None for v in [self.const_multiple, self.p, self.beta, self.rho, self.t, self.X0]):
            return (False, "SKIP: missing givens for theta_rms recomputation")
        t_gcm2 = self.rho * self.t
        theta_rms_calc = (self.const_multiple / (self.p * self.beta)) * math.sqrt(t_gcm2 / self.X0)
        if math.isclose(theta_rms_calc, self.theta_rms, rel_tol=0.15, abs_tol=1e-4):
            return (True, f"theta_rms computed {theta_rms_calc:.5f} matches provided {self.theta_rms:.5f}")
        return (False, f"theta_rms mismatch: computed {theta_rms_calc:.5f}, provided {self.theta_rms:.5f}")

    # -----------------------------------------------------------------
    # Monotonicity test (perturb thickness)
    # -----------------------------------------------------------------
    def test_monotonicity_theta_rms_increases_with_thickness(self) -> Tuple[bool, str]:
        """Increase t by 10% and check theta_rms increases."""
        if self.theta_rms is None:
            return (False, "SKIP: theta_rms missing")
        if any(v is None for v in [self.const_multiple, self.p, self.beta, self.rho, self.t, self.X0]):
            return (False, "SKIP: missing givens for monotonicity test")
        t_gcm2_base = self.rho * self.t
        t_gcm2_pert = t_gcm2_base * 1.10
        theta_pert = (self.const_multiple / (self.p * self.beta)) * math.sqrt(t_gcm2_pert / self.X0)
        if theta_pert > self.theta_rms:
            return (True, f"Perturbed theta_rms {theta_pert:.5f} > original {self.theta_rms:.5f}")
        return (False, f"Monotonicity violated: perturbed {theta_pert:.5f} <= original {self.theta_rms:.5f}")

    # -----------------------------------------------------------------
    # Optional weak answer sanity (not dominant)
    # -----------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self) -> Tuple[bool, str]:
        """Weak sanity: check that all numeric answers are within reasonable orders of magnitude."""
        # Only rough ranges to avoid dominating checks
        if self.dsigma_30 is not None:
            if not (1e-30 < self.dsigma_30 < 1e-20):
                return (False, f"dsigma_30={self.dsigma_30:.3e} order of magnitude suspicious")
        if self.N3 is not None:
            if not (1 < self.N3 < 1e8):
                return (False, f"N3={self.N3:.3e} order of magnitude suspicious")
        if self.sigma_gt5 is not None:
            if not (1e-30 < self.sigma_gt5 < 1e-20):
                return (False, f"sigma_gt5={self.sigma_gt5:.3e} order of magnitude suspicious")
        if self.N5 is not None:
            if not (1 < self.N5 < 1e12):
                return (False, f"N5={self.N5:.3e} order of magnitude suspicious")
        if self.theta_rms is not None:
            if not (1e-5 < self.theta_rms < 1.0):
                return (False, f"theta_rms={self.theta_rms:.5f} order of magnitude suspicious")
        return (True, "All answers within plausible order-of-magnitude ranges")

    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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