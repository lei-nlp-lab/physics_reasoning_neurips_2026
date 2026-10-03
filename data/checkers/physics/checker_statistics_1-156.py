import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (use None for missing)
        self.q_dot = getattr(solution_instance, 'q_dot', None)            # cal/(g·s)
        self.rho = getattr(solution_instance, 'rho', None)                # g/cm³
        self.k = getattr(solution_instance, 'k', None)                    # cal/(deg·cm·s)
        self.R = getattr(solution_instance, 'R', None)                    # cm
        self.sigma = getattr(solution_instance, 'sigma', None)            # cal/(cm²·s·K⁴)

        # Extract results (from a results dict or direct attribute)
        results = getattr(solution_instance, 'results', None) or {}
        self.T_s = results.get('T_s', None)
        self.T_c = results.get('T_c', None)

        # Derived volumetric heat generation rate
        self.Q_dot = None
        if self.rho is not None and self.q_dot is not None:
            self.Q_dot = self.rho * self.q_dot

        # Baseline values (computed from givens if all are present)
        self.computed_T_s = None
        self.computed_T_c = None
        if all(x is not None for x in [self.Q_dot, self.R, self.sigma, self.k]):
            self.computed_T_s = (self.Q_dot * self.R / (3.0 * self.sigma)) ** 0.25
            self.computed_T_c = self.computed_T_s + (self.Q_dot * self.R ** 2) / (6.0 * self.k)

    # ------------------------------------------------------------------
    # Interface checks (required keys exist)
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.T_s is None:
            return False, "T_s not found in results"
        if self.T_c is None:
            return False, "T_c not found in results"
        return True, "Both T_s and T_c are present"

    def test_interface_numeric_values(self):
        if self.T_s is not None and not isinstance(self.T_s, (int, float)):
            return False, f"T_s is not numeric: {type(self.T_s)}"
        if self.T_c is not None and not isinstance(self.T_c, (int, float)):
            return False, f"T_c is not numeric: {type(self.T_c)}"
        return True, "T_s and T_c are numeric"

    # ------------------------------------------------------------------
    # Invariant / theorem-based checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_radiation_balance(self):
        """4πR² σ T_s⁴ should equal total internal heat generation (4/3 π R³ Q_dot)"""
        if any(x is None for x in [self.T_s, self.Q_dot, self.R, self.sigma]):
            return False, "SKIP: missing givens or T_s"
        total_heat = (4.0 / 3.0) * math.pi * self.R ** 3 * self.Q_dot
        radiated = 4.0 * math.pi * self.R ** 2 * self.sigma * self.T_s ** 4
        if math.isclose(total_heat, radiated, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Radiation balance: total_heat={total_heat:.3e}, radiated={radiated:.3e}"
        return False, f"Radiation balance mismatch: total_heat={total_heat:.3e}, radiated={radiated:.3e}"

    def test_invariant_center_surface_relation(self):
        """T_c = T_s + (Q_dot R²)/(6k)"""
        if any(x is None for x in [self.T_s, self.T_c, self.Q_dot, self.R, self.k]):
            return False, "SKIP: missing data"
        expected_delta = (self.Q_dot * self.R ** 2) / (6.0 * self.k)
        actual_delta = self.T_c - self.T_s
        if math.isclose(actual_delta, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ΔT = {actual_delta:.3f}, expected {expected_delta:.3f}"
        return False, f"ΔT = {actual_delta:.3f}, expected {expected_delta:.3f}"

    def test_invariant_flux_consistency(self):
        """-k dT/dr at r=R (approximated from profile) should equal σ T_s⁴.
        dT/dr at surface = -(Q_dot R)/(3k), so -k dT/dr = Q_dot R/3"""
        if any(x is None for x in [self.T_s, self.Q_dot, self.R, self.k, self.sigma]):
            return False, "SKIP: missing data"
        flux_conducted = self.Q_dot * self.R / 3.0               # cal/(cm²·s)
        flux_radiated = self.sigma * self.T_s ** 4
        if math.isclose(flux_conducted, flux_radiated, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Surface flux conducted={flux_conducted:.3e}, radiated={flux_radiated:.3e}"
        return False, f"Surface flux mismatch: conducted={flux_conducted:.3e}, radiated={flux_radiated:.3e}"

    # ------------------------------------------------------------------
    # Modeling / trace checks (derived quantities)
    # ------------------------------------------------------------------
    def test_modeling_surface_temperature_formula(self):
        """Self-consistency: T_s computed from givens should match solution's T_s"""
        if any(x is None for x in [self.computed_T_s, self.T_s]):
            return False, "SKIP: cannot compute baseline T_s"
        if math.isclose(self.computed_T_s, self.T_s, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"T_s solution={self.T_s:.3f}, computed={self.computed_T_s:.3f}"
        return False, f"T_s mismatch: solution={self.T_s:.3f}, computed={self.computed_T_s:.3f}"

    def test_modeling_center_temperature_formula(self):
        """Self-consistency: T_c computed from givens should match solution's T_c"""
        if any(x is None for x in [self.computed_T_c, self.T_c]):
            return False, "SKIP: cannot compute baseline T_c"
        if math.isclose(self.computed_T_c, self.T_c, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"T_c solution={self.T_c:.3f}, computed={self.computed_T_c:.3f}"
        return False, f"T_c mismatch: solution={self.T_c:.3f}, computed={self.computed_T_c:.3f}"

    def test_modeling_finite_center_gradient(self):
        """From spherical symmetry, dT/dr = 0 at r=0 → T_c is the maximum temperature"""
        if any(x is None for x in [self.T_s, self.T_c]):
            return False, "SKIP: T_s or T_c missing"
        if self.T_c > self.T_s + 1e-9:   # strict inequality
            return True, f"T_c ({self.T_c:.3f}) > T_s ({self.T_s:.3f})"
        return False, f"T_c ({self.T_c:.3f}) is not greater than T_s ({self.T_s:.3f})"

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb one input, check direction)
    # ------------------------------------------------------------------
    def test_monotonicity_increase_with_R(self):
        """If radius increases (by 10%), both T_s and T_c should increase."""
        # We need baseline values. R is the only givens we vary.
        if any(x is None for x in [self.Q_dot, self.R, self.sigma, self.k]):
            return False, "SKIP: missing givens for R perturbation"
        R_pert = self.R * 1.1
        T_s_pert = (self.Q_dot * R_pert / (3.0 * self.sigma)) ** 0.25
        T_c_pert = T_s_pert + (self.Q_dot * R_pert ** 2) / (6.0 * self.k)
        # Use computed baseline from __init__, if available
        if self.computed_T_s is None or self.computed_T_c is None:
            return False, "SKIP: cannot compute baseline temperatures"
        if T_s_pert > self.computed_T_s and T_c_pert > self.computed_T_c:
            return True, f"Perturbed R: T_s={T_s_pert:.3f} (base={self.computed_T_s:.3f}), T_c={T_c_pert:.3f} (base={self.computed_T_c:.3f})"
        return False, f"Monotonicity failure: perturbed T_s={T_s_pert:.3f} <= base T_s={self.computed_T_s:.3f} or T_c={T_c_pert:.3f} <= base T_c={self.computed_T_c:.3f}"

    def test_monotonicity_increase_with_Qdot(self):
        """If volumetric heat generation increases (by 10%), both T_s and T_c increase."""
        if any(x is None for x in [self.Q_dot, self.R, self.sigma, self.k]):
            return False, "SKIP: missing givens for Q_dot perturbation"
        Q_dot_pert = self.Q_dot * 1.1
        T_s_pert = (Q_dot_pert * self.R / (3.0 * self.sigma)) ** 0.25
        T_c_pert = T_s_pert + (Q_dot_pert * self.R ** 2) / (6.0 * self.k)
        if self.computed_T_s is None or self.computed_T_c is None:
            return False, "SKIP: cannot compute baseline"
        if T_s_pert > self.computed_T_s and T_c_pert > self.computed_T_c:
            return True, f"Perturbed Q_dot: T_s={T_s_pert:.3f} (base={self.computed_T_s:.3f}), T_c={T_c_pert:.3f} (base={self.computed_T_c:.3f})"
        return False, f"Monotonicity failure: perturbed T_s={T_s_pert:.3f} <= base T_s={self.computed_T_s:.3f} or T_c={T_c_pert:.3f} <= base T_c={self.computed_T_c:.3f}"

    # ------------------------------------------------------------------
    # Weak answer sanity (do not re-derive or use isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Only sign, non-NaN, broad range, and ordering."""
        if any(x is None for x in [self.T_s, self.T_c]):
            return False, "SKIP: T_s or T_c missing"
        if math.isnan(self.T_s) or math.isnan(self.T_c):
            return False, "Temperature is NaN"
        if self.T_s <= 0:
            return False, f"T_s = {self.T_s} is not positive"
        if self.T_c <= 0:
            return False, f"T_c = {self.T_c} is not positive"
        if self.T_c <= self.T_s:
            return False, f"T_c ({self.T_c}) is not greater than T_s ({self.T_s})"
        if self.T_c > 1e6:   # far above any plausible rock asteroid
            return False, f"T_c ({self.T_c}) seems unrealistically high"
        return True, f"T_s={self.T_s:.3f}, T_c={self.T_c:.3f}, both positive and ordered"

    # ------------------------------------------------------------------
    # Run all tests (discover and execute)
    # ------------------------------------------------------------------
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
        return all_pass, logs