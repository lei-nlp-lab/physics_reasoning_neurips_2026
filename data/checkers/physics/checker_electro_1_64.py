import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the problem:
    A dielectric sphere in uniform electric field – induced surface charge density.
    """

    def __init__(self, solution_instance):
        # Safely retrieve required results and givens using .get(), defaulting to None.
        self.A = solution_instance.results.get('A', None)
        self.B = solution_instance.results.get('B', None)
        self.sigma_theta = solution_instance.results.get('sigma(theta)', None)
        self.magnetic_field = solution_instance.results.get('magnetic_field', None)

        # Givens – try results dict first, then direct attributes.
        self.epsilon = self._safe_get(solution_instance, 'epsilon')
        self.epsilon0 = self._safe_get(solution_instance, 'epsilon0')
        self.E0 = self._safe_get(solution_instance, 'E0')
        self.R = self._safe_get(solution_instance, 'R')
        self.theta = self._safe_get(solution_instance, 'theta')   # optional

        self.cos_theta = math.cos(self.theta) if self.theta is not None else None

    def _safe_get(self, obj, key):
        """Try to get from results dict first, then from object attribute."""
        try:
            return obj.results.get(key, None)
        except AttributeError:
            return getattr(obj, key, None)

    # ------------------------------------------------------------------
    # Helper: evaluate sigma(theta) expression for given parameters
    # ------------------------------------------------------------------
    def _evaluate_sigma(self, epsilon_val, epsilon0_val, E0_val, cos_val):
        """
        Evaluate the solution's sigma(theta) expression at the given parameters.
        Returns a float or None if evaluation fails.
        """
        if self.sigma_theta is None:
            return None
        # If sigma_theta is a string, we assume it contains the symbolic expression.
        # If it's a number, we directly return it (formula independent of parameters)
        if isinstance(self.sigma_theta, (int, float)):
            return float(self.sigma_theta)
        # Try to treat as a sympy expression or string
        try:
            if isinstance(self.sigma_theta, str):
                # Build a sympy expression from the string – be safe: evaluate only known variables
                # Use sp.sympify with local_dict to restrict substitution
                local = {
                    'epsilon': sp.Symbol('epsilon'),
                    'epsilon0': sp.Symbol('epsilon0'),
                    'E0': sp.Symbol('E0'),
                    'R': sp.Symbol('R'),
                    'theta': sp.Symbol('theta'),
                    'cos': sp.cos,
                }
                expr = sp.sympify(self.sigma_theta, locals=local)
            else:
                # Assume it's a sympy expression or callable
                expr = self.sigma_theta
            # Substitute numerical values
            sub_dict = {
                sp.Symbol('epsilon'): epsilon_val,
                sp.Symbol('epsilon0'): epsilon0_val,
                sp.Symbol('E0'): E0_val,
                sp.Symbol('R'): self.R if self.R is not None else 1.0,  # R may not appear
                sp.Symbol('theta'): math.acos(cos_val) if cos_val != 1.0 else 0.0,
            }
            val = float(expr.subs(sub_dict).evalf())
            return val
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four required result keys exist."""
        missing = []
        if self.A is None:
            missing.append('A')
        if self.B is None:
            missing.append('B')
        if self.sigma_theta is None:
            missing.append('sigma(theta)')
        if self.magnetic_field is None:
            missing.append('magnetic_field')
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_available(self):
        """Check that at least the essential givens are present (not None)."""
        missing = []
        if self.epsilon is None:
            missing.append('epsilon')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if self.E0 is None:
            missing.append('E0')
        if self.R is None:
            missing.append('R')
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_potential_continuity(self):
        """Check φ_in(R,θ) = φ_out(R,θ) using solved constants A, B."""
        if self.A is None or self.B is None or self.R is None or self.E0 is None or self.cos_theta is None:
            return (False, "SKIP: missing data for potential continuity")

        phi_in = self.A * self.R * self.cos_theta
        phi_out = (-self.E0 * self.R + self.B / (self.R ** 2)) * self.cos_theta
        if math.isclose(phi_in, phi_out, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Potential continuity holds")
        else:
            return (False, f"Potential mismatch: φ_in={phi_in}, φ_out={phi_out}")

    def test_invariant_d_continuity(self):
        """Check ε0 ∂φ_out/∂r |_R = ε ∂φ_in/∂r |_R."""
        if self.A is None or self.B is None or self.R is None or self.E0 is None or self.epsilon is None or self.epsilon0 is None or self.cos_theta is None:
            return (False, "SKIP: missing data for D continuity")

        dphi_in_dr = self.A * self.cos_theta
        dphi_out_dr = (-self.E0 - 2 * self.B / (self.R ** 3)) * self.cos_theta

        lhs = self.epsilon0 * dphi_out_dr
        rhs = self.epsilon * dphi_in_dr
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Normal D continuity holds")
        else:
            return (False, f"D mismatch: ε0 dφ_out/dr = {lhs}, ε dφ_in/dr = {rhs}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_sigma_formula_derivation(self):
        """Verify that sigma(theta) from results is consistent with derived expression from A,B."""
        if self.A is None or self.B is None or self.sigma_theta is None or self.epsilon is None or self.epsilon0 is None or self.E0 is None or self.R is None or self.cos_theta is None:
            return (False, "SKIP: missing data for sigma derivation")

        # Compute sigma from boundary conditions
        dphi_out_dr = (-self.E0 - 2 * self.B / (self.R ** 3)) * self.cos_theta
        dphi_in_dr = self.A * self.cos_theta
        sigma_computed = self.epsilon0 * dphi_out_dr - self.epsilon * dphi_in_dr

        # Evaluate the provided sigma_theta at the same parameters
        sigma_provided = self._evaluate_sigma(self.epsilon, self.epsilon0, self.E0, self.cos_theta)
        if sigma_provided is None:
            return (False, "Cannot evaluate provided sigma(theta) expression")

        if math.isclose(sigma_computed, sigma_provided, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "sigma(theta) consistent with derived A,B")
        else:
            return (False, f"sigma mismatch: computed {sigma_computed}, provided {sigma_provided}")

    def test_modeling_dipole_moment_consistency(self):
        """Check that the total dipole moment P = (4π/3)R^3 p, with p=(ε-ε0)E_int."""
        if self.epsilon is None or self.epsilon0 is None or self.E0 is None or self.R is None:
            return (False, "SKIP: missing data for dipole moment check")
        epsilon_num = float(self.epsilon)
        epsilon0_num = float(self.epsilon0)
        E0_num = float(self.E0)
        R_num = float(self.R)
        E_int = (3 * epsilon0_num) / (epsilon_num + 2 * epsilon0_num) * E0_num
        p = (epsilon_num - epsilon0_num) * E_int
        P_total = (4 * math.pi / 3) * (R_num ** 3) * p
        if P_total >= 0:
            return (True, f"Dipole moment P = {P_total} is non‑negative (consistent)")
        else:
            return (True, f"P = {P_total} negative, but could be if ε<ε0.")

    def test_modeling_magnetic_field_static(self):
        """Check that magnetic_field indicates no field (string '0' or similar)."""
        if self.magnetic_field is None:
            return (False, "SKIP: magnetic_field not provided")
        msg = str(self.magnetic_field).lower()
        if '0' in msg or 'no' in msg or 'none' in msg:
            return (True, "Magnetic field correctly stated as zero/none")
        else:
            return (False, f"Unexpected magnetic_field value: {self.magnetic_field}")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks by explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_increases_with_epsilon(self):
        """
        For fixed E0, R, θ=0, sigma should increase with epsilon (for ε>ε0).
        Perturb epsilon by +5% and +10%, evaluate sigma via the solution's expression,
        and verify monotonic increase.
        """
        if self.epsilon is None or self.epsilon0 is None or self.E0 is None or self.R is None or self.sigma_theta is None:
            return (False, "SKIP: missing data for sigma monotonicity")

        cos0 = 1.0  # θ=0
        epsilon_base = float(self.epsilon)
        epsilon0_val = float(self.epsilon0)
        E0_val = float(self.E0)

        # Evaluate baseline sigma using solution's expression
        sigma_base = self._evaluate_sigma(epsilon_base, epsilon0_val, E0_val, cos0)
        if sigma_base is None:
            return (False, "Cannot evaluate sigma at baseline epsilon")

        # Perturb epsilon +5%
        eps_pert1 = epsilon_base * 1.05
        sigma_pert1 = self._evaluate_sigma(eps_pert1, epsilon0_val, E0_val, cos0)
        if sigma_pert1 is None:
            return (False, "Cannot evaluate sigma at perturbed epsilon +5%")

        # Perturb epsilon +10%
        eps_pert2 = epsilon_base * 1.10
        sigma_pert2 = self._evaluate_sigma(eps_pert2, epsilon0_val, E0_val, cos0)
        if sigma_pert2 is None:
            return (False, "Cannot evaluate sigma at perturbed epsilon +10%")

        # Check monotonic increase
        if sigma_pert1 > sigma_base and sigma_pert2 > sigma_base:
            return (True, f"sigma increases with epsilon: base={sigma_base:.4f}, +5%={sigma_pert1:.4f}, +10%={sigma_pert2:.4f}")
        else:
            return (False, f"sigma does not monotonically increase: base={sigma_base:.4f}, +5%={sigma_pert1:.4f}, +10%={sigma_pert2:.4f}")

    def test_monotonicity_sigma_scales_with_E0(self):
        """
        sigma should be proportional to E0 for fixed ε, ε0, θ.
        Perturb E0 by +5% and +10% and verify that sigma scales proportionally.
        """
        if self.epsilon is None or self.epsilon0 is None or self.E0 is None or self.R is None or self.sigma_theta is None:
            return (False, "SKIP: missing data for E0 scaling")

        cos0 = 1.0  # θ=0
        epsilon_val = float(self.epsilon)
        epsilon0_val = float(self.epsilon0)
        E0_base = float(self.E0)

        # Baseline sigma
        sigma_base = self._evaluate_sigma(epsilon_val, epsilon0_val, E0_base, cos0)
        if sigma_base is None or sigma_base == 0.0:
            return (False, "Cannot evaluate sigma at baseline E0")

        # Perturb E0 +5%
        E0_pert1 = E0_base * 1.05
        sigma_pert1 = self._evaluate_sigma(epsilon_val, epsilon0_val, E0_pert1, cos0)
        if sigma_pert1 is None:
            return (False, "Cannot evaluate sigma at perturbed E0 +5%")

        # Perturb E0 +10%
        E0_pert2 = E0_base * 1.10
        sigma_pert2 = self._evaluate_sigma(epsilon_val, epsilon0_val, E0_pert2, cos0)
        if sigma_pert2 is None:
            return (False, "Cannot evaluate sigma at perturbed E0 +10%")

        # Check proportionality: sigma_pert1 / sigma_base ≈ 1.05, sigma_pert2 / sigma_base ≈ 1.10
        if (math.isclose(sigma_pert1 / sigma_base, 1.05, rel_tol=1e-2) and
            math.isclose(sigma_pert2 / sigma_base, 1.10, rel_tol=1e-2)):
            return (True, f"sigma scales linearly with E0: base={sigma_base:.4f}, +5%={sigma_pert1:.4f}, +10%={sigma_pert2:.4f}")
        else:
            return (False, f"sigma does not scale linearly: base={sigma_base:.4f}, +5% ratio={sigma_pert1/sigma_base:.4f}, +10% ratio={sigma_pert2/sigma_base:.4f}")

    # ------------------------------------------------------------------
    # Weak Answer Sanity (at most one such test)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self):
        """
        Weak sanity: sigma_theta must be finite and non-NaN.
        Do NOT re-derive expected values.
        """
        if self.sigma_theta is None:
            return (False, "SKIP: sigma(theta) not provided")
        # Evaluate at given parameters (if available) or just check numeric
        if self.epsilon is not None and self.epsilon0 is not None and self.E0 is not None and self.cos_theta is not None:
            val = self._evaluate_sigma(float(self.epsilon), float(self.epsilon0), float(self.E0), self.cos_theta)
            if val is None:
                return (False, "Failed to evaluate sigma(theta)")
            if math.isfinite(val):
                return (True, f"sigma(theta) is finite: {val:.4f}")
            else:
                return (False, f"sigma(theta) is not finite: {val}")
        # If givens missing, just try to interpret sigma_theta as number
        try:
            val = float(self.sigma_theta)
            if math.isfinite(val):
                return (True, f"sigma(theta) is finite: {val:.4f}")
            else:
                return (False, f"sigma(theta) is not finite: {val}")
        except (TypeError, ValueError):
            return (True, "sigma(theta) expression present (cannot evaluate numerically, but weak sanity passes)")

    # ------------------------------------------------------------------
    # Master test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)