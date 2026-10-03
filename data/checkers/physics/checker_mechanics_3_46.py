import math
import sympy as sp
from collections.abc import Mapping

class PhysicsChecker:
    """
    Audits a solution for the charged particle in a magnetic dipole field problem.
    Checks conservation laws, turning point equations, ordering, monotonicity, and modeling consistency.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, Mapping):
            results = {}
        self.results = results
        self.r_max = results.get('r_max', None)
        self.r_min = results.get('r_min', None)

        # Extract givens from various possible sources
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, Mapping):
            givens = {}
        # Also try direct attributes as fallback
        self.e = givens.get('e', getattr(solution_instance, 'e', None))
        self.M = givens.get('M', getattr(solution_instance, 'M', None))
        self.c = givens.get('c', getattr(solution_instance, 'c', None))
        self.m = givens.get('m', getattr(solution_instance, 'm', None))
        self.v = givens.get('v', getattr(solution_instance, 'v', None))
        self.R = givens.get('R', getattr(solution_instance, 'R', None))

        # Derived parameter α = v*c*m/(e*M)
        self.alpha = None
        if all(x is not None and x != 0 for x in [self.v, self.c, self.m, self.e, self.M]):
            self.alpha = self.v * self.c * self.m / (self.e * self.M)

    def _get(self, name):
        """Get a given value by name."""
        return getattr(self, name, None)

    def test_interface_required_keys(self):
        """Check that r_max and r_min keys exist and are numeric."""
        if 'r_max' not in self.results or 'r_min' not in self.results:
            return (False, "Missing required result keys: 'r_max' and/or 'r_min'")
        r_max = self.r_max
        r_min = self.r_min
        if not (isinstance(r_max, (int, float)) and isinstance(r_min, (int, float))):
            return (False, "r_max and r_min must be numeric")
        if math.isnan(r_max) or math.isnan(r_min):
            return (False, "r_max or r_min is NaN")
        if math.isinf(r_max) or math.isinf(r_min):
            return (False, "r_max or r_min is infinite")
        return (True, "Required result keys present and numeric")

    def test_invariant_turning_point_eq_r_max(self):
        """
        Check that r_max satisfies the turning point quadratic from the positive branch:
        α * R * r_max^2 - r_max + R = 0, where α = v*c*m/(e*M).
        """
        if self.alpha is None:
            return (False, "SKIP: insufficient givens to compute alpha")
        if self.r_max is None:
            return (False, "SKIP: r_max not available")
        R = self.R
        alpha = self.alpha
        val = alpha * R * self.r_max**2 - self.r_max + R
        if math.isclose(val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r_max satisfies quadratic: {val:.6e}")
        return (False, f"r_max fails quadratic: value = {val:.6e}")

    def test_invariant_turning_point_eq_r_min(self):
        """
        Check that r_min satisfies the turning point quadratic from the negative branch:
        α * R * r_min^2 + r_min - R = 0.
        """
        if self.alpha is None:
            return (False, "SKIP: insufficient givens to compute alpha")
        if self.r_min is None:
            return (False, "SKIP: r_min not available")
        R = self.R
        alpha = self.alpha
        val = alpha * R * self.r_min**2 + self.r_min - R
        if math.isclose(val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r_min satisfies quadratic: {val:.6e}")
        return (False, f"r_min fails quadratic: value = {val:.6e}")

    def test_invariant_ordering(self):
        """Check expected ordering: r_min < R < r_max, and both positive."""
        if None in (self.r_max, self.r_min, self.R):
            return (False, "SKIP: missing r_max, r_min, or R")
        r_max, r_min, R = self.r_max, self.r_min, self.R
        if not (r_min > 0 and r_max > 0):
            return (False, "Radii must be positive")
        if not (r_min < R < r_max):
            return (False, f"Expected r_min < R < r_max, got r_min={r_min}, R={R}, r_max={r_max}")
        return (True, "Ordering constraints satisfied")

    def test_invariant_boundedness_condition(self):
        """
        Check that α >= 4*R^2 for the orbit to be bounded.
        If alpha is available and R is known, verify the inequality holds.
        """
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if self.R is None:
            return (False, "SKIP: R not available")
        if self.alpha < 4 * self.R**2:
            return (False, f"Boundedness condition α >= 4R^2 violated: α={self.alpha}, 4R^2={4*self.R**2}")
        return (True, f"Boundedness condition holds: α={self.alpha} >= {4*self.R**2}")

    def test_modeling_energy_at_turning_points(self):
        """
        Modeling/trace test: For both r_max and r_min, compute the effective potential
        U_eff(r) = (e^2 * M^2 / (2 * c^2 * m)) * (1/R - 1/r)^2 / r^2
        and verify it equals the kinetic energy (1/2) * m * v^2.
        This directly checks energy conservation at turning points, a core modeling step.
        """
        if None in (self.e, self.M, self.c, self.m, self.v, self.R, self.r_max, self.r_min):
            return (False, "SKIP: missing required parameters")
        # Compute prefactor: e^2 * M^2 / (2 * c^2 * m)
        prefactor = (self.e**2 * self.M**2) / (2 * self.c**2 * self.m)
        expected_KE = 0.5 * self.m * self.v**2
        for label, r in [('r_max', self.r_max), ('r_min', self.r_min)]:
            # U_eff(r) = prefactor * (1/R - 1/r)^2 / r^2
            term = (1.0 / self.R - 1.0 / r)
            U_eff = prefactor * term**2 / (r**2)
            if not math.isclose(U_eff, expected_KE, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Energy mismatch at {label}: U_eff={U_eff}, expected KE={expected_KE}")
        return (True, "Energy conservation holds at both turning points")

    def test_modeling_sqrt_non_negative_r_max(self):
        """
        Modeling/trace test: The formula for r_max requires that the expression under the square root
        be non-negative: 1 - 4*R^2/alpha >= 0.  Verify this from the given parameters and the
        computed alpha.  This ensures the modeling assumptions (bounded orbit) are valid.
        """
        if self.alpha is None or self.R is None:
            return (False, "SKIP: missing alpha or R")
        discriminant = 1 - 4 * self.R**2 / self.alpha
        if discriminant < 0:
            return (False, f"Discriminant for r_max formula is negative: {discriminant:.6e}")
        return (True, f"Discriminant for r_max is non-negative: {discriminant:.6e}")

    def test_monotonicity_r_max(self):
        """
        Check that r_max increases when initial speed v is increased.
        Compute a perturbed r_max using the analytic formula with a slightly larger v,
        and verify it is larger than the original r_max.
        """
        if None in (self.v, self.alpha, self.r_max, self.R, self.e, self.M, self.c, self.m):
            return (False, "SKIP: missing required parameters")
        # Perturb v upward by 5%
        v_new = self.v * 1.05
        # Recompute α_new
        alpha_new = v_new * self.c * self.m / (self.e * self.M)
        if alpha_new is None or alpha_new <= 0:
            return (False, "SKIP: invalid perturbed alpha")
        # Check boundedness condition for new alpha
        if alpha_new < 4 * self.R**2:
            return (False, "SKIP: perturbed orbit not bounded")
        # Compute new r_max from analytic formula (positive branch)
        r_max_new = (alpha_new / (2 * self.R)) * (1 + math.sqrt(1 - 4 * self.R**2 / alpha_new))
        if r_max_new > self.r_max:
            return (True, f"r_max increased from {self.r_max} to {r_max_new} when v increased")
        return (False, f"r_max did not increase when v increased: old={self.r_max}, new={r_max_new}")

    def test_monotonicity_r_min(self):
        """
        Check that r_min increases when initial speed v is increased.
        Use same perturbation as above.
        """
        if None in (self.v, self.alpha, self.r_min, self.R, self.e, self.M, self.c, self.m):
            return (False, "SKIP: missing required parameters")
        v_new = self.v * 1.05
        alpha_new = v_new * self.c * self.m / (self.e * self.M)
        if alpha_new is None or alpha_new <= 0:
            return (False, "SKIP: invalid perturbed alpha")
        # r_min formula from negative branch
        r_min_new = (alpha_new / (2 * self.R)) * (-1 + math.sqrt(1 + 4 * self.R**2 / alpha_new))
        if r_min_new > self.r_min:
            return (True, f"r_min increased from {self.r_min} to {r_min_new} when v increased")
        return (False, f"r_min did not increase when v increased: old={self.r_min}, new={r_min_new}")

    def test_answer_sanity_positivity(self):
        """Weak sanity: r_max and r_min must be positive and finite."""
        if None in (self.r_max, self.r_min):
            return (False, "SKIP: missing radii")
        if not (math.isfinite(self.r_max) and math.isfinite(self.r_min)):
            return (False, "Radii must be finite")
        if not (self.r_max > 0 and self.r_min > 0):
            return (False, "Radii must be positive")
        return (True, "Radii positive and finite")

    def run_all_tests(self):
        """Discover all test methods and run them."""
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