import math
import sympy as sp  # optional, for symbolic checks if needed
import traceback

class PhysicsChecker:
    """
    Audits a solution for quantum mechanical scattering problem:
    deriving bounds for elastic cross section sigma_el^(ℓ) in terms of sigma_in^(ℓ).
    """
    def __init__(self, solution_instance):
        # Safe extraction from results dict
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.bound_lower = results.get('bound_lower', None)
        self.bound_upper = results.get('bound_upper', None)

        # Safely extract givens and intermediate quantities from the solution instance
        # Use getattr with None default
        self.k = getattr(solution_instance, 'k', None)
        self.ell = getattr(solution_instance, 'ell', None)
        self.eta = getattr(solution_instance, 'eta', None)
        self.delta = getattr(solution_instance, 'delta', None)
        self.sigma_el = getattr(solution_instance, 'sigma_el', None)
        self.sigma_in = getattr(solution_instance, 'sigma_in', None)

        # Also try to get from results if not present as attributes
        if self.sigma_el is None and 'sigma_el' in results:
            self.sigma_el = results['sigma_el']
        if self.sigma_in is None and 'sigma_in' in results:
            self.sigma_in = results['sigma_in']
        if self.eta is None and 'eta' in results:
            self.eta = results['eta']
        if self.delta is None and 'delta' in results:
            self.delta = results['delta']

        # Keep track of which quantities are available for tests
        self.has_numeric_data = (self.k is not None and self.ell is not None and 
                                 self.eta is not None and self.delta is not None and 
                                 self.sigma_in is not None and self.sigma_el is not None)

    # =================== Interface Tests ===================
    def test_interface_required_keys(self):
        """Check that bound_lower and bound_upper exist in results."""
        if self.bound_lower is None or self.bound_upper is None:
            return (False, "Missing required keys: bound_lower or bound_upper")
        return (True, "Both bound_lower and bound_upper present")

    # =================== Invariant / Theorem-based Tests ===================
    def test_invariant_lower_upper_ordering(self):
        """Lower bound must be <= upper bound (if numeric)."""
        if self.bound_lower is None or self.bound_upper is None:
            return (False, "SKIP: bounds not available")
        # Attempt numeric comparison if possible
        try:
            lower = float(self.bound_lower)
            upper = float(self.bound_upper)
        except (TypeError, ValueError):
            # Maybe symbolic strings; attempt sympy numerical evaluation
            try:
                lower = float(sp.N(self.bound_lower))
                upper = float(sp.N(self.bound_upper))
            except:
                return (False, "Cannot convert bounds to numeric values")
        if lower <= upper:
            return (True, f"Lower bound {lower} <= Upper bound {upper}")
        else:
            return (False, f"Lower bound {lower} > Upper bound {upper}")

    def test_invariant_bounds_nonnegative(self):
        """Both bounds must be nonnegative (cross sections are nonnegative)."""
        if self.bound_lower is None or self.bound_upper is None:
            return (False, "SKIP: bounds not available")
        try:
            lower = float(self.bound_lower)
            upper = float(self.bound_upper)
        except (TypeError, ValueError):
            try:
                lower = float(sp.N(self.bound_lower))
                upper = float(sp.N(self.bound_upper))
            except:
                return (False, "Cannot convert bounds to numeric")
        if lower >= 0 and upper >= 0:
            return (True, f"Bounds nonnegative: lower={lower}, upper={upper}")
        else:
            return (False, f"Bounds negative: lower={lower}, upper={upper}")

    def test_invariant_sigma_el_between_bounds(self):
        """If sigma_el, sigma_in, eta given, verify sigma_el lies between derived bounds."""
        if not self.has_numeric_data:
            return (False, "SKIP: insufficient numeric data")
        # Recompute bounds from formula using sigma_in and eta
        try:
            eta = float(self.eta)
            sigma_in = float(self.sigma_in)
            sigma_el = float(self.sigma_el)
            lower = sigma_in * (1 - eta)**2 / (1 - eta**2)
            upper = sigma_in * (1 + eta)**2 / (1 - eta**2)
        except (ValueError, ZeroDivisionError):
            return (False, "Numeric conversion error (eta=1? division by zero)")
        # Check that given sigma_el is within [lower, upper]
        if lower <= sigma_el <= upper:
            return (True, f"sigma_el={sigma_el} lies in [{lower}, {upper}]")
        else:
            return (False, f"sigma_el={sigma_el} outside bounds [{lower}, {upper}]")

    # =================== Modeling / Trace Tests ===================
    def test_modeling_lower_bound_formula(self):
        """If sigma_in and eta available, check bound_lower matches formula using math.isclose."""
        if self.sigma_in is None or self.eta is None or self.bound_lower is None:
            return (False, "SKIP: missing sigma_in, eta, or bound_lower")
        try:
            eta = float(self.eta)
            sigma_in = float(self.sigma_in)
            expected_lower = sigma_in * (1 - eta)**2 / (1 - eta**2)
            actual_lower = float(self.bound_lower)
        except (TypeError, ValueError, ZeroDivisionError):
            return (False, "Cannot convert to numbers (eta=1? division by zero)")
        if math.isclose(actual_lower, expected_lower, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lower bound matches: {actual_lower} ≈ {expected_lower}")
        else:
            return (False, f"Lower bound mismatch: {actual_lower} vs expected {expected_lower}")

    def test_modeling_upper_bound_formula(self):
        """If sigma_in and eta available, check bound_upper matches formula using math.isclose."""
        if self.sigma_in is None or self.eta is None or self.bound_upper is None:
            return (False, "SKIP: missing sigma_in, eta, or bound_upper")
        try:
            eta = float(self.eta)
            sigma_in = float(self.sigma_in)
            expected_upper = sigma_in * (1 + eta)**2 / (1 - eta**2)
            actual_upper = float(self.bound_upper)
        except (TypeError, ValueError, ZeroDivisionError):
            return (False, "Cannot convert to numbers (eta=1? division by zero)")
        if math.isclose(actual_upper, expected_upper, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Upper bound matches: {actual_upper} ≈ {expected_upper}")
        else:
            return (False, f"Upper bound mismatch: {actual_upper} vs expected {expected_upper}")

    def test_modeling_lower_upper_ratio(self):
        """Check that bound_upper / bound_lower >= 1 (since bounds are ordered)."""
        if self.bound_lower is None or self.bound_upper is None:
            return (False, "SKIP: bounds not available")
        try:
            lower = float(self.bound_lower)
            upper = float(self.bound_upper)
        except (TypeError, ValueError):
            return (False, "Cannot convert bounds to numeric")
        if lower == 0:
            return (False, "Lower bound is zero, ratio undefined")
        ratio = upper / lower
        if ratio >= 1.0:
            return (True, f"Upper/Lower ratio = {ratio} >= 1")
        else:
            return (False, f"Upper/Lower ratio = {ratio} < 1")

    # =================== Monotonicity Tests ===================
    def test_monotonicity_lower_bound_with_sigma_in(self):
        """Perturb sigma_in upward; lower bound should increase (proportional)."""
        if self.sigma_in is None or self.eta is None or self.bound_lower is None:
            return (False, "SKIP: needed data missing")
        try:
            sigma_in = float(self.sigma_in)
            eta = float(self.eta)
            lower_orig = float(self.bound_lower)
        except (TypeError, ValueError):
            return (False, "Cannot convert to numeric")
        # perturb sigma_in by +10%
        sigma_in_pert = sigma_in * 1.1
        # recompute lower bound for perturbed sigma_in (same eta)
        lower_pert = sigma_in_pert * (1 - eta)**2 / (1 - eta**2)
        if lower_pert > lower_orig:
            return (True, f"Lower bound increases from {lower_orig} to {lower_pert} when sigma_in +10%")
        else:
            return (False, f"Lower bound did not increase: orig {lower_orig}, perturbed {lower_pert}")

    def test_monotonicity_upper_bound_with_sigma_in(self):
        """Perturb sigma_in upward; upper bound should increase (proportional)."""
        if self.sigma_in is None or self.eta is None or self.bound_upper is None:
            return (False, "SKIP: needed data missing")
        try:
            sigma_in = float(self.sigma_in)
            eta = float(self.eta)
            upper_orig = float(self.bound_upper)
        except (TypeError, ValueError):
            return (False, "Cannot convert to numeric")
        sigma_in_pert = sigma_in * 1.1
        upper_pert = sigma_in_pert * (1 + eta)**2 / (1 - eta**2)
        if upper_pert > upper_orig:
            return (True, f"Upper bound increases from {upper_orig} to {upper_pert} when sigma_in +10%")
        else:
            return (False, f"Upper bound did not increase: orig {upper_orig}, perturbed {upper_pert}")

    def test_monotonicity_lower_bound_with_eta(self):
        """For fixed sigma_in, as eta increases, lower bound should decrease (derivative negative for eta in (0,1))."""
        if self.sigma_in is None or self.eta is None or self.bound_lower is None:
            return (False, "SKIP: needed data missing")
        try:
            sigma_in = float(self.sigma_in)
            eta = float(self.eta)
            lower_orig = float(self.bound_lower)
        except (TypeError, ValueError):
            return (False, "Cannot convert to numeric")
        # ensure eta is not too close to 1 to avoid numerical issues
        if eta >= 0.95:
            return (False, "SKIP: eta too close to 1 for perturbation")
        eta_pert = eta + 0.05  # increase by 0.05
        if eta_pert >= 1.0:
            return (False, "SKIP: perturbed eta >= 1")
        lower_pert = sigma_in * (1 - eta_pert)**2 / (1 - eta_pert**2)
        # lower bound should decrease as eta increases (for eta in [0,1))
        if lower_pert < lower_orig:
            return (True, f"Lower bound decreases from {lower_orig} to {lower_pert} when eta increases from {eta} to {eta_pert}")
        else:
            return (False, f"Lower bound did not decrease: orig {lower_orig}, perturbed {lower_pert}")

    def test_monotonicity_upper_bound_with_eta(self):
        """For fixed sigma_in, as eta increases, upper bound should increase (derivative positive for eta in (0,1))."""
        if self.sigma_in is None or self.eta is None or self.bound_upper is None:
            return (False, "SKIP: needed data missing")
        try:
            sigma_in = float(self.sigma_in)
            eta = float(self.eta)
            upper_orig = float(self.bound_upper)
        except (TypeError, ValueError):
            return (False, "Cannot convert to numeric")
        if eta >= 0.95:
            return (False, "SKIP: eta too close to 1")
        eta_pert = eta + 0.05
        if eta_pert >= 1.0:
            return (False, "SKIP: perturbed eta >= 1")
        upper_pert = sigma_in * (1 + eta_pert)**2 / (1 - eta_pert**2)
        if upper_pert > upper_orig:
            return (True, f"Upper bound increases from {upper_orig} to {upper_pert} when eta increases from {eta} to {eta_pert}")
        else:
            return (False, f"Upper bound did not increase: orig {upper_orig}, perturbed {upper_pert}")

    # =================== Answer Sanity (Weak) ===================
    def test_answer_sanity_bounds_nonnegative(self):
        """Weak sanity: bounds are nonnegative (should never be negative)."""
        if self.bound_lower is None or self.bound_upper is None:
            return (False, "SKIP: bounds missing")
        try:
            lower = float(self.bound_lower)
            upper = float(self.bound_upper)
        except (TypeError, ValueError):
            return (False, "Cannot convert to numeric")
        if lower >= 0 and upper >= 0:
            return (True, "Bounds are nonnegative")
        else:
            return (False, "Negative bound found")

    # =================== Orchestrator ===================
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