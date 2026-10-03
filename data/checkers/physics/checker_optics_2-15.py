import math
import sys

class PhysicsChecker:
    """
    Audits a solution instance for the thin film interference problem.
    Expects solution_instance to have attributes t, n (or in results),
    and results to contain keys 'lambdas' and 'm_values'.
    """
    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', None) or {}
        self.lambdas = results.get('lambdas', None)
        self.m_values = results.get('m_values', None)
        # Extract physical inputs with None defaults
        self.t = getattr(solution_instance, 't', None)
        if self.t is None:
            self.t = results.get('t', None)
        self.n = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = results.get('n', None)

    # -----------------------------------------------------------------
    # Helper: compute λ (in nm) from formula 2*n*t/(m+0.5)
    # -----------------------------------------------------------------
    def _lambda_from_m(self, m, t=None, n=None):
        if t is None:
            t = self.t
        if n is None:
            n = self.n
        if t is None or n is None:
            return None
        return 2.0 * n * t * 1e9 / (m + 0.5)   # convert t to nm

    # -----------------------------------------------------------------
    # Interface checks
    # -----------------------------------------------------------------
    def test_interface_required_keys_present(self):
        if self.lambdas is None:
            return (False, "SKIP: 'lambdas' missing from results")
        if self.m_values is None:
            return (False, "SKIP: 'm_values' missing from results")
        if not isinstance(self.lambdas, list):
            return (False, "lambdas must be a list")
        if not isinstance(self.m_values, list):
            return (False, "m_values must be a list")
        if len(self.lambdas) != len(self.m_values):
            return (False, f"Length mismatch: {len(self.lambdas)} lambdas vs {len(self.m_values)} m_values")
        return (True, "All required keys present and valid")

    def test_interface_lambdas_are_numbers(self):
        if self.lambdas is None:
            return (False, "SKIP")
        for i, lam in enumerate(self.lambdas):
            if not isinstance(lam, (int, float)):
                return (False, f"lambdas[{i}] is not a number")
        return (True, "All lambdas are numeric")

    def test_interface_m_values_are_integers(self):
        if self.m_values is None:
            return (False, "SKIP")
        for i, m in enumerate(self.m_values):
            if not isinstance(m, int):
                try:
                    _ = int(m)
                except (ValueError, TypeError):
                    return (False, f"m_values[{i}] is not an integer")
        return (True, "All m_values are integers")

    # -----------------------------------------------------------------
    # Invariant / theorem-based checks (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_constructive_condition(self):
        """Check 2*n*t == (m+0.5)*λ for each pair, using given t,n if available."""
        if self.lambdas is None or self.m_values is None:
            return (False, "SKIP: missing data")
        if self.t is None or self.n is None:
            return (False, "SKIP: missing t or n")
        D = 2.0 * self.n * self.t * 1e9   # nm
        rel_tol = 1e-2
        abs_tol = 1e-2
        for i, (lam, m) in enumerate(zip(self.lambdas, self.m_values)):
            expected = D / (m + 0.5)
            if not math.isclose(lam, expected, rel_tol=rel_tol, abs_tol=abs_tol):
                return (False, f"Index {i}: λ={lam} nm, expected ~{expected:.2f} nm")
        return (True, "All pairs satisfy constructive interference condition")

    def test_invariant_wavelengths_in_visible_range(self):
        """Check 400 nm ≤ λ ≤ 700 nm."""
        if self.lambdas is None:
            return (False, "SKIP")
        for i, lam in enumerate(self.lambdas):
            if lam < 400 or lam > 700:
                return (False, f"λ[{i}]={lam} nm is not in [400,700]")
        return (True, "All wavelengths lie in visible range")

    def test_invariant_product_constant(self):
        """Check that (m+0.5)*λ equals 2*n*t for every pair."""
        if self.lambdas is None or self.m_values is None:
            return (False, "SKIP")
        if self.t is None or self.n is None:
            return (False, "SKIP: missing t or n")
        D = 2.0 * self.n * self.t * 1e9
        rel_tol = 1e-2
        abs_tol = 1e-2
        for i, (lam, m) in enumerate(zip(self.lambdas, self.m_values)):
            product = (m + 0.5) * lam
            if not math.isclose(product, D, rel_tol=rel_tol, abs_tol=abs_tol):
                return (False, f"Index {i}: (m+0.5)*λ = {product:.2f} nm, expected {D:.2f} nm")
        return (True, "All products equal 2*n*t within tolerance")

    # -----------------------------------------------------------------
    # Modeling / trace checks
    # -----------------------------------------------------------------
    def test_modeling_correct_number_of_modes(self):
        """Expect exactly four wavelengths in the visible range."""
        if self.lambdas is None:
            return (False, "SKIP")
        expected_count = 4
        if len(self.lambdas) != expected_count:
            return (False, f"Expected {expected_count} wavelengths, found {len(self.lambdas)}")
        return (True, f"Found {expected_count} wavelengths")

    def test_modeling_monotonic_relation(self):
        """
        Check that as λ increases, m decreases (consistent with λ = 2nt/(m+0.5)).
        """
        if self.lambdas is None or self.m_values is None:
            return (False, "SKIP")
        n = len(self.lambdas)
        for i in range(n-1):
            lam_i, lam_j = self.lambdas[i], self.lambdas[i+1]
            m_i, m_j = self.m_values[i], self.m_values[i+1]
            if lam_i < lam_j and m_i >= m_j:
                return (False, f"Non‑monotonic: λ↑ but m not ↓ (index {i} -> {i+1})")
            if lam_i > lam_j and m_i <= m_j:
                return (False, f"Non‑monotonic: λ↓ but m not ↑ (index {i} -> {i+1})")
        return (True, "Monotonic relation between λ and m holds")

    def test_modeling_all_m_values_positive(self):
        if self.m_values is None:
            return (False, "SKIP")
        for i, m in enumerate(self.m_values):
            if m < 0:
                return (False, f"m[{i}] = {m} is negative")
        return (True, "All m values are non‑negative")

    # -----------------------------------------------------------------
    # Monotonicity test (perturb an input and compare)
    # -----------------------------------------------------------------
    def test_monotonicity_thickness_perturbation(self):
        """
        Perturb thickness by +1% and check that each λ scales proportionally.
        Baseline λ are the reported lambdas.
        """
        if self.lambdas is None or self.m_values is None:
            return (False, "SKIP: missing lambdas or m_values")
        if self.t is None or self.n is None:
            return (False, "SKIP: missing t or n")
        t_base = self.t
        t_pert = t_base * 1.01
        ratio_expected = t_pert / t_base
        rel_tol = 1e-2
        abs_tol = 1e-2
        for i, (lam_base, m) in enumerate(zip(self.lambdas, self.m_values)):
            lam_pert = 2.0 * self.n * t_pert * 1e9 / (m + 0.5)
            ratio = lam_pert / lam_base
            if not math.isclose(ratio, ratio_expected, rel_tol=rel_tol, abs_tol=abs_tol):
                return (False, f"Index {i}: λ ratio = {ratio:.4f}, expected {ratio_expected:.4f}")
        return (True, "Wavelengths scale correctly with +1% thickness perturbation")

    # -----------------------------------------------------------------
    # Weak answer sanity (at most one such method)
    # -----------------------------------------------------------------
    def test_answer_sanity_weak_checks(self):
        """
        Combined weak sanity for wavelengths and m_values.
        No derived expected values; only positivity, finiteness, and basic type.
        """
        if self.lambdas is None or self.m_values is None:
            return (False, "SKIP: missing data")
        # Check each wavelength is positive and finite
        for i, lam in enumerate(self.lambdas):
            if not (lam > 0 and math.isfinite(lam)):
                return (False, f"λ[{i}] = {lam} is not positive/finite")
        # Check each m_value is a non‑negative integer
        for i, m in enumerate(self.m_values):
            if not (isinstance(m, int) and m >= 0):
                try:
                    _m = int(m)
                    if _m < 0:
                        return (False, f"m[{i}] = {m} is negative")
                except (ValueError, TypeError):
                    return (False, f"m[{i}] = {m} is not an integer")
        return (True, "All wavelengths positive/finite, all m non‑negative integers")

    # -----------------------------------------------------------------
    # Main runner
    # -----------------------------------------------------------------
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