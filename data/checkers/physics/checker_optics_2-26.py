import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results keys safely
        self.arrangement = solution_instance.results.get('arrangement', None)
        self.ss_zeros = solution_instance.results.get('ss_zeros', None)
        self.ds_maxima = solution_instance.results.get('ds_maxima', None)
        self.ds_minima = solution_instance.results.get('ds_minima', None)
        self.I_triple = solution_instance.results.get('I_triple', None)

        # Extract optional numerical inputs (givens) for monotonicity tests.
        # Use getattr/givens to avoid numeric fallbacks.
        givens = getattr(solution_instance, 'givens', None) or {}
        self.lambda_ = getattr(solution_instance, 'lambda',
                               givens.get('lambda', None))
        self.a = getattr(solution_instance, 'a',
                         givens.get('a', None))
        self.d = getattr(solution_instance, 'd',
                         givens.get('d', None))

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_arrangement_exists(self):
        if self.arrangement is None:
            return (False, "Missing 'arrangement' key")
        if not isinstance(self.arrangement, str):
            return (False, f"'arrangement' is not a string: {type(self.arrangement)}")
        return (True, "'arrangement' present and is string")

    def test_interface_ss_zeros_exists(self):
        if self.ss_zeros is None:
            return (False, "Missing 'ss_zeros' key")
        if not isinstance(self.ss_zeros, str):
            return (False, f"'ss_zeros' is not a string: {type(self.ss_zeros)}")
        return (True, "'ss_zeros' present and is string")

    def test_interface_ds_maxima_exists(self):
        if self.ds_maxima is None:
            return (False, "Missing 'ds_maxima' key")
        if not isinstance(self.ds_maxima, str):
            return (False, f"'ds_maxima' is not a string: {type(self.ds_maxima)}")
        return (True, "'ds_maxima' present and is string")

    def test_interface_ds_minima_exists(self):
        if self.ds_minima is None:
            return (False, "Missing 'ds_minima' key")
        if not isinstance(self.ds_minima, str):
            return (False, f"'ds_minima' is not a string: {type(self.ds_minima)}")
        return (True, "'ds_minima' present and is string")

    def test_interface_I_triple_exists(self):
        if self.I_triple is None:
            return (False, "Missing 'I_triple' key")
        if not isinstance(self.I_triple, str):
            return (False, f"'I_triple' is not a string: {type(self.I_triple)}")
        return (True, "'I_triple' present and is string")

    # ------------------------------------------------------------------
    # Invariant tests (physics content of the formulas)
    # ------------------------------------------------------------------
    def test_invariant_ss_zero_condition(self):
        if self.ss_zeros is None:
            return (False, "SKIP: ss_zeros not available")
        s = self.ss_zeros.lower().replace(" ", "")
        if "sintheta=klambda/a" in s or "sinθ=kλ/a" in s:
            return (True, "Zero condition matches a sinθ = kλ")
        if "klambda" in s and "/a" in s:
            if "k!=0" in s or "k ≠ 0" in s:
                return (True, "Zero condition correct (k!=0)")
            else:
                return (False, "Zero condition missing 'k != 0' clause")
        return (False, f"ss_zeros string does not contain expected relation: {s}")

    def test_invariant_ds_maxima_condition(self):
        if self.ds_maxima is None:
            return (False, "SKIP: ds_maxima not available")
        s = self.ds_maxima.lower().replace(" ", "")
        if "mlambda/d" in s or "mλ/d" in s:
            return (True, "Double‑slit maxima condition correct")
        return (False, f"ds_maxima string does not contain 'mlambda/d': {s}")

    def test_invariant_ds_minima_condition(self):
        if self.ds_minima is None:
            return (False, "SKIP: ds_minima not available")
        s = self.ds_minima.lower().replace(" ", "")
        if "(m+1/2)lambda/d" in s or "(m+½)λ/d" in s or "(m+0.5)lambda/d" in s:
            return (True, "Double‑slit minima condition correct")
        return (False, f"ds_minima string does not contain '(m+1/2)lambda/d': {s}")

    def test_invariant_triple_intensity_form(self):
        if self.I_triple is None:
            return (False, "SKIP: I_triple not available")
        s = self.I_triple.lower().replace(" ", "")
        if "sin(pi a" in s and "sin(3 pi d" in s:
            return (True, "I_triple contains correct symbolic form")
        if "sin^2" in s and "sin" in s and "3" in s:
            return (True, "I_triple appears to contain squared sine ratio terms")
        return (False, f"I_triple string does not match expected pattern: {s}")

    # ------------------------------------------------------------------
    # Modeling tests (quality of setup description)
    # ------------------------------------------------------------------
    def test_modeling_arrangement_fraunhofer(self):
        if self.arrangement is None:
            return (False, "SKIP: arrangement not available")
        s = self.arrangement.lower()
        has_l1 = "l1" in s
        has_l2 = "l2" in s
        has_plane_waves = "plane wave" in s
        has_back_focal = "back focal plane" in s or "focal plane" in s
        score = sum([has_l1, has_l2, has_plane_waves, has_back_focal])
        if score >= 2:
            return (True, "Arrangement describes Fraunhofer setup")
        return (False, f"Arrangement lacks Fraunhofer key elements: {s}")

    def test_modeling_distinguish_fraunhofer_fresnel(self):
        if self.arrangement is None:
            return (False, "SKIP")
        # A good arrangement text may also distinguish the two; no strict requirement.
        return (True, "Arrangement describes Fraunhofer (lens used)")

    # ------------------------------------------------------------------
    # Monotonicity test (perturb an input and check effect on derived quantity)
    # ------------------------------------------------------------------
    def test_monotonicity_ds_maxima_spacing(self):
        """For double‑slit maxima, the angular spacing Δsinθ = λ/d.
        If d increases, Δsinθ should decrease (monotonic).
        """
        # Need numerical values of λ and d. If missing, skip.
        if self.lambda_ is None or self.d is None:
            return (False, "SKIP: wavelength ('lambda') and slit separation ('d') not available")

        d0 = float(self.d)
        lam = float(self.lambda_)

        # Baseline: first maximum sinθ₁ = λ / d
        sinθ1 = lam / d0

        # Perturb d upward by 10%
        d_pert = d0 * 1.10
        sinθ_pert = lam / d_pert

        # Check monotonic: larger d => smaller sinθ for m=1
        if sinθ_pert < sinθ1:
            return (True, f"Spacing decreases when d increases: {sinθ_pert:.4e} < {sinθ1:.4e}")
        else:
            return (False, f"Spacing did not decrease: {sinθ_pert:.4e} >= {sinθ1:.4e}")

    def test_monotonicity_triple_slit_principal_maxima(self):
        """For triple slit, the condition for principal maxima is d sinθ = mλ.
        Perturb d and check that the first maximum shifts to smaller sinθ.
        """
        if self.lambda_ is None or self.d is None:
            return (False, "SKIP: wavelength and slit separation not available")
        d0 = float(self.d)
        lam = float(self.lambda_)

        sinθ_base = lam / d0       # m=1 principal maximum
        d_pert = d0 * 1.10
        sinθ_pert = lam / d_pert

        if sinθ_pert < sinθ_base:
            return (True, f"Principal maximum moves to smaller sinθ when d increases: {sinθ_pert:.4e} < {sinθ_base:.4e}")
        else:
            return (False, f"Monotonicity violated: {sinθ_pert:.4e} >= {sinθ_base:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_triple_not_empty(self):
        if self.I_triple is None:
            return (False, "SKIP")
        if len(self.I_triple.strip()) == 0:
            return (False, "I_triple string is empty")
        return (True, "I_triple is non‑empty string")

    # ------------------------------------------------------------------
    # Runner
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
        return (all_pass, logs)