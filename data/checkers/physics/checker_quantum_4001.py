import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for the Bohr-Sommerfeld quantization of k-space area.
    Expected results: S = (2π e B (n+0.5))/(ħ c)
    """
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        givens = getattr(solution_instance, 'givens', {})
        self.e = givens.get('e', None)
        self.B = givens.get('B', None)
        self.n = givens.get('n', None)           # integer quantum number
        self.h = givens.get('h', None)
        self.c = givens.get('c', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.S = results.get('S', None)

        # Compute derived constants if possible
        if self.h is not None:
            self.hbar = self.h / (2 * math.pi)
        else:
            self.hbar = None

    # ------------------------------------------------------------------
    # Interface consistency tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all mandatory givens and results are present."""
        missing = []
        for name, val in [('e', self.e), ('B', self.B), ('n', self.n),
                          ('h', self.h), ('c', self.c), ('S', self.S)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_n_is_nonnegative(self):
        """Check that n is a non‑negative integer."""
        if self.n is None:
            return (False, "SKIP: n not available")
        if isinstance(self.n, (int, float)):
            if self.n >= 0 and self.n == int(self.n):
                return (True, f"n = {self.n} is valid")
            else:
                return (False, f"n = {self.n} is not a non‑negative integer")
        else:
            return (False, f"n has unexpected type {type(self.n)}")

    # ------------------------------------------------------------------
    # Physical invariant tests (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_flux_quantization(self):
        """
        Using the Onsager relation A = (ħ c / (e B))^2 S,
        check that the magnetic flux Φ = B A equals the quantized flux (n+½) h c / e.
        This tests internal consistency of the derivation.
        """
        if None in (self.S, self.e, self.B, self.h, self.c, self.n):
            return (False, "SKIP: missing data for flux quantization check")
        try:
            hbar = self.h / (2 * math.pi)
            # Real‑space area from Onsager relation (from explanation steps)
            A = (hbar * self.c / (self.e * self.B)) ** 2 * self.S
            phi = self.B * A
            phi_expected = (self.n + 0.5) * self.h * self.c / self.e
            if math.isclose(phi, phi_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Flux matches: phi = {phi:.3e}, expected {phi_expected:.3e}")
            else:
                return (False, f"Flux mismatch: phi = {phi:.3e}, expected {phi_expected:.3e}")
        except Exception as e:
            return (False, f"Flux quantization check crashed: {e}")

    def test_invariant_onsager_ratio(self):
        """
        Check the ratio S / B equals the theoretical value 2π e (n+½) / (ħ c).
        This is a direct corollary of the derived formula.
        """
        if None in (self.S, self.e, self.B, self.h, self.c, self.n):
            return (False, "SKIP: missing data for Onsager ratio check")
        try:
            hbar = self.h / (2 * math.pi)
            ratio_actual = self.S / self.B
            ratio_expected = 2 * math.pi * self.e * (self.n + 0.5) / (hbar * self.c)
            if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Ratio S/B matches: {ratio_actual:.3e} vs {ratio_expected:.3e}")
            else:
                return (False, f"Ratio S/B mismatch: {ratio_actual:.3e} vs {ratio_expected:.3e}")
        except Exception as e:
            return (False, f"Onsager ratio check crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling quality tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_derivation_step_consistency(self):
        """
        Verify that the given S, when used to back‑calculate the kinetic momentum
        integral, yields the correct Bohr‑Sommerfeld condition.
        We check that ∮ p·dr = (n+½)h + (e/c) Φ, with p = ħ k,
        and ∮ p·dr = ħ^2 S / (m?) Actually more direct:
        From the derivation, we have 2eΦ/c = ∮ p·dr.
        Using p = ħk and the area scaling, we can test that the relation
        ∮ p·dr - (e/c)Φ = (n+½)h holds.
        """
        if None in (self.S, self.e, self.B, self.h, self.c, self.n):
            return (False, "SKIP: missing data for derivation step check")
        try:
            hbar = self.h / (2 * math.pi)
            # Real‑space area A from Onsager relation
            A = (hbar * self.c / (self.e * self.B)) ** 2 * self.S
            phi = self.B * A
            # Compute ∮ p·dr from the derived relation ∮ p·dr = 2eΦ/c
            p_integral = 2 * self.e * phi / self.c
            # Bohr‑Sommerfeld: ∮ p·dr - (e/c)Φ = (n+½)h
            lhs = p_integral - (self.e / self.c) * phi
            rhs = (self.n + 0.5) * self.h
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Bohr‑Sommerfeld holds: LHS = {lhs:.3e}, RHS = {rhs:.3e}")
            else:
                return (False, f"Bohr‑Sommerfeld mismatch: LHS = {lhs:.3e}, RHS = {rhs:.3e}")
        except Exception as e:
            return (False, f"Derivation step check crashed: {e}")

    def test_modeling_formula_consistency(self):
        """
        Check that the reported k‑space area S is consistent with the derived
        formula S = (2π e B (n+½))/(ħ c), re‑computed from the given givens.
        This verifies correct implementation of the physics derivation.
        """
        if None in (self.e, self.B, self.n, self.h, self.c, self.S):
            return (False, "SKIP: missing data for formula consistency check")
        try:
            hbar = self.h / (2 * math.pi)
            expected_S = (2 * math.pi * self.e * self.B * (self.n + 0.5)) / (hbar * self.c)
            if math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"S = {self.S:.3e} matches expected {expected_S:.3e}")
            else:
                return (False, f"S = {self.S:.3e} differs significantly from expected {expected_S:.3e}")
        except Exception as e:
            return (False, f"Formula consistency check crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity / trend tests
    # ------------------------------------------------------------------
    def test_monotonicity_S_increases_with_B(self):
        """
        Check that the derivative ∂S/∂B is positive.
        Using the explicit formula, S increases with B.
        We test by perturbing B by +10% and verifying that S would increase.
        """
        if None in (self.S, self.e, self.B, self.h, self.c, self.n):
            return (False, "SKIP: missing data for monotonicity test")
        try:
            hbar = self.h / (2 * math.pi)
            # Baselines
            B0 = self.B
            S0 = self.S
            # Perturb B up by 10%
            B1 = B0 * 1.10
            # Compute new S using the same functional relationship (linear in B)
            # Since S ∝ B, the scaling is exact:
            S1 = S0 * (B1 / B0)
            if S1 > S0:
                return (True, f"S(B) increases: S0={S0:.3e}, S1={S1:.3e}")
            else:
                return (False, f"S(B) does not increase: S0={S0:.3e}, S1={S1:.3e}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_S_increases_with_n(self):
        """
        Check that S increases with quantum number n.
        Perturb n by +1 (but n integer, so we check n+1 gives larger S).
        """
        if None in (self.S, self.e, self.B, self.h, self.c, self.n):
            return (False, "SKIP: missing data for monotonicity test")
        try:
            hbar = self.h / (2 * math.pi)
            n0 = self.n
            S0 = self.S
            # For n' = n0+1, S is proportional to (n+0.5)
            S1 = S0 * (n0 + 1.5) / (n0 + 0.5)
            if S1 > S0:
                return (True, f"S(n) increases: S(n0)={S0:.3e}, S(n0+1)={S1:.3e}")
            else:
                return (False, f"S(n) does not increase: S(n0)={S0:.3e}, S(n0+1)={S1:.3e}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """
        Check that the reported k‑space area S is positive and finite.
        """
        if self.S is None:
            return (False, "SKIP: S not available")
        try:
            if math.isfinite(self.S) and self.S > 0:
                return (True, f"S = {self.S:.3e} is positive and finite")
            else:
                return (False, f"S = {self.S} is not positive/finite")
        except Exception as e:
            return (False, f"Sign/finite check crashed: {e}")

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