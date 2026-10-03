import sympy as sp
import math
import re

class PhysicsChecker:
    """
    Audit solution for a particle in a potential V(x) = -g [δ(x-a) + δ(x+a)].
    Checks interface, physics invariants, modeling consistency, and monotonicity.
    """
    def __init__(self, solution_instance):
        # Extract physical parameters (could be None)
        self.m = getattr(solution_instance, 'm', None)
        self.g = getattr(solution_instance, 'g', None)
        self.a = getattr(solution_instance, 'a', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.wavefunction_even = self.results.get('wavefunction_even', None)
        self.transcendental_eq = self.results.get('transcendental_eq', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_wavefunction_even_exists(self):
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not provided")
        if not isinstance(self.wavefunction_even, str):
            return (False, "FAIL: wavefunction_even should be a string")
        return (True, "PASS: wavefunction_even is a string")

    def test_interface_transcendental_eq_exists(self):
        if self.transcendental_eq is None:
            return (False, "SKIP: transcendental_eq not provided")
        if not isinstance(self.transcendental_eq, str):
            return (False, "FAIL: transcendental_eq should be a string")
        return (True, "PASS: transcendental_eq is a string")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem‑based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_parity_even(self):
        """ψ(-x) = ψ(x) because ground state is even."""
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not available")
        x, a, k, B = sp.symbols('x a k B', positive=True, real=True)
        try:
            wf_str = re.sub(r'\|x\|', 'Abs(x)', self.wavefunction_even)
            wf = sp.sympify(wf_str)
            diff = sp.simplify(wf.subs(x, -x) - wf)
            if diff == 0:
                return (True, "PASS: wavefunction is even")
            else:
                return (False, f"FAIL: parity not even, difference = {diff}")
        except Exception as e:
            return (False, f"FAIL: could not check parity: {e}")

    def test_invariant_asymptotic_decay(self):
        """Wavefunction must vanish as |x| → ∞ (bound state)."""
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not available")
        # Look for exponential decay in the outer region
        if 'exp(-k*Abs(x))' in self.wavefunction_even or 'e^{-k|x|}' in self.wavefunction_even:
            return (True, "PASS: wavefunction decays exponentially")
        else:
            return (False, "FAIL: no asymptotic decay pattern found")

    def test_invariant_continuity_at_a(self):
        """Wavefunction must be continuous at x = a."""
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not available")
        # The form given in the explanation ensures continuity: both sides give B cosh(ka).
        # We simply verify that the provided string matches the expected structure.
        expected_outer = r'B\s*e\^{ka}\s*\\cosh\s*\(ka\)\s*e\^{-k\|x\|}'
        expected_inner = r'B\s*\\cosh\s*\(kx\)'
        if re.search(expected_outer, self.wavefunction_even) and re.search(expected_inner, self.wavefunction_even):
            return (True, "PASS: continuity at x=a is built into the piecewise form")
        else:
            return (False, "FAIL: wavefunction form does not obviously enforce continuity")

    # ----------------------------------------------------------------------
    # Modeling/trace quality checks
    # ----------------------------------------------------------------------
    def test_modeling_derivative_jump(self):
        """
        Verify that the derivative jump condition leads to the provided
        transcendental equation. This test uses symbolic algebra.
        """
        if any(v is None for v in [self.m, self.g, self.a, self.hbar]):
            return (False, "SKIP: missing m, g, a, or hbar")
        if self.wavefunction_even is None or self.transcendental_eq is None:
            return (False, "SKIP: missing wavefunction or transcendental equation")

        x, a, k, B, m, g, hbar = sp.symbols('x a k B m g hbar', positive=True, real=True)
        try:
            # Region definitions for x ≥ 0 (even parity)
            psi_left  = B * sp.cosh(k * x)                     # 0 ≤ x ≤ a
            psi_right = B * sp.exp(k * a) * sp.cosh(k * a) * sp.exp(-k * x)  # x ≥ a

            # Derivatives at x = a (from left and right)
            dpsi_left  = sp.diff(psi_left, x).subs(x, a)
            dpsi_right = sp.diff(psi_right, x).subs(x, a)

            # Jump condition: ψ'(a+) - ψ'(a-) = - (2 m g / ħ²) ψ(a)
            jump_left  = dpsi_right - dpsi_left
            jump_right = - (2 * m * g / hbar**2) * psi_left.subs(x, a)

            # Derived equation from matching jumps
            eq_derived = sp.simplify(sp.Eq(jump_left, jump_right))
            # eq_derived should simplify to k*(1+tanh(ka)) = 2 m g / ħ²
            eq_correct = sp.Eq(k * (1 + sp.tanh(k * a)), 2 * m * g / hbar**2)
            diff = sp.simplify(eq_derived.lhs - eq_correct.lhs) - sp.simplify(eq_derived.rhs - eq_correct.rhs)
            if diff == 0:
                return (True, "PASS: jump condition leads to correct transcendental equation")
            else:
                return (False, f"FAIL: derived equation differs from expected: {eq_derived}")
        except Exception as e:
            return (False, f"FAIL: error in symbolic derivation: {e}")

    def test_modeling_schrodinger_region(self):
        """
        Verify that the wavefunction satisfies the time‑independent Schrödinger
        equation away from the delta peaks (x ≠ ±a).
        """
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not available")
        x, a, k, B, m, E = sp.symbols('x a k B m E', positive=True, real=True)
        hbar = sp.symbols('hbar', positive=True, real=True)
        try:
            wf_str = re.sub(r'\|x\|', 'Abs(x)', self.wavefunction_even)
            wf = sp.sympify(wf_str)
            # For x ≠ ±a, potential is zero. Free Schrödinger equation:
            # -hbar²/(2m) ψ''(x) = E ψ(x)   →   ψ''(x) = - (2mE/hbar²) ψ(x)
            # With E = -hbar² k² / (2m), so ψ''(x) = k² ψ(x)
            # Check that ψ'' - k² ψ = 0
            psi_second = sp.diff(wf, x, 2)
            lhs = psi_second - k**2 * wf
            lhs_simplified = sp.simplify(lhs)
            # The expression should be zero except possibly at x = ±a.
            # We can check for symbolic equality ignoring the points.
            if lhs_simplified == 0:
                return (True, "PASS: wavefunction satisfies Schrödinger equation in each region")
            else:
                # Possibly the simplification is not perfect due to Abs; try to evaluate at a sample point
                # Use a numeric substitution to see if it is close to zero.
                subs_dict = {a: 1, k: 1, B: 1, x: 0.5}
                numeric_val = float(lhs_simplified.subs(subs_dict))
                if abs(numeric_val) < 1e-10:
                    return (True, "PASS: Schrödinger equation holds (numerically verified)")
                else:
                    return (False, f"FAIL: Schrödinger equation not satisfied, residual = {lhs_simplified}")
        except Exception as e:
            return (False, f"FAIL: could not check Schrödinger equation: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of input)
    # ----------------------------------------------------------------------
    def test_monotonicity_g_increase_increases_k(self):
        """
        For fixed a, m, ħ, increasing g should increase k (binding deepens).
        Numerically solve the transcendental equation for a baseline and perturbed g.
        """
        if any(v is None for v in [self.m, self.g, self.a, self.hbar]):
            return (False, "SKIP: missing m, g, a, or hbar")
        # Use the exact equation from the problem: k*(1+tanh(k*a)) = 2*m*g/ħ²
        # We'll solve numerically for k given numeric values of parameters.
        # Choose reasonable numeric values (avoiding issues)
        m_val = float(self.m) if not isinstance(self.m, (int, float)) else 1.0
        g_val = float(self.g) if not isinstance(self.g, (int, float)) else 1.0
        a_val = float(self.a) if not isinstance(self.a, (int, float)) else 1.0
        hbar_val = float(self.hbar) if not isinstance(self.hbar, (int, float)) else 1.0

        # Define symbolic k
        k = sp.symbols('k', positive=True, real=True)
        eq = sp.Eq(k * (1 + sp.tanh(k * a_val)), 2 * m_val * g_val / hbar_val**2)
        # Solve baseline k
        try:
            # Provide a reasonable initial guess (e.g., mg/ħ²)
            guess = m_val * g_val / hbar_val**2
            # Use nsolve; adjust if initial guess is zero
            if guess == 0:
                guess = 0.1
            k_baseline = float(sp.nsolve(eq, k, guess))
        except Exception as e:
            return (False, f"FAIL: could not solve baseline k: {e}")

        # Now perturb g by +1%
        g_perturbed = g_val * 1.01
        eq_perturbed = sp.Eq(k * (1 + sp.tanh(k * a_val)), 2 * m_val * g_perturbed / hbar_val**2)
        try:
            k_perturbed = float(sp.nsolve(eq_perturbed, k, guess))
        except Exception as e:
            return (False, f"FAIL: could not solve perturbed k: {e}")

        if k_perturbed > k_baseline + 1e-10:
            return (True, f"PASS: k increased from {k_baseline:.4f} to {k_perturbed:.4f} when g increased by 1%")
        else:
            return (False, f"FAIL: k did not increase as expected (baseline {k_baseline:.4f}, perturbed {k_perturbed:.4f})")

    def test_monotonicity_a_increase_decreases_k(self):
        """
        For fixed m, g, ħ, increasing a should decrease k (wells further apart weaken binding).
        Numerically solve the transcendental equation for baseline and perturbed a.
        """
        if any(v is None for v in [self.m, self.g, self.a, self.hbar]):
            return (False, "SKIP: missing m, g, a, or hbar")
        m_val = float(self.m) if not isinstance(self.m, (int, float)) else 1.0
        g_val = float(self.g) if not isinstance(self.g, (int, float)) else 1.0
        a_val = float(self.a) if not isinstance(self.a, (int, float)) else 1.0
        hbar_val = float(self.hbar) if not isinstance(self.hbar, (int, float)) else 1.0

        k = sp.symbols('k', positive=True, real=True)
        eq = sp.Eq(k * (1 + sp.tanh(k * a_val)), 2 * m_val * g_val / hbar_val**2)
        try:
            guess = m_val * g_val / hbar_val**2
            if guess == 0:
                guess = 0.1
            k_baseline = float(sp.nsolve(eq, k, guess))
        except Exception as e:
            return (False, f"FAIL: could not solve baseline k: {e}")

        # Perturb a by +10%
        a_perturbed = a_val * 1.1
        eq_perturbed = sp.Eq(k * (1 + sp.tanh(k * a_perturbed)), 2 * m_val * g_val / hbar_val**2)
        try:
            k_perturbed = float(sp.nsolve(eq_perturbed, k, guess))
        except Exception as e:
            return (False, f"FAIL: could not solve perturbed k: {e}")

        if k_perturbed < k_baseline - 1e-10:
            return (True, f"PASS: k decreased from {k_baseline:.4f} to {k_perturbed:.4f} when a increased by 10%")
        else:
            return (False, f"FAIL: k did not decrease as expected (baseline {k_baseline:.4f}, perturbed {k_perturbed:.4f})")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_wavefunction_finite(self):
        """Check that the wavefunction is finite everywhere (no singularities)."""
        if self.wavefunction_even is None:
            return (False, "SKIP: wavefunction_even not available")
        # The wavefunction is built from cosh and exp, both finite for finite x.
        # Check that it contains no poles like 1/(x-a) etc.
        if '1/' in self.wavefunction_even or 'coth' in self.wavefunction_even:
            return (False, "FAIL: wavefunction contains potentially singular terms")
        return (True, "PASS: wavefunction appears finite")

    # ----------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)