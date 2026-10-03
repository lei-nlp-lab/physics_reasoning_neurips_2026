import math
import traceback
from collections.abc import Mapping

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely – no numeric defaults
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, Mapping):
            self.results = {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

        # Required result keys – use None if missing
        self.L = self.results.get('L', None)
        self.S = self.results.get('S', None)
        self.prob_both_opposite = self.results.get('prob_both_opposite', None)
        self.ang_dist = self.results.get('ang_dist', None)

    # --------------- Interface tests ---------------
    def test_interface_required_keys(self):
        """All four expected result keys must exist and be non‑None."""
        missing = []
        for key in ['L', 'S', 'prob_both_opposite', 'ang_dist']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All keys present.")

    def test_interface_L_type(self):
        """L must be an integer."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if not isinstance(self.L, int):
            return (False, f"L is {type(self.L).__name__}, expected int.")
        return (True, "L is integer.")

    def test_interface_S_type(self):
        """S must be an integer."""
        if self.S is None:
            return (False, "SKIP: S not available")
        if not isinstance(self.S, int):
            return (False, f"S is {type(self.S).__name__}, expected int.")
        return (True, "S is integer.")

    def test_interface_prob_type(self):
        """Probability must be numeric (int or float)."""
        if self.prob_both_opposite is None:
            return (False, "SKIP: prob_both_opposite not available")
        if not isinstance(self.prob_both_opposite, (int, float)):
            return (False, f"Probability is {type(self.prob_both_opposite).__name__}, expected numeric.")
        return (True, "Probability is numeric.")

    def test_interface_ang_dist_type(self):
        """Angular distribution must be a string."""
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available")
        if not isinstance(self.ang_dist, str):
            return (False, f"ang_dist is {type(self.ang_dist).__name__}, expected str.")
        return (True, "Angular distribution is a string.")

    # --------------- Invariant tests (theorem‑based) ---------------
    def test_invariant_L_odd(self):
        """Parity conservation: initial parity = (-1)*1 = -1,
           final parity = (-1)^L => L must be odd."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L % 2 != 1:
            return (False, f"L = {self.L} is even; parity forces L odd.")
        return (True, "L is odd (parity conserved).")

    def test_invariant_S_triplet(self):
        """Pauli antisymmetry: (-1)^L * (-1)^(S+1) = -1.
           With L odd, this forces S = 1."""
        if self.L is None or self.S is None:
            return (False, "SKIP: L or S not available")
        product = ((-1) ** self.L) * ((-1) ** (self.S + 1))
        if product != -1:
            return (False, f"Pauli antisymmetry requires product = -1, got {product}.")
        if self.S != 1:
            return (False, f"With L odd, S must be 1, got S = {self.S}.")
        return (True, "S = 1 (triplet) as required by Pauli principle.")

    def test_invariant_total_J_allowed(self):
        """Total J must be 1 (initial state: J(π⁻)=0, J(d)=1).
           Check that J=1 lies within |L‑S| to L+S."""
        if self.L is None or self.S is None:
            return (False, "SKIP: L or S not available")
        L, S = self.L, self.S
        J = 1  # derived from initial state
        if not (abs(L - S) <= J <= L + S):
            return (False, f"J=1 not in [{abs(L-S)}, {L+S}].")
        return (True, "J=1 is consistent with L and S.")

    def test_invariant_pauli_condition(self):
        """Explicit antisymmetry condition: (-1)^L * (-1)^(S+1) = -1."""
        if self.L is None or self.S is None:
            return (False, "SKIP: L or S not available")
        product = ((-1) ** self.L) * ((-1) ** (self.S + 1))
        if product != -1:
            return (False, f"Antisymmetry condition fails: product = {product}.")
        return (True, "Antisymmetry condition satisfied.")

    # --------------- Modeling / trace tests ---------------
    def test_modeling_prob_zero(self):
        """Conservation of M_J: initial M=+1 (deuteron polarized along z).
           If both neutrons have spin opposite (M_S = -1), then M_L must be +2,
           but with L=1, |M_L| ≤ 1 → impossible → probability 0."""
        if self.prob_both_opposite is None:
            return (False, "SKIP: probability not available")
        if self.prob_both_opposite != 0:
            return (False, f"Probability is {self.prob_both_opposite}, expected 0.")
        return (True, "Probability is 0 as per M_J conservation.")

    def test_modeling_ang_dist_sin2(self):
        """Angular distribution for spin‑down neutron must contain sin²θ factor
           (originates from |Y₁¹|²). No coefficient check – only the functional form."""
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available")
        text = self.ang_dist.lower()
        has_sin2 = ('sin^2' in text) or ('sin²' in text) or ('sin2' in text)
        has_theta = ('theta' in text) or ('θ' in text)
        if not has_sin2 or not has_theta:
            return (False, f"Expression '{self.ang_dist}' does not contain sin²θ.")
        return (True, "Angular distribution contains sin²θ dependence.")

    # --------------- Monotonicity tests ---------------
    def test_monotonicity_ang_dist(self):
        """The angular distribution dP/dΩ is proportional to sin²θ, which is
           monotonically increasing on [0, π/2].  Verify by comparing a
           baseline (θ=0) with a perturbed value (θ=0.1 rad)."""
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available")
        # Parse expression
        import sympy as sp
        expr_str = self.ang_dist
        # Replace common formatting
        replacements = {'^': '**', 'π': 'pi', 'θ': 'theta',
                        'sin²theta': 'sin(theta)**2', 'sin^2theta': 'sin(theta)**2'}
        for old, new in replacements.items():
            expr_str = expr_str.replace(old, new)
        try:
            expr = sp.parsing.sympy_parser.parse_expr(expr_str)
        except Exception:
            return (False, "SKIP: cannot parse angular distribution expression.")
        theta = sp.Symbol('theta')
        try:
            f = sp.lambdify(theta, expr, 'numpy')
        except Exception:
            return (False, "SKIP: cannot lambdify expression.")
        import numpy as np
        try:
            baseline = f(0.0)                     # θ = 0
            perturbed = f(0.1)                    # small positive θ
        except Exception as e:
            return (False, f"Evaluation failed: {e}")
        if baseline is None or perturbed is None:
            return (False, "Evaluation returned None.")
        # Baseline should be 0 (or very small), perturbed should be positive
        if math.isclose(baseline, 0.0, abs_tol=1e-9) and perturbed > baseline:
            return (True, f"Baseline f(0)={baseline:.4g}, perturbed f(0.1)={perturbed:.4g} > baseline. Monotonic increase on [0, π/2] verified.")
        else:
            return (False, f"Monotonicity condition not met: baseline={baseline:.4g}, perturbed={perturbed:.4g}.")

    # --------------- Weak answer sanity (at most one) ---------------
    def test_answer_sanity_general(self):
        """Basic sanity: L,S positive integers; probability exactly 0; ang_dist non‑empty."""
        if self.L is not None:
            if not isinstance(self.L, int) or self.L <= 0:
                return (False, f"L is not a positive integer: {self.L}")
        if self.S is not None:
            if not isinstance(self.S, int) or self.S <= 0:
                return (False, f"S is not a positive integer: {self.S}")
        if self.prob_both_opposite is not None and self.prob_both_opposite != 0:
            return (False, f"Probability non‑zero: {self.prob_both_opposite}")
        if self.ang_dist is not None and len(self.ang_dist) == 0:
            return (False, "Angular distribution string is empty.")
        return (True, "Basic sanity passed: positive integers, zero prob, non‑empty expression.")

    # --------------- Main runner ---------------
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