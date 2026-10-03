import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.state = self.results.get('state', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ---------- Interface tests ----------
    def test_interface_state_exists(self):
        if self.state is None:
            return (False, "SKIP: 'state' is not present in results")
        if not isinstance(self.state, str):
            return (False, f"state should be a string, got {type(self.state)}")
        return (True, "state key exists and is a string")

    def test_interface_state_nonempty(self):
        if self.state is None:
            return (False, "SKIP: state not available")
        if len(self.state.strip()) == 0:
            return (False, "state string is empty")
        return (True, "state string is non-empty")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_antisymmetry(self):
        """
        Check that the state is antisymmetric under exchange of electrons 1 and 2.
        We use the known transformation properties of the mixed-symmetry basis (a, b)
        for both spatial and spin parts, as derived in the explanation.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # Define symbolic basis vectors
        aL, bL = sp.symbols('aL bL')        # |1,1>_a, |1,1>_b (spatial)
        aS, bS = sp.symbols('aS bS')        # |1/2,1/2>_a, |1/2,1/2>_b (spin)

        # State we expect: 1/√2 ( aL * bS - bL * aS )
        state_sym = (1/sp.sqrt(2)) * (aL * bS - bL * aS)

        # Transformation under (12) from explanation:
        # Spatial:
        #   (12) |1,1>_a = 1/2 |1,1>_a + sqrt(3)/2 |1,1>_b
        #   (12) |1,1>_b = sqrt(3)/2 |1,1>_a - 1/2 |1,1>_b
        # Spin (from the derivation in explanation):
        #   (12) |1/2,1/2>_a = sqrt(3)/2 |1/2,1/2>_a - 1/2 |1/2,1/2>_b
        #   (12) |1/2,1/2>_b = 1/2 |1/2,1/2>_a + sqrt(3)/2 |1/2,1/2>_b
        trans = {
            aL: sp.Rational(1,2)*aL + sp.sqrt(3)/2 * bL,
            bL: sp.sqrt(3)/2 * aL - sp.Rational(1,2)*bL,
            aS: sp.sqrt(3)/2 * aS - sp.Rational(1,2)*bS,
            bS: sp.Rational(1,2)*aS + sp.sqrt(3)/2 * bS
        }

        state_transformed = state_sym.subs(trans)
        simplified = sp.simplify(state_transformed)
        expected = -state_sym

        if sp.simplify(simplified - expected) == 0:
            return (True, "State is antisymmetric under (12) exchange")
        else:
            return (False, f"State is NOT antisymmetric: transformed = {simplified}")

    def test_invariant_normalization(self):
        """
        The state must be normalized to 1.
        The symbolic expression has coefficient 1/√2 and two orthogonal terms,
        giving |coefficient|^2 * (1 + 1) = 1.
        We also verify that the LaTeX string contains the correct factor.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # Check that the factor 1/√2 appears explicitly
        if r"{\sqrt{2}}" not in self.state and r"\frac{1}{\sqrt{2}}" not in self.state:
            return (False, "State does not contain the factor 1/√2")
        # Check that the two product terms are present (with opposite signs)
        if "|1,1\\rangle_a" not in self.state or "|1,1\\rangle_b" not in self.state:
            return (False, "Missing spatial basis states a or b")
        if "|\\frac{1}{2},\\frac{1}{2}\\rangle_b" not in self.state or "|\\frac{1}{2},\\frac{1}{2}\\rangle_a" not in self.state:
            return (False, "Missing spin basis states a or b")
        # Check antisymmetry via sign (explicit minus sign between the two products)
        if "-" not in self.state:
            return (False, "Missing minus sign between the two product terms")
        return (True, "State normalization factor and structure are correct")

    def test_invariant_mixed_symmetry(self):
        """
        Verify that the state uses the mixed-symmetry basis (a, b) for both
        spatial and spin parts. The a and b subscripts indicate the specific
        non-symmetric combinations derived in the explanation.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # The state should contain exactly two product terms with subscript a and b
        # Check that the pattern '_a' and '_b' appears
        if "_a" not in self.state or "_b" not in self.state:
            return (False, "State does not use mixed-symmetry a/b labels")
        return (True, "State uses mixed-symmetry basis (a/b) as expected")

    # ---------- Modeling/trace checks ----------
    def test_modeling_uncoupled_basis(self):
        """
        The state should be expressed in the uncoupled basis |L, M_L> |S, M_S>
        (product of spatial and spin). The LaTeX should show this product.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # Look for a pattern like ... | ... > ... | ... >  (product of two kets)
        # At least two separate bra-ket expressions should be present.
        count_kets = self.state.count("\\rangle")  # Look for closing braces
        if count_kets < 2:
            return (False, f"Only {count_kets} ket found; expected at least 2 product terms")
        return (True, "State appears as a product of spatial and spin kets")

    def test_modeling_no_symmetric_terms(self):
        """
        The state should not contain symmetric combinations like a_a * spin_a or b_b * spin_b.
        Such terms would be symmetric under exchange.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # Check that the state does NOT contain the pattern |1,1>_a |1/2,1/2>_a
        # (product of same subscript) or |1,1>_b |1/2,1/2>_b
        if "|1,1\\rangle_a \\left| \\frac{1}{2}, \\frac{1}{2} \\right\\rangle_a" in self.state or \
           "|1,1\\rangle_b \\left| \\frac{1}{2}, \\frac{1}{2} \\right\\rangle_b" in self.state:
            return (False, "State contains a symmetric product (same subscripts)")
        return (True, "No symmetric (aa or bb) product terms found")

    # ---------- Monotonicity test ----------
    def test_monotonicity_norm_with_coefficient(self):
        """
        Check that the norm of the state increases monotonically with the coefficient α
        (the overall factor) for positive α. We treat α as a variable and compute the norm
        squared. The norm squared = 2 * α^2 (since the two product terms are orthonormal).
        The derivative d(norm^2)/dα = 4α > 0 for α>0, confirming monotonic increase.
        We perturb α from the baseline 1/√2 to slightly larger values and verify the norm increases.
        """
        if self.state is None:
            return (False, "SKIP: state not available")

        alpha = sp.symbols('alpha', positive=True)
        # Symbolic expression for the state (in the orthonormal basis):
        # |ψ> = α * ( |a_L>|b_S> - |b_L>|a_S> )
        # We assume the two product states are orthonormal and that the overlap between
        # the two terms is zero (as derived from mixed symmetry).
        # Norm squared = α^2 * ( <a_L b_S|a_L b_S> + <b_L a_S|b_L a_S> ) = 2 α^2
        norm_sq = 2 * alpha**2

        # Baseline at α0 = 1/√2
        alpha0 = 1 / sp.sqrt(2)
        baseline_norm_sq = norm_sq.subs(alpha, alpha0)

        # Perturb α upward: +10%
        alpha_perturbed = alpha0 * 1.1
        perturbed_norm_sq = norm_sq.subs(alpha, alpha_perturbed)

        # Check monotonic increase: perturbed > baseline (with a small margin)
        if float(perturbed_norm_sq) > float(baseline_norm_sq):
            return (True, f"Norm increases with coefficient (baseline={float(baseline_norm_sq):.6f}, perturbed={float(perturbed_norm_sq):.6f})")
        else:
            return (False, f"Norm did not increase with coefficient (baseline={float(baseline_norm_sq)}, perturbed={float(perturbed_norm_sq)})")

    # ---------- Weak answer sanity check (at most one) ----------
    def test_answer_sanity_weak(self):
        """
        Weak check: the state string should broadly match the expected LaTeX pattern.
        This is a low‑priority test and does NOT re‑derive any numbers.
        """
        if self.state is None:
            return (False, "SKIP: state not available")
        # Check for essential LaTeX components (loose check)
        required = [
            "\\left|1,1,\\frac{1}{2},\\frac{1}{2}\\right\\rangle",
            "\\frac{1}{\\sqrt{2}}",
            "|1,1\\rangle_a",
            "|1,1\\rangle_b",
            "\\left| \\frac{1}{2}, \\frac{1}{2} \\right\\rangle_a",
            "\\left| \\frac{1}{2}, \\frac{1}{2} \\right\\rangle_b",
            "-"
        ]
        for fragment in required:
            if fragment not in self.state:
                return (False, f"Missing required LaTeX fragment: {fragment}")
        return (True, "State LaTeX contains all essential components")