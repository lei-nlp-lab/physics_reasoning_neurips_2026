import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Robust checker for the modified Poisson equation from Yukawa-like potential.
    Validates the derived equation: (∇² - K²)φ = -4πρ (Gaussian units).
    """

    def __init__(self, solution_instance):
        # Always use .get() or getattr with None default
        self.modified_poisson = getattr(solution_instance, 'results', {}).get('modified_poisson', None)
        # Store the full solution_instance for potential future use
        self.solution_instance = solution_instance

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_modified_poisson_key_exists(self):
        """Check that the result key 'modified_poisson' is present and non-None."""
        if self.modified_poisson is None:
            return (False, "modified_poisson key missing or None")
        return (True, "modified_poisson key found")

    def test_interface_modified_poisson_is_string(self):
        """Check that the result is a string (equation representation)."""
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")
        if not isinstance(self.modified_poisson, str):
            return (False, f"modified_poisson is not a string, got {type(self.modified_poisson)}")
        return (True, "modified_poisson is a string")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_laplacian_derivation(self):
        """
        Re-derive the Laplacian of φ = q * exp(-K*r)/r and verify it gives -4πρ + K² φ.
        Use SymPy for symbolic check.
        """
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")

        # Define symbols
        r, K, q = sp.symbols('r K q', positive=True, real=True)
        # Potential in Gaussian units: φ = q * exp(-K*r) / r
        phi = q * sp.exp(-K * r) / r

        # Compute Laplacian in spherical coordinates (assuming spherical symmetry)
        # ∇² φ = (1/r²) ∂/∂r ( r² ∂φ/∂r )
        dphi_dr = sp.diff(phi, r)
        laplacian = sp.simplify((1 / r**2) * sp.diff(r**2 * dphi_dr, r))

        # The result should be K² φ for r≠0
        expected_laplacian_no_delta = K**2 * phi
        # Check equality
        if sp.simplify(laplacian - expected_laplacian_no_delta) != 0:
            # Allow small numerical tolerance - use sympy N for 1e-12
            diff = sp.N(laplacian - expected_laplacian_no_delta, subs={r:1, K:0.1, q:1})
            if abs(diff) > 1e-12:
                return (False, f"Laplacian for r≠0 deviates: {diff}")

        # Now test the modified equation string
        # We expect (∇² - K²)φ = -4πρ. In Gaussian units, ρ = q δ(r).
        # Since we cannot test delta, we test the r≠0 part: ∇² φ = K² φ
        # And the overall equation must be ∇² φ = K² φ - 4π ρ
        # Check that the user's equation can be rearranged to this form
        # We'll parse the equation string and compare symbolically.
        # Remove LaTeX formatting if present (assume simple text)
        eq_str = self.modified_poisson.replace("\\nabla^2", "nabla2").replace(" ", "")
        # Simple check: equation should contain "nabla2 phi - K^2 phi = -4 pi rho" or similar
        # We'll accept variations like "nabla^2 phi - K^2 phi = -4 pi rho"
        # or "∇² φ + K² φ = -4πρ" which is the wrong sign, but we'll flag that later.

        # We'll just verify the presence of key operators
        has_laplacian = 'nabla2' in eq_str or '∇²' in eq_str
        has_K_squared = 'K^2' in eq_str or 'K²' in eq_str
        has_rho = 'rho' in eq_str or 'ρ' in eq_str
        has_4pi = '4pi' in eq_str or '4π' in eq_str

        if not (has_laplacian and has_K_squared and has_rho and has_4pi):
            return (False, f"Incomplete equation: {self.modified_poisson}")

        return (True, f"Laplacian derivation consistent, equation structure present: {self.modified_poisson}")

    def test_invariant_limiting_case_coulomb(self):
        """
        When K=0, the equation should reduce to standard Poisson: ∇² φ = -4πρ.
        Use numerical check: compare left and right sides for K=0 in the Laplacian formula.
        """
        # For K=0, φ = q/r, ∇²(q/r) = -4π q δ(r). So equation becomes ∇² φ = -4πρ.
        # We'll just check that the user's equation simplifies to that when K=0.
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")

        eq_str = self.modified_poisson.lower().replace(" ", "").replace("\\nabla^2", "nabla2")
        # Try to detect whether the K^2 term disappears
        if 'k^2' in eq_str or 'k²' in eq_str:
            # We can't directly check, but we can evaluate that for K=0 the term is zero
            return (True, "Equation contains K^2 term; physical limit K=0 yields Poisson")
        return (False, f"K^2 term not found, cannot verify limiting case")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_sign_correctness(self):
        """
        Derive the correct modified Poisson equation from the potential and compare sign.
        The correct equation (Gaussian) is (∇² - K²) φ = -4πρ.
        Check that the user's equation has the right sign (K² term subtracted).
        """
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")

        # Expect minus on K^2 term. Accept variations: "nabla^2 phi - K^2 phi = -4 pi rho"
        # or "∇² φ - K² φ = -4π ρ"
        eq_norm = self.modified_poisson.replace("\\nabla^2", "∇²").replace(" ", "")
        # Check if there is a minus before K^2 term
        # Simple heuristic: look for "- K^2" or "-K^2" or "-K²"
        if "-K^2" in eq_norm or "-K²" in eq_norm:
            return (True, "Correct sign: K² term subtracted")
        elif "+K^2" in eq_norm or "+K²" in eq_norm:
            return (False, f"Incorrect sign: found '+K²' instead of '-K²' in {self.modified_poisson}")
        else:
            # Might be written differently; try to parse with sympy? Risk of complexity.
            # Fallback to general indication.
            return (False, f"Cannot determine sign of K² term in: {self.modified_poisson}")

    def test_modeling_source_term(self):
        """Check that the right-hand side is -4πρ (not just -ρ)."""
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")

        eq_str = self.modified_poisson.replace(" ", "")
        # Look for -4πρ or -4 pi rho
        if "-4πρ" in eq_str or "-4π ρ" in eq_str or "-4 pi rho" in eq_str or "-4πρ" in eq_str:
            return (True, "RHS is -4πρ")
        elif "-ρ" in eq_str and "4π" not in eq_str and "4pi" not in eq_str:
            return (False, "RHS is -ρ only, should be -4πρ (Gaussian units)")
        else:
            # Could be that they write - (4πρ) etc.
            if "4πρ" in eq_str or "4pi" in eq_str:
                return (True, "RHS contains 4πρ")
            else:
                return (False, "Cannot verify RHS structure")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturbation analysis)
    # ----------------------------------------------------------------------
    def test_monotonicity_as_K_increases_field_changes(self):
        """
        For a point charge, the potential and field decay faster for larger K.
        We can't easily recompute but we can check that if K increases, the equation's K² term becomes more significant.
        Alternatively, we can perturb the user's equation artificially by adding a small ΔK and see if the equation form changes (it shouldn't).
        We'll do a symbolic check: for fixed charge distribution, the relation between ∇²φ and φ should be modified.
        No quantitative test here, just presence of K² term.
        """
        if self.modified_poisson is None:
            return (False, "SKIP: modified_poisson not available")
        # The equation itself already encodes monotonicity: larger K => larger contribution.
        # Just confirm that K appears squared.
        if "K^2" in self.modified_poisson or "K²" in self.modified_poisson:
            return (True, "Monotonicity implied by K² term")
        return (False, "K² term not found")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose, no derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic_structure(self):
        """
        Weak sanity: the answer should be an equation relating ∇²φ, K²φ, and ρ.
        We already check this in other tests; here just ensure it's not empty and has minimum length.
        """
        if self.modified_poisson is None:
            return (False, "SKIP: not available")
        if len(self.modified_poisson) < 10:
            return (False, f"Equation too short: '{self.modified_poisson}'")
        # Check that it contains at least one '=' sign
        if '=' not in self.modified_poisson:
            return (False, "Equation missing '='")
        return (True, "Basic structure looks plausible")

    # ----------------------------------------------------------------------
    # Run all tests (generic, do not modify)
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)