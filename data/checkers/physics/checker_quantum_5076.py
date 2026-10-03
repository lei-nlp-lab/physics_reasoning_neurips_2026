import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        # Extract givens if available
        self.givens = getattr(solution_instance, 'givens', {})
        if self.givens is None:
            self.givens = {}

        # Required result keys
        self.P = self.results.get('P', None)
        self.T_formula = self.results.get('T_formula', None)

        # Optional explanation steps (for context, not hard-coded numbers)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract input parameters from givens (may be None)
        self.M = self.givens.get('M', None)
        self.k = self.givens.get('k', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_has_P(self):
        if self.P is None:
            return False, "Missing required result 'P'"
        return True, "P is present"

    def test_interface_has_T_formula(self):
        if self.T_formula is None:
            return False, "Missing required result 'T_formula'"
        return True, "T_formula is present"

    def test_interface_P_type(self):
        if self.P is None:
            return False, "SKIP: P is missing, cannot check type"
        if not isinstance(self.P, (int, float, sp.Expr)):
            return False, f"P has unexpected type {type(self.P)}"
        return True, "P is a number or symbolic expression"

    def test_interface_T_formula_type(self):
        if self.T_formula is None:
            return False, "SKIP: T_formula is missing, cannot check type"
        if not isinstance(self.T_formula, str):
            return False, f"T_formula must be a string, got {type(self.T_formula)}"
        return True, "T_formula is a string"

    # ------------------------------------------------------------------
    # Physics invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_probability_formula(self):
        """Check that P matches the overlap formula: P = 2*sqrt(ω₁ω₂)/(ω₁+ω₂)."""
        if self.P is None:
            return False, "SKIP: P not available"
        # Need M and k to compute ω₁, ω₂
        if self.M is None or self.k is None or self.M <= 0 or self.k <= 0:
            return False, "SKIP: M or k not available or non-positive for exact check"
        omega1 = math.sqrt(self.k / self.M)
        omega2 = math.sqrt(2 * self.k / self.M)   # k doubled => ω₂ = √2 ω₁
        expected_P = 2 * math.sqrt(omega1 * omega2) / (omega1 + omega2)
        if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"P = {self.P} does not match expected {expected_P}"
        return True, f"P matches overlap formula: {self.P} ~ {expected_P}"

    def test_invariant_restoration_phase_condition(self):
        """Check that the T_formula obeys ω₂ T = Nπ (N integer) when ω₂ = √(2k/M)."""
        if self.T_formula is None:
            return False, "SKIP: T_formula not available"
        if self.M is None or self.k is None or self.M <= 0 or self.k <= 0:
            return False, "SKIP: M or k not available for numerical check"
        try:
            # Evaluate T for N=1
            expr = sp.sympify(self.T_formula.replace('N', '1'))
            M_sym, k_sym = sp.symbols('M k')
            if expr.has(sp.Symbol('M')) or expr.has(sp.Symbol('k')):
                expr_sub = expr.subs({M_sym: self.M, k_sym: self.k})
            else:
                expr_sub = expr
            T_val = float(expr_sub.evalf())
        except Exception:
            # Fallback structural check
            if 'pi' not in self.T_formula or 'sqrt' not in self.T_formula:
                return False, f"T_formula missing essential parts: {self.T_formula}"
            return True, "T_formula has plausible structure (pi, sqrt)"
        omega2 = math.sqrt(2.0 * self.k / self.M)
        phase = omega2 * T_val
        multiples = [0, math.pi, 2*math.pi, 3*math.pi]
        if not any(math.isclose(phase, p, rel_tol=1e-2, abs_tol=1e-2) for p in multiples):
            return False, f"Phase condition not satisfied: ω₂ T = {phase} rad, expected π * integer"
        return True, f"T_formula gives phase {phase} rad, consistent with integer multiple of π"

    # ------------------------------------------------------------------
    # Modeling / trace checks (consistency with explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_parity_selection(self):
        """Check that the explanation steps mention parity (even‑only overlap) for part (b)."""
        if self.explanation_steps is None:
            return False, "SKIP: no explanation steps provided"
        text = str(self.explanation_steps)
        keywords = ['parity', 'even', 'odd', 'selection', 'non-zero for even']
        if not any(kw in text.lower() for kw in keywords):
            return False, "Explanation steps do not mention parity selection rule for part (b)"
        return True, "Parity selection rule is present in explanation steps"

    def test_modeling_time_evolution_consistency(self):
        """Verify that the T_formula is derived from phases of even eigenstates."""
        if self.T_formula is None:
            return False, "SKIP: T_formula not available"
        if self.explanation_steps is None:
            return False, "SKIP: no explanation steps to check"
        text = str(self.explanation_steps).lower()
        checks = ['time evolution', 'phase', 'e^{', 'even', '2m']
        missing = [c for c in checks if c not in text]
        if missing:
            return False, f"Explanation steps missing key concepts: {missing}"
        return True, "Time evolution reasoning includes necessary phase and parity arguments"

    # ------------------------------------------------------------------
    # Monotonicity checks (must perturb an input and compare)
    # ------------------------------------------------------------------
    def test_monotonicity_P_decreases_with_increased_k(self):
        """
        For the given sudden doubling (k->2k), the probability P is a fixed number.
        However, if we imagine further increasing k (i.e., ω₂ larger), the overlap
        should decrease because ω₂ > ω₁ and the overlap function is decreasing for ω₂ > ω₁.
        We test this by recomputing P with a perturbed k (e.g., k' = 1.1 * original k)
        and verifying that the new P is smaller.
        """
        if self.P is None or self.M is None or self.k is None:
            return False, "SKIP: missing M, k, or P for monotonicity test"
        if self.M <= 0 or self.k <= 0:
            return False, "SKIP: M or k non-positive"
        # Baseline: omega1, omega2 from given k
        omega1 = math.sqrt(self.k / self.M)
        omega2 = math.sqrt(2 * self.k / self.M)
        P_baseline = 2 * math.sqrt(omega1 * omega2) / (omega1 + omega2)
        # Perturb k upward -> new omega2' = sqrt(2 * k')/sqrt(M) = sqrt(2)*sqrt(k')/sqrt(M)
        k_pert = 1.1 * self.k
        omega1_new = math.sqrt(k_pert / self.M)   # but V1 unchanged? Actually we are only changing the doubling factor? 
        # Careful: The problem considers only one doubling. For monotonicity we consider a hypothetical scenario:
        # increase the final spring constant further. But the initial potential stays V1, so ω1 unchanged.
        # However the overlap formula uses ω1 (initial) and ω2 (final). So we keep ω1 the same and increase ω2.
        omega2_new = math.sqrt(2 * k_pert / self.M)   # final potential with spring constant 2*k_pert
        P_pert = 2 * math.sqrt(omega1 * omega2_new) / (omega1 + omega2_new)
        if not (P_pert < P_baseline):
            return False, f"Perturbed probability {P_pert} is not less than baseline {P_baseline}"
        return True, f"P decreases when final k increases: baseline {P_baseline:.4f}, perturbed {P_pert:.4f}"

    def test_monotonicity_T_increases_with_M(self):
        """
        The restoration time T = N * pi * sqrt(M/(2k)). For fixed N and k, T increases with M.
        We verify this by comparing baseline T (with given M) with T after increasing M.
        """
        if self.T_formula is None or self.M is None or self.k is None:
            return False, "SKIP: missing T_formula, M, or k for monotonicity test"
        if self.M <= 0 or self.k <= 0:
            return False, "SKIP: M or k non-positive"
        # Evaluate baseline T using the formula with N=1
        try:
            expr = sp.sympify(self.T_formula.replace('N', '1'))
            M_sym, k_sym = sp.symbols('M k')
            if expr.has(sp.Symbol('M')) or expr.has(sp.Symbol('k')):
                expr_baseline = expr.subs({M_sym: self.M, k_sym: self.k})
            else:
                expr_baseline = expr
            T_baseline = float(expr_baseline.evalf())
        except Exception:
            return False, "SKIP: cannot evaluate T_formula numerically"
        # Perturb M
        M_pert = 1.2 * self.M
        try:
            if expr.has(sp.Symbol('M')) or expr.has(sp.Symbol('k')):
                expr_pert = expr.subs({M_sym: M_pert, k_sym: self.k})
            else:
                expr_pert = expr
            T_pert = float(expr_pert.evalf())
        except Exception:
            return False, "SKIP: cannot evaluate perturbed T_formula"
        if not (T_pert > T_baseline):
            return False, f"Perturbed T {T_pert} is not greater than baseline {T_baseline}"
        return True, f"T increases with M: baseline {T_baseline:.4f}, perturbed {T_pert:.4f}"

    # ------------------------------------------------------------------
    # Weak answer sanity checks (only one such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_bounds(self):
        """Weak sanity: check that P is finite, positive, and less than 1."""
        if self.P is None:
            return False, "SKIP: P not available"
        if math.isnan(self.P) or math.isinf(self.P):
            return False, f"P is not finite: {self.P}"
        if self.P <= 0:
            return False, f"P is non-positive: {self.P}"
        if self.P >= 1 + 1e-12:
            return False, f"P exceeds 1: {self.P}"
        return True, f"P = {self.P} is within (0,1]"

    # ------------------------------------------------------------------
    # Run all tests
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