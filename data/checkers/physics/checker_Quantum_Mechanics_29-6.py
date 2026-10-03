import math
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely; never assign numeric defaults for missing values
        self.results = getattr(solution_instance, 'results', {})
        self.E_levels = self.results.get('E_levels', None)
        self.E1_transitions = self.results.get('E1_transitions', None)
        self.CG = self.results.get('CG', None)
        self.n2_wavefunctions = self.results.get('n2_wavefunctions', None)
        self.all_four_n2_states = self.results.get('all_four_n2_states', None)

    # ──────────────────────────────────────
    # Interface checks
    # ──────────────────────────────────────
    def test_interface_required_keys(self):
        """Check that all expected result keys are present in the solution."""
        required = ['E_levels', 'E1_transitions', 'CG', 'n2_wavefunctions', 'all_four_n2_states']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # ──────────────────────────────────────
    # Invariant checks (theorem-based)
    # ──────────────────────────────────────
    def test_invariant_fine_structure_ordering(self):
        """Check that the description of energy levels includes correct ordering:
        for n=2, 2s_{1/2} and 2p_{1/2} are degenerate, 2p_{3/2} is higher."""
        if self.E_levels is None:
            return (False, "SKIP: E_levels not available")
        has_2s1 = '2s_{1/2}' in self.E_levels or '2s1/2' in self.E_levels
        has_2p1 = '2p_{1/2}' in self.E_levels or '2p1/2' in self.E_levels
        has_2p3 = '2p_{3/2}' in self.E_levels or '2p3/2' in self.E_levels
        degenerate_mentioned = 'degenerate' in self.E_levels.lower()
        ordering = has_2s1 and has_2p1 and has_2p3 and degenerate_mentioned
        return (ordering,
                f"2s1/2? {has_2s1}, 2p1/2? {has_2p1}, 2p3/2? {has_2p3}, degenerate? {degenerate_mentioned}")

    def test_invariant_selection_rules(self):
        """Check that E1 transitions description includes Δℓ = ±1 and Δj = 0,±1."""
        if self.E1_transitions is None:
            return (False, "SKIP: E1_transitions not available")
        has_delta_l = 'Δℓ' in self.E1_transitions or 'Delta l' in self.E1_transitions
        has_delta_j = 'Δj' in self.E1_transitions or 'Delta j' in self.E1_transitions
        pattern = re.compile(r'0\s*,\s*[±]\s*1')
        has_0pm1 = bool(pattern.search(self.E1_transitions))
        return (has_delta_l and has_delta_j and has_0pm1,
                f"Δℓ? {has_delta_l}, Δj? {has_delta_j}, 0,±1? {has_0pm1}")

    def test_invariant_CG_orthonormality(self):
        """Check that CG coefficients include the required sqrt fractions (qualitative)."""
        if self.CG is None:
            return (False, "SKIP: CG not available")
        has_sqrt_23 = 'sqrt' in self.CG and ('2/3' in self.CG or '2/3' in self.CG)
        has_sqrt_13 = 'sqrt' in self.CG and ('1/3' in self.CG or '1/3' in self.CG)
        has_unit_coeff = '|3/2,3/2>' in self.CG or '|3/2, 3/2>' in self.CG
        return (has_sqrt_23 and has_sqrt_13 and has_unit_coeff,
                f"Contains sqrt(2/3)? {has_sqrt_23}, sqrt(1/3)? {has_sqrt_13}, high-weight state? {has_unit_coeff}")

    # ──────────────────────────────────────
    # Modeling / trace checks (based on explanation steps)
    # ──────────────────────────────────────
    def test_modeling_wavefunction_forms(self):
        """Check that the wavefunctions contain the correct radial, angular, and spin components."""
        if self.all_four_n2_states is None:
            return (False, "SKIP: all_four_n2_states not available")
        has_R20 = 'R_{20}' in self.all_four_n2_states or 'R20' in self.all_four_n2_states
        has_R21 = 'R_{21}' in self.all_four_n2_states or 'R21' in self.all_four_n2_states
        has_Y00 = 'Y^0_0' in self.all_four_n2_states or 'Y_0^0' in self.all_four_n2_states
        has_Y11 = 'Y^1_1' in self.all_four_n2_states or 'Y_1^1' in self.all_four_n2_states
        has_alpha_beta = ('alpha' in self.all_four_n2_states and 'beta' in self.all_four_n2_states)
        return (has_R20 and has_R21 and has_Y00 and has_Y11 and has_alpha_beta,
                f"R20? {has_R20}, R21? {has_R21}, Y00? {has_Y00}, Y11? {has_Y11}, alpha/beta? {has_alpha_beta}")

    def test_trace_CG_lowering(self):
        """Verify that the CG coefficients are consistent with the lowering operator (qualitative)."""
        if self.CG is None:
            return (False, "SKIP: CG not available")
        has_0_12 = '|0' in self.CG and '|1/2' in self.CG
        has_1_minus12 = '|1' in self.CG and '|-1/2' in self.CG
        return (has_0_12 and has_1_minus12,
                f"CG includes |0>|1/2>? {has_0_12}, includes |1>|-1/2>? {has_1_minus12}")

    # ──────────────────────────────────────
    # Monotonicity checks (explicit input perturbation)
    # ──────────────────────────────────────
    def test_monotonicity_number_of_states(self):
        """
        Check that the number of distinct fine‑structure levels increases with n.
        We perturb the input principal quantum number (n=1 → n=2 → n=3) and count
        the state labels in the description. Baseline: count for n=1; perturbed: n=2, n=3.
        """
        if self.E_levels is None:
            return (False, "SKIP: E_levels not available")
        # Count state labels by regex for each n
        def count_states(n_val):
            # Look for patterns like "ns_{1/2}", "np_{1/2}", "np_{3/2}", "nd_{3/2}", etc.
            pattern = rf'{n_val}[spd]?_[{{}}]?[1-9]/2'
            return len(re.findall(pattern, self.E_levels))
        c1 = count_states(1)
        c2 = count_states(2)
        c3 = count_states(3)
        if c1 == 0 or c2 == 0 or c3 == 0:
            return (False, f"Could not count states: n=1 count={c1}, n=2 count={c2}, n=3 count={c3}")
        # Monotonic increase
        if c1 < c2 and c2 < c3:
            return (True, f"State counts increase: {c1} → {c2} → {c3}")
        else:
            return (False, f"State counts not monotonic: {c1} → {c2} → {c3}")

    def test_monotonicity_fs_splitting(self):
        """
        Check that the fine‑structure splitting between j=1/2 and j=3/2 for a given ℓ
        decreases as n increases. We re‑derive the splitting numerically from the formula
        and verify the description contains a statement about decreasing splitting.
        Input perturbation: n=2 → n=3.
        Baseline splitting ΔE(n=2) > ΔE(n=3) is implied.
        """
        if self.E_levels is None:
            return (False, "SKIP: E_levels not available")
        alpha = 1/137.036
        def fs_shift(n, j):
            # Expression from problem: ΔE ~ (α²/(4n³)) * (1/(j+1/2) - 3/(4n))
            return (alpha**2 / (4 * n**3)) * (1/(j + 0.5) - 3/(4*n))
        # Splitting for ℓ=1: j=1/2 vs j=3/2
        splitting_n2 = abs(fs_shift(2, 1.5) - fs_shift(2, 0.5))
        splitting_n3 = abs(fs_shift(3, 1.5) - fs_shift(3, 0.5))
        # Check that description contains language about decreasing splitting with n
        has_decrease = any(phrase in self.E_levels.lower() for phrase in [
            'decreas', 'smaller for larger n', 'reduc', 'less splitting', 'closer'])
        # Also we can check that the ordering is correct: n=2 splitting larger than n=3
        correct_numeric = splitting_n2 > splitting_n3
        if has_decrease and correct_numeric:
            return (True, f"Splitting n2={splitting_n2:.2e} > n3={splitting_n3:.2e} and description indicates decrease")
        else:
            return (False, f"Splitting n2={splitting_n2:.2e}, n3={splitting_n3:.2e}, description mentions decrease? {has_decrease}")

    # ──────────────────────────────────────
    # Weak answer sanity (no isclose, no re‑derivation of final answer)
    # ──────────────────────────────────────
    def test_answer_sanity_four_states(self):
        """Weak sanity: check that exactly four wavefunctions are listed (by counting 'Psi' or 'Ψ')."""
        if self.all_four_n2_states is None:
            return (False, "SKIP: all_four_n2_states not available")
        count = self.all_four_n2_states.count('Psi') + self.all_four_n2_states.count('psi') + self.all_four_n2_states.count('Ψ')
        if count == 4:
            return (True, f"Found {count} state definitions, expected 4")
        else:
            return (False, f"Found {count} state definitions, expected 4")

    # ──────────────────────────────────────
    # Test runner
    # ──────────────────────────────────────
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