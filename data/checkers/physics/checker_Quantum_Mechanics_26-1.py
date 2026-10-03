import math
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Auditor for atomic structure problem – central field approximation,
    quantum numbers, filling order, LS coupling, and carbon ground state.
    """

    def __init__(self, solution_instance: Any):
        results = getattr(solution_instance, 'results', {}) or {}
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None)

        # Safely extract result keys (default None)
        self.quantum_numbers = results.get('quantum_numbers', None)
        self.filling_order = results.get('filling_order', None)
        self.config = results.get('config', None)
        self.allowed_terms = results.get('allowed_terms', None)
        self.ground_state = results.get('ground_state', None)

        # Z – must be None if not given (never a numeric default)
        self.Z = givens.get('Z', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """All five mandatory result keys must be present and not None."""
        missing = []
        for key in ['quantum_numbers', 'filling_order', 'config',
                     'allowed_terms', 'ground_state']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All five required keys are present.")

    def test_interface_quantum_numbers_type(self) -> Tuple[bool, str]:
        """quantum_numbers must be a non‑empty string."""
        if not isinstance(self.quantum_numbers, str) or len(self.quantum_numbers) == 0:
            return (False, f"quantum_numbers is not a non‑empty string: {self.quantum_numbers}")
        return (True, "quantum_numbers is a valid string.")

    def test_interface_filling_order_type(self) -> Tuple[bool, str]:
        if not isinstance(self.filling_order, str) or len(self.filling_order) == 0:
            return (False, f"filling_order is not a non‑empty string: {self.filling_order}")
        return (True, "filling_order is a valid string.")

    def test_interface_config_type(self) -> Tuple[bool, str]:
        if not isinstance(self.config, str) or len(self.config) == 0:
            return (False, f"config is not a non‑empty string: {self.config}")
        return (True, "config is a valid string.")

    def test_interface_allowed_terms_type(self) -> Tuple[bool, str]:
        if not isinstance(self.allowed_terms, list) or len(self.allowed_terms) == 0:
            return (False, f"allowed_terms is not a non‑empty list: {self.allowed_terms}")
        for term in self.allowed_terms:
            if not isinstance(term, str) or len(term) == 0:
                return (False, f"allowed_terms contains invalid element: {term}")
        return (True, "allowed_terms is a list of non‑empty strings.")

    def test_interface_ground_state_type(self) -> Tuple[bool, str]:
        if not isinstance(self.ground_state, str) or len(self.ground_state) == 0:
            return (False, f"ground_state is not a non‑empty string: {self.ground_state}")
        return (True, "ground_state is a valid string.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_pauli_allowed_terms(self) -> Tuple[bool, str]:
        """
        For two equivalent p electrons (carbon 2p²), Pauli principle
        allows only ¹S₀, ¹D₂, ³P₀,₁,₂. Check that the provided list
        contains exactly these three term symbols.
        """
        if self.allowed_terms is None:
            return (False, "SKIP: allowed_terms not available")
        # Normalise
        combined = ''.join(self.allowed_terms).replace(' ', '').lower()
        required = ['1s0', '1d2', '3p']  # 3p covers all J values
        for r in required:
            if r not in combined:
                return (False, f"Missing required term: {r}")
        forbidden = ['3d', '1p', '3s', '1f']
        for f in forbidden:
            if f in combined:
                return (False, f"Forbidden term present: {f}")
        return (True, "All allowed terms satisfy the Pauli principle for p².")

    def test_invariant_ground_state_in_allowed_list(self) -> Tuple[bool, str]:
        """Ground state must appear in the allowed terms list."""
        if self.ground_state is None or self.allowed_terms is None:
            return (False, "SKIP: ground_state or allowed_terms missing")
        gs = self.ground_state.replace(' ', '').lower()
        allowed_norm = [t.replace(' ', '').lower() for t in self.allowed_terms]
        if gs not in allowed_norm:
            return (False, f"Ground state '{self.ground_state}' not in allowed_terms")
        return (True, "Ground state is listed among allowed terms.")

    def test_invariant_central_field_quantum_numbers(self) -> Tuple[bool, str]:
        """The answer must mention quantum numbers n, ℓ, m_ℓ, m_s."""
        if self.quantum_numbers is None:
            return (False, "SKIP: quantum_numbers not available")
        qn = self.quantum_numbers.lower()
        # Accept various representations
        required_terms = ['n', 'l', 'm_l', 'm_s']
        # also accept 'ml', 'mℓ', 'ms', etc.
        found = [t for t in required_terms if t in qn or t.replace('_','') in qn]
        if len(found) < 4:
            return (False, f"Missing quantum number descriptor(s). Got: {self.quantum_numbers}")
        return (True, "All four quantum numbers are mentioned.")

    def test_invariant_hund_ground_state_carbon(self) -> Tuple[bool, str]:
        """
        For carbon (Z=6, 2p², less than half‑filled), Hund’s rules give ³P₀.
        Re‑derive the ground state if Z is given and equals 6.
        """
        if self.Z is None or self.Z != 6:
            # Inconclusive; skip if we don’t know the atom
            return (False, "SKIP: Z unavailable or not 6")
        gs = self.ground_state
        if gs is None:
            return (False, "ground_state is None")
        # Expected: ^3P_0 (or variations)
        expected_patterns = ['3p0', '^3p_0', '³p₀']
        gs_norm = gs.replace(' ', '').lower()
        if not any(p in gs_norm for p in expected_patterns):
            return (False, f"Ground state '{gs}' does not match expected ³P₀ for carbon")
        return (True, "Hund’s rules predict ³P₀ for carbon – matches provided ground state.")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_filling_order_energy_reason(self) -> Tuple[bool, str]:
        """Filling order explanation should invoke energy order (lower ℓ first)."""
        if self.filling_order is None:
            return (False, "SKIP: filling_order not available")
        words = self.filling_order.lower()
        # at least two of these concepts should appear
        concepts = ['low', 'energ', 'shield', 'penetrat', 'first', 'smaller']
        hits = sum(1 for c in concepts if c in words)
        if hits < 2:
            return (False, f"Filling order lacks physical reasoning: '{self.filling_order}'")
        return (True, "Filling order includes energy‑based explanation.")

    def test_modeling_config_carbon_structure(self) -> Tuple[bool, str]:
        """
        Verify that the configuration contains the correct subshells for carbon:
        1s, 2s, 2p. (Does not rely on Z; it checks common pattern.)
        """
        if self.config is None:
            return (False, "SKIP: config not available")
        conf_norm = self.config.replace(' ', '').lower()
        for sub in ['1s', '2s', '2p']:
            if sub not in conf_norm:
                return (False, f"Missing subshell '{sub}' in config: '{self.config}'")
        # Total electron count should be 6 (carbon) or at least appear as 2+2+2
        # But we cannot rigorously parse; just check presence of three '2' digits
        if conf_norm.count('2') < 3:
            return (False, f"Electron numbers seem inconsistent: '{self.config}'")
        return (True, "Configuration shows 1s² 2s² 2p² pattern.")

    def test_modeling_ls_coupling_notation(self) -> Tuple[bool, str]:
        """Allowed terms must be in spectroscopic notation (e.g., ^1S_0)."""
        if self.allowed_terms is None:
            return (False, "SKIP: allowed_terms not available")
        import re
        pattern = r'\^?\d+[SPDFGHIJKLMN]_\d+'
        for term in self.allowed_terms:
            clean = term.replace(' ', '')
            if not re.match(pattern, clean):
                return (False, f"Term '{term}' is not in spectroscopic notation")
        return (True, "All allowed terms are correctly formatted.")

    def test_modeling_central_field_explanation(self) -> Tuple[bool, str]:
        """If explanation_steps exist, they must mention the central potential."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = str(self.explanation_steps).lower()
        keywords = ['central', 'potential', 'v(r)', 'radial', 'effective']
        if sum(1 for k in keywords if k in text) < 2:
            return (False, "Explanation steps lack central field detail.")
        return (True, "Explanation steps reference the central field approximation.")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_ground_state_J_increases_with_filling(self) -> Tuple[bool, str]:
        """
        Perturb the number of 2p electrons from 2 (carbon) to 3 (nitrogen).
        Baseline: carbon ground state J (should be 0 for ³P₀).
        Perturbed: nitrogen ground state J (should be 3/2 for ⁴S_{3/2}).
        The monotonicity condition: baseline J < perturbed J.
        This test only runs if Z=6 and the carbon ground state is correctly given.
        """
        if self.Z is None or self.Z != 6:
            return (False, "SKIP: Z not 6, cannot perform carbon‑based monotonicity")
        if self.ground_state is None:
            return (False, "SKIP: ground_state not available")

        # Parse baseline J from ground state string
        gs = self.ground_state.replace(' ', '')
        # Expected format: ^3P_0, ^3P_0, ³p₀, etc.
        try:
            # split on underscore, take last part
            j_part = gs.split('_')[-1] if '_' in gs else gs
            # handle superscripts: '3P0' -> extract trailing number
            import re
            j_match = re.search(r'(\d+(?:/\d+)?)\s*$', j_part)
            if not j_match:
                return (False, f"Cannot parse J from ground state: '{self.ground_state}'")
            j_base = float(eval(j_match.group(1)))  # e.g. '3/2' -> 1.5
        except Exception:
            return (False, f"Failed to parse J from '{self.ground_state}'")

        # Perturbed scenario: Z=7 (nitrogen) ground state from Hund’s rules
        # p³ half‑filled → ⁴S_{3/2} → J=1.5
        j_perturbed = 1.5

        if j_base < j_perturbed:
            return (True, f"Monotonic: baseline J={j_base} < perturbed J={j_perturbed}")
        else:
            return (False, f"Expected increase, got baseline J={j_base}, perturbed J={j_perturbed}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_ground_state_not_nan(self) -> Tuple[bool, str]:
        """Weak check: ground state must be a finite string, not empty/NaN."""
        if self.ground_state is None:
            return (False, "SKIP: ground_state not available")
        if not isinstance(self.ground_state, str) or len(self.ground_state.strip()) == 0:
            return (False, "Ground state is empty.")
        if self.ground_state.strip().lower() == 'nan':
            return (False, "Ground state is 'NaN'.")
        return (True, "Ground state is a finite, non‑empty string.")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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