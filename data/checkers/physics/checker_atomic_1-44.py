import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution instance for the atomic Li/H energy level and transition problem.
    Checks interface completeness, physical invariants (selection rules, energy ordering),
    modeling quality, and optionally a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract required data from the solution instance safely.
        Expected structure: solution_instance.results['allowed_abs'] (a string).
        """
        # Default empty results dict if attribute missing
        results: Dict = getattr(solution_instance, 'results', {}) or {}
        if not isinstance(results, dict):
            results = {}

        # Extract allowed_abs string; default None if missing
        self.allowed_abs: Optional[str] = results.get('allowed_abs', None)

    # ──────────────────────────────────────────────
    # Interface Checks
    # ──────────────────────────────────────────────
    def test_interface_allowed_abs_exists(self) -> Tuple[bool, str]:
        """Check that the key 'allowed_abs' exists, is a string, and is non-empty."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs key missing or None")
        if not isinstance(self.allowed_abs, str):
            return (False, f"allowed_abs is not a string, got {type(self.allowed_abs)}")
        if not self.allowed_abs.strip():
            return (False, "allowed_abs is an empty string after strip")
        return (True, f"allowed_abs exists as non-empty string: '{self.allowed_abs}'")

    def test_interface_allowed_abs_format(self) -> Tuple[bool, str]:
        """Check that the string contains '->' to indicate transitions and mentions s and p."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        if '->' not in self.allowed_abs:
            return (False, "allowed_abs missing '->' transition notation")
        if 's' not in self.allowed_abs.lower() or 'p' not in self.allowed_abs.lower():
            return (False, "allowed_abs must mention 's' and 'p' orbitals")
        return (True, "Format looks correct (contains '->', s and p)")

    def test_interface_allowed_abs_contains_absorption(self) -> Tuple[bool, str]:
        """Check that 'absorption' or 'abs' appears in the string (case‑insensitive)."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        lower = self.allowed_abs.lower()
        if 'absorption' not in lower and 'abs' not in lower:
            return (False, "String does not mention 'absorption' or 'abs'")
        return (True, "Contains 'absorption' or 'abs'")

    # ──────────────────────────────────────────────
    # Invariant Checks (theorem‑based physical sanity)
    # ──────────────────────────────────────────────
    def _parse_transitions(self) -> List[Tuple[int, int]]:
        """Helper: extract (n_initial, n_final) from allowed_abs lines.
        Assumes format like '2s -> 2p', '2s -> 3p', etc.
        Returns empty list on failure.
        """
        if self.allowed_abs is None:
            return []
        # Split on comma or newline to get individual transitions
        parts = self.allowed_abs.replace(',', ' ').split()
        transitions = []
        for i, token in enumerate(parts):
            if token == '->':
                # look left for initial state (like '2s') and right for final (like '2p')
                if i > 0 and i + 1 < len(parts):
                    left = parts[i-1].strip()
                    right = parts[i+1].strip()
                    # Use helper to extract n and l
                    init = self._parse_state(left)
                    final = self._parse_state(right)
                    if init is not None and final is not None:
                        transitions.append((init, final))
        return transitions

    def _parse_state(self, token: str) -> Optional[Tuple[int, int]]:
        """Parse token like '2s' -> (n=2, l=0). Return None if fails."""
        if not token:
            return None
        # Expect form: digit(s) + letter (s,p,d,f)
        n_str = token.rstrip('spdf')
        l_char = token[len(n_str):] if len(token) > len(n_str) else ''
        if not n_str.isdigit() or l_char not in ('s', 'p', 'd', 'f'):
            return None
        n = int(n_str)
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        l = l_map.get(l_char)
        if l is None:
            return None
        return (n, l)

    def test_invariant_selection_rule_delta_l_plus_minus_one(self) -> Tuple[bool, str]:
        """Check that all listed transitions obey Δl = ±1.
        For absorption from 2s (l=0), only final l=1 (p) is allowed.
        """
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        transitions = self._parse_transitions()
        if not transitions:
            return (False, "Could not parse any transitions from allowed_abs")
        for (n_init, l_init), (n_final, l_final) in transitions:
            if abs(l_final - l_init) != 1:
                return (False, f"Found transition with Δl = {l_final - l_init}, expected ±1")
        return (True, f"All {len(transitions)} parsed transitions obey Δl = ±1")

    def test_invariant_no_l_to_l_transition(self) -> Tuple[bool, str]:
        """For a valid absorption from 2s, there should be no s->s or p->p transitions."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        transitions = self._parse_transitions()
        for (n_init, l_init), (n_final, l_final) in transitions:
            if l_init == l_final:
                return (False, f"Found forbidden Δl = 0 transition")
        return (True, "No Δl = 0 transitions found")

    def test_invariant_energy_increase_allowed_abs(self) -> Tuple[bool, str]:
        """Absorption must go from lower to higher energy.
        In Li, 2s is the ground state; all np with n≥2 have higher energy.
        Check that all final n are ≥ 2 and that for n=2 the transition is s->p.
        """
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        transitions = self._parse_transitions()
        if not transitions:
            return (False, "No transitions to check")
        for (n_init, l_init), (n_final, l_final) in transitions:
            if n_final < 2:
                return (False, f"Final state n={n_final} < 2, not a real absorption")
            if n_init != 2:
                return (False, f"Initial state should be n=2 (ground state of Li valence), got n={n_init}")
            if l_init != 0:
                return (False, f"Initial l must be 0 (s), got l={l_init}")
            if l_final != 1:
                return (False, f"Final l must be 1 (p), got l={l_final}")
        return (True, f"All transitions indicate energy increase from 2s to np with n≥2")

    # ──────────────────────────────────────────────
    # Modeling Quality Checks
    # ──────────────────────────────────────────────
    def test_modeling_includes_2p_3p_4p(self) -> Tuple[bool, str]:
        """Check that the allowed absorption lines include 2s→2p, 2s→3p, 2s→4p."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        lower = self.allowed_abs.lower()
        expected_n = [2, 3, 4]
        missing = []
        for n in expected_n:
            pattern = f"2s -> {n}p"
            if pattern not in lower and pattern.replace(' ','') not in lower:
                missing.append(f"2s->{n}p")
        if missing:
            return (False, f"Missing expected transitions: {', '.join(missing)}")
        return (True, f"Contains 2s→2p, 2s→3p, 2s→4p")

    def test_modeling_no_forbidden_transitions(self) -> Tuple[bool, str]:
        """Check that no forbidden transitions (e.g., 2s→ns, 2s→nd) are listed."""
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        # Parse transitions and look for any not matching s->p
        transitions = self._parse_transitions()
        for (n_init, l_init), (n_final, l_final) in transitions:
            if l_init != 0 or l_final != 1:
                return (False, f"Forbidden transition type: (n={n_init},l={l_init}) -> (n={n_final},l={l_final})")
        # Also check that no extra transitions like 2s->5p appear (only n=2,3,4 expected)
        # But this is a modeling expectation, not a physics violation. We'll flag as warning.
        allowed_n = {2, 3, 4}
        for (_, _), (n_final, _) in transitions:
            if n_final not in allowed_n:
                return (False, f"Transition to n={n_final} is not among expected {allowed_n}; may be extraneous")
        return (True, "No forbidden transitions; only s->p with n=2,3,4")

    # ──────────────────────────────────────────────
    # Monotonicity Check (discrete parameter)
    # ──────────────────────────────────────────────
    def test_monotonicity_adding_forbidden_transition_breaks_rule(self) -> Tuple[bool, str]:
        """Perturb the selection rule by considering a forbidden transition (e.g., 2s→3s).
        The checker would reject it. This is a negative monotonicity test on the logic.
        """
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        # We can't modify the solution; instead we check that if we artificially add a forbidden
        # transition to the string, the selection_rule test would fail.
        # Since we cannot modify at runtime, we do a simulated check:
        test_string = self.allowed_abs + ", 2s -> 3s"
        # We'll temporarily replace and test the invariant.
        original = self.allowed_abs
        try:
            self.allowed_abs = test_string
            passed, msg = self.test_invariant_selection_rule_delta_l_plus_minus_one()
            if passed:
                self.allowed_abs = original
                return (False, "Adding forbidden transition did not fail the invariant (unexpected)")
            self.allowed_abs = original
            return (True, "Adding a forbidden transition correctly causes invariant failure")
        except Exception:
            self.allowed_abs = original
            return (False, "Exception during perturbation test")

    # ──────────────────────────────────────────────
    # Weak Answer Sanity Check (optional, only one)
    # ──────────────────────────────────────────────
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Weak sanity: check that the answer is a non‑empty string with plausible content.
        No numeric comparison, no isclose, no derivation.
        """
        if self.allowed_abs is None:
            return (False, "SKIP: allowed_abs not available")
        if not isinstance(self.allowed_abs, str) or not self.allowed_abs.strip():
            return (False, "allowed_abs is not a non‑empty string")
        # Check that it contains expected words
        lower = self.allowed_abs.lower()
        if 'absorption' not in lower and 'abs' not in lower:
            return (False, "Does not contain 'absorption' or 'abs'")
        if '2s' not in lower:
            return (False, "Does not contain '2s'")
        return (True, "Weak sanity passed: contains 'absorption' and '2s'")

    # ──────────────────────────────────────────────
    # Run all tests
    # ──────────────────────────────────────────────
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collect results."""
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