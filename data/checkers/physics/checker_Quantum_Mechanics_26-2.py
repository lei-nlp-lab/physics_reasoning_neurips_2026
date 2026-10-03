import math
import re

class PhysicsChecker:
    """
    Audits atomic term symbols for ground states of first ten elements.
    Checks interface existence, Hund's rule invariants, modeling/trace quality,
    and monotonicity by perturbing input (atomic number Z).
    """

    def __init__(self, solution_instance):
        # Extract results safely – use None for missing keys
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        elif isinstance(solution_instance, dict):
            results = solution_instance
        else:
            results = {}

        self.term_H  = results.get('term_H', None)
        self.term_He = results.get('term_He', None)
        self.term_Li = results.get('term_Li', None)
        self.term_Be = results.get('term_Be', None)
        self.term_B  = results.get('term_B', None)
        self.term_C  = results.get('term_C', None)
        self.term_N  = results.get('term_N', None)
        self.term_O  = results.get('term_O', None)
        self.term_F  = results.get('term_F', None)
        self.term_Ne = results.get('term_Ne', None)

        # Map atomic number Z to term string
        self.all_terms = {
            1:  self.term_H,
            2:  self.term_He,
            3:  self.term_Li,
            4:  self.term_Be,
            5:  self.term_B,
            6:  self.term_C,
            7:  self.term_N,
            8:  self.term_O,
            9:  self.term_F,
            10: self.term_Ne
        }

        # L‑letter mapping for internal derivation
        self.L_letter = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _parse_term(self, term_str):
        """Return (S, L, J) from a term string like ^2S_{1/2}, or None."""
        if not term_str or not isinstance(term_str, str):
            return None
        m = re.match(r'\^(\d+)([SPDF])_(\d+)(?:/(\d+))?$', term_str.strip())
        if not m:
            return None
        mult = int(m.group(1))
        L = {'S':0, 'P':1, 'D':2, 'F':3}.get(m.group(2), None)
        if L is None:
            return None
        num = int(m.group(3))
        den = int(m.group(4)) if m.group(4) else 1
        if den == 0:
            return None
        J = num / den
        S = (mult - 1) / 2.0
        return (S, L, J)

    def _get_term_parsed(self, Z):
        """Return parsed (S,L,J) for element Z, or None if missing/unparseable."""
        term = self.all_terms.get(Z, None)
        if term is None:
            return None
        return self._parse_term(term)

    # --------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------
    def test_interface_keys_exist(self):
        """All ten expected term keys must be present and non‑empty strings."""
        for key in ['term_H','term_He','term_Li','term_Be','term_B',
                     'term_C','term_N','term_O','term_F','term_Ne']:
            val = getattr(self, key, None)
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"Missing or empty key: {key}")
        return (True, "All term keys exist and are non‑empty.")

    def test_interface_format(self):
        """Every term string must match standard notation ^2S_{1/2} etc."""
        for Z, term in self.all_terms.items():
            if term is None:
                return (False, f"Z={Z}: term missing.")
            if self._parse_term(term) is None:
                return (False, f"Z={Z}: term '{term}' does not match required format.")
        return (True, "All terms have valid format.")

    # --------------------------------------------------------
    # Invariant tests (theorem-based)
    # --------------------------------------------------------
    def test_invariant_closed_shells(self):
        """He, Be, Ne (closed subshells) must be ^1S_0."""
        for Z in [2,4,10]:
            term = self.all_terms.get(Z)
            if term is None:
                return (False, f"Z={Z} missing.")
            if term != '^1S_0':
                return (False, f"Z={Z}: expected ^1S_0, got {term}")
        return (True, "Closed shells give ^1S_0.")

    def test_invariant_single_s_electron(self):
        """H and Li (single s electron) must be ^2S_{1/2}."""
        for Z in [1,3]:
            term = self.all_terms.get(Z)
            if term is None:
                return (False, f"Z={Z} missing.")
            if term != '^2S_{1/2}':
                return (False, f"Z={Z}: expected ^2S_{{1/2}}, got {term}")
        return (True, "H and Li give ^2S_{1/2}.")

    def test_invariant_half_filled_p(self):
        """Nitrogen (p³) must be ^4S_{3/2}."""
        term = self.term_N
        if term is None:
            return (False, "N term missing.")
        if term != '^4S_{3/2}':
            return (False, f"N: expected ^4S_{{3/2}}, got {term}")
        return (True, "N gives ^4S_{3/2}.")

    def test_invariant_p1_less_half(self):
        """Boron (p¹) must be ^2P_{1/2}."""
        term = self.term_B
        if term is None:
            return (False, "B term missing.")
        if term != '^2P_{1/2}':
            return (False, f"B: expected ^2P_{{1/2}}, got {term}")
        return (True, "B gives ^2P_{1/2}.")

    def test_invariant_p2_less_half(self):
        """Carbon (p²) must be ^3P_0."""
        term = self.term_C
        if term is None:
            return (False, "C term missing.")
        if term != '^3P_0':
            return (False, f"C: expected ^3P_0, got {term}")
        return (True, "C gives ^3P_0.")

    def test_invariant_p4_more_half(self):
        """Oxygen (p⁴) must be ^3P_2."""
        term = self.term_O
        if term is None:
            return (False, "O term missing.")
        if term != '^3P_2':
            return (False, f"O: expected ^3P_2, got {term}")
        return (True, "O gives ^3P_2.")

    def test_invariant_p5_more_half(self):
        """Fluorine (p⁵) must be ^2P_{3/2}."""
        term = self.term_F
        if term is None:
            return (False, "F term missing.")
        if term != '^2P_{3/2}':
            return (False, f"F: expected ^2P_{{3/2}}, got {term}")
        return (True, "F gives ^2P_{3/2}.")

    # --------------------------------------------------------
    # Modeling / trace quality tests
    # --------------------------------------------------------
    def test_modeling_hunds_rules_consistent(self):
        """Re‑derive each term from Hund’s rules and compare quantum numbers (internal derivation)."""
        def expected_term(Z):
            # same as _compute_expected_term but embedded locally
            if Z == 1:   S, L, J = 0.5, 0, 0.5
            elif Z == 2: S, L, J = 0, 0, 0
            elif Z == 3: S, L, J = 0.5, 0, 0.5
            elif Z == 4: S, L, J = 0, 0, 0
            elif 5 <= Z <= 10:
                n = Z - 4
                if n == 6:  S, L, J = 0, 0, 0
                else:
                    S = n/2.0 if n <= 3 else (6-n)/2.0
                    if n in (1,2,4,5): L = 1
                    elif n == 3: L = 0
                    else: L = 0
                    if n < 3:       J = abs(L - S)
                    elif n == 3:    J = S
                    else:           J = L + S
            else:
                return None
            mult = int(2*S + 1)
            L_letter = self.L_letter[int(L)]
            if J == int(J): J_str = str(int(J))
            else:           J_str = f"{int(J*2)}/2"
            return (f"^{mult}{L_letter}_{{{J_str}}}", S, L, J)

        for Z in range(1, 11):
            term = self.all_terms.get(Z)
            if term is None:
                return (False, f"Z={Z}: term missing.")
            result = expected_term(Z)
            if result is None:
                return (False, f"Z={Z}: could not derive expected term.")
            expected_str, S_exp, L_exp, J_exp = result
            parsed = self._parse_term(term)
            if parsed is None:
                return (False, f"Z={Z}: parsing failure.")
            S, L, J = parsed
            if not math.isclose(S, S_exp, abs_tol=1e-9):
                return (False, f"Z={Z}: S mismatch {S} vs {S_exp}.")
            if L != L_exp:
                return (False, f"Z={Z}: L mismatch {L} vs {L_exp}.")
            if not math.isclose(J, J_exp, abs_tol=1e-9):
                return (False, f"Z={Z}: J mismatch {J} vs {J_exp}.")
        return (True, "All terms match Hund’s rules re‑derivation.")

    def test_modeling_hole_formalism_oxygen(self):
        """Oxygen (p⁴) should be equivalent to p² with reverse J rule (hole formalism)."""
        term_O = self.term_O
        if term_O is None:
            return (False, "O term missing.")
        parsed_O = self._parse_term(term_O)
        if parsed_O is None:
            return (False, "O term not parseable.")
        S_O, L_O, J_O = parsed_O
        exp_S = 1.0   # two holes, same as p²
        exp_L = 1
        exp_J = L_O + S_O   # more than half -> J = L+S
        if not math.isclose(S_O, exp_S, abs_tol=1e-9):
            return (False, f"O: S should be 1, got {S_O}.")
        if L_O != exp_L:
            return (False, f"O: L should be 1, got {L_O}.")
        if not math.isclose(J_O, 2.0, abs_tol=1e-9):
            return (False, f"O: J should be 2, got {J_O}.")
        return (True, "Oxygen hole formalism consistent.")

    def test_modeling_aufbau_ordering(self):
        """Aufbau: s‑block elements (Z=1-4) have only S‑terms; p‑block (Z=5-9) have P terms except half‑filled N."""
        for Z in range(1,5):
            term = self.all_terms.get(Z)
            if term is None:
                return (False, f"Z={Z} missing for ordering test.")
            if 'P' in term or 'D' in term or 'F' in term:
                return (False, f"Z={Z} has non‑S term but should be from s subshell.")
        for Z in range(5,10):
            term = self.all_terms.get(Z)
            if term is None:
                continue
            if Z == 7:   # half‑filled p -> S term
                if 'S' not in term or 'P' in term:
                    return (False, f"N (Z=7) should have S term, got {term}.")
            else:
                if 'P' not in term:
                    return (False, f"Z={Z} should have P term, got {term}.")
        return (True, "Aufbau ordering consistent: s‑block gives S terms, p‑block gives P terms (except N).")

    def test_modeling_J_parity_for_half_filled(self):
        """Half‑filled subshell (N) must have J = S = 3/2."""
        parsed_N = self._get_term_parsed(7)
        if parsed_N is None:
            return (False, "N term missing or unparseable.")
        S_N, L_N, J_N = parsed_N
        exp_S = 1.5
        exp_L = 0
        exp_J = exp_S
        if not math.isclose(S_N, exp_S, abs_tol=1e-9):
            return (False, f"N: expected S=1.5, got {S_N}.")
        if L_N != exp_L:
            return (False, f"N: expected L=0, got {L_N}.")
        if not math.isclose(J_N, exp_J, abs_tol=1e-9):
            return (False, f"N: J should equal S = 1.5, got {J_N}.")
        return (True, "N half‑filled rule satisfied.")

    # --------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input Z by +1)
    # --------------------------------------------------------
    def test_monotonicity_S_increase_from_5_to_6(self):
        """Increase Z from 5 (B) to 6 (C) → S increases from 0.5 to 1.0."""
        base = self._get_term_parsed(5)   # B
        perturb = self._get_term_parsed(6) # C
        if base is None:
            return (False, "Baseline Z=5 (B) missing.")
        if perturb is None:
            return (False, "Perturbed Z=6 (C) missing.")
        S_base, S_pert = base[0], perturb[0]
        if not (S_pert > S_base):
            return (False, f"S not increasing: Z=5 S={S_base}, Z=6 S={S_pert}.")
        return (True, f"S increases from {S_base} to {S_pert} when Z increases by 1.")

    def test_monotonicity_S_increase_from_6_to_7(self):
        """Increase Z from 6 (C) to 7 (N) → S increases from 1.0 to 1.5."""
        base = self._get_term_parsed(6)   # C
        perturb = self._get_term_parsed(7) # N
        if base is None:
            return (False, "Baseline Z=6 (C) missing.")
        if perturb is None:
            return (False, "Perturbed Z=7 (N) missing.")
        S_base, S_pert = base[0], perturb[0]
        if not (S_pert > S_base):
            return (False, f"S not increasing: Z=6 S={S_base}, Z=7 S={S_pert}.")
        return (True, f"S increases from {S_base} to {S_pert} when Z increases by 1.")

    def test_monotonicity_J_decrease_from_5_to_6(self):
        """Increase Z from 5 (B) to 6 (C) → J decreases from 0.5 to 0."""
        base = self._get_term_parsed(5)   # B
        perturb = self._get_term_parsed(6) # C
        if base is None:
            return (False, "Baseline Z=5 (B) missing.")
        if perturb is None:
            return (False, "Perturbed Z=6 (C) missing.")
        J_base, J_pert = base[2], perturb[2]
        if not (J_pert < J_base):
            return (False, f"J not decreasing: Z=5 J={J_base}, Z=6 J={J_pert}.")
        return (True, f"J decreases from {J_base} to {J_pert} when Z increases by 1.")

    def test_monotonicity_J_increase_from_6_to_7(self):
        """Increase Z from 6 (C) to 7 (N) → J increases from 0 to 1.5."""
        base = self._get_term_parsed(6)   # C
        perturb = self._get_term_parsed(7) # N
        if base is None:
            return (False, "Baseline Z=6 (C) missing.")
        if perturb is None:
            return (False, "Perturbed Z=7 (N) missing.")
        J_base, J_pert = base[2], perturb[2]
        if not (J_pert > J_base):
            return (False, f"J not increasing: Z=6 J={J_base}, Z=7 J={J_pert}.")
        return (True, f"J increases from {J_base} to {J_pert} when Z increases by 1.")

    def test_monotonicity_L_decrease_from_5_to_6(self):
        """Increase Z from 5 (B) to 6 (C) → L remains 1 (no change).  Verify L stays same for these two."""
        base = self._get_term_parsed(5)   # B
        perturb = self._get_term_parsed(6) # C
        if base is None:
            return (False, "Baseline Z=5 (B) missing.")
        if perturb is None:
            return (False, "Perturbed Z=6 (C) missing.")
        L_base, L_pert = base[1], perturb[1]
        if L_base != L_pert:
            return (False, f"L should remain constant (L={L_base}, L_pert={L_pert}).")
        return (True, f"L stays at {L_base} for Z=5 and Z=6 (expected).")

    def test_monotonicity_L_decrease_from_6_to_7(self):
        """Increase Z from 6 (C) to 7 (N) → L decreases from 1 to 0."""
        base = self._get_term_parsed(6)   # C
        perturb = self._get_term_parsed(7) # N
        if base is None:
            return (False, "Baseline Z=6 (C) missing.")
        if perturb is None:
            return (False, "Perturbed Z=7 (N) missing.")
        L_base, L_pert = base[1], perturb[1]
        if not (L_pert < L_base):
            return (False, f"L not decreasing: Z=6 L={L_base}, Z=7 L={L_pert}.")
        return (True, f"L decreases from {L_base} to {L_pert} when Z increases by 1.")

    # --------------------------------------------------------
    # Weak answer sanity (at most one)
    # --------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Weak sanity: all terms start with '^', contain letter S,P,D,F, and have a J value."""
        for Z, term in self.all_terms.items():
            if term is None:
                return (False, f"Z={Z}: term missing.")
            if not isinstance(term, str) or len(term) < 5:
                return (False, f"Z={Z}: term too short.")
            if not term.startswith('^'):
                return (False, f"Z={Z}: term must start with '^'.")
            if not term[1].isdigit():
                return (False, f"Z={Z}: multiplicity digit missing.")
            if term[2] not in 'SPDF':
                return (False, f"Z={Z}: expected letter S,P,D,F, got '{term[2]}'.")
            if '_' not in term:
                return (False, f"Z={Z}: missing underscore for J.")
        return (True, "All terms pass trivial format sanity.")

    # --------------------------------------------------------
    # Test runner
    # --------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)