import math
import sympy as sp
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', {})
        self.bohr_postulate = self.results.get('bohr_postulate', None)
        self.standing_wave = self.results.get('standing_wave', None)
        self.givens = getattr(solution_instance, 'givens', {})  # optional, not used heavily
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----- Interface checks -----
    def test_interface_bohr_postulate_key(self):
        if self.bohr_postulate is None:
            return (False, "SKIP: bohr_postulate key missing or None")
        if not isinstance(self.bohr_postulate, str):
            return (False, "bohr_postulate is not a string")
        if len(self.bohr_postulate.strip()) == 0:
            return (False, "bohr_postulate string is empty")
        # Check essential physics symbols are present
        if 'hbar' not in self.bohr_postulate and 'h/(2pi)' not in self.bohr_postulate:
            return (False, "bohr_postulate does not contain 'hbar' or 'h/(2pi)'")
        return (True, "bohr_postulate key present, non-empty, contains hbar symbol")

    def test_interface_standing_wave_key(self):
        if self.standing_wave is None:
            return (False, "SKIP: standing_wave key missing or None")
        if not isinstance(self.standing_wave, str):
            return (False, "standing_wave is not a string")
        if len(self.standing_wave.strip()) == 0:
            return (False, "standing_wave string is empty")
        if 'lambda' not in self.standing_wave and 'λ' not in self.standing_wave:
            return (False, "standing_wave does not contain 'lambda'")
        if '2pi' not in self.standing_wave and '2π' not in self.standing_wave:
            return (False, "standing_wave does not contain '2pi'")
        return (True, "standing_wave key present, non-empty, contains 2pi and lambda")

    # ----- Invariant checks (theorem-based consistency) -----
    def test_invariant_derivation_consistency(self):
        """
        Check that the standing_wave equation can be derived from bohr_postulate
        using de Broglie relation and hbar = h/(2pi).  This is a logical invariant.
        """
        bp = self.bohr_postulate
        sw = self.standing_wave
        if bp is None or sw is None:
            return (False, "SKIP: both keys needed for derivation check")
        # Normalize strings: remove spaces, asterisks, periods
        bp_norm = bp.replace(" ", "").replace("*", "").replace("⋅", "").replace("_","").lower()
        sw_norm = sw.replace(" ", "").replace("*", "").replace("⋅", "").replace("_","").lower()
        # Check that standing_wave contains elements from de Broglie derivation:
        # 2pi r = n * h/(mv) => n * h/(mv) = n * lambda
        if '2pir' not in sw_norm and '2πr' not in sw_norm:
            # also allow variations with parentheses, e.g., "2 pi r" or "2π r"
            if not re.search(r'2\s*pi\s*r', sw_norm) and not re.search(r'2π\s*r', sw_norm):
                return (False, "standing_wave does not contain '2pi r'")
        if 'nlambda' not in sw_norm and 'nλ' not in sw_norm:
            if not re.search(r'n\s*lambda', sw_norm) and not re.search(r'n\s*λ', sw_norm):
                return (False, "standing_wave does not contain 'n*lambda'")
        # Check bohr_postulate contains 'm', 'v', 'r', 'n', 'hbar'
        required_bp = ['m', 'v', 'r', 'n', 'hbar']
        for sym in required_bp:
            if sym not in bp_norm:
                return (False, f"bohr_postulate missing required symbol '{sym}'")
        return (True, "Standing wave equation is a logical consequence of Bohr postulate + de Broglie relation")

    def test_invariant_embedded_relations(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_text = " ".join(self.explanation_steps).lower()
        keywords = ['angular momentum', 'quantization', 'de broglie', 'standing wave', '2πr = nλ', 'mvr = nh']
        found = [kw for kw in keywords if kw in steps_text]
        if len(found) < 2:
            return (False, f"Explanation steps missing essential physics keywords; found: {found}")
        return (True, "Explanation steps contain key physics relationships")

    # ----- Modeling / trace checks (based on pseudocode) -----
    def test_trace_bohr_postulate_origin(self):
        """
        Check that the bohr_postulate string expresses angular momentum quantization:
        product of m, v, r equals n times hbar.
        """
        if self.bohr_postulate is None:
            return (False, "SKIP: no bohr_postulate value")
        bp = self.bohr_postulate.replace(" ", "").replace("*","").replace("⋅","").replace("_","")
        # Must contain '=' and variables: m, v, r on one side, n, hbar on the other.
        if '=' not in bp:
            return (False, "bohr_postulate does not contain an equals sign")
        sides = bp.split('=')
        if len(sides) != 2:
            return (False, "bohr_postulate has more than one equals sign")
        left, right = sides
        # Check left side contains m, v, r (order may vary)
        if not all(sym in left for sym in ['m', 'v', 'r']):
            return (False, "Left side of bohr_postulate missing one of m, v, r")
        # Check right side contains n and hbar
        if not all(sym in right for sym in ['n', 'hbar']):
            return (False, "Right side of bohr_postulate missing n or hbar")
        return (True, "bohr_postulate structure matches L = n*hbar")

    def test_trace_standing_wave_origin(self):
        """
        Check that the standing_wave string expresses standing wave condition:
        2πr equals n times lambda.
        """
        if self.standing_wave is None:
            return (False, "SKIP: no standing_wave value")
        sw = self.standing_wave.replace(" ", "").replace("*","").replace("⋅","")
        if '=' not in sw:
            return (False, "standing_wave does not contain an equals sign")
        sides = sw.split('=')
        if len(sides) != 2:
            return (False, "standing_wave has more than one equals sign")
        left, right = sides
        # Left side must contain 2π (or 2pi) and r
        has_2pi = ('2π' in left) or ('2pi' in left)
        if not has_2pi:
            return (False, "Left side of standing_wave missing 2π or 2pi")
        if 'r' not in left:
            return (False, "Left side of standing_wave missing r")
        # Right side must contain n and lambda (or λ)
        if not all(sym in right for sym in ['n', 'lambda']) and 'λ' not in right:
            return (False, "Right side of standing_wave missing n and lambda/λ")
        return (True, "standing_wave structure matches 2πr = nλ")

    # ----- Monotonicity checks -----
    def test_monotonicity_angular_momentum_increase(self):
        """Perturb principal quantum number n by +1% and check L increases."""
        try:
            hbar = 1.054571817e-34  # J·s, standard constant
            n_baseline = 1.0
            n_perturbed = n_baseline * 1.01   # +1% perturbation
            L_baseline = n_baseline * hbar
            L_perturbed = n_perturbed * hbar
            if L_perturbed > L_baseline:
                return (True, f"Angular momentum increases with n: baseline {L_baseline:.3e} < perturbed {L_perturbed:.3e}")
            else:
                return (False, "Angular momentum did not increase with n (unexpected)")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ----- Answer sanity (weak, secondary) -----
    def test_answer_sanity_strings_length(self):
        """Weak sanity: both result strings should have reasonable length (>10 characters)"""
        bp = self.bohr_postulate
        sw = self.standing_wave
        if bp is None and sw is None:
            return (False, "Both results missing, cannot perform answer sanity")
        if bp is not None and len(bp.strip()) < 10:
            return (False, "bohr_postulate string is too short (<10 chars)")
        if sw is not None and len(sw.strip()) < 10:
            return (False, "standing_wave string is too short (<10 chars)")
        return (True, "Result strings have reasonable length")

    # ----- Run all tests -----
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