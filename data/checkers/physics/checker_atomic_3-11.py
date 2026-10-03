import sympy as sp
import re
import math
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    """
    Auditor for particle physics problem: lepton list, muon decay, lepton number assignments,
    and heavy lepton production.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract all required result keys; default to None.
        self.lepton_list = getattr(solution_instance, 'lepton_list', None)
        self.mu_plus_decay = getattr(solution_instance, 'mu_plus_decay', None)
        self.L_e_assignment = getattr(solution_instance, 'L_e_assignment', None)
        self.L_mu_assignment = getattr(solution_instance, 'L_mu_assignment', None)
        self.heavy_lepton_reaction = getattr(solution_instance, 'heavy_lepton_reaction', None)

        # Precompile some regex for later use
        self.particle_pattern = re.compile(r'[A-Za-z_\\^]+')

    # ------------------------------------------------------------------
    # Helper: check conservation of charge and lepton numbers for a decay string
    # ------------------------------------------------------------------
    def _check_decay_conservation(self, decay_str: str) -> Tuple[bool, str]:
        """
        Given a decay string (e.g., '\\mu^+ \\rightarrow e^+ + \\nu_e + \\bar{\\nu}_\\mu'),
        return (True, "") if charge and both lepton numbers are conserved,
        otherwise (False, error_message).
        Uses standard lepton number assignments.
        """
        # Define lepton numbers for known particles
        lepton_numbers = {
            'e^-':    (1, 0),
            '\\nu_e': (1, 0),
            'e^+':    (-1, 0),
            '\\bar{\\nu}_e': (-1, 0),
            '\\mu^-':   (0, 1),
            '\\nu_\\mu': (0, 1),
            '\\mu^+':   (0, -1),
            '\\bar{\\nu}_\\mu': (0, -1),
        }
        # Charges: e=-1, mu=-1, tau=-1; antiparticles positive
        charges = {
            'e^-': -1, '\\nu_e':0, 'e^+': +1, '\\bar{\\nu}_e':0,
            '\\mu^-':-1, '\\nu_\\mu':0, '\\mu^+':+1, '\\bar{\\nu}_\\mu':0,
        }

        # Parse reaction string: split by '\\rightarrow'
        if '\\rightarrow' not in decay_str:
            return (False, "No '\\rightarrow' found")
        parts = decay_str.split('\\rightarrow')
        left = parts[0].strip()
        right = parts[1].strip()
        # Tokenize right-hand side: split by '+' signs
        # Remove LaTeX spacing
        rhs_str = right.replace(' ', '').replace('\\,', '')
        rhs_particles = [p.strip() for p in rhs_str.split('+') if p.strip()]
        lhs_particles = [left]  # left side is a single particle

        # Compute initial sums
        sum_L_e_i = sum_L_mu_i = sum_Q_i = 0
        for p in lhs_particles:
            if p in lepton_numbers:
                Le, Lmu = lepton_numbers[p]
                sum_L_e_i += Le
                sum_L_mu_i += Lmu
            else:
                return (False, f"Unknown particle '{p}' on left side")
            if p in charges:
                sum_Q_i += charges[p]
            else:
                return (False, f"Unknown charge for '{p}'")

        # Compute final sums
        sum_L_e_f = sum_L_mu_f = sum_Q_f = 0
        for p in rhs_particles:
            if p in lepton_numbers:
                Le, Lmu = lepton_numbers[p]
                sum_L_e_f += Le
                sum_L_mu_f += Lmu
            else:
                return (False, f"Unknown particle '{p}' on right side")
            if p in charges:
                sum_Q_f += charges[p]
            else:
                return (False, f"Unknown charge for '{p}'")

        # Check conservation
        if sum_L_e_i != sum_L_e_f:
            return (False, f"L_e not conserved: initial {sum_L_e_i} vs final {sum_L_e_f}")
        if sum_L_mu_i != sum_L_mu_f:
            return (False, f"L_mu not conserved: initial {sum_L_mu_i} vs final {sum_L_mu_f}")
        if sum_Q_i != sum_Q_f:
            return (False, f"Charge not conserved: initial {sum_Q_i} vs final {sum_Q_f}")
        return (True, "All conservation laws satisfied")

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, returning (passed, logs)."""
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

    # ------------------------------------------------------------------
    #  Interface tests – check that required keys exist and are not empty
    # ------------------------------------------------------------------
    def test_interface_lepton_list(self) -> Tuple[bool, str]:
        if self.lepton_list is None:
            return (False, "SKIP: lepton_list not provided")
        if not isinstance(self.lepton_list, str) or len(self.lepton_list.strip()) == 0:
            return (False, "lepton_list is not a non-empty string")
        return (True, "lepton_list is present and non-empty string")

    def test_interface_mu_plus_decay(self) -> Tuple[bool, str]:
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not provided")
        if not isinstance(self.mu_plus_decay, str) or len(self.mu_plus_decay.strip()) == 0:
            return (False, "mu_plus_decay is not a non-empty string")
        return (True, "mu_plus_decay is present and non-empty string")

    def test_interface_L_e_assignment(self) -> Tuple[bool, str]:
        if self.L_e_assignment is None:
            return (False, "SKIP: L_e_assignment not provided")
        if not isinstance(self.L_e_assignment, str) or len(self.L_e_assignment.strip()) == 0:
            return (False, "L_e_assignment is not a non-empty string")
        return (True, "L_e_assignment is present and non-empty string")

    def test_interface_L_mu_assignment(self) -> Tuple[bool, str]:
        if self.L_mu_assignment is None:
            return (False, "SKIP: L_mu_assignment not provided")
        if not isinstance(self.L_mu_assignment, str) or len(self.L_mu_assignment.strip()) == 0:
            return (False, "L_mu_assignment is not a non-empty string")
        return (True, "L_mu_assignment is present and non-empty string")

    def test_interface_heavy_lepton_reaction(self) -> Tuple[bool, str]:
        if self.heavy_lepton_reaction is None:
            return (False, "SKIP: heavy_lepton_reaction not provided")
        if not isinstance(self.heavy_lepton_reaction, str) or len(self.heavy_lepton_reaction.strip()) == 0:
            return (False, "heavy_lepton_reaction is not a non-empty string")
        return (True, "heavy_lepton_reaction is present and non-empty string")

    # ----------------------------------------------------------------
    #  Invariant tests – physics conservation laws
    # ----------------------------------------------------------------
    def test_invariant_muon_decay_lepton_conservation(self) -> Tuple[bool, str]:
        """
        Check that the muon decay reaction conserves electric charge and
        both lepton numbers L_e and L_mu.
        """
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not available")
        ok, msg = self._check_decay_conservation(self.mu_plus_decay)
        if ok:
            return (True, "Muon decay conserves all relevant quantum numbers")
        else:
            return (False, msg)

    def test_invariant_forbidden_reaction_logical(self) -> Tuple[bool, str]:
        """
        Check that the muon decay is not the forbidden reaction ν_μ + n → e⁻ + p.
        """
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not available")
        forbidden_prefix = r'\\nu_\\mu + n \\rightarrow e^- + p'
        if forbidden_prefix.lstrip() in self.mu_plus_decay:
            return (False, "Decay appears to be a forbidden reaction, not the standard muon decay")
        return (True, "Reaction does not match the forbidden ν_μ + n → e⁻ + p")

    # ----------------------------------------------------------------
    #  Modeling tests – check that the emitted strings match expected patterns
    # ----------------------------------------------------------------
    def test_modeling_lepton_list_contains_essentials(self) -> Tuple[bool, str]:
        """
        Check that the lepton list contains e, μ, τ and their neutrinos and antiparticles.
        """
        if self.lepton_list is None:
            return (False, "SKIP: lepton_list not available")
        text = self.lepton_list
        required_parts = ['e^-', '\\nu_e', '\\mu^-', '\\nu_\\mu', '\\tau^-', 
                          'e^+', '\\bar{\\nu}_e', '\\mu^+', '\\bar{\\nu}_\\mu', '\\tau^+']
        missing = [p for p in required_parts if p not in text]
        if missing:
            return (False, f"Missing particles in lepton list: {missing}")
        return (True, "Lepton list contains all required fundamental leptons and antiparticles")

    def test_modeling_mu_plus_decay_structure(self) -> Tuple[bool, str]:
        """
        Check that the muon decay string has the correct structure:
        left side: mu^+, right side: e^+, ν_e, and ν̄_μ.
        """
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not available")
        text = self.mu_plus_decay
        if '\\mu^+' not in text:
            return (False, "Decay does not involve mu+ on left side")
        if '\\rightarrow' not in text:
            return (False, "Decay missing 'rightarrow'")
        rhs = text.split('\\rightarrow')[-1]
        if 'e^+' not in rhs:
            return (False, "Right side missing e^+")
        if '\\nu_e' not in rhs:
            return (False, "Right side missing ν_e")
        if '\\bar{\\nu}_\\mu' not in rhs and '\\bar{\\nu}' not in rhs:
            return (False, "Right side missing ν̄_μ (anti-ν_μ)")
        return (True, "Muon decay has correct particle content")

    def test_modeling_L_e_assignment_content(self) -> Tuple[bool, str]:
        if self.L_e_assignment is None:
            return (False, "SKIP: L_e_assignment not available")
        text = self.L_e_assignment
        if not ('L_e' in text and ('1' in text)):
            return (False, "Does not specify L_e = 1 assignment")
        if not (('e^-' in text or 'e^' in text) and ('\\nu_e' in text or 'nu_e' in text)):
            return (False, "Does not mention both e- and ν_e")
        return (True, "L_e assignment appears correct")

    def test_modeling_L_mu_assignment_content(self) -> Tuple[bool, str]:
        if self.L_mu_assignment is None:
            return (False, "SKIP: L_mu_assignment not available")
        text = self.L_mu_assignment
        if not ('L_\\mu' in text and ('1' in text)):
            return (False, "Does not specify L_μ = 1")
        if not (('\\mu^-' in text or 'mu^-' in text) and ('\\nu_\\mu' in text or 'nu_mu' in text)):
            return (False, "Does not mention both μ- and ν_μ")
        return (True, "L_μ assignment appears correct")

    def test_modeling_heavy_lepton_reaction_structure(self) -> Tuple[bool, str]:
        """
        Check that the heavy lepton reaction contains: ν_μ, n, A^+, ν_A, μ^-, X.
        """
        if self.heavy_lepton_reaction is None:
            return (False, "SKIP: heavy_lepton_reaction not available")
        text = self.heavy_lepton_reaction
        if '\\nu_\\mu' not in text:
            return (False, "Missing ν_μ")
        if 'n' not in text:
            return (False, "Missing neutron n")
        if 'A^+' not in text:
            return (False, "Missing new heavy lepton A^+")
        if '\\nu_A' not in text:
            return (False, "Missing associated neutrino ν_A")
        if '\\mu^-' not in text:
            return (False, "Missing muon μ⁻")
        if 'X' not in text:
            return (False, "Missing baryonic remainder X")
        if '\\rightarrow' not in text:
            return (False, "Missing arrow")
        left, right = text.split('\\rightarrow', 1)
        if '\\nu_\\mu' not in left or 'n' not in left:
            return (False, "Left side must contain ν_μ + n")
        rhs_parts = [p.strip() for p in right.split('+')]
        needed = ['A^+', '\\nu_A', '\\mu^-', 'X']
        for n in needed:
            if not any(n in p for p in rhs_parts):
                return (False, f"Right side missing {n}")
        return (True, "Heavy lepton reaction has correct particles and ordering")

    # ----------------------------------------------------------------
    #  Monotonicity test – explicitly perturb input and compare to baseline
    # ----------------------------------------------------------------
    def test_monotonicity_perturb_muon_decay(self) -> Tuple[bool, str]:
        """
        Perturb the muon decay string by flipping e⁺ to e⁻. The baseline (original)
        should conserve lepton numbers; the perturbed version should violate L_e conservation.
        This demonstrates monotonicity: changing the charge sign flips conservation
        from true to false.
        """
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not available")
        original = self.mu_plus_decay
        # Perturb: replace e⁺ with e⁻ (also need to adjust antineutrino? Actually, the
        # charge of e⁻ is -1, so the reaction would become μ⁺ → e⁻ + ν_e + ν̄_μ.
        # That violates L_e because initial L_e=0, final L_e=+1 for e⁻.
        perturbed = original.replace('e^+', 'e^-')
        if perturbed == original:
            # If no replacement happened (e.g., already contains e^-), try replacing mu+ with mu-
            perturbed = original.replace('\\mu^+', '\\mu^-')
        if perturbed == original:
            return (False, "Could not create a perturbed version (no e^+ or mu^+ found)")

        # Check conservation on original (should pass)
        original_ok, _ = self._check_decay_conservation(original)
        if not original_ok:
            return (False, "Baseline (original) does not even conserve quantum numbers; monotonicity test invalid")

        # Check conservation on perturbed (should fail)
        perturbed_ok, error_msg = self._check_decay_conservation(perturbed)
        if perturbed_ok:
            return (False, f"Perturbed version still conserves; expected violation. Perturbed: {perturbed}")

        # Monotonicity satisfied: original passes, perturbed fails
        return (True, "Monotonicity valid: flipping e⁺ to e⁻ changes conservation from True to False")

    # ----------------------------------------------------------------
    #  Weak answer sanity check (only one allowed, must be weak)
    # ----------------------------------------------------------------
    def test_answer_sanity_muon_decay_sign(self) -> Tuple[bool, str]:
        """
        Very weak sanity: check that muon decay has a plus sign on both sides
        (indication of charge conservation) and no negative charges aside from antimuon on left.
        Just a superficial check.
        """
        if self.mu_plus_decay is None:
            return (False, "SKIP: mu_plus_decay not available")
        text = self.mu_plus_decay
        if 'e^+' in text and '\\nu' in text:
            return (True, "Weak sanity: decay contains e^+ and a neutrino")
        return (False, "Decay does not obviously contain e^+ and a neutrino")

if __name__ == "__main__":
    # Example usage
    class MockSolution:
        pass
    sol = MockSolution()
    sol.lepton_list = '{e^-, \\nu_e, \\mu^-, \\nu_\\mu, \\tau^-, e^+, \\bar{\\nu}_e, \\mu^+, \\bar{\\nu}_\\mu, \\tau^+}'
    sol.mu_plus_decay = '\\mu^+ \\rightarrow e^+ + \\nu_e + \\bar{\\nu}_\\mu'
    sol.L_e_assignment = 'L_e = 1 \\quad \\text{for} \\quad e^-, \\nu_e'
    sol.L_mu_assignment = 'L_\\mu = 1 \\quad \\text{for} \\quad \\mu^-, \\nu_\\mu'
    sol.heavy_lepton_reaction = '\\nu_\\mu + n \\rightarrow A^+ + \\nu_A + \\mu^- + X'

    checker = PhysicsChecker(sol)
    passed, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"Overall: {'PASS' if passed else 'FAIL'}")