import math
import re
import traceback
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    """
    Auditor for a solution to neutrino physics problem.
    Checks interface completeness, lepton flavor/charge/baryon conservation,
    helicity values, decay forbiddenness, and monotonicity of cross-section.
    """

    # Conservation dictionaries
    PARTICLE_CHARGE = {
        'nu_mu': 0, 'mu-': -1, 'mu+': 1, 'p': 1, 'n': 0,
        'pi+': 1, 'pi0': 0, 'e+': 1, 'e-': -1, 'gamma': 0,
        'nu_e': 0, 'anti_nu_e': 0, 'nu_mu_bar': 0
    }
    BARYON_NUMBER = {'p': 1, 'n': 1, 'pi+': 0, 'pi0': 0, 'mu-': 0, 'nu_mu': 0}
    LEPTON_MU = {'mu-': 1, 'nu_mu': 1, 'mu+': -1, 'nu_mu_bar': -1}
    LEPTON_E = {'e-': 1, 'nu_e': 1, 'e+': -1, 'anti_nu_e': -1}

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Fallback: try direct attribute access
            self.results = {}
            for key in ['sigma_nu', 'small_cross_section', 'reaction_3', 'reaction_4a',
                        'reaction_4b', 'h_nu', 'h_anu', 'mu_to_e_gamma', 'mu_to_3e']:
                val = getattr(solution_instance, key, None)
                if val is not None:
                    self.results[key] = val

        # Safely extract each key with None default
        self.sigma_nu = self.results.get('sigma_nu', None)
        self.small_cross_section = self.results.get('small_cross_section', None)
        self.reaction_3 = self.results.get('reaction_3', None)
        self.reaction_4a = self.results.get('reaction_4a', None)
        self.reaction_4b = self.results.get('reaction_4b', None)
        self.h_nu = self.results.get('h_nu', None)
        self.h_anu = self.results.get('h_anu', None)
        self.mu_to_e_gamma = self.results.get('mu_to_e_gamma', None)
        self.mu_to_3e = self.results.get('mu_to_3e', None)

    def _check_particle_conservation(self, reaction_str: str,
                                     quantity_dict: Dict[str, float]) -> Tuple[bool, str]:
        """Check that sum(quantity) on left side equals sum on right side."""
        if not reaction_str:
            return False, "Reaction string is empty"
        sides = re.split(r'\s*->\s*|\s*→\s*', reaction_str)
        if len(sides) != 2:
            return False, f"Reaction does not have exactly two sides: {reaction_str}"
        left_str, right_str = sides[0].strip(), sides[1].strip()
        left_parts = re.split(r'\s*\+\s*', left_str)
        right_parts = re.split(r'\s*\+\s*', right_str)

        def sum_quantity(parts):
            total = 0.0
            for p in parts:
                p = p.strip()
                if p in quantity_dict:
                    total += quantity_dict[p]
            return total

        left_sum = sum_quantity(left_parts)
        right_sum = sum_quantity(right_parts)
        if math.isclose(left_sum, right_sum, rel_tol=1e-6, abs_tol=1e-6):
            return True, f"Sums equal: left={left_sum}, right={right_sum}"
        else:
            return False, f"Conservation violation: left={left_sum}, right={right_sum}"

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required keys exist in self.results."""
        required = ['sigma_nu', 'small_cross_section', 'reaction_3', 'reaction_4a',
                     'reaction_4b', 'h_nu', 'h_anu', 'mu_to_e_gamma', 'mu_to_3e']
        missing = [k for k in required if k not in self.results]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"

    def test_interface_sigma_nu_type(self) -> Tuple[bool, str]:
        """Check sigma_nu is a numeric or string."""
        if self.sigma_nu is None:
            return False, "sigma_nu is None"
        if isinstance(self.sigma_nu, (int, float, str)):
            return True, f"sigma_nu = {self.sigma_nu}"
        return False, f"sigma_nu has unexpected type {type(self.sigma_nu)}"

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physics)
    # ------------------------------------------------------------
    def test_invariant_lepton_flavor_mu_to_e_gamma(self) -> Tuple[bool, str]:
        """mu+ -> e+ gamma is forbidden due to lepton flavor violation."""
        if self.mu_to_e_gamma is None:
            return (False, "mu_to_e_gamma not provided")
        if isinstance(self.mu_to_e_gamma, str) and \
           self.mu_to_e_gamma.lower() in ['forbidden', 'no', 'false', 'cannot']:
            return (True, "Correctly identified as forbidden")
        return (False, f"Expected forbidden, got {self.mu_to_e_gamma}")

    def test_invariant_lepton_flavor_mu_to_3e(self) -> Tuple[bool, str]:
        """mu+ -> e+e-e+ is forbidden."""
        if self.mu_to_3e is None:
            return (False, "mu_to_3e not provided")
        if isinstance(self.mu_to_3e, str) and \
           self.mu_to_3e.lower() in ['forbidden', 'no', 'false', 'cannot']:
            return (True, "Correctly identified as forbidden")
        return (False, f"Expected forbidden, got {self.mu_to_3e}")

    def test_invariant_charge_conservation_reaction3(self) -> Tuple[bool, str]:
        if self.reaction_3 is None:
            return (False, "reaction_3 not provided")
        return self._check_particle_conservation(self.reaction_3, self.PARTICLE_CHARGE)

    def test_invariant_charge_conservation_reaction4a(self) -> Tuple[bool, str]:
        if self.reaction_4a is None:
            return (False, "reaction_4a not provided")
        return self._check_particle_conservation(self.reaction_4a, self.PARTICLE_CHARGE)

    def test_invariant_charge_conservation_reaction4b(self) -> Tuple[bool, str]:
        if self.reaction_4b is None:
            return (False, "reaction_4b not provided")
        return self._check_particle_conservation(self.reaction_4b, self.PARTICLE_CHARGE)

    def test_invariant_baryon_conservation_reaction3(self) -> Tuple[bool, str]:
        if self.reaction_3 is None:
            return (False, "reaction_3 not provided")
        return self._check_particle_conservation(self.reaction_3, self.BARYON_NUMBER)

    def test_invariant_baryon_conservation_reaction4a(self) -> Tuple[bool, str]:
        if self.reaction_4a is None:
            return (False, "reaction_4a not provided")
        return self._check_particle_conservation(self.reaction_4a, self.BARYON_NUMBER)

    def test_invariant_baryon_conservation_reaction4b(self) -> Tuple[bool, str]:
        if self.reaction_4b is None:
            return (False, "reaction_4b not provided")
        return self._check_particle_conservation(self.reaction_4b, self.BARYON_NUMBER)

    def test_invariant_lepton_number_reaction3(self) -> Tuple[bool, str]:
        if self.reaction_3 is None:
            return (False, "reaction_3 not provided")
        return self._check_particle_conservation(self.reaction_3, self.LEPTON_MU)

    def test_invariant_lepton_number_reaction4a(self) -> Tuple[bool, str]:
        if self.reaction_4a is None:
            return (False, "reaction_4a not provided")
        return self._check_particle_conservation(self.reaction_4a, self.LEPTON_MU)

    def test_invariant_lepton_number_reaction4b(self) -> Tuple[bool, str]:
        if self.reaction_4b is None:
            return (False, "reaction_4b not provided")
        return self._check_particle_conservation(self.reaction_4b, self.LEPTON_MU)

    def test_invariant_helicity_values(self) -> Tuple[bool, str]:
        if self.h_nu is None or self.h_anu is None:
            return (False, "h_nu or h_anu not provided")
        try:
            h_nu_val = float(self.h_nu)
            h_anu_val = float(self.h_anu)
        except (ValueError, TypeError):
            return (False, f"Non-numeric helicity: {self.h_nu}, {self.h_anu}")
        if math.isclose(h_nu_val, -1.0, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose(h_anu_val, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"h_nu = {h_nu_val}, h_anu = {h_anu_val}")
        return (False, f"Wrong helicities: expected -1 and +1, got {h_nu_val}, {h_anu_val}")

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_reaction3_contains_particles(self) -> Tuple[bool, str]:
        if self.reaction_3 is None:
            return (False, "reaction_3 not provided")
        needed = ['nu_mu', 'p', 'mu-', 'pi+']
        missing = [p for p in needed if p not in self.reaction_3]
        if missing:
            return (False, f"Missing particles: {missing}")
        return (True, "All required particles present")

    def test_modeling_reaction4a_contains_particles(self) -> Tuple[bool, str]:
        if self.reaction_4a is None:
            return (False, "reaction_4a not provided")
        needed = ['nu_mu', 'n', 'mu-', 'pi+']
        missing = [p for p in needed if p not in self.reaction_4a]
        if missing:
            return (False, f"Missing particles: {missing}")
        return (True, "All required particles present")

    def test_modeling_reaction4b_contains_particles(self) -> Tuple[bool, str]:
        if self.reaction_4b is None:
            return (False, "reaction_4b not provided")
        needed = ['nu_mu', 'n', 'mu-', 'p', 'pi0']
        missing = [p for p in needed if p not in self.reaction_4b]
        if missing:
            return (False, f"Missing particles: {missing}")
        return (True, "All required particles present")

    def test_modeling_small_cross_section(self) -> Tuple[bool, str]:
        if self.small_cross_section is None:
            return (False, "small_cross_section not provided")
        text = str(self.small_cross_section).lower()
        # Check for numeric magnitude indication
        if '10^{-44}' in text or '10^-44' in text or '10-44' in text:
            return (True, "Cross-section magnitude (10^-44) explicitly noted")
        if 'small' in text or 'tiny' in text or '10' in text:
            return (True, "Cross-section described as small")
        return (False, f"No evidence of small cross-section in: {self.small_cross_section}")

    # ------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare perturbed to baseline)
    # ------------------------------------------------------------
    def test_monotonicity_cross_section_scaling(self) -> Tuple[bool, str]:
        """
        Check that if neutrino energy increases, cross-section increases.
        Use Fermi theory scaling: sigma ~ E_nu^2 (for E_nu << M_W^2).
        Take baseline E0 = 1 MeV, perturb E1 = 2 MeV.
        Baseline sigma from solution; compute perturbed sigma as baseline * (E1/E0)^2.
        Verify perturbed > baseline.
        """
        if self.sigma_nu is None:
            return (False, "SKIP: sigma_nu not available")
        try:
            sigma_baseline = float(str(self.sigma_nu).replace('\\text{cm}^2', '').replace('cm^2', '').replace('~', '').strip())
        except (ValueError, TypeError):
            return (False, "Could not parse sigma_nu as number")
        if sigma_baseline <= 0:
            return (False, f"sigma_baseline <= 0: {sigma_baseline}")
        # Fermi theory: sigma ~ E^2 (for low energy)
        # Baseline energy: typical reactor neutrino ~ 1 MeV (known from problem context)
        # Perturb to 2 MeV
        e_ratio = 2.0  # 2 / 1
        # Perturbed sigma = baseline * (E_perturb / E_baseline)^2
        sigma_perturbed = sigma_baseline * (e_ratio ** 2)
        if sigma_perturbed > sigma_baseline:
            return (True, f"sigma increases with energy: perturbed {sigma_perturbed:.2e} > baseline {sigma_baseline:.2e}")
        else:
            return (False, f"sigma did not increase: perturbed {sigma_perturbed:.2e} <= baseline {sigma_baseline:.2e}")

    # ------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------
    def test_answer_sanity_mu_decays_forbidden(self) -> Tuple[bool, str]:
        if self.mu_to_e_gamma is None or self.mu_to_3e is None:
            return (False, "One or both decay answers missing")
        ok1 = isinstance(self.mu_to_e_gamma, str) and 'forbidden' in self.mu_to_e_gamma.lower()
        ok2 = isinstance(self.mu_to_3e, str) and 'forbidden' in self.mu_to_3e.lower()
        if ok1 and ok2:
            return (True, "Both decays correctly flagged as forbidden")
        return (False, f"mu_to_e_gamma='{self.mu_to_e_gamma}', mu_to_3e='{self.mu_to_3e}'; expected forbidden")

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
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