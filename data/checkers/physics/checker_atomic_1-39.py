import math
import re
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results; use None for missing keys
        results = getattr(solution_instance, 'results', {})
        self.E_nu_typical = results.get('E_nu_typical', None)
        self.tau_scaling   = results.get('tau_scaling', None)
        self.config        = results.get('config', None)
        self.S             = results.get('S', None)
        self.L             = results.get('L', None)
        self.J             = results.get('J', None)

    # ----------------------------------------------------------------------
    # INTERFACE TESTS – check required keys and types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.E_nu_typical is None:
            return (False, "E_nu_typical is missing")
        if self.tau_scaling is None:
            return (False, "tau_scaling is missing")
        if self.config is None:
            return (False, "config is missing")
        if self.S is None:
            return (False, "S is missing")
        if self.L is None:
            return (False, "L is missing")
        if self.J is None:
            return (False, "J is missing")
        return (True, "All required keys present")

    def test_interface_E_nu_typical_type(self):
        if self.E_nu_typical is None:
            return (False, "E_nu_typical missing")
        if not isinstance(self.E_nu_typical, str) or not self.E_nu_typical.strip():
            return (False, f"E_nu_typical must be a non-empty string, got {type(self.E_nu_typical)}")
        return (True, "E_nu_typical is a string")

    def test_interface_tau_scaling_type(self):
        if self.tau_scaling is None:
            return (False, "tau_scaling missing")
        if not isinstance(self.tau_scaling, str) or not self.tau_scaling.strip():
            return (False, f"tau_scaling must be a non-empty string, got {type(self.tau_scaling)}")
        return (True, "tau_scaling is a string")

    def test_interface_config_type(self):
        if self.config is None:
            return (False, "config missing")
        if not isinstance(self.config, str) or not self.config.strip():
            return (False, f"config must be a non-empty string, got {type(self.config)}")
        return (True, "config is a string")

    def test_interface_S_L_J_types(self):
        if None in (self.S, self.L, self.J):
            return (False, "One or more of S, L, J is missing")
        for name, val in [('S', self.S), ('L', self.L), ('J', self.J)]:
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (got {type(val)})")
            if val <= 0:
                return (False, f"{name} is not positive (got {val})")
        return (True, "S, L, J are positive numeric values")

    # ----------------------------------------------------------------------
    # INVARIANT TESTS – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_beta_decay_energy(self):
        if self.E_nu_typical is None:
            return (False, "SKIP: E_nu_typical not available")
        match = re.search(r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', self.E_nu_typical)
        if not match:
            return (False, "Could not extract numeric value from E_nu_typical")
        try:
            energy_mev = float(match.group(1))
        except ValueError:
            return (False, "Extracted value is not a valid float")
        if not math.isfinite(energy_mev):
            return (False, f"Energy is not finite: {energy_mev}")
        if not (5e-1 <= energy_mev <= 2.0):
            return (False, f"Energy {energy_mev} MeV is not typical (~1 MeV)")
        if "MeV" not in self.E_nu_typical:
            return (False, "String does not indicate MeV units")
        return (True, f"Neutrino energy ~{energy_mev} MeV is typical")

    def test_invariant_tau_scaling_power(self):
        if self.tau_scaling is None:
            return (False, "SKIP: tau_scaling not available")
        expr_str = self.tau_scaling.replace('^', '**').replace('{', '(').replace('}', ')')
        expr_str = re.sub(r'\s+', '', expr_str)
        try:
            expr = sp.sympify(expr_str)
        except (sp.SympifyError, TypeError):
            return (False, f"Could not sympify tau_scaling string: '{self.tau_scaling}'")
        Z = sp.symbols('Z')
        try:
            simplified = sp.simplify(expr / (Z**(-4)))
        except Exception:
            return (False, "Failed to compare with Z^{-4}")
        if simplified.free_symbols:
            return (False, f"tau_scaling does not match Z^{-4}; simplified ratio: {simplified}")
        if simplified <= 0:
            return (False, "Constant factor in scaling is non-positive")
        return (True, "tau_scaling is proportional to Z^{-4}")

    def test_invariant_hund_rules(self):
        if any(v is None for v in [self.S, self.L, self.J]):
            return (False, "SKIP: one of S, L, J missing")
        if not math.isclose(self.S, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S = {self.S} is not 1 (within tolerance)")
        if not math.isclose(self.L, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"L = {self.L} is not 1 (within tolerance)")
        if not math.isclose(self.J, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"J = {self.J} is not 2 (within tolerance)")
        return (True, "Hund's rules satisfied: S=1, L=1, J=2")

    def test_invariant_config_aufbau(self):
        if self.config is None:
            return (False, "SKIP: config not available")
        cfg_norm = re.sub(r'\s+', '', self.config)
        expected = r'1s\^?2\s*2s\^?2\s*2p\^?4'
        if re.search(expected, cfg_norm, re.IGNORECASE) is None:
            return (False, f"config '{self.config}' does not match expected '1s²2s²2p⁴'")
        try:
            electrons = sum(int(num) for num in re.findall(r'(\d+)(?:$|\s|s|p|d)', cfg_norm[cfg_norm.find('1s'):]))
        except ValueError:
            return (False, "Could not parse electron counts")
        if electrons != 8:
            return (False, f"Total electrons = {electrons}, expected 8 for oxygen")
        return (True, "Config matches Aufbau with 8 electrons")

    # ----------------------------------------------------------------------
    # MODELING/TRACE TESTS
    # ----------------------------------------------------------------------
    def test_modeling_beta_decay_range(self):
        if self.E_nu_typical is None:
            return (False, "SKIP: E_nu_typical not available")
        match = re.search(r'([-+]?\d+(?:\.\d+)?)(?:\s*MeV)?', self.E_nu_typical)
        if not match:
            return (False, "Could not extract numeric value")
        try:
            value = float(match.group(1))
        except ValueError:
            return (False, "Invalid number")
        if value < 0:
            return (False, f"Negative neutrino energy ({value}) not physical")
        if value > 2.0:
            return (False, f"Neutrino energy {value} seems too high for typical beta decay")
        return (True, f"Neutrino energy {value} MeV within expected range 0–2 MeV")

    def test_modeling_tau_scaling_proportionality(self):
        if self.tau_scaling is None:
            return (False, "SKIP: tau_scaling not available")
        if "Z" not in self.tau_scaling:
            return (False, "tau_scaling does not contain Z")
        if not re.search(r'(-4|−4)', self.tau_scaling):
            return (False, f"tau_scaling '{self.tau_scaling}' does not indicate Z^{-4}")
        return (True, "tau_scaling expression matches Z^{-4}")

    def test_modeling_oxygen_term_symbol(self):
        if self.config is None:
            return (False, "SKIP: config not available")
        if not re.search(r'3P[^a-z]*2', self.config):
            if not ('3' in self.config and 'P' in self.config and '2' in self.config):
                return (False, "Config does not hint at ³P₂ term symbol")
        return (True, "Config suggests ³P₂ term symbol")

    # ----------------------------------------------------------------------
    # MONOTONICITY TESTS – explicitly perturb an input and compare output
    # ----------------------------------------------------------------------
    def test_monotonicity_tau_scaling(self):
        """Check that increasing Z decreases lifetime by the correct factor: τ ∝ Z^{-4}."""
        if self.tau_scaling is None:
            return (False, "SKIP: tau_scaling not available")
        expr_str = self.tau_scaling.replace('^', '**').replace('{', '(').replace('}', ')')
        expr_str = re.sub(r'\s+', '', expr_str)
        try:
            expr = sp.sympify(expr_str)
        except (sp.SympifyError, TypeError):
            return (False, "Cannot parse tau_scaling")
        Z = sp.symbols('Z')
        # Baseline: Z=1, perturbed: Z=1.1 (10% increase)
        try:
            tau0 = float(expr.subs(Z, 1.0))
            tau1 = float(expr.subs(Z, 1.1))
        except Exception as e:
            return (False, f"Failed to evaluate expression at Z=1, 1.1: {e}")
        if not math.isfinite(tau0) or not math.isfinite(tau1):
            return (False, "Lifetime values are not finite")
        if tau0 <= 0 or tau1 <= 0:
            return (False, "Lifetime values must be positive")
        # Expected ratio: tau1/tau0 = (Z0/Z1)^4 = (1/1.1)^4 ≈ 0.683
        expected_ratio = (1.0 / 1.1)**4
        actual_ratio = tau1 / tau0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (False, f"Ratio tau1/tau0 = {actual_ratio:.4f}, expected {expected_ratio:.4f} (from Z^{-4})")
        return (True, f"tau(Z=1.1)/tau(Z=1) = {actual_ratio:.4f} ≈ expected {expected_ratio:.4f}")

    # ----------------------------------------------------------------------
    # ANSWER SANITY TEST (weak only – at most one such method)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basics(self):
        """Weak sanity checks for all three sub-answers (no computed expectations)."""
        checks = []
        # 1) Neutrino energy: string containing "MeV" and a positive number
        if self.E_nu_typical is not None:
            if not isinstance(self.E_nu_typical, str):
                checks.append("E_nu_typical not string")
            else:
                if "MeV" not in self.E_nu_typical:
                    checks.append("E_nu_typical missing 'MeV'")
                match = re.search(r'(\d+(?:\.\d+)?)', self.E_nu_typical)
                if match:
                    try:
                        v = float(match.group(1))
                        if v <= 0:
                            checks.append("E_nu_typical non-positive")
                    except ValueError:
                        checks.append("E_nu_typical invalid number")
                else:
                    checks.append("E_nu_typical missing numeric part")
        else:
            checks.append("E_nu_typical missing")

        # 2) tau_scaling: string containing Z and some negative exponent
        if self.tau_scaling is not None:
            if not isinstance(self.tau_scaling, str):
                checks.append("tau_scaling not string")
            else:
                if "Z" not in self.tau_scaling:
                    checks.append("tau_scaling missing Z")
                if not re.search(r'-4|−4', self.tau_scaling):
                    checks.append("tau_scaling missing exponent -4")
        else:
            checks.append("tau_scaling missing")

        # 3) config: string containing 1s, 2s, 2p and some numbers
        if self.config is not None:
            if not isinstance(self.config, str):
                checks.append("config not string")
            else:
                if not ('1s' in self.config and '2s' in self.config and '2p' in self.config):
                    checks.append("config missing required orbitals")
        else:
            checks.append("config missing")

        # 4) S, L, J: positive finite numbers
        for name, val in [('S', self.S), ('L', self.L), ('J', self.J)]:
            if val is None:
                checks.append(f"{name} missing")
            elif not isinstance(val, (int, float)):
                checks.append(f"{name} not numeric")
            elif not math.isfinite(val) or val <= 0:
                checks.append(f"{name} not positive finite")

        if checks:
            return (False, "; ".join(checks))
        return (True, "All weak sanity checks passed")

    # ----------------------------------------------------------------------
    # RUNNER
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
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)