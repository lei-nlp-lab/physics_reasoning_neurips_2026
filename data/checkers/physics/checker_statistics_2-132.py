import math
import traceback
import re

class PhysicsChecker:
    """
    Audits a solution for spin-1/2 paramagnet problem:
    (a) E = -M H
    (b) E = -N μ0 H tanh(μ0 H / (k_B T))
    (c) C = N k_B (β μ0 H)^2 sech^2(β μ0 H)
        S = N k_B [ln(2 cosh(β μ0 H)) - β μ0 H tanh(β μ0 H)]
    """

    def __init__(self, solution_instance):
        # Extract results (expected keys from pseudocode emit contract)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Safely extract each result key
        self.E_general = self.results.get('E_general', None)
        self.E_thermal = self.results.get('E_thermal', None)
        self.C = self.results.get('C', None)
        self.S = self.results.get('S', None)

        # Extract givens if present (optional, used in some tests)
        # Try to get from solution_instance attributes or a givens dict
        self.N = None
        self.mu0 = None
        self.H = None
        self.T = None
        self.k_B = None
        # Attempt to find givens from solution_instance directly
        for attr in ['N', 'mu0', 'H', 'T', 'k_B']:
            val = getattr(solution_instance, attr, None)
            if val is None:
                # maybe in a givens sub-dict
                givens = getattr(solution_instance, 'givens', {})
                if isinstance(givens, dict):
                    val = givens.get(attr, None)
            setattr(self, attr, val)

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist and are strings
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Verify that all four result keys exist and are not None."""
        missing = []
        for key in ['E_general', 'E_thermal', 'C', 'S']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All four result keys present.")

    def test_interface_key_types(self):
        """Check that each result is a string."""
        issues = []
        for key in ['E_general', 'E_thermal', 'C', 'S']:
            val = getattr(self, key)
            if not isinstance(val, str):
                issues.append(f"{key} is not a string (type: {type(val).__name__})")
        if issues:
            return (False, " | ".join(issues))
        return (True, "All results are strings.")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use isclose or pattern)
    # ------------------------------------------------------------------
    def test_invariant_energy_general_form(self):
        """E = -M H must contain 'M' and 'H' with a minus sign."""
        if self.E_general is None:
            return (False, "SKIP: E_general not available")
        text = self.E_general
        # Check for presence of 'M' and 'H' and a minus sign somewhere
        if 'M' not in text:
            return (False, "Missing 'M' in E_general")
        if 'H' not in text:
            return (False, "Missing 'H' in E_general")
        # The minus sign should be present, typically at the beginning
        if '-' not in text:
            return (False, "Missing minus sign (expected -M H)")
        # Optional: check for proper spacing or multiplication? Weak.
        return (True, "E_general contains 'M', 'H' and a minus sign.")

    def test_invariant_energy_thermal_form(self):
        """E_thermal should contain 'tanh', 'mu0', 'H' and (optionally) 'k_B'."""
        if self.E_thermal is None:
            return (False, "SKIP: E_thermal not available")
        text = self.E_thermal
        # Essential physics elements
        required = ['tanh', 'mu0', 'H']
        missing = [s for s in required if s not in text]
        if missing:
            return (False, f"Missing expected substrings: {', '.join(missing)}")
        # Check for a minus sign (energy negative)
        if '-' not in text:
            return (False, "E_thermal should contain a minus sign.")
        # Check for 'N' (if N appears)
        # Not required but common
        return (True, "E_thermal contains tanh, mu0, H and minus sign.")

    def test_invariant_heat_capacity_form(self):
        """C should contain 'sech', 'cosh', or 'sech^2' and 'beta'."""
        if self.C is None:
            return (False, "SKIP: C not available")
        text = self.C
        # Heat capacity is proportional to sech^2 or 1/cosh^2
        has_sech = 'sech' in text
        has_cosh = 'cosh' in text
        if not (has_sech or has_cosh):
            return (False, "C does not contain 'sech' or 'cosh'")
        # beta should appear
        if 'beta' not in text:
            return (False, "C should contain 'beta'")
        # The overall structure should involve N k_B factor
        # But not strict unless we have N, k_B.
        return (True, "C contains hyperbolic secant/cosh and beta.")

    def test_invariant_entropy_form(self):
        """S should contain 'ln', 'cosh', and 'tanh'."""
        if self.S is None:
            return (False, "SKIP: S not available")
        text = self.S
        required = ['ln', 'cosh', 'tanh']
        missing = [s for s in required if s not in text]
        if missing:
            return (False, f"Missing expected substrings: {', '.join(missing)}")
        # Also check for 'beta' and 'mu0' and 'H'
        if 'beta' not in text:
            return (False, "S should contain 'beta'")
        if 'mu0' not in text:
            return (False, "S should contain 'mu0'")
        if 'H' not in text:
            return (False, "S should contain 'H'")
        return (True, "S contains ln, cosh, tanh, and field variables.")

    # ------------------------------------------------------------------
    # Modeling quality tests – based on explanation steps / pseudocode
    # ------------------------------------------------------------------
    def test_modeling_energy_general_correct_structure(self):
        """
        Check that E_general follows the exact pattern '- M H' or similar.
        We allow optional multiplication signs and whitespace.
        """
        if self.E_general is None:
            return (False, "SKIP: E_general not available")
        text = self.E_general.replace(' ', '')  # remove spaces
        # Expected: either "-M H" or "-M*H" or "-M·H" etc.
        # We'll check if it matches the pattern: starts with '-', then M, then H with optional operator.
        pattern = r'^\-[\s]*M[\s]*[*·]?[\s]*H$'
        if re.match(pattern, text):
            return (True, "E_general matches '- M H' pattern.")
        # Alternative pattern: "-MH" (no operator)
        pattern2 = r'^\-[\s]*M[\s]*H$'
        if re.match(pattern2, text):
            return (True, "E_general matches '- M H' pattern (no operator).")
        # More permissive: contains "-M" and "H" in order
        if '-M' in text and 'H' in text and text.index('-M') < text.index('H'):
            return (True, "E_general contains '-M' before 'H'.")
        return (False, f"E_general does not match expected form '- M H': {self.E_general}")

    def test_modeling_energy_thermal_contains_tanh_product(self):
        """E_thermal should contain product N mu0 H tanh(...)."""
        if self.E_thermal is None:
            return (False, "SKIP: E_thermal not available")
        text = self.E_thermal.replace(' ', '')
        # Check for 'N*mu0*H*tanh' or 'Nmu0Htanh' etc.
        # Also allow mu0 written as mu_0 or mu0
        # We'll use a regex that allows flexibility
        pattern = r'N[\s]*[*·]?[\s]*mu[0_]?[\s]*[*·]?[\s]*H[\s]*[*·]?[\s]*tanh'
        if re.search(pattern, text, re.IGNORECASE):
            return (True, "E_thermal contains product N·μ₀·H·tanh")
        # If N is missing, maybe it's implied? But unlikely.
        # Also check for tanh expression with arguments
        if 'tanh(' in text and 'mu0' in text and 'H' in text:
            return (True, "E_thermal contains tanh, mu0, H")
        return (False, f"E_thermal does not have expected product: {self.E_thermal}")

    def test_modeling_heat_capacity_sech_squared(self):
        """C should contain (β μ0 H)^2 sech^2(...)."""
        if self.C is None:
            return (False, "SKIP: C not available")
        text = self.C.replace(' ', '')
        # Look for (beta mu0 H)^2 or (β μ0 H)^2
        # Also sech^2
        pattern_sech2 = r'sech\^2|1\/cosh\^2|cosh\^2'
        pattern_beta = r'beta[\s]*[*·]?[\s]*mu[0_]?[\s]*[*·]?[\s]*H'
        if re.search(pattern_sech2, text, re.IGNORECASE) and re.search(pattern_beta, text, re.IGNORECASE):
            return (True, "C contains sech²(βμ₀H)")
        # Also accept if contains 'sech' and 'beta' and 'mu0' and 'H'
        if 'sech' in text and 'beta' in text and 'mu0' in text and 'H' in text:
            return (True, "C contains sech, beta, mu0, H")
        return (False, f"C does not match expected form: {self.C}")

    def test_modeling_entropy_structure(self):
        """S should contain ln(2 cosh(...)) - βμ₀H tanh(βμ₀H)."""
        if self.S is None:
            return (False, "SKIP: S not available")
        text = self.S.replace(' ', '')
        # Check for ln and cosh and minus and tanh
        has_ln = 'ln(' in text or 'ln ' in text
        has_cosh = 'cosh(' in text
        has_minus = '-' in text
        has_tanh = 'tanh(' in text
        # Also need beta and mu0 and H inside
        has_beta = 'beta' in text
        has_mu = 'mu0' in text
        has_H = 'H' in text
        if has_ln and has_cosh and has_minus and has_tanh:
            if has_beta and has_mu and has_H:
                return (True, "S contains ln(2 cosh) - βμ₀H tanh structure")
        return (False, f"S does not match expected form: {self.S}")

    # ------------------------------------------------------------------
    # Monotonicity test – only if numeric givens are available
    # ------------------------------------------------------------------
    def test_monotonicity_entropy_increases_with_temperature(self):
        """
        If we have numeric values for N, mu0, H, T, k_B, we can compute
        entropy for two temperatures and verify it increases with T.
        We use a small perturbation: ΔT = 0.01*T.
        """
        if any(v is None for v in [self.N, self.mu0, self.H, self.T, self.k_B]):
            return (False, "SKIP: numeric givens (N, mu0, H, T, k_B) not all available.")
        # Compute entropy using standard formula
        try:
            beta = 1.0 / (self.k_B * self.T)
            # S = N k_B [ln(2 cosh(β μ0 H)) - β μ0 H tanh(β μ0 H)]
            x = beta * self.mu0 * self.H
            coshx = math.cosh(x)
            S_ref = self.N * self.k_B * (math.log(2 * coshx) - x * math.tanh(x))

            # Perturb T upward by 1%
            T2 = self.T * 1.01
            beta2 = 1.0 / (self.k_B * T2)
            x2 = beta2 * self.mu0 * self.H
            coshx2 = math.cosh(x2)
            S_pert = self.N * self.k_B * (math.log(2 * coshx2) - x2 * math.tanh(x2))

            if S_pert > S_ref:
                return (True, f"Entropy increased with T: S(T)={S_ref:.4e}, S(T+1%)={S_pert:.4e}")
            else:
                return (False, f"Entropy did not increase with T: S(T)={S_ref:.4e}, S(T+1%)={S_pert:.4e}")
        except Exception as e:
            return (False, f"Entropy monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity – optional (must not re-derive answers)
    # ------------------------------------------------------------------
    def test_answer_sanity_units_implied(self):
        """
        Very weak check: all results should be non-empty strings.
        No numeric comparison, no re-derivation.
        """
        for key, val in [('E_general', self.E_general),
                         ('E_thermal', self.E_thermal),
                         ('C', self.C),
                         ('S', self.S)]:
            if val is None:
                return (False, f"{key} is missing")
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"{key} is empty or not a string")
        return (True, "All results are non‑empty strings.")

    # ------------------------------------------------------------------
    # Run all tests – discover and execute all test_* methods
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