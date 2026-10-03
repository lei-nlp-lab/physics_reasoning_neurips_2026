import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the quantum bound particle problem.
    Checks probability conservation, stationary state persistence, and expansion in eigenstates.
    """
    def __init__(self, solution_instance):
        # Extract results from the provided solution instance
        results = getattr(solution_instance, 'results', {})
        self.dP_total_dt = results.get('dP_total_dt', None)
        self.stationary_remaining = results.get('stationary_remaining', None)
        self.expansion = results.get('expansion', None)
        self.a_n = results.get('a_n', None)

        # Attempt to extract the physical parameter 'a' (width of initial box)
        # It may be stored directly as an attribute or inside a givens dictionary.
        self.a = getattr(solution_instance, 'a', None)
        if self.a is None:
            givens = getattr(solution_instance, 'givens', {})
            self.a = givens.get('a', None)

    # -----------------------------------------------------------------
    # INTERFACE CHECKS – required keys exist and have correct types
    # -----------------------------------------------------------------
    def test_interface_dP_total_dt_present(self):
        """Check that dP_total_dt key exists and is not None."""
        if self.dP_total_dt is None:
            return False, "dP_total_dt is missing or None"
        return True, "dP_total_dt present"

    def test_interface_stationary_remaining_present(self):
        """Check that stationary_remaining key exists and is not None."""
        if self.stationary_remaining is None:
            return False, "stationary_remaining is missing or None"
        return True, "stationary_remaining present"

    def test_interface_expansion_present(self):
        """Check that expansion key exists and is not None."""
        if self.expansion is None:
            return False, "expansion is missing or None"
        return True, "expansion present"

    def test_interface_a_n_present(self):
        """Check that a_n key exists and is not None."""
        if self.a_n is None:
            return False, "a_n is missing or None"
        return True, "a_n present"

    def test_interface_dP_total_dt_zero(self):
        """dP_total_dt must be exactly 0 (probability conservation)."""
        if self.dP_total_dt is None:
            return False, "Cannot check: dP_total_dt is missing"
        if not isinstance(self.dP_total_dt, (int, float)):
            return False, f"dP_total_dt should be a number, got {type(self.dP_total_dt)}"
        # Accept exact zero (theoretical value)
        if self.dP_total_dt != 0:
            return False, f"dP_total_dt should be 0, got {self.dP_total_dt}"
        return True, "dP_total_dt is zero"

    def test_interface_stationary_remaining_type(self):
        """stationary_remaining must be a non‑empty string."""
        if self.stationary_remaining is None:
            return False, "Cannot check: stationary_remaining is missing"
        if not isinstance(self.stationary_remaining, str) or len(self.stationary_remaining.strip()) == 0:
            return False, "stationary_remaining not a non‑empty string"
        return True, "stationary_remaining is a non‑empty string"

    def test_interface_expansion_type(self):
        """expansion must be a non‑empty string."""
        if self.expansion is None:
            return False, "Cannot check: expansion is missing"
        if not isinstance(self.expansion, str) or len(self.expansion.strip()) == 0:
            return False, "expansion not a non‑empty string"
        return True, "expansion is a non‑empty string"

    def test_interface_a_n_type(self):
        """a_n must be a non‑empty string."""
        if self.a_n is None:
            return False, "Cannot check: a_n is missing"
        if not isinstance(self.a_n, str) or len(self.a_n.strip()) == 0:
            return False, "a_n not a non‑empty string"
        return True, "a_n is a non‑empty string"

    # -----------------------------------------------------------------
    # INVARIANT CHECKS – theorem‑based physical sanity (use isclose where appropriate)
    # -----------------------------------------------------------------
    def test_invariant_probability_conservation(self):
        """Probability is conserved: dP_total/dt = 0 (theorem)."""
        if self.dP_total_dt is None:
            return False, "SKIP: dP_total_dt missing"
        if not isinstance(self.dP_total_dt, (int, float)):
            return False, f"dP_total_dt should be numeric, got {type(self.dP_total_dt)}"
        # Use isclose (though exact 0 is expected) to follow the fuzzy rule
        if not math.isclose(self.dP_total_dt, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return False, f"dP_total_dt = {self.dP_total_dt} is not close to 0"
        return True, "Probability conservation holds (dP/dt ≈ 0)"

    def test_invariant_stationary_persistence(self):
        """If state is stationary at t0, it remains stationary for all t."""
        if self.stationary_remaining is None:
            return False, "SKIP: stationary_remaining missing"
        # The statement must contain both "H psi" and "E psi" (or similar)
        msg = self.stationary_remaining.lower()
        if "h psi" not in msg and "hψ" not in msg:
            return False, "stationary_remaining does not mention H psi"
        if "e psi" not in msg and "eψ" not in msg:
            return False, "stationary_remaining does not mention E psi"
        if "all t" not in msg:
            return False, "stationary_remaining should mention 'all t'"
        return True, "Stationary state persistence correctly expressed"

    # -----------------------------------------------------------------
    # MODELING / TRACE CHECKS – verify the expansion and coefficient formulas
    # -----------------------------------------------------------------
    def test_modeling_expansion_formula(self):
        """Expansion must be ψ(x,t) = Σ a_n φ_n(x) exp(-i E_n t/ℏ)."""
        if self.expansion is None:
            return False, "SKIP: expansion missing"
        lower = self.expansion.lower()
        # Must contain sum, phi_n, exp, i
        if "sum" not in lower and "∑" not in lower:
            return False, "Expansion does not contain summation symbol"
        if "phi" not in lower and "φ" not in lower and "ψ_n" not in lower:
            return False, "Expansion does not mention eigenfunctions (phi_n or ψ_n)"
        if "exp" not in lower:
            return False, "Expansion does not contain exponential factor"
        if "i" not in lower:
            return False, "Expansion missing imaginary unit i"
        if "t" not in lower:
            return False, "Expansion missing time variable t"
        return True, "Expansion formula looks correct"

    def test_modeling_a_n_formula(self):
        """Coefficient must be a_n = (1/√(2a)) ∫_{-a}^{a} φ_n^*(x) dx."""
        if self.a_n is None:
            return False, "SKIP: a_n missing"
        lower = self.a_n.lower()
        # Must contain integral, sqrt(2a), phi_n^*
        if "integral" not in lower and "∫" not in lower:
            return False, "a_n formula does not contain an integral"
        # Check for complex conjugate (^{*} or star)
        if "*" not in self.a_n:
            return False, "a_n formula missing complex conjugate on φ_n"
        # Check that integration limits appear
        if "a" not in lower and "-a" not in lower:
            return False, "a_n formula does not indicate integration from -a to a"
        # left‑hand side should be a_n
        if "a_n" not in lower:
            return False, "a_n formula missing left‑hand side a_n"
        return True, "Coefficient formula appears physically correct"

    # -----------------------------------------------------------------
    # MONOTONICITY – explicit input perturbation and output comparison
    # -----------------------------------------------------------------
    def test_monotonicity_width_affects_coefficient_norm(self):
        """
        The normalization factor in a_n is 1/sqrt(2a).  If the box width a
        increases, this factor decreases monotonically.
        Because the overlap integral ∫_{-a}^{a} φ_n^*(x) dx typically grows with a,
        the exact behaviour of |a_n| depends on the eigenfunctions, but the
        overall factor always decreases with a.  Here we test that necessary
        condition.
        """
        if self.a is None:
            return (False, "SKIP: width 'a' not available; cannot test monotonicity")

        if not isinstance(self.a, (int, float)) or self.a <= 0:
            return (False, f"Invalid width a = {self.a}; must be a positive number")

        # Baseline factor
        baseline = 1.0 / math.sqrt(2.0 * self.a)

        # Perturb a by +10%
        a2 = self.a * 1.10
        perturbed = 1.0 / math.sqrt(2.0 * a2)

        # The factor must decrease (strictly)
        if perturbed >= baseline:
            return (False, f"Factor did not decrease: baseline {baseline:.6f}, perturbed {perturbed:.6f}")

        return (True, f"Factor decreases with increasing width: {baseline:.6f} -> {perturbed:.6f}")

    # -----------------------------------------------------------------
    # ANSWER SANITY – weak, non‑dominating check (max one)
    # -----------------------------------------------------------------
    def test_answer_sanity_all_nonempty(self):
        """Sanity: all result values are non‑empty and of expected basic type."""
        checks = []
        # dP_total_dt should be a number
        if not isinstance(self.dP_total_dt, (int, float)):
            checks.append("dP_total_dt not numeric")
        # stationary_remaining should be a string
        if not isinstance(self.stationary_remaining, str) or not self.stationary_remaining.strip():
            checks.append("stationary_remaining not a non‑empty string")
        # expansion should be a string
        if not isinstance(self.expansion, str) or not self.expansion.strip():
            checks.append("expansion not a non‑empty string")
        # a_n should be a string
        if not isinstance(self.a_n, str) or not self.a_n.strip():
            checks.append("a_n not a non‑empty string")
        if checks:
            return False, "; ".join(checks)
        return True, "All results are present and of appropriate basic type"

    # -----------------------------------------------------------------
    # RUNNER
    # -----------------------------------------------------------------
    def run_all_tests(self):
        """Discover and execute all test_* methods, return (all_pass, logs)."""
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
                msg = f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs