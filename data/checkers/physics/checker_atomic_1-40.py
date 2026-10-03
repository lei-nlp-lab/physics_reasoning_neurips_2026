import math
import traceback

class PhysicsChecker:
    """Audit checker for parity-violating hydrogen 2S1/2 state decay."""

    def __init__(self, solution_instance):
        # Extract results from solution_instance, supporting both dict and attribute interfaces
        self.allowed_decay = None
        self.M = None

        # First try the results dict (most common)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.allowed_decay = solution_instance.results.get('allowed_decay', None)
            self.M = solution_instance.results.get('M', None)
        else:
            # Fallback to direct attribute access
            self.allowed_decay = getattr(solution_instance, 'allowed_decay', None)
            self.M = getattr(solution_instance, 'M', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both required keys exist and are not None."""
        if self.allowed_decay is None:
            return (False, "Missing 'allowed_decay' in results")
        if self.M is None:
            return (False, "Missing 'M' in results")
        return (True, "Both required keys present")

    def test_interface_allowed_decay_type(self):
        """Check that allowed_decay is a string."""
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        if not isinstance(self.allowed_decay, str):
            return (False, f"allowed_decay is not a string, got {type(self.allowed_decay)}")
        return (True, "allowed_decay is a string")

    def test_interface_M_type(self):
        """Check that M is numeric (int or float) and finite."""
        if self.M is None:
            return (False, "SKIP: M not available")
        if not isinstance(self.M, (int, float)):
            return (False, f"M is not numeric, got {type(self.M)}")
        if math.isnan(self.M) or math.isinf(self.M):
            return (False, "M is NaN or inf")
        return (True, "M is a finite number")

    # ----------------------------------------------------------------------
    # Physical invariance & theorem checks
    # ----------------------------------------------------------------------
    def test_invariant_parity_selection(self):
        """E1 transition requires parity change; the allowed decay must go through the p-wave part (odd parity) to the s-wave (even)."""
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        adm = self.allowed_decay.strip()
        if 'P' not in adm or 'S' not in adm:
            return (False, f"allowed_decay '{adm}' does not contain both 'P' and 'S' (indicating p->s transition)")
        parts = adm.split('->')
        if len(parts) != 2:
            return (False, f"allowed_decay '{adm}' does not contain exactly one '->' separating initial and final states")
        initial, final = parts[0].strip(), parts[1].strip()
        if 'P' not in initial:
            return (False, f"Initial state '{initial}' does not contain 'P' (meaning l=1)")
        if 'S' not in final:
            return (False, f"Final state '{final}' does not contain 'S' (meaning l=0)")
        return (True, "Parity selection rules satisfied: p (odd) to s (even) transition")

    def test_invariant_e1_selection_rules(self):
        """Check that j is conserved (Δj=0) and that the transition is E1 (containing 'E1')."""
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        adm = self.allowed_decay.strip()
        if 'E1' not in adm:
            return (False, f"allowed_decay '{adm}' does not mention 'E1' (electric dipole)")
        parts = adm.split('->')
        if len(parts) != 2:
            return (False, "Cannot parse allowed_decay into two parts")
        initial, final = parts[0].strip(), parts[1].strip()
        if '_{1/2}' not in initial or '_{1/2}' not in final:
            return (False, f"Not both states have j=1/2 subscript: initial='{initial}', final='{final}'")
        return (True, "Δj=0 and E1 label present")

    def test_invariant_s_state_forbidden_direct(self):
        """The pure s‑state (ε→0) gives zero E1 matrix element because initial and final parity are both even."""
        if self.M is None:
            return (False, "SKIP: M not available")
        if not math.isclose(self.M, 0.0, rel_tol=0, abs_tol=1e-15):
            return (False, f"M = {self.M} is not zero; expected integer 0 for pure s-state")
        return (True, "M=0, consistent with parity selection rule forbidding s→s E1 transition")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency checks (at least two)
    # ----------------------------------------------------------------------
    def test_modeling_allowed_decay_specific_channel(self):
        """The allowed decay should be exactly the one given in the problem's reasoning plan."""
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        expected_base = "2P_{1/2} -> 1S_{1/2} E1"
        adm = self.allowed_decay.replace(" ", "")
        expected = expected_base.replace(" ", "")
        if adm != expected:
            return (False, f"allowed_decay '{self.allowed_decay}' does not match expected '{expected_base}'")
        return (True, "allowed_decay matches the physically expected channel")

    def test_modeling_allowed_decay_contains_physical_states(self):
        """Modeling check: the decay must involve the correct atomic levels (n=2 P and n=1 S)."""
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        adm = self.allowed_decay.strip()
        parts = adm.split('->')
        if len(parts) != 2:
            return (False, "Cannot split allowed_decay into two parts")
        initial, final = parts[0].strip(), parts[1].strip()
        # Check initial contains "2" and "P"
        if '2' not in initial or 'P' not in initial:
            return (False, f"Initial state '{initial}' does not contain '2' (n=2) or 'P' (l=1)")
        # Check final contains "1" and "S"
        if '1' not in final or 'S' not in final:
            return (False, f"Final state '{final}' does not contain '1' (n=1) or 'S' (l=0)")
        return (True, "Allowed decay correctly uses n=2 and n=1 with the right orbital labels")

    # ----------------------------------------------------------------------
    # Monotonicity check
    # ----------------------------------------------------------------------
    def test_monotonicity_allowed_decay_state_perturbation(self):
        """
        Perturb the initial state from p-wave (l=1) to s-wave (l=0) in the
        allowed_decay string.  The perturbed transition must be forbidden by
        E1 parity selection rules (baseline is allowed).  This checks the
        monotonic dependence on the parity of the initial state.
        """
        if self.allowed_decay is None:
            return (False, "SKIP: allowed_decay not available")
        parts = self.allowed_decay.split('->')
        if len(parts) != 2:
            return (False, "Cannot parse allowed_decay to perform perturbation")
        initial_base, final_base = parts[0].strip(), parts[1].strip()
        # Check that baseline is a p->s transition (as expected)
        if 'P' not in initial_base or 'S' not in final_base:
            return (False, f"Baseline initial state '{initial_base}' does not contain 'P' or final state '{final_base}' does not contain 'S'")
        # Perturbation: replace initial state's 'P' with 'S' (change l from 1 to 0)
        initial_perturbed = initial_base.replace('P', 'S', 1)
        perturbed_transition = f"{initial_perturbed} -> {final_base}"
        # The perturbed transition (s->s) should be parity-forbidden.
        perturbed_parts = perturbed_transition.split('->')
        if len(perturbed_parts) != 2:
            return (False, "Perturbed transition parse error")
        init_pert, final_pert = perturbed_parts[0].strip(), perturbed_parts[1].strip()
        # Both should be S states (no P in either)
        if 'P' in init_pert or 'P' in final_pert:
            return (False, f"Perturbation resulted in unexpected state labels: {perturbed_transition}")
        # The perturbed transition is not allowed for E1 (same parity). This is a discrete
        # monotonic change: moving initial from odd to even parity eliminates the decay.
        return (True, f"Perturbation of initial state from p to s yields forbidden transition '{perturbed_transition}'")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, do not re-derive)
    # ----------------------------------------------------------------------
    def test_answer_sanity_M_zero(self):
        """For the pure s-state (no mixing), the E1 matrix element must be exactly zero."""
        if self.M is None:
            return (False, "SKIP: M not available")
        # Exact zero: no tolerance needed (integer)
        if self.M != 0:
            return (False, f"M = {self.M} is not zero (expected integer 0)")
        return (True, "M = 0, consistent with the answer for ε→0")

    # ----------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)