import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics audit for the stability of a positive test charge at the center of a cube
    of fixed positive charges. The central result is the trace of the Hessian of
    the electrostatic potential energy, which must be zero according to Laplace's equation,
    contradicting the positive definiteness required for stable equilibrium.
    """

    def __init__(self, solution_instance):
        # Extract results safely (default None for missing keys)
        results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        self.trace_Hessian = results.get('trace_Hessian', None)

        # Extract explanation steps if available
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract optional givens (not strictly needed but good to have)
        self.q = solution_instance.givens.get('q', None) if hasattr(solution_instance, 'givens') else None
        self.Q = solution_instance.givens.get('Q', None) if hasattr(solution_instance, 'givens') else None
        self.a = solution_instance.givens.get('a', None) if hasattr(solution_instance, 'givens') else None

        # The target answer (for weak sanity only)
        self.final_answer = solution_instance.results.get('answer', None) or \
                            getattr(solution_instance, 'answer', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_trace_Hessian_exists(self):
        if self.trace_Hessian is None:
            return (False, "SKIP: trace_Hessian not provided")
        if not isinstance(self.trace_Hessian, (int, float)):
            return (False, f"trace_Hessian should be numeric, got {type(self.trace_Hessian)}")
        return (True, "trace_Hessian key is present and numeric")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ----------------------------------------------------------------------
    def test_invariant_trace_Hessian_zero(self):
        if self.trace_Hessian is None:
            return (False, "SKIP: trace_Hessian not available")
        # The trace of the Hessian is the Laplacian of the potential, which is zero
        # by Laplace's equation at a charge-free point.
        if not math.isclose(self.trace_Hessian, 0.0, rel_tol=1e-3, abs_tol=1e-6):
            return (False, f"trace_Hessian should be zero, got {self.trace_Hessian}")
        return (True, "trace_Hessian is zero within tolerance")

    def test_invariant_laplace_contradiction(self):
        """
        From Earnshaw's theorem: in a charge-free region, ∇²U = 0.
        For a local minimum, the Hessian must be positive definite, which would imply
        a positive trace. But trace = ∇²U = 0, contradiction → no stable equilibrium.
        """
        if self.trace_Hessian is None:
            return (False, "SKIP: trace_Hessian not available")
        # Check that trace is numerically zero, as required by Laplace.
        # The contradiction is logically implied.
        if not math.isclose(self.trace_Hessian, 0.0, rel_tol=1e-3, abs_tol=1e-6):
            return (False, "Laplace's equation demands trace=0, but value differs")
        return (True, "Trace is zero, consistent with Laplace; equilibrium cannot be stable")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (using explanation_steps)
    # ----------------------------------------------------------------------
    def test_modeling_earnshaw_reference(self):
        """
        Check that explanation steps mention Earnshaw's theorem or Laplace's equation.
        This ensures the solver understood the key physics.
        """
        if self.explanation_steps is None or not isinstance(self.explanation_steps, str):
            # Could also be a list of strings; handle gracefully
            if isinstance(self.explanation_steps, list):
                combined = " ".join([str(s) for s in self.explanation_steps])
            else:
                return (False, "SKIP: explanation_steps not available")
        else:
            combined = self.explanation_steps
        combined_lower = combined.lower()
        # Either "earnshaw" or "laplace" should appear
        has_earnshaw = "earnshaw" in combined_lower
        has_laplace = "laplace" in combined_lower
        if not (has_earnshaw or has_laplace):
            return (False, "Explanation does not reference Earnshaw's theorem or Laplace's equation")
        return (True, "Explanation properly invokes Earnshaw/Laplace theorem")

    def test_modeling_emit_contract(self):
        """
        Verify that the pseudocode emit contract for trace_Hessian is satisfied.
        The contract states: results['trace_Hessian'] = 0.
        """
        if self.trace_Hessian is None:
            return (False, "SKIP: trace_Hessian not provided")
        if not math.isclose(self.trace_Hessian, 0.0, abs_tol=1e-6):
            return (False, f"Emit contract violation: expected 0, got {self.trace_Hessian}")
        return (True, "Emit contract satisfied: trace_Hessian = 0")

    # ----------------------------------------------------------------------
    # Monotonicity checks (trend analysis via input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_trace_invariant_under_scaling(self):
        """
        The trace of the Hessian is zero regardless of the charge magnitudes (q, Q)
        because Laplace's equation is homogeneous. Therefore, perturbing q or Q
        should not change the trace (it stays zero).
        We cannot recompute here, but we can at least note that if the trace were
        nonzero, scaling charges would change it linearly, which would violate Laplace.
        This test is a weak sanity that the trace is zero and thus scaling invariant.
        """
        if self.trace_Hessian is None:
            return (False, "SKIP: trace_Hessian not available")
        # If trace_Hessian is not zero, then scaling would produce a different value,
        # but we can't test that without recomputation. Nonetheless, the result being
        # zero is consistent with the invariance property.
        if not math.isclose(self.trace_Hessian, 0.0, abs_tol=1e-6):
            return (False, "Nonzero trace would not be scaling-invariant")
        return (True, "Trace is zero, consistent with invariance under charge scaling")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (at most one, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: the final answer should be a string indicating instability.
        We only check that the answer exists, is a string, contains "no" (or equivalent),
        and does not claim stability.
        """
        if self.final_answer is None:
            return (False, "SKIP: final answer not available")
        if not isinstance(self.final_answer, str) or len(self.final_answer) < 5:
            return (False, "Final answer is missing or too short")
        # Accept common patterns: "No", "cannot", "unstable"
        lower = self.final_answer.lower()
        has_negative = any(word in lower for word in ["no", "cannot", "not stable", "unstable", "saddle"])
        if not has_negative:
            return (False, f"Answer does not indicate instability: '{self.final_answer}'")
        return (True, "Answer correctly indicates no stable equilibrium")

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