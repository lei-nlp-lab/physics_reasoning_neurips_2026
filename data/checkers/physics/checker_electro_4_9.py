import sympy as sp
import math
import traceback
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract results safely, default to None
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        
        # Interface keys expected from the emit contract
        self.helmholtz_phi = results.get('helmholtz_phi', None)
        self.helmholtz_A = results.get('helmholtz_A', None)
        self.phi_omega = results.get('phi_omega', None)
        self.A_omega = results.get('A_omega', None)

        # Also store the full instance for accessing explanation_steps if needed
        self.solution_instance = solution_instance
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ==================== Interface Tests ====================
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        missing = []
        for key in ['helmholtz_phi', 'helmholtz_A', 'phi_omega', 'A_omega']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_helmholtz_phi_type(self) -> Tuple[bool, str]:
        if self.helmholtz_phi is None:
            return (False, "SKIP: helmholtz_phi not available")
        if not isinstance(self.helmholtz_phi, str):
            return (False, f"helmholtz_phi should be a string, got {type(self.helmholtz_phi).__name__}")
        return (True, "helmholtz_phi is a string")

    def test_interface_helmholtz_A_type(self) -> Tuple[bool, str]:
        if self.helmholtz_A is None:
            return (False, "SKIP: helmholtz_A not available")
        if not isinstance(self.helmholtz_A, str):
            return (False, f"helmholtz_A should be a string, got {type(self.helmholtz_A).__name__}")
        return (True, "helmholtz_A is a string")

    def test_interface_phi_omega_type(self) -> Tuple[bool, str]:
        if self.phi_omega is None:
            return (False, "SKIP: phi_omega not available")
        if not isinstance(self.phi_omega, str):
            return (False, f"phi_omega should be a string, got {type(self.phi_omega).__name__}")
        return (True, "phi_omega is a string")

    def test_interface_A_omega_type(self) -> Tuple[bool, str]:
        if self.A_omega is None:
            return (False, "SKIP: A_omega not available")
        if not isinstance(self.A_omega, str):
            return (False, f"A_omega should be a string, got {type(self.A_omega).__name__}")
        return (True, "A_omega is a string")

    # ==================== Invariant Tests ====================
    def test_invariant_helmholtz_phi_sign(self) -> Tuple[bool, str]:
        """
        Physically, (nabla^2 + K^2) phi_omega = -rho_omega/epsilon_0.
        Check the string contains the correct sign and components.
        """
        if self.helmholtz_phi is None:
            return (False, "SKIP: helmholtz_phi not available")
        expr = self.helmholtz_phi
        needed = ['nabla^2', 'K^2', 'phi_omega', 'rho_omega', 'epsilon_0']
        for sym in needed:
            if sym not in expr:
                return (False, f"Expected symbol '{sym}' missing in helmholtz_phi: {expr}")
        if '-rho_omega/epsilon_0' not in expr and '-\\frac{\\rho_\\omega}{\\epsilon_0}' not in expr:
            if 'rho_omega' in expr:
                idx = expr.find('rho_omega')
                before = expr[:idx].rstrip()
                if not before.endswith('-') and not before.endswith('- \\\\frac') and not before.endswith('-\\\\frac'):
                    if '-rho_omega/epsilon_0' not in expr and '-\\frac{\\rho_\\omega}{\\epsilon_0}' not in expr:
                        return (False, f"Expected negative sign before rho_omega term in helmholtz_phi: {expr}")
        return (True, "helmholtz_phi has correct sign and components")

    def test_invariant_helmholtz_A_sign(self) -> Tuple[bool, str]:
        if self.helmholtz_A is None:
            return (False, "SKIP: helmholtz_A not available")
        expr = self.helmholtz_A
        needed = ['nabla^2', 'K^2', 'A_omega', 'J_omega', 'mu_0']
        for sym in needed:
            if sym not in expr:
                return (False, f"Expected symbol '{sym}' missing in helmholtz_A: {expr}")
        if '-mu_0' not in expr and '-\\mu_0' not in expr:
            if 'J_omega' in expr:
                idx = expr.find('J_omega')
                before = expr[:idx].rstrip()
                if not before.endswith('-') and not before.endswith('- \\\\mu_0') and not before.endswith('-\\\\mu_0'):
                    if '-mu_0 J_omega' not in expr and '-\\mu_0 J_\\omega' not in expr:
                        return (False, f"Expected negative sign before mu_0 J_omega term in helmholtz_A: {expr}")
        return (True, "helmholtz_A has correct sign and components")

    def test_invariant_phi_omega_form(self) -> Tuple[bool, str]:
        if self.phi_omega is None:
            return (False, "SKIP: phi_omega not available")
        expr = self.phi_omega
        if '4*pi*epsilon_0' not in expr and '4\\pi \\epsilon_0' not in expr and '4\\pi\\epsilon_0' not in expr:
            return (False, f"Missing factor 1/(4π ε₀) in phi_omega: {expr}")
        if 'e^{iK' not in expr and 'exp(i*K' not in expr and 'e^{i\\,K' not in expr:
            return (False, f"Missing exponential e^{{iK|r-r'|}} in phi_omega: {expr}")
        if '|r-r' not in expr and '|\\mathbf{r} - \\mathbf{r}' not in expr:
            return (False, f"Missing denominator |r-r'| in phi_omega: {expr}")
        return (True, "phi_omega has correct integral form")

    def test_invariant_A_omega_form(self) -> Tuple[bool, str]:
        if self.A_omega is None:
            return (False, "SKIP: A_omega not available")
        expr = self.A_omega
        if 'mu_0/(4*pi)' not in expr and '\\frac{\\mu_0}{4\\pi}' not in expr and '\\mu_0/ (4\\pi)' not in expr:
            if 'mu_0/(4\\pi)' not in expr:
                return (False, f"Missing factor μ₀/(4π) in A_omega: {expr}")
        if 'e^{iK' not in expr and 'exp(i*K' not in expr:
            return (False, f"Missing exponential e^{{iK|r-r'|}} in A_omega: {expr}")
        if '|r-r' not in expr and '|\\mathbf{r} - \\mathbf{r}' not in expr:
            return (False, f"Missing denominator |r-r'| in A_omega: {expr}")
        if 'J_omega' not in expr and '\\mathbf{J}_\\omega' not in expr:
            return (False, f"Missing J_omega in A_omega: {expr}")
        return (True, "A_omega has correct integral form")

    def test_invariant_continuity_law(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if isinstance(self.explanation_steps, list):
            steps_str = ' '.join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            steps_str = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps not in expected format")
        import re
        patterns = [
            r'nabla\s*cdot\s*J_omega\s*-\s*i\s*omega\s*rho_omega\s*=\s*0',
            r'\\nabla\\cdot\\mathbf{J}_\\omega\s*-\s*i\\omega\\rho_\\omega\s*=\s*0',
            r'continuity',
            r'charge-current conservation'
        ]
        for pat in patterns:
            if re.search(pat, steps_str, re.IGNORECASE):
                return (True, "Continuity equation found in explanation_steps")
        return (False, "Continuity equation not found in explanation_steps")

    # ==================== Modeling / Trace Tests ====================
    def test_trace_far_zone_approximation(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps mentions the far-zone approximation
        and the simplification |r - r'| ≈ r - (r·r')/r.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if isinstance(self.explanation_steps, list):
            steps_str = ' '.join(str(s) for s in self.explanation_steps)
        else:
            steps_str = str(self.explanation_steps)
        if 'far zone' in steps_str.lower() or 'r \\to \\infty' in steps_str or 'far-zone' in steps_str.lower():
            if '\\frac{\\mathbf{r} \\cdot \\mathbf{r}' in steps_str or 'r · r' in steps_str:
                return (True, "Far-zone approximation present")
            return (False, "Far-zone mentioned but missing dot product approximation")
        return (False, "Far-zone approximation not found in explanation_steps")

    def test_trace_green_function_mention(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps mentions the Green's function for Helmholtz
        equation and the outgoing wave condition.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if isinstance(self.explanation_steps, list):
            steps_str = ' '.join(str(s) for s in self.explanation_steps)
        else:
            steps_str = str(self.explanation_steps)
        # Check for Green's function or Helmholtz equation
        if 'Green' in steps_str or 'Helmholtz' in steps_str:
            # Also check outgoing wave or Sommerfeld
            if 'outgoing' in steps_str.lower() or 'Sommerfeld' in steps_str or 'radiation' in steps_str.lower():
                return (True, "Green's function and outgoing wave condition mentioned")
            else:
                # Still accept Green's function mention alone as important modeling step
                return (True, "Green's function mentioned")
        return (False, "Neither Green's function nor Helmholtz equation found in explanation_steps")

    # ==================== Monotonicity Test ====================
    def test_monotonicity_green_func_distance(self) -> Tuple[bool, str]:
        """
        The magnitude of the Green's function kernel, 1/|r - r'|, decreases
        monotonically with distance. This property underlies the potential
        expressions. We check it numerically: baseline at d=1, perturb to d=1.1.
        """
        d = sp.Symbol('d', positive=True)
        kernel = 1 / d
        baseline = kernel.subs(d, 1.0).evalf()
        perturbed = kernel.subs(d, 1.1).evalf()
        passed = float(perturbed) < float(baseline)
        msg = f"Kernel magnitude at d=1: {baseline:.4f}, at d=1.1: {perturbed:.4f}, decrease: {passed}"
        return (passed, msg)

    # ==================== Answer Sanity (weak) ====================
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that the integral expressions are non-empty and plausible.
        """
        if self.phi_omega is None or self.A_omega is None:
            return (False, "SKIP: one of the potentials missing")
        if not self.phi_omega.strip():
            return (False, "phi_omega expression is empty")
        if not self.A_omega.strip():
            return (False, "A_omega expression is empty")
        if 'int' not in self.phi_omega and '\\int' not in self.phi_omega:
            return (False, "phi_omega does not contain integral sign")
        if 'int' not in self.A_omega and '\\int' not in self.A_omega:
            return (False, "A_omega does not contain integral sign")
        return (True, "Potentials have non-empty integral expressions")

    # ==================== Runner ====================
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)