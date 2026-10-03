import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract E and B vectors (list of 3) – accept None if not available
        self.E = getattr(solution_instance, 'E', None)
        self.B = getattr(solution_instance, 'B', None)

        # Results dictionary (should contain 'invariant1' and 'invariant2')
        self.results = getattr(solution_instance, 'results', {})

        # Velocity of boost (optional)
        self.v = getattr(solution_instance, 'v', None)

        # Speed of light (optional)
        self.c = getattr(solution_instance, 'c', None)

        # Field tensor as 4x4 list of lists (optional)
        self.F = getattr(solution_instance, 'F', None)

        # Metric signature (optional, default Minkowski)
        self.eta = getattr(solution_instance, 'eta', [1, -1, -1, -1])

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def dot(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def cross(v1, v2):
        return [
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ]

    @staticmethod
    def norm_sq(v):
        return PhysicsChecker.dot(v, v)

    @staticmethod
    def epsilon(mu, nu, rho, sigma):
        """Levi-Civita symbol with epsilon[0,1,2,3] = 1.
        Returns 0 if any index repeated or out of [0,1,2,3]."""
        indices = [mu, nu, rho, sigma]
        if len(set(indices)) != 4:
            return 0
        # Calculate sign of permutation
        sign = 1
        for i in range(4):
            for j in range(i+1, 4):
                if indices[i] > indices[j]:
                    sign *= -1
        return sign

    # ------------------------------------------------------------------
    # Test: Interface – required keys exist
    # ------------------------------------------------------------------
    def test_interface_has_keys(self):
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")
        required = ['invariant1', 'invariant2']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, f"Keys present: {required}")

    # ------------------------------------------------------------------
    # Test: Invariant – result forms are non‑empty strings or finite numbers
    # ------------------------------------------------------------------
    def test_invariant_forms(self):
        inv1 = self.results.get('invariant1')
        inv2 = self.results.get('invariant2')
        if inv1 is None or inv2 is None:
            return (False, "One or both invariants are None")
        for name, val in [('invariant1', inv1), ('invariant2', inv2)]:
            if isinstance(val, str):
                if not val.strip():
                    return (False, f"{name} is empty string")
            elif not isinstance(val, (int, float)):
                return (False, f"{name} has unexpected type {type(val)}")
            elif not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "Valid forms")

    # ------------------------------------------------------------------
    # Test: Invariant – numerical consistency from E,B
    # ------------------------------------------------------------------
    def test_invariant_numerical(self):
        if self.E is None or self.B is None:
            return (True, "SKIP: E or B not provided")
        try:
            inv1_comp = self.norm_sq(self.E) - self.norm_sq(self.B)
            inv2_comp = self.dot(self.E, self.B)
        except Exception as e:
            return (False, f"Cannot compute invariants: {e}")

        inv1_res = self.results.get('invariant1')
        inv2_res = self.results.get('invariant2')
        ok1 = not isinstance(inv1_res, (int, float)) or math.isclose(inv1_comp, inv1_res, rel_tol=1e-2, abs_tol=1e-2)
        ok2 = not isinstance(inv2_res, (int, float)) or math.isclose(inv2_comp, inv2_res, rel_tol=1e-2, abs_tol=1e-2)
        if ok1 and ok2:
            return (True, "Numerical consistency with E,B")
        else:
            parts = []
            if not ok1:
                parts.append(f"invariant1 computed {inv1_comp:.6f} != given {inv1_res}")
            if not ok2:
                parts.append(f"invariant2 computed {inv2_comp:.6f} != given {inv2_res}")
            return (False, "; ".join(parts))

    # ------------------------------------------------------------------
    # Test: Invariant – under Lorentz boost
    # ------------------------------------------------------------------
    def test_invariant_boost(self):
        if self.E is None or self.B is None or self.v is None or self.c is None:
            return (True, "SKIP: E, B, v, or c missing")
        # Ensure vectors length 3
        if not (len(self.E) == 3 and len(self.B) == 3 and len(self.v) == 3):
            return (False, "Vectors must be length 3")
        try:
            v_mag = math.sqrt(self.dot(self.v, self.v))
            if v_mag == 0:
                return (True, "Zero boost – trivial")
            gamma = 1.0 / math.sqrt(1.0 - v_mag**2 / self.c**2)
            v_hat = [x / v_mag for x in self.v]

            # Decompose
            E_par = [self.dot(self.E, v_hat) * x for x in v_hat]
            E_perp = [self.E[i] - E_par[i] for i in range(3)]
            B_par = [self.dot(self.B, v_hat) * x for x in v_hat]
            B_perp = [self.B[i] - B_par[i] for i in range(3)]

            # Cross products
            v_cross_Bperp = self.cross(self.v, B_perp)
            v_cross_Eperp = self.cross(self.v, E_perp)

            # Transformed perpendicular (Gaussian)
            E_perp_prime = [gamma * (E_perp[i] + v_cross_Bperp[i] / self.c**2) for i in range(3)]
            B_perp_prime = [gamma * (B_perp[i] - v_cross_Eperp[i] / self.c**2) for i in range(3)]

            # Total transformed
            E_prime = [E_par[i] + E_perp_prime[i] for i in range(3)]
            B_prime = [B_par[i] + B_perp_prime[i] for i in range(3)]

            # Invariants
            inv1_orig = self.norm_sq(self.E) - self.norm_sq(self.B)
            inv2_orig = self.dot(self.E, self.B)
            inv1_prime = self.norm_sq(E_prime) - self.norm_sq(B_prime)
            inv2_prime = self.dot(E_prime, B_prime)

            tol = 1e-2
            ok1 = math.isclose(inv1_orig, inv1_prime, rel_tol=tol, abs_tol=tol)
            ok2 = math.isclose(inv2_orig, inv2_prime, rel_tol=tol, abs_tol=tol)
            msg = f"inv1 orig {inv1_orig:.6f} prime {inv1_prime:.6f} | inv2 orig {inv2_orig:.6f} prime {inv2_prime:.6f}"
            if ok1 and ok2:
                return (True, "Boost invariance holds: " + msg)
            else:
                return (False, "Boost invariance fails: " + msg)
        except Exception as e:
            return (False, f"Boost test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Invariant – from field tensor contraction
    # ------------------------------------------------------------------
    def test_invariant_tensor_contraction(self):
        if self.F is None:
            return (True, "SKIP: field tensor F not provided")
        if len(self.F) != 4 or any(len(row) != 4 for row in self.F):
            return (False, "F must be 4x4")
        try:
            # Lower indices with diag(1,-1,-1,-1) metric
            eta = [1, -1, -1, -1]  # default
            # Compute F_{mu nu} = eta[mu]*eta[nu]*F^{mu nu} because eta is diagonal
            contraction = 0.0
            for mu in range(4):
                for nu in range(4):
                    contraction += self.F[mu][nu] * (eta[mu] * eta[nu] * self.F[mu][nu])
            # contraction = 2(B^2 - E^2) => inv1 = -0.5*contraction
            inv1_from_tensor = -0.5 * contraction

            inv1_res = self.results.get('invariant1')
            if isinstance(inv1_res, (int, float)):
                if not math.isclose(inv1_from_tensor, inv1_res, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Tensor contraction gives {inv1_from_tensor:.6f} != result {inv1_res}")
            if math.isnan(inv1_from_tensor):
                return (False, "invariant1 from tensor is NaN")
            return (True, "Tensor contraction consistent")
        except Exception as e:
            return (False, f"Tensor test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Modeling – transformation of perpendicular components
    # ------------------------------------------------------------------
    def test_modeling_perp_transformation(self):
        """Verify that the Gaussian transformation of perpendicular components
        follows the correct cross‑product relation."""
        if self.E is None or self.B is None or self.v is None or self.c is None:
            return (True, "SKIP: E, B, v, or c missing")
        if not (len(self.E) == 3 and len(self.B) == 3 and len(self.v) == 3):
            return (False, "Vectors must be length 3")
        try:
            v_mag = math.sqrt(self.dot(self.v, self.v))
            if v_mag == 0:
                return (True, "Zero boost")
            # Decompose
            v_hat = [x / v_mag for x in self.v]
            E_par = [self.dot(self.E, v_hat) * x for x in v_hat]
            E_perp = [self.E[i] - E_par[i] for i in range(3)]
            B_perp = [self.B[i] - self.dot(self.B, v_hat) * v_hat[i] for i in range(3)]

            # Compute the perpendicular transformation formula
            gamma = 1.0 / math.sqrt(1.0 - v_mag**2 / self.c**2)
            v_cross_Bperp = self.cross(self.v, B_perp)
            E_perp_prime_formula = [gamma * (E_perp[i] + v_cross_Bperp[i] / self.c**2) for i in range(3)]

            # Build full E_prime from formula and extract its perpendicular part
            E_par_prime = E_par[:]  # unchanged
            E_full = [E_par_prime[i] + E_perp_prime_formula[i] for i in range(3)]
            # Project out the parallel component to get E_full_perp
            E_full_par = [self.dot(E_full, v_hat) * x for x in v_hat]
            E_full_perp = [E_full[i] - E_full_par[i] for i in range(3)]

            # Compare E_full_perp with E_perp_prime_formula
            diff = math.sqrt(self.norm_sq([E_full_perp[i] - E_perp_prime_formula[i] for i in range(3)]))
            if diff > 1e-8:
                return (False, f"Perpendicular transformation mismatch: diff = {diff}")
            return (True, "Perpendicular transformation consistent")
        except Exception as e:
            return (False, f"Modeling test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Modeling – dual tensor and field tensor invariance
    # ------------------------------------------------------------------
    def test_modeling_dual_contraction(self):
        """Check that *F^{μν} F_{μν} = -4 E·B (if possible)."""
        if self.F is None:
            return (True, "SKIP: field tensor F not provided")
        if len(self.F) != 4 or any(len(row) != 4 for row in self.F):
            return (False, "F must be 4x4")
        try:
            eta = [1, -1, -1, -1]
            F_lower = [[eta[mu]*eta[nu]*self.F[mu][nu] for nu in range(4)] for mu in range(4)]

            # Compute dual contraction: 1/2 ε^{μνρσ} F_{ρσ} F_{μν}
            dual_contraction = 0.0
            for mu in range(4):
                for nu in range(4):
                    dual_mu_nu = 0.0
                    for rho in range(4):
                        for sigma in range(4):
                            dual_mu_nu += 0.5 * self.epsilon(mu, nu, rho, sigma) * F_lower[rho][sigma]
                    dual_contraction += dual_mu_nu * F_lower[mu][nu]

            # If we have E and B vectors, verify against -4 E·B
            if self.E is not None and self.B is not None and len(self.E)==3 and len(self.B)==3:
                dot_EB = self.dot(self.E, self.B)
                expected = -4 * dot_EB
                if not math.isclose(dual_contraction, expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Dual contraction {dual_contraction:.6f} != -4 E·B {expected:.6f}")
                return (True, "Dual contraction matches -4 E·B")
            else:
                # Without E,B, just check that the contraction is finite
                if not math.isfinite(dual_contraction):
                    return (False, "Dual contraction is not finite")
                return (True, "Dual contraction computed (cannot verify against E·B)")
        except Exception as e:
            return (False, f"Dual contraction test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Monotonicity – invariant1 increases when E magnitude increases
    # ------------------------------------------------------------------
    def test_monotonicity_invariant1_increases_with_E(self):
        """Increase the magnitude of E by 10% and verify that E² - B² increases."""
        if self.E is None or self.B is None:
            return (True, "SKIP: E or B not provided")
        try:
            inv1_baseline = self.norm_sq(self.E) - self.norm_sq(self.B)
            # Perturb E: multiply each component by 1.1
            E_perturbed = [x * 1.1 for x in self.E]
            inv1_perturbed = self.norm_sq(E_perturbed) - self.norm_sq(self.B)
            if not (inv1_perturbed > inv1_baseline):
                return (False, f"E² - B² did not increase: baseline {inv1_baseline:.6f}, perturbed {inv1_perturbed:.6f}")
            return (True, f"Monotonic increase: baseline {inv1_baseline:.6f} -> {inv1_perturbed:.6f}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Monotonicity – invariant2 scales quadratically when both vectors scaled
    # ------------------------------------------------------------------
    def test_monotonicity_invariant2_scaling(self):
        """Scale both E and B by a factor f>1 and check that E·B scales by f^2.
        This demonstrates a monotonic relationship (strictly increasing if baseline>0,
        strictly decreasing if baseline<0)."""
        if self.E is None or self.B is None:
            return (True, "SKIP: E or B not provided")
        try:
            scale = 1.2  # 20% increase
            inv2_baseline = self.dot(self.E, self.B)
            E_scaled = [scale * x for x in self.E]
            B_scaled = [scale * y for y in self.B]
            inv2_scaled = self.dot(E_scaled, B_scaled)

            expected = scale**2 * inv2_baseline
            if not math.isclose(inv2_scaled, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"E·B scaling failed: baseline {inv2_baseline:.6f}, scaled {inv2_scaled:.6f}, expected {expected:.6f}")

            # Verify monotonicity: if baseline > 0, scaled > baseline; if baseline < 0, scaled < baseline
            if inv2_baseline > 0 and not (inv2_scaled > inv2_baseline):
                return (False, f"E·B should increase (baseline positive) but didn't: {inv2_baseline:.6f} -> {inv2_scaled:.6f}")
            if inv2_baseline < 0 and not (inv2_scaled < inv2_baseline):
                return (False, f"E·B should decrease (baseline negative) but didn't: {inv2_baseline:.6f} -> {inv2_scaled:.6f}")
            # If baseline = 0, both should be 0 (already checked via isclose)
            return (True, f"E·B scales quadratically: {inv2_baseline:.6f} -> {inv2_scaled:.6f} (expected {expected:.6f})")
        except Exception as e:
            return (False, f"Monotonicity scaling test crashed: {e}")

    # ------------------------------------------------------------------
    # Test: Weak sanity – invariants are plausible (non‑None, finite)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        inv1 = self.results.get('invariant1')
        inv2 = self.results.get('invariant2')
        if inv1 is None or inv2 is None:
            return (False, "One or both invariants are None")
        for name, val in [('invariant1', inv1), ('invariant2', inv2)]:
            if isinstance(val, (int, float)):
                if not math.isfinite(val):
                    return (False, f"{name} is not finite")
        return (True, "Weak sanity passed")

    # ------------------------------------------------------------------
    # Run all tests
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