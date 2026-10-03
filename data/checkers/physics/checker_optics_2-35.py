import math
import traceback

class PhysicsChecker:
    """
    Checker for Fresnel zone plate intensity ratio problem.
    """
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.ratio_lens_zp = results.get('ratio_lens_zp', None)
        
        # Extract given quantities
        givens = getattr(solution_instance, 'givens', None) or {}
        self.num_open_zones = givens.get('num_zones_zp', None)    # 100
        self.num_total_zones = givens.get('num_zones_total', None) # 200
        
        # Optional
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    def _safe_float(self, val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ---- Interface Tests ----
    def test_interface_ratio_key_exists(self):
        """Check that 'ratio_lens_zp' is present in results and is numeric."""
        if self.ratio_lens_zp is None:
            return (False, "Missing key 'ratio_lens_zp' in results")
        val = self._safe_float(self.ratio_lens_zp)
        if val is None:
            return (False, f"'ratio_lens_zp' is not numeric: {self.ratio_lens_zp}")
        return (True, f"'ratio_lens_zp' = {val}")

    def test_interface_given_zone_numbers(self):
        """Check that given zone numbers are positive integers (if provided)."""
        if self.num_open_zones is None or self.num_total_zones is None:
            return (False, "SKIP: zone numbers not provided in givens")
        if not isinstance(self.num_open_zones, int) or self.num_open_zones <= 0:
            return (False, f"num_open_zones not a positive integer: {self.num_open_zones}")
        if not isinstance(self.num_total_zones, int) or self.num_total_zones <= 0:
            return (False, f"num_total_zones not a positive integer: {self.num_total_zones}")
        if self.num_total_zones <= self.num_open_zones:
            return (False, f"num_total_zones ({self.num_total_zones}) must exceed num_open_zones ({self.num_open_zones})")
        return (True, f"Zone numbers: open={self.num_open_zones}, total={self.num_total_zones}")

    # ---- Invariant Tests (theorem-based physical sanity) ----
    def test_invariant_intensity_ratio_from_zone_counts(self):
        """
        Invariant: For an ideal zone plate and lens with equal-amplitude in-phase zones,
        the intensity ratio I_lens / I_zp equals (N_total / N_open)^2.
        """
        if self.num_open_zones is None or self.num_total_zones is None:
            return (False, "SKIP: zone numbers missing, cannot derive invariant")
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio_lens_zp not available")
        try:
            n_open = float(self.num_open_zones)
            n_total = float(self.num_total_zones)
            expected_ratio = (n_total / n_open) ** 2
            actual_ratio = self._safe_float(self.ratio_lens_zp)
            if actual_ratio is None:
                return (False, "ratio_lens_zp not numeric")
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Ratio {actual_ratio} matches expected {expected_ratio} from zone counts")
            else:
                return (False, f"Ratio {actual_ratio} does not match expected {expected_ratio}")
        except Exception as e:
            return (False, f"test_invariant_intensity_ratio_from_zone_counts crashed: {e}")

    def test_invariant_ratio_consistency(self):
        """
        Invariant: Using the given zone numbers and the standard Fresnel result that
        the unobstructed aperture amplitude is A1/2, we can derive two expressions for
        I_zp / I_noplate. They must be consistent given the observed ratio_lens_zp.
        """
        if self.num_open_zones is None or self.num_total_zones is None:
            return (False, "SKIP: zone numbers missing")
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio_lens_zp not available")
        try:
            n_open = float(self.num_open_zones)
            n_total = float(self.num_total_zones)
            actual_ratio = self._safe_float(self.ratio_lens_zp)
            if actual_ratio is None:
                return (False, "ratio_lens_zp not numeric")

            # Compute I_lens / I_noplate from amplitude theory (A_noplate = A1/2)
            # I_lens ∝ (n_total * A1)^2, I_noplate ∝ (A1/2)^2 => ratio = (2 * n_total)^2
            ratio_lens_noplate_theory = (2.0 * n_total) ** 2  # e.g., (400)^2 = 160000

            # Compute I_zp / I_noplate from zone plate theory (amplitude = n_open * A1)
            ratio_zp_noplate_theory = (2.0 * n_open) ** 2   # e.g., (200)^2 = 40000

            # From the measured ratio_lens_zp, we can infer ratio_zp_noplate:
            ratio_zp_noplate_inferred = ratio_lens_noplate_theory / actual_ratio

            # The two expressions for I_zp / I_noplate must match.
            if math.isclose(ratio_zp_noplate_theory, ratio_zp_noplate_inferred, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Consistency check: theory {ratio_zp_noplate_theory} vs inferred {ratio_zp_noplate_inferred} agree")
            else:
                return (False, f"Inconsistency: theory {ratio_zp_noplate_theory} vs inferred {ratio_zp_noplate_inferred}")
        except Exception as e:
            return (False, f"test_invariant_ratio_consistency crashed: {e}")

    # ---- Modeling / Trace Tests ----
    def test_modeling_amplitude_sum_in_explanation(self):
        """
        Check that explanation steps mention adding amplitudes of open zones.
        """
        steps_text = " ".join(self.explanation_steps) if self.explanation_steps else ""
        required_phrases = ["amplitud", "add", "phase", "100", "200"]
        found = [p for p in required_phrases if p.lower() in steps_text.lower()]
        if len(found) >= 2:
            return (True, f"Explanation mentions amplitude addition: found {found}")
        else:
            return (False, f"Explanation missing key phrases for amplitude addition. Found: {found}")

    def test_modeling_ratio_positivity_and_finiteness(self):
        """Check that the ratio is positive and finite."""
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio not available")
        val = self._safe_float(self.ratio_lens_zp)
        if val is None:
            return (False, "ratio_lens_zp not numeric")
        if not math.isfinite(val):
            return (False, f"ratio_lens_zp is not finite: {val}")
        if val <= 0:
            return (False, f"ratio_lens_zp must be positive, got {val}")
        return (True, f"Ratio is positive finite: {val}")

    # ---- Monotonicity Tests ----
    def test_monotonicity_increasing_open_zones_decreases_ratio(self):
        """
        If we increase the number of open zones (while keeping total zones fixed),
        the ratio I_lens / I_zp should decrease because I_zp increases.
        """
        if self.num_open_zones is None or self.num_total_zones is None:
            return (False, "SKIP: zone numbers unavailable")
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio_lens_zp not available")
        try:
            n_open = float(self.num_open_zones)
            n_total = float(self.num_total_zones)
            # Perturb: increase open zones by 1
            pert_n_open = n_open + 1
            expected_ratio_pert = (n_total / pert_n_open) ** 2
            actual_ratio = self._safe_float(self.ratio_lens_zp)
            if actual_ratio is None:
                return (False, "ratio_lens_zp not numeric")
            # Monotonicity: actual ratio should be larger than the perturbed ratio
            if actual_ratio > expected_ratio_pert:
                return (True, f"Actual ratio {actual_ratio} > {expected_ratio_pert} (expected when open zones increase to {pert_n_open})")
            else:
                return (False, f"Actual ratio {actual_ratio} should be > {expected_ratio_pert}")
        except Exception as e:
            return (False, f"test_monotonicity_increasing_open_zones crashed: {e}")

    def test_monotonicity_decreasing_total_zones_decreases_ratio(self):
        """
        If we decrease the total number of zones (while keeping open zones fixed),
        the ratio should decrease because I_lens decreases faster.
        """
        if self.num_open_zones is None or self.num_total_zones is None:
            return (False, "SKIP: zone numbers unavailable")
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio not available")
        try:
            n_open = float(self.num_open_zones)
            n_total = float(self.num_total_zones)
            # Perturb: decrease total zones by 1
            pert_n_total = n_total - 1
            if pert_n_total <= n_open:
                return (False, "SKIP: perturbed total zones not > open zones")
            expected_ratio_pert = (pert_n_total / n_open) ** 2
            actual_ratio = self._safe_float(self.ratio_lens_zp)
            if actual_ratio is None:
                return (False, "ratio_lens_zp not numeric")
            if actual_ratio > expected_ratio_pert:
                return (True, f"Actual ratio {actual_ratio} > {expected_ratio_pert} (expected when total zones decrease to {pert_n_total})")
            else:
                return (False, f"Actual ratio {actual_ratio} should be > {expected_ratio_pert}")
        except Exception as e:
            return (False, f"test_monotonicity_decreasing_total_zones crashed: {e}")

    # ---- Weak Answer Sanity (only one) ----
    def test_answer_sanity_ratio_reasonable_magnitude(self):
        """
        Very weak sanity: the ratio should be >0 and <10 (rounded magnitude).
        """
        if self.ratio_lens_zp is None:
            return (False, "SKIP: ratio not available")
        val = self._safe_float(self.ratio_lens_zp)
        if val is None:
            return (False, "ratio_lens_zp not numeric")
        if val <= 0:
            return (False, f"Ratio must be positive, got {val}")
        if val > 10:
            return (False, f"Ratio is too large ({val}), expected around 4")
        return (True, f"Ratio {val} is positive and within reasonable magnitude")

    def run_all_tests(self):
        """Collect and run all test methods, returning (success, logs)."""
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)