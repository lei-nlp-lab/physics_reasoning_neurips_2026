import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.cause = results.get('cause', None)
        self.balance = results.get('balance', None)
        self.spreading = results.get('spreading', None)
        self.intensity_law = results.get('intensity_law', None)

        # Extract any additional givens or explanations (safely)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.givens = getattr(solution_instance, 'givens', {})

        # For temperature trends, try to infer from explanation_steps
        self.temp_tropo_decreases = None
        self.temp_strato_increases = None
        if self.explanation_steps:
            if 'decreases' in self.explanation_steps.lower() and 'troposphere' in self.explanation_steps.lower():
                self.temp_tropo_decreases = True
            if 'increases' in self.explanation_steps.lower() and 'stratosphere' in self.explanation_steps.lower():
                self.temp_strato_increases = True

    # --- Interface Tests ---
    def test_interface_cause_key(self):
        if self.cause is None:
            return (False, "SKIP: 'cause' key missing")
        if not isinstance(self.cause, str):
            return (False, f"cause is not a string, got {type(self.cause)}")
        return (True, "cause key exists and is string")

    def test_interface_balance_key(self):
        if self.balance is None:
            return (False, "SKIP: 'balance' key missing")
        if not isinstance(self.balance, str):
            return (False, f"balance is not a string, got {type(self.balance)}")
        return (True, "balance key exists and is string")

    def test_interface_spreading_key(self):
        if self.spreading is None:
            return (False, "SKIP: 'spreading' key missing")
        if not isinstance(self.spreading, str):
            return (False, f"spreading is not a string, got {type(self.spreading)}")
        return (True, "spreading key exists and is string")

    def test_interface_intensity_law_key(self):
        if self.intensity_law is None:
            return (False, "SKIP: 'intensity_law' key missing")
        if not isinstance(self.intensity_law, str):
            return (False, f"intensity_law is not a string, got {type(self.intensity_law)}")
        return (True, "intensity_law key exists and is string")

    # --- Invariant Tests (theorem-based physical sanity) ---
    def test_invariant_ozone_absorption(self):
        if self.cause is None:
            return (False, "SKIP: cause not available")
        keywords = ['ozone', 'ultraviolet', 'uv', 'absorption']
        if any(kw in self.cause.lower() for kw in keywords):
            return (True, "Ozone absorption mentioned")
        return (False, "Expected mention of ozone/UV absorption, got: " + self.cause)

    def test_invariant_radiative_equilibrium(self):
        if self.balance is None:
            return (False, "SKIP: balance not available")
        lower = self.balance.lower()
        has_abs = any(kw in lower for kw in ['uv', 'ultraviolet', 'absorb'])
        has_em = any(kw in lower for kw in ['ir', 'infrared', 'emit'])
        has_eq = any(kw in lower for kw in ['equilibrium', 'balance'])
        if has_abs and has_em and has_eq:
            return (True, "Radiative equilibrium described")
        return (False, "Expected description of radiative equilibrium (absorbed UV = emitted IR), got: " + self.balance)

    def test_invariant_sound_speed_refraction(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        lower = self.explanation_steps.lower()
        if 'refract' in lower and 'lower' in lower and ('speed' in lower or 'temperature' in lower):
            return (True, "Refraction toward lower speed/temperature mentioned")
        return (False, "Expected mention of refraction toward lower sound speed/temperature")

    def test_invariant_temperature_troposphere(self):
        if self.temp_tropo_decreases is None:
            return (False, "SKIP: troposphere trend not known")
        if self.temp_tropo_decreases:
            return (True, "Troposphere temperature decreases with altitude")
        return (False, "Expected troposphere temperature to decrease with altitude")

    def test_invariant_temperature_stratosphere(self):
        if self.temp_strato_increases is None:
            return (False, "SKIP: stratosphere trend not known")
        if self.temp_strato_increases:
            return (True, "Stratosphere temperature increases with altitude")
        return (False, "Expected stratosphere temperature to increase with altitude")

    def test_invariant_cylindrical_spreading(self):
        if self.spreading is None:
            return (False, "SKIP: spreading not available")
        if 'cylindrical' in self.spreading.lower():
            return (True, "Spreading type identified as cylindrical")
        return (False, "Expected cylindrical spreading, got: " + self.spreading)

    def test_invariant_intensity_law(self):
        if self.intensity_law is None:
            return (False, "SKIP: intensity_law not available")
        law = self.intensity_law.replace(" ", "").lower()
        if "1/r" in law:
            return (True, "Intensity law includes 1/R")
        return (False, "Expected intensity law I ∝ 1/R, got: " + self.intensity_law)

    # --- Modeling/Trace Checks ---
    def test_modeling_duct_explained(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        lower = self.explanation_steps.lower()
        duct_keywords = ['confine', 'layer', 'fan', 'lateral', 'waveguide', 'duct']
        if any(kw in lower for kw in duct_keywords):
            return (True, "Ducting mechanism mentioned")
        return (False, "Expected explanation of sound ducting (waveguide/fan-shaped propagation)")

    def test_modeling_ozone_role(self):
        if self.cause is None:
            return (False, "SKIP: cause not available")
        lower = self.cause.lower()
        if 'absorb' in lower and ('temperature' in lower or 'heat' in lower or 'warm' in lower):
            return (True, "Ozone absorption linked to temperature increase")
        return (False, "Required: ozone absorption causes temperature rise, got: " + self.cause)

    # --- Monotonicity Tests (explicit perturbation of an input and comparison) ---
    def test_monotonicity_sound_speed_temperature(self):
        """Sound speed increases monotonically with temperature: c = sqrt(gamma * R * T)."""
        # Standard constants for air
        gamma = 1.4
        R_gas = 287.0  # J/(kg·K)
        # Baseline temperature (e.g., 250 K typical for tropopause)
        T_baseline = 250.0
        c_baseline = math.sqrt(gamma * R_gas * T_baseline)

        # Perturb: increase temperature by 10%
        T_perturbed = T_baseline * 1.1
        c_perturbed = math.sqrt(gamma * R_gas * T_perturbed)

        # Monotonicity: if T increases, c should increase (positive derivative)
        if c_perturbed > c_baseline:
            return (True, "Sound speed increases when temperature increases (monotonic increasing).")
        else:
            return (False, "Sound speed did not increase with temperature; check formula.")

    def test_monotonicity_intensity_distance_cylindrical(self):
        """For cylindrical spreading, intensity decreases monotonically with distance: I ∝ 1/R."""
        # Check if the solution indicates cylindrical spreading
        if self.intensity_law is None:
            return (False, "SKIP: intensity_law not available")
        # We only apply this test if the law is consistent with cylindrical spreading (1/R)
        if "1/r" not in self.intensity_law.replace(" ", "").lower():
            return (False, "SKIP: intensity law does not indicate 1/R dependence")

        # Use a simple model: I = 1/R (proportional to 1/R)
        R_baseline = 1000.0  # meters
        I_baseline = 1.0 / R_baseline

        # Perturb: increase distance by 20%
        R_perturbed = R_baseline * 1.2
        I_perturbed = 1.0 / R_perturbed

        # Monotonicity: as R increases, I decreases
        if I_perturbed < I_baseline:
            return (True, "Intensity decreases when distance increases (monotonic decreasing).")
        else:
            return (False, "Intensity did not decrease with distance; check 1/R formula.")

    # --- Weak Answer Sanity (only one) ---
    def test_answer_sanity_simple(self):
        """Very weak sanity: all four answer fields are non-empty strings."""
        fields = [('cause', self.cause), ('balance', self.balance), ('spreading', self.spreading), ('intensity_law', self.intensity_law)]
        for name, val in fields:
            if val is None:
                return (False, f"Field '{name}' is missing")
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"Field '{name}' is empty or not a string")
        return (True, "All four answer fields present and non-empty")

    # --- Run all tests ---
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)