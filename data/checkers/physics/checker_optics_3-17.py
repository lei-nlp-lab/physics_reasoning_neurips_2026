import sys
import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits the solution for a qualitative molecular absorption problem.
    Checks interface, modeling, invariant energy ordering and wavelength consistency, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        self.expected_keys: List[str] = ['Q1_process', 'Q2_process', 'Q3_process', 'Q4_process']

        # Extract results from solution_instance
        self.results: Optional[Dict[str, Any]] = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = {}
            for key in self.expected_keys:
                self.results[key] = getattr(solution_instance, key, None)

        # Safely extract each process
        self.Q1 = self.results.get('Q1_process', None) if self.results else None
        self.Q2 = self.results.get('Q2_process', None) if self.results else None
        self.Q3 = self.results.get('Q3_process', None) if self.results else None
        self.Q4 = self.results.get('Q4_process', None) if self.results else None

    # ------------------------------------------------------------------
    # Helper: energy rank and approximate wavelength from description
    # ------------------------------------------------------------------
    @staticmethod
    def _process_rank(desc: str) -> Optional[int]:
        """Return an integer rank (1=highest energy, 4=lowest) based on keywords.
           Returns None if description is missing or cannot be classified."""
        if desc is None:
            return None
        text = desc.lower()
        if 'inner shell' in text or 'inner-shell' in text:
            return 1
        if 'valence' in text or 'orbiting' in text or 'electronic' in text:
            return 2
        if 'vibrational' in text or 'vibration' in text:
            return 3
        if 'rotational' in text or 'rotation' in text:
            return 4
        return None

    @staticmethod
    def _process_energy_ev(desc: str) -> Optional[float]:
        """Return approximate energy in eV for a process.
           None if classification fails."""
        rank = PhysicsChecker._process_rank(desc)
        if rank is None:
            return None
        # Typical energies (re-derived from physics knowledge, not from answer sheet)
        mapping = {1: 10000.0, 2: 10.0, 3: 0.1, 4: 0.001}
        return mapping[rank]

    @staticmethod
    def _process_wavelength_m(desc: str) -> Optional[float]:
        """Return approximate wavelength in meters from photon energy (E=hc/λ).
           Uses standard constants: h = 4.135667696e-15 eV·s, c = 2.99792458e8 m/s.
           Returns None if energy cannot be determined."""
        e_ev = PhysicsChecker._process_energy_ev(desc)
        if e_ev is None or e_ev == 0.0:
            return None
        # hc = 1240 eV·nm = 1.24e-6 eV·m
        hc_eVm = 1.23984193e-6  # eV·m
        wavelength = hc_eVm / e_ev  # in meters
        return wavelength

    @staticmethod
    def _expected_wavelength_range(rank: int) -> Tuple[float, float]:
        """
        Return (min_m, max_m) of typical wavelength ranges for each spectral region.
        Derived from standard physics, not from answer sheet.
        Ranks: 1=X-ray (0.01–10 nm => 1e-11–1e-8 m)
               2=UV/vis (10–400 nm => 1e-8–4e-7 m)
               3=near-IR (0.7–2.5 µm => 7e-7–2.5e-6 m)
               4=far-IR/radio (2.5 µm–1 m => 2.5e-6–1 m)
        """
        if rank == 1:
            return (1e-11, 1e-8)
        elif rank == 2:
            return (1e-8, 4e-7)
        elif rank == 3:
            return (7e-7, 2.5e-6)
        elif rank == 4:
            return (2.5e-6, 1.0)  # up to radio waves
        else:
            return (0.0, float('inf'))

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four result keys are present and non-None."""
        if self.results is None:
            return (False, "No results dictionary found.")
        missing = [k for k in self.expected_keys if k not in self.results or self.results[k] is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_keys_nonempty_strings(self) -> Tuple[bool, str]:
        """Check that each result is a non-empty string."""
        for key, val in [('Q1_process', self.Q1), ('Q2_process', self.Q2),
                         ('Q3_process', self.Q3), ('Q4_process', self.Q4)]:
            if val is None:
                return (False, f"{key} is None")
            if not isinstance(val, str):
                return (False, f"{key} is not a string (type: {type(val).__name__})")
            if len(val.strip()) == 0:
                return (False, f"{key} is empty after stripping whitespace")
        return (True, "All keys contain non-empty strings.")

    # ------------------------------------------------------------------
    # Modeling tests (trace/description correctness)
    # ------------------------------------------------------------------
    def test_modeling_xray_process(self) -> Tuple[bool, str]:
        """Check that X-ray absorption is described as inner-shell electronic transition."""
        if self.Q1 is None:
            return (False, "SKIP: Q1_process not available")
        text = self.Q1.lower()
        if not any(term in text for term in ['inner shell', 'inner-shell', 'inner shell electron', 'inner-shell electron']):
            return (False, "X-ray description missing 'inner shell' concept")
        if 'electron' not in text:
            return (False, "X-ray description missing 'electron'")
        return (True, "X-ray description correctly references inner-shell electron transition.")

    def test_modeling_uvvis_process(self) -> Tuple[bool, str]:
        """Check that UV/visible absorption is described as valence electronic transition."""
        if self.Q2 is None:
            return (False, "SKIP: Q2_process not available")
        text = self.Q2.lower()
        if 'transition' not in text:
            return (False, "UV/visible description missing 'transition'")
        if not any(term in text for term in ['electron', 'orbiting', 'energy states']):
            return (False, "UV/visible description missing reference to electronic states")
        return (True, "UV/visible description correctly references electronic transition.")

    def test_modeling_nearir_process(self) -> Tuple[bool, str]:
        """Check that near-IR absorption is described as vibrational transition."""
        if self.Q3 is None:
            return (False, "SKIP: Q3_process not available")
        text = self.Q3.lower()
        if 'vibrational' not in text and 'vibration' not in text:
            return (False, "Near-IR description missing 'vibrational' or 'vibration'")
        return (True, "Near-IR description correctly references vibrational transition.")

    def test_modeling_farir_radio_process(self) -> Tuple[bool, str]:
        """Check that far-IR/radio absorption is described as rotational transition."""
        if self.Q4 is None:
            return (False, "SKIP: Q4_process not available")
        text = self.Q4.lower()
        if 'rotational' not in text and 'rotation' not in text:
            return (False, "Far-IR/radio description missing 'rotational' or 'rotation'")
        return (True, "Far-IR/radio description correctly references rotational transition.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_energy_order(self) -> Tuple[bool, str]:
        """Check that the processes are ordered by decreasing transition energy.
           This follows from photon energy = transition energy, and different spectral bands."""
        ranks = []
        for i, desc in enumerate([self.Q1, self.Q2, self.Q3, self.Q4], start=1):
            if desc is None:
                return (False, f"SKIP: Q{i}_process is None")
            rank = self._process_rank(desc)
            if rank is None:
                return (False, f"Q{i}_process '{desc}' cannot be classified for energy rank")
            ranks.append(rank)
        if ranks != [1, 2, 3, 4]:
            return (False, f"Energy order violated: got ranks {ranks}, expected [1,2,3,4]")
        return (True, "Processes are in correct decreasing energy order (X-ray > UV/vis > near-IR > far-IR).")

    def test_invariant_wavelength_bands(self) -> Tuple[bool, str]:
        """
        Invariant: For each process description, the implied photon wavelength (from E = hc/λ)
        must lie within the typical spectral band for that region.
        This uses the physical constant hc = 1.24e-6 eV·m and typical band boundaries derived from standard knowledge.
        """
        for i, desc in enumerate([self.Q1, self.Q2, self.Q3, self.Q4], start=1):
            if desc is None:
                return (False, f"SKIP: Q{i}_process is None")
            rank = self._process_rank(desc)
            if rank is None:
                return (False, f"Q{i}_process '{desc}' cannot be classified")
            wavelength = self._process_wavelength_m(desc)
            if wavelength is None:
                return (False, f"Cannot compute wavelength for Q{i}_process")
            min_w, max_w = self._expected_wavelength_range(rank)
            if not (min_w <= wavelength <= max_w):
                return (False, f"Q{i}_process wavelength {wavelength:.2e} m is outside expected range [{min_w:.2e}, {max_w:.2e}] for rank {rank}")
        return (True, "All process descriptions imply wavelengths consistent with their spectral band (energy conservation holds).")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb wavelength region and check energy change
    # ------------------------------------------------------------------
    def test_monotonicity_transition_energy_decreases_with_wavelength(self) -> Tuple[bool, str]:
        """
        Monotonicity: as we move from one spectral region to the next (increasing wavelength),
        the transition energy (proportional to 1/λ) must decrease.
        Baseline: Q1 energy; perturbed: Q2 energy, etc.
        """
        energies = []
        for i, desc in enumerate([self.Q1, self.Q2, self.Q3, self.Q4], start=1):
            if desc is None:
                return (False, f"SKIP: Q{i}_process is None")
            e = self._process_energy_ev(desc)
            if e is None:
                return (False, f"Q{i}_process '{desc}' cannot be assigned an energy")
            energies.append(e)
        # Check monotonic decrease for each adjacent pair (baseline -> perturbed)
        for i in range(len(energies) - 1):
            if not (energies[i] > energies[i+1]):
                return (False, f"Monotonicity violated: energy at region {i+1} ({energies[i]} eV) is not greater than energy at region {i+2} ({energies[i+1]} eV)")
        return (True, "Transition energy decreases monotonically with increasing wavelength (X-ray → far-IR/radio).")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_processes_plausible(self) -> Tuple[bool, str]:
        """
        Weak sanity: each description is plausible (length > 10 characters,
        not a placeholder like 'None' or '0.0').
        """
        for name, val in [('Q1_process', self.Q1), ('Q2_process', self.Q2),
                          ('Q3_process', self.Q3), ('Q4_process', self.Q4)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, str):
                return (False, f"{name} is not a string")
            stripped = val.strip()
            if len(stripped) < 10:
                return (False, f"{name} is too short (length {len(stripped)})")
            # Reject common dummy values
            if stripped.lower() in ['none', '0.0', '0', 'null', 'undefined']:
                return (False, f"{name} contains a placeholder value: '{stripped}'")
        return (True, "All answers appear plausible (length > 10, not placeholder).")

    # ------------------------------------------------------------------
    # Test runner (do not modify)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, collect logs."""
        logs: List[str] = []
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