import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely
        for key in ['U1', 'U2', 'd', 'L1', 'L2', 'm', 'e']:
            self.inputs[key] = getattr(solution_instance, key, 0.0)
        
        # Extract results directly from results dict
        results_dict = getattr(solution_instance, 'results', {})
        
        # Map expected result keys - use the actual keys from the solution
        # The solution uses keys like 'v0', 'vx', 'vy', 'accel', 'po_final', etc.
        # Map them to the internal names used by the checker
        self.results['v0'] = results_dict.get('v0', 0.0)
        self.results['vx'] = results_dict.get('vx', 0.0)
        self.results['vy'] = results_dict.get('vy', 0.0)
        self.results['accel'] = results_dict.get('accel', 0.0)
        self.results['po_final'] = results_dict.get('po_final', 0.0)
        self.results['t1'] = results_dict.get('t_1', 0.0)
        self.results['t2'] = results_dict.get('t_2', 0.0)
        self.results['y_displacement'] = results_dict.get('y', 0.0)
        self.results['y2'] = results_dict.get('y_2', 0.0)
        self.results['ke_initial'] = results_dict.get('1/2*m*v_02', 0.0)
        
        # Store original inputs for perturbation tests
        self.original_inputs = self.inputs.copy()
        
    def test_invariants(self):
        """Test fundamental physics invariants"""
        logs = []
        successes = []
        
        try:
            # 1. Check Work-Energy Theorem: Kinetic energy gained = e*U1
            U1, e, m = self.inputs['U1'], self.inputs['e'], self.inputs['m']
            v0 = self.results['v0']
            
            ke_calculated = 0.5 * m * v0**2
            ke_expected = e * U1
            
            if math.isclose(ke_calculated, ke_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Work-Energy Theorem satisfied: KE gained equals e*U1")
                successes.append(True)
            else:
                logs.append(f"✗ Work-Energy Theorem violation: KE={ke_calculated:.6e}, e*U1={ke_expected:.6e}, diff={abs(ke_calculated-ke_expected):.6e}")
                successes.append(False)
            
            # 2. Check Newton's Second Law for deflection: a = e*U2/(m*d)
            U2, d = self.inputs['U2'], self.inputs['d']
            accel = self.results['accel']
            
            a_expected = e * U2 / (m * d)
            
            if math.isclose(accel, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's Second Law satisfied: a = e*U2/(m*d)")
                successes.append(True)
            else:
                logs.append(f"✗ Newton's Second Law violation: a={accel:.6e}, e*U2/(m*d)={a_expected:.6e}, diff={abs(accel-a_expected):.6e}")
                successes.append(False)
            
            # 3. Check conservation of horizontal velocity: vx should equal v0
            vx = self.results['vx']
            
            if math.isclose(vx, v0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Horizontal velocity conserved: vx = v0")
                successes.append(True)
            else:
                logs.append(f"✗ Horizontal velocity not conserved: vx={vx:.6e}, v0={v0:.6e}, diff={abs(vx-v0):.6e}")
                successes.append(False)
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            successes.append(False)
            
        return all(successes), logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        logs = []
        successes = []
        
        try:
            # Extract values
            U1, U2, d, L1, L2, m, e = [self.inputs[k] for k in ['U1', 'U2', 'd', 'L1', 'L2', 'm', 'e']]
            v0 = self.results['v0']
            t1 = self.results['t1']
            accel = self.results['accel']
            y_displacement = self.results['y_displacement']
            vy = self.results['vy']
            vx = self.results['vx']
            t2 = self.results['t2']
            y2 = self.results['y2']
            po_final = self.results['po_final']
            
            # 1. Check t1 = L1/v0
            t1_expected = L1 / v0 if v0 != 0 else 0
            if math.isclose(t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ t1 consistent: L1/v0")
                successes.append(True)
            else:
                logs.append(f"✗ t1 inconsistent: t1={t1:.6e}, L1/v0={t1_expected:.6e}, diff={abs(t1-t1_expected):.6e}")
                successes.append(False)
            
            # 2. Check vy = a*t1
            vy_expected = accel * t1
            if math.isclose(vy, vy_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ vy consistent: a*t1")
                successes.append(True)
            else:
                logs.append(f"✗ vy inconsistent: vy={vy:.6e}, a*t1={vy_expected:.6e}, diff={abs(vy-vy_expected):.6e}")
                successes.append(False)
            
            # 3. Check y = 0.5*a*t1^2
            y_expected = 0.5 * accel * t1**2
            if math.isclose(y_displacement, y_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ y displacement consistent: 0.5*a*t1^2")
                successes.append(True)
            else:
                logs.append(f"✗ y displacement inconsistent: y={y_displacement:.6e}, 0.5*a*t1^2={y_expected:.6e}, diff={abs(y_displacement-y_expected):.6e}")
                successes.append(False)
            
            # 4. Check t2 = L2/vx
            t2_expected = L2 / vx if vx != 0 else 0
            if math.isclose(t2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ t2 consistent: L2/vx")
                successes.append(True)
            else:
                logs.append(f"✗ t2 inconsistent: t2={t2:.6e}, L2/vx={t2_expected:.6e}, diff={abs(t2-t2_expected):.6e}")
                successes.append(False)
            
            # 5. Check y2 = vy*t2
            y2_expected = vy * t2
            if math.isclose(y2, y2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ y2 consistent: vy*t2")
                successes.append(True)
            else:
                logs.append(f"✗ y2 inconsistent: y2={y2:.6e}, vy*t2={y2_expected:.6e}, diff={abs(y2-y2_expected):.6e}")
                successes.append(False)
            
            # 6. Check PO = y + y2
            po_expected = y_displacement + y2
            if math.isclose(po_final, po_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Final PO consistent: y + y2")
                successes.append(True)
            else:
                logs.append(f"✗ Final PO inconsistent: PO={po_final:.6e}, y+y2={po_expected:.6e}, diff={abs(po_final-po_expected):.6e}")
                successes.append(False)
            
            # 7. Check against target formulas
            # v0 = sqrt(2eU1/m)
            v0_expected = math.sqrt(2 * e * U1 / m) if m > 0 else 0
            if math.isclose(v0, v0_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v0 matches target formula")
                successes.append(True)
            else:
                logs.append(f"✗ v0 doesn't match target: v0={v0:.6e}, sqrt(2eU1/m)={v0_expected:.6e}, diff={abs(v0-v0_expected):.6e}")
                successes.append(False)
            
            # y = U2*L1^2/(4dU1)
            y_formula = (U2 * L1**2) / (4 * d * U1) if d * U1 != 0 else 0
            if math.isclose(y_displacement, y_formula, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ y matches target formula")
                successes.append(True)
            else:
                logs.append(f"✗ y doesn't match target: y={y_displacement:.6e}, U2*L1^2/(4dU1)={y_formula:.6e}, diff={abs(y_displacement-y_formula):.6e}")
                successes.append(False)
            
            # PO = U2*L1*(L1+2L2)/(4dU1)
            po_formula = (U2 * L1 * (L1 + 2*L2)) / (4 * d * U1) if d * U1 != 0 else 0
            if math.isclose(po_final, po_formula, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ PO matches target formula")
                successes.append(True)
            else:
                logs.append(f"✗ PO doesn't match target: PO={po_final:.6e}, U2*L1*(L1+2L2)/(4dU1)={po_formula:.6e}, diff={abs(po_final-po_formula):.6e}")
                successes.append(False)
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            successes.append(False)
            
        return all(successes), logs
    
    def test_monotonicity(self):
        """Test how outputs change with input perturbations"""
        logs = []
        successes = []
        
        try:
            # Store original values
            original_inputs = self.inputs.copy()
            U1, U2, d, L1, L2, m, e = [original_inputs[k] for k in ['U1', 'U2', 'd', 'L1', 'L2', 'm', 'e']]
            original_po = self.results['po_final']
            
            # Test 1: Increase U2 by 1%, PO should increase
            perturbed_U2 = U2 * 1.01
            po_perturbed_U2 = (perturbed_U2 * L1 * (L1 + 2*L2)) / (4 * d * U1) if d * U1 != 0 else 0
            
            if po_perturbed_U2 > original_po:
                logs.append(f"✓ Monotonicity: Increasing U2 increases PO (PO_U2+1%={po_perturbed_U2:.6e} > original={original_po:.6e})")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: Increasing U2 should increase PO but got {po_perturbed_U2:.6e} <= {original_po:.6e}")
                successes.append(False)
            
            # Test 2: Increase U1 by 1%, PO should decrease
            perturbed_U1 = U1 * 1.01
            po_perturbed_U1 = (U2 * L1 * (L1 + 2*L2)) / (4 * d * perturbed_U1) if d * perturbed_U1 != 0 else 0
            
            if po_perturbed_U1 < original_po:
                logs.append(f"✓ Monotonicity: Increasing U1 decreases PO (PO_U1+1%={po_perturbed_U1:.6e} < original={original_po:.6e})")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: Increasing U1 should decrease PO but got {po_perturbed_U1:.6e} >= {original_po:.6e}")
                successes.append(False)
            
            # Test 3: Increase d by 1%, PO should decrease
            perturbed_d = d * 1.01
            po_perturbed_d = (U2 * L1 * (L1 + 2*L2)) / (4 * perturbed_d * U1) if perturbed_d * U1 != 0 else 0
            
            if po_perturbed_d < original_po:
                logs.append(f"✓ Monotonicity: Increasing d decreases PO (PO_d+1%={po_perturbed_d:.6e} < original={original_po:.6e})")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: Increasing d should decrease PO but got {po_perturbed_d:.6e} >= {original_po:.6e}")
                successes.append(False)
            
            # Test 4: Increase L1 by 1%, PO should increase (since L1 appears in numerator squared-ish)
            perturbed_L1 = L1 * 1.01
            po_perturbed_L1 = (U2 * perturbed_L1 * (perturbed_L1 + 2*L2)) / (4 * d * U1) if d * U1 != 0 else 0
            
            if po_perturbed_L1 > original_po:
                logs.append(f"✓ Monotonicity: Increasing L1 increases PO (PO_L1+1%={po_perturbed_L1:.6e} > original={original_po:.6e})")
                successes.append(True)
            else:
                logs.append(f"✗ Monotonicity violation: Increasing L1 should increase PO but got {po_perturbed_L1:.6e} <= {original_po:.6e}")
                successes.append(False)
            
            # Restore original inputs
            self.inputs = original_inputs.copy()
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            successes.append(False)
            # Restore inputs even if error
            self.inputs = original_inputs.copy()
            
        return all(successes), logs
    
    def run_all_tests(self):
        """Run all tests and return overall result"""
        all_logs = []
        all_success = []
        
        # Run each test
        for test_name, test_method in [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_method()
                all_logs.extend(logs)
                all_success.append(success)
            except Exception as e:
                all_logs.append(f"Test {test_name} failed with exception: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_success.append(False)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        passed = sum(all_success)
        total = len(all_success)
        all_logs.append(f"Passed {passed}/{total} test suites")
        
        overall_success = all(all_success)
        if overall_success:
            all_logs.append("✓ All tests passed!")
        else:
            all_logs.append("✗ Some tests failed")
        
        return overall_success, all_logs