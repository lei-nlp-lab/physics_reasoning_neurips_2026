import traceback
import multiprocessing
import threading
import contextlib
import io
import sympy as sp
import math


def _silent_print(*args, **kwargs):
    return None

class MockSolution:
    """
    高级适配器层：
    1. 动态属性获取：支持 solution.U 访问。
    2. 属性提升：在初始化后会将 results 里的键值对直接设为对象属性。
    3. 模拟字典行为：解决 checker 中调用 solution.inputs 或 solution.get() 报错的问题。
    """
    def __init__(self, results_dict, trace_dict=None):
        self.results = results_dict
        self.trace = trace_dict if trace_dict is not None else {}
        self.params = results_dict
        self.inputs = results_dict  # [FIX] 解决 AttributeError: MockSolution has no attribute 'inputs'

    def get(self, key, default=None):
        # [FIX] 解决 AttributeError: MockSolution has no attribute 'get'
        return self.results.get(key, default)

    def __getattr__(self, name):
        # 兜底逻辑：如果 setattr 没覆盖到，从字典找
        if name in self.results:
            return self.results[name]
        # 兼容 solution.solve() 调用
        if name == 'solve':
            return lambda *args, **kwargs: self.results
        # 抛出 AttributeError，让 getattr(obj, 'x', default) 的默认值机制正常工作
        raise AttributeError(f"MockSolution has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Subprocess-based execution for generated code (killable on timeout)
# ---------------------------------------------------------------------------

def _exec_in_subprocess(code: str, result_queue, timeout_seconds: int):
    """Execute generated code in a child process and send results back via queue."""
    import sympy as _sp
    import math as _math

    namespace = {
        "sp": _sp,
        "sympy": _sp,
        "math": _math,
        "results": {},
        "trace": {},
        "g": 10.0,
        "__builtins__": {
            # Allow safe builtins only
            "abs": abs, "all": all, "any": any, "bool": bool,
            "complex": complex, "dict": dict, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float,
            "format": format, "frozenset": frozenset, "hasattr": hasattr,
            "hash": hash, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len,
            "locals": locals, "globals": globals,
            "list": list, "map": map, "max": max, "min": min,
            "next": next, "object": object, "pow": pow, "print": _silent_print,
            "range": range, "repr": repr, "reversed": reversed,
            "round": round, "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip, "True": True, "False": False, "None": None,
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "NameError": NameError,
            "KeyError": KeyError, "IndexError": IndexError,
            "RuntimeError": RuntimeError, "ArithmeticError": ArithmeticError,
            "__import__": __import__,  # needed for sympy internals
        },
    }

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, namespace)
    except Exception as e:
        result_queue.put(("error", type(e).__name__, str(e),
                          ''.join(traceback.format_exception(type(e), e, e.__traceback__))))
        return

    # Collect results and trace, serialise as plain dicts
    results = namespace.get("results", {})
    if not isinstance(results, dict):
        results = {}

    # Convert all values to serialisable forms
    safe = {}
    for k, v in namespace.items():
        if k.startswith('_') or k in ('sp', 'sympy', 'math', 'results', 'trace', 'g'):
            continue
        try:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                safe[k] = ("num", float(v))
            elif isinstance(v, str):
                safe[k] = ("str", v)
            elif isinstance(v, _sp.Basic) and v.is_number:
                safe[k] = ("num", float(v.evalf()))
            elif isinstance(v, (list, tuple)) and v and all(isinstance(item, str) for item in v):
                safe[k] = ("strlist", list(v))
        except Exception:
            pass

    # Also serialize results dict
    res_safe = {}
    for k, v in results.items():
        try:
            if isinstance(v, bool):
                res_safe[k] = ("bool", v)
            elif isinstance(v, (int, float)):
                res_safe[k] = ("num", float(v))
            elif isinstance(v, str):
                res_safe[k] = ("str", v)
            elif isinstance(v, _sp.Basic) and v.is_number:
                res_safe[k] = ("num", float(v.evalf()))
            elif isinstance(v, _sp.Basic) and not v.is_number:
                res_safe[k] = ("symbolic", str(v))
            elif isinstance(v, _sp.Equality):
                res_safe[k] = ("num", float(v.rhs.evalf()))
            elif isinstance(v, dict):
                res_safe[k] = ("dict", {str(dk): float(dv) if hasattr(dv, '__float__') else str(dv)
                                        for dk, dv in v.items()})
            elif isinstance(v, (list, tuple)):
                items = []
                for item in v:
                    if isinstance(item, str):
                        items.append(("str", item))
                    elif hasattr(item, 'evalf'):
                        items.append(("num", float(item.evalf())))
                    elif hasattr(item, '__float__'):
                        items.append(("num", float(item)))
                    else:
                        items.append(("other", str(item)))
                res_safe[k] = ("list", items)
            elif isinstance(v, complex):
                res_safe[k] = ("complex", v.real, v.imag)
            elif hasattr(v, 'evalf'):
                res_safe[k] = ("num", float(v.evalf()))
            elif hasattr(v, '__float__'):
                res_safe[k] = ("num", float(v))
            else:
                res_safe[k] = ("other", str(v))
        except Exception as exc:
            res_safe[k] = ("error", str(exc))

    trace = namespace.get("trace", {})
    trace_safe = {}
    if isinstance(trace, dict):
        for k, v in trace.items():
            try:
                trace_safe[str(k)] = str(v)
            except Exception:
                pass

    result_queue.put(("ok", res_safe, safe, trace_safe))


def run_physics_sandbox(generated_code: str, checker_code: str, timeout_seconds: int = 30):
    """
    运行物理沙盒，包含四个防御层级。
    """

    # ==========================================
    # Layer 1: 子进程执行模型生成代码 (Killable)
    # ==========================================
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_exec_in_subprocess,
                       args=(generated_code, result_queue, timeout_seconds))
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        return False, f"[Sandbox Timeout]: Execution exceeded {timeout_seconds}s."

    if result_queue.empty():
        return False, "[Sandbox Error]: Process exited without producing results."

    msg = result_queue.get_nowait()

    if msg[0] == "error":
        _, etype, emsg, etb = msg
        if etype == "IndexError":
            return False, (
                f"[Code Execution Failed - IndexError]: You tried to access '[0]' on an empty list. "
                f"This happens when `sp.solve()` returns no solution. "
                f"Fix it by adding a fallback (e.g., `if sol: val = sol[0] else: val = 0.0`).\n"
                f"Traceback:\n{etb}"
            )
        return False, f"[Code Execution Failed]:\n{etb}"

    _, res_safe, ns_safe, trace_safe = msg

    # ==========================================
    # Layer 2: 反序列化与变量捞取 (Context Recovery)
    # ==========================================
    computed_results = {}

    # Deserialize results dict
    for k, packed in res_safe.items():
        tag = packed[0]
        if tag == "num":
            computed_results[k] = packed[1]
        elif tag == "str":
            computed_results[k] = packed[1]
        elif tag == "bool":
            # skip booleans to avoid pollution
            continue
        elif tag == "symbolic":
            # Skip unsolved symbolic expressions — they can't be reliably
            # reconstructed via sympify and would fail the checker anyway.
            continue
        elif tag == "dict":
            computed_results[k] = packed[1]
        elif tag == "list":
            items = []
            for item_tag, item_val in packed[1]:
                if item_tag == "num":
                    items.append(item_val)
                elif item_tag == "str":
                    items.append(item_val)
                else:
                    items.append(item_val)
            computed_results[k] = items
        elif tag == "complex":
            real, imag = packed[1], packed[2]
            if abs(imag) < 1e-9:
                computed_results[k] = real
            # else skip complex
        elif tag == "error":
            pass  # skip values that couldn't be serialized
        else:
            computed_results[k] = str(packed[1]) if len(packed) > 1 else None

    # Recover top-level variables from namespace
    extra_attributes = {}
    for k, packed in ns_safe.items():
        if k not in computed_results:
            tag = packed[0]
            if tag == "num":
                computed_results[k] = packed[1]
            elif tag == "str":
                computed_results[k] = packed[1]
            elif tag == "strlist":
                extra_attributes[k] = packed[1]

    # ==========================================
    # Layer 3: 数值标准化与精度修约 (Normalization)
    # ==========================================
    safe_results = {}
    symbolic_failures = []

    for k, v in computed_results.items():
        safe_key = k if isinstance(k, str) else str(k)
        try:
            if not isinstance(k, str):
                symbolic_failures.append(
                    f"Result key {k!r} is not a string. Keys in `results` must be strings."
                )

            # 1. 字符串处理
            if isinstance(v, str):
                # [FIX] 如果模型用 str() 包装了列表，在这里自动反序列化还原为真正的 list！
                if v.strip().startswith('[') and v.strip().endswith(']'):
                    try:
                        import ast
                        parsed_list = ast.literal_eval(v)
                        if isinstance(parsed_list, list):
                            safe_results[safe_key] = parsed_list
                            continue
                    except Exception:
                        pass
                safe_results[safe_key] = v
                continue

            # 2. 元组处理
            if isinstance(v, tuple):
                symbolic_failures.append(
                    f"'{safe_key}' is a tuple {v}. You MUST extract a single specific numeric float value (or a list/string if explicitly required)."
                )
                safe_results[safe_key] = v
                continue

            # 3. 字典解包
            if isinstance(v, dict):
                matched = [val for sym, val in v.items() if str(sym) == safe_key]
                if matched:
                    v = matched[0]
                elif v:
                    v = list(v.values())[0]
                else:
                    symbolic_failures.append(f"'{safe_key}' is an empty dict.")
                    safe_results[safe_key] = v
                    continue

            # 4. 列表解包
            if isinstance(v, (list, sp.Matrix)):
                if len(v) == 0:
                    symbolic_failures.append(f"'{safe_key}' is an empty list.")
                    safe_results[safe_key] = v
                    continue
                elif len(v) == 1:
                    v = v[0]
                else:
                    clean_list = []
                    for item in v:
                        if isinstance(item, str):
                            clean_list.append(item)
                        elif hasattr(item, 'evalf'):
                            clean_list.append(float(item.evalf()))
                        elif hasattr(item, '__float__'):
                            clean_list.append(float(item))
                        else:
                            clean_list.append(item)
                    safe_results[safe_key] = clean_list
                    continue

            # 5. 等式解包
            if isinstance(v, sp.Equality):
                v = v.rhs

            # 6. 未求值的 SymPy 表达式拦截
            if isinstance(v, sp.sets.sets.Set):
                symbolic_failures.append(f"'{safe_key}' is a SymPy Set. Extract numeric value with float(list(sol)[0]).")
                safe_results[safe_key] = v
                continue
            if isinstance(v, sp.Basic) and not v.is_number:
                symbolic_failures.append(
                    f"'{safe_key}' is a symbolic expression ({v}), not a concrete number. "
                    f"You forgot to substitute constants using .subs() before evaluating."
                )
                safe_results[safe_key] = v
                continue

            # 7. 浮点数强转
            if hasattr(v, 'evalf'):
                val = float(v.evalf())
            elif hasattr(v, '__float__'):
                val = float(v)
            else:
                val = float(v)

            # 8. 忽略极小虚部
            if isinstance(v, complex) or (hasattr(v, 'is_complex') and v.is_complex and not v.is_real):
                if abs(v.imag) < 1e-9:
                    val = float(v.real)
                else:
                    symbolic_failures.append(f"'{safe_key}' resulted in a complex number: {v}")
                    safe_results[safe_key] = v
                    continue

            safe_results[safe_key] = val

        except (TypeError, ValueError) as e:
            symbolic_failures.append(f"'{safe_key}' could not be converted to float/string. Error: {str(e)}")
            safe_results[safe_key] = v
        except Exception as e:
            safe_results[safe_key] = v

    if symbolic_failures:
        failure_report = "[Format/Symbolic Value Error]: Your code did not output proper values.\n"
        failure_report += "\n".join(f"  - {msg}" for msg in symbolic_failures)
        failure_report += "\n\nFix: Ensure `results` dictionary contains Python `float` types for physical quantities, or explicit `str` / `list` values when text answers are required."
        return False, failure_report

    solution_instance = MockSolution(safe_results, trace_safe)
    for k, v in safe_results.items():
        if not isinstance(k, str):
            continue
        if not hasattr(solution_instance, k):
            setattr(solution_instance, k, v)
    for k, v in extra_attributes.items():
        if not isinstance(k, str):
            continue
        if not hasattr(solution_instance, k):
            setattr(solution_instance, k, v)
    # ==========================================
    # Layer 4: Checker 适配与语义化报错 (Auditing)
    # ==========================================
    checker_namespace = {}
    checker_result = [None]  # (success, report) or None if timeout/error

    def _checker_target():
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(checker_code, checker_namespace)
                if 'PhysicsChecker' not in checker_namespace:
                    checker_result[0] = (False, "[Sandbox Error]: Checker code missing 'PhysicsChecker' class.")
                    return

                checker = checker_namespace['PhysicsChecker'](solution_instance)

                try:
                    success, report = checker.run_all_tests()
                except TypeError as te:
                    checker_result[0] = (False, (
                        f"[Interface Match Failed]: Missing a required variable in your results. "
                        f"Error: {str(te)}\n"
                        f"Please ensure all intermediate physical quantities are assigned to 'results'."
                    ))
                    return
                except ZeroDivisionError as ze:
                    checker_result[0] = (False, (
                        f"[Checker ZeroDivisionError]: A value you computed was exactly 0.0, causing a division-by-zero in the unit test.\n"
                        f"Computed results: {safe_results}\n"
                        f"Diagnosis: Did your value underflow (e.g., 1e-40 converted to 0.0)? Or did you solve for the wrong physics phase?"
                    ))
                    return
                except Exception as e:
                    checker_result[0] = (False, f"[Checker Runtime Error]: {str(e)}\n{traceback.format_exc()}")
                    return

            if isinstance(report, list):
                report_str = "\n".join([str(item) for item in report])
            else:
                report_str = str(report)

            results_summary = f">>> Computed results: {safe_results}\n"
            checker_result[0] = (success, results_summary + report_str)

        except Exception:
            checker_result[0] = (False, f"[Checker Execution Error]:\n{traceback.format_exc()}")

    checker_worker = threading.Thread(target=_checker_target, daemon=True)
    checker_worker.start()
    checker_worker.join(timeout=timeout_seconds)

    if checker_worker.is_alive():
        return False, f"[Checker Timeout]: Checker execution exceeded {timeout_seconds}s."

    if checker_result[0] is None:
        return False, "[Checker Error]: Checker produced no result."

    return checker_result[0]
