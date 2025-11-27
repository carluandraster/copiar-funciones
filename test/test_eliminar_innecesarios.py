import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import eliminar_innecesarios as ei
import unittest

class EliminarInnecesariosTest(unittest.TestCase):
    def testInitCodeAnalyzer(self):
        analyzer = ei.CodeAnalyzer()
        self.assertEqual(analyzer.functions, set())
        self.assertEqual(analyzer.classes, set())
        self.assertEqual(analyzer.calls, {})
        self.assertEqual(analyzer.used, set())
        self.assertIsNone(analyzer.current_context)
    
    def testVisitFunctionDef(self):
        analyzer = ei.CodeAnalyzer()
        func_node = ei.ast.FunctionDef(name="mi_funcion", args=ei.ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[], decorator_list=[])
        analyzer.visit_FunctionDef(func_node)
        self.assertIn("mi_funcion", analyzer.functions)
        self.assertIn("mi_funcion", analyzer.calls)
    
    def testVisitClassDef(self):
        analyzer = ei.CodeAnalyzer()
        class_node = ei.ast.ClassDef(name="MiClase", bases=[], keywords=[], body=[], decorator_list=[])
        analyzer.visit_ClassDef(class_node)
        self.assertIn("MiClase", analyzer.classes)
        self.assertIn("MiClase", analyzer.calls)
    
    def testVisitCall1(self): # El atributo function de node es un ast.Name
        analyzer = ei.CodeAnalyzer()
        call_node = ei.ast.Call(func=ei.ast.Name(id="mi_funcion", ctx=ei.ast.Load()), args=[], keywords=[])
        analyzer.current_context = "otra_funcion"
        analyzer.visit_Call(call_node)
        self.assertIn("mi_funcion", analyzer.used)
        self.assertIn("mi_funcion", analyzer.calls["otra_funcion"])
    
    def testVisitCall2(self): # El atributo function de node no es un ast.Name
        analyzer = ei.CodeAnalyzer()
        call_node = ei.ast.Call(func=ei.ast.Attribute(value=ei.ast.Name(id="objeto", ctx=ei.ast.Load()), attr="metodo", ctx=ei.ast.Load()), args=[], keywords=[])
        analyzer.current_context = "otra_funcion"
        analyzer.visit_Call(call_node)
        self.assertNotIn("metodo", analyzer.used)
        self.assertNotIn("metodo", analyzer.calls["otra_funcion"])
    
    def testVisitCall3(self): # current_context es None
        analyzer = ei.CodeAnalyzer()
        call_node = ei.ast.Call(func=ei.ast.Name(id="mi_funcion", ctx=ei.ast.Load()), args=[], keywords=[])
        analyzer.current_context = None
        analyzer.visit_Call(call_node)
        self.assertIn("mi_funcion", analyzer.used)
    
    def testExpandUsedSymbols1(self): # Se usa a la función func_a que llama a func_b, que a su vez llama a func_c
        used = {"func_a"}
        calls: dict[str, set[str]] = {
            "func_a": {"func_b"},
            "func_b": {"func_c"},
            "func_c": set()
        }
        expanded = ei.expand_used_symbols(used, calls)
        self.assertEqual(expanded, {"func_a", "func_b", "func_c"})
    
    def testExpandUsedSymbols2(self): # Se usa a la función func_a que no llama a nadie
        used = {"func_a"}
        calls: dict[str, set[str]] = {
            "func_a": set(),
            "func_b": {"func_c"},
            "func_c": set()
        }
        expanded = ei.expand_used_symbols(used, calls)
        self.assertEqual(expanded, {"func_a"})
    
    def testExpandUsedSymbols3(self): # No se usa ninguna función
        used: set[str] = set()
        calls: dict[str, set[str]] = {
            "func_a": {"func_b"},
            "func_b": {"func_c"},
            "func_c": set()
        }
        expanded = ei.expand_used_symbols(used, calls)
        self.assertEqual(expanded, set())
    
    def testKeepNodes1(self): # Mantiene solo la función usada
        source = """def func_a():
                        pass
                    def func_b():
                        pass
                    class ClaseC:
                        pass
                    func_a()"""
        tree = ei.ast.parse(source)
        used_syms = {"func_a"}
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = """def func_a():
                                pass
                            func_a()"""
        self.assertEqual(new_source.strip(), expected_source.strip())
    
    def testKeepNodes2(self): # Mantiene la clase usada y la función que llama a la clase
        source = """def func_a():
                        pass
                    def func_b():
                        instancia = ClaseC()
                    class ClaseC:
                        pass
                    func_b()"""
        tree = ei.ast.parse(source)
        used_syms = {"func_b", "ClaseC"}
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = """def func_b():
                                instancia = ClaseC()
                            class ClaseC:
                                pass
                            func_b()"""
        self.assertEqual(new_source.strip(), expected_source.strip())
    
    def testKeepNodes3(self): # No mantiene nada si no se usa nada
        source = """def func_a():
                        pass
                    def func_b():
                        pass
                    class ClaseC:
                        pass"""
        tree = ei.ast.parse(source)
        used_syms: set[str] = set()
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = ""
        self.assertEqual(new_source.strip(), expected_source.strip())