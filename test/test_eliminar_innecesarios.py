import sys
import os
import textwrap
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import eliminar_innecesarios as ei
import unittest

def normalize_code(code: str) -> str:
    """Elimina líneas vacías para comparar código sin importar el formato de líneas en blanco."""
    return "\n".join([line for line in code.splitlines() if line.strip()])

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
        # Usar get para evitar KeyError si "otra_funcion" no fue creada
        self.assertNotIn("metodo", analyzer.calls.get("otra_funcion", set()))
    
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
    
    def testExpandUsedSymbols4(self):
        used = {"Clase"}
        calls: dict[str, set[str]] = {
            "Clase": {"metodo_a"},
            "metodo_a": set()
        }
        expanded = ei.expand_used_symbols(used, calls)
        self.assertEqual(expanded, {"Clase", "metodo_a"})
    
    def testKeepNodes1(self): # Mantiene solo la función usada
        source = textwrap.dedent("""
            def func_a():
                pass
            def func_b():
                pass
            class ClaseC:
                pass
            func_a()
        """).strip()
        tree = ei.ast.parse(source)
        used_syms = {"func_a"}
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = textwrap.dedent("""
            def func_a():
                pass
            func_a()
        """).strip()
        self.assertEqual(normalize_code(new_source), normalize_code(expected_source))
    
    def testKeepNodes2(self): # Mantiene la clase usada y la función que llama a la clase
        source = textwrap.dedent("""
            def func_a():
                pass
            def func_b():
                instancia = ClaseC()
            class ClaseC:
                pass
            func_b()
        """).strip()
        tree = ei.ast.parse(source)
        used_syms = {"func_b", "ClaseC"}
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = textwrap.dedent("""
            def func_b():
                instancia = ClaseC()
            class ClaseC:
                pass
            func_b()
        """).strip()
        self.assertEqual(normalize_code(new_source), normalize_code(expected_source))
    
    def testKeepNodes3(self): # No mantiene nada si no se usa nada
        source = textwrap.dedent("""
            def func_a():
                pass
            def func_b():
                pass
            class ClaseC:
                pass
        """).strip()
        tree = ei.ast.parse(source)
        used_syms: set[str] = set()
        new_tree = ei.keep_nodes(tree, used_syms)
        new_source = ei.ast.unparse(new_tree)
        expected_source = ""
        self.assertEqual(new_source.strip(), expected_source.strip())
    
    def testEliminarInnecesarios1(self):
        source = textwrap.dedent("""
            def func_a():
                func_b()
            def func_b():
                pass
            def func_c():
                pass
            class ClaseD:
                pass
            func_a()
        """).strip()
        with open("temp_test_file.py", "w", encoding="utf-8") as f:
            f.write(source)
        
        ei.eliminar_innecesarios("temp_test_file.py", "temp_test_file.py")
        
        with open("temp_test_file.py", "r", encoding="utf-8") as f:
            new_source = f.read().strip()
        
        expected_source = textwrap.dedent("""
            def func_a():
                func_b()
            def func_b():
                pass
            func_a()
        """).strip()
        
        self.assertEqual(normalize_code(new_source), normalize_code(expected_source))
        
        os.remove("temp_test_file.py")
    
    def testEliminarInnecesarios2(self):
        source = textwrap.dedent("""
            def func_x():
                pass
            class ClaseY:
                def metodo_z(self):
                    pass
                def metodo_w(self):
                    func_x()
            objY = ClaseY()
            objY.metodo_w()
        """).strip()
        with open("temp_test_file2.py", "w", encoding="utf-8") as f:
            f.write(source)
        
        ei.eliminar_innecesarios("temp_test_file2.py", "temp_test_file2.py")
        
        with open("temp_test_file2.py", "r", encoding="utf-8") as f:
            new_source = f.read().strip()
        print("Codigo resultante:")
        print(new_source)
        print("Codigo original:")
        print(source)
        self.assertEqual(normalize_code(new_source), normalize_code(source))
        
        os.remove("temp_test_file2.py")
    
    def testEliminarInnecesarios3(self):
        source = textwrap.dedent("""
            def func_x():
                pass
            class ClaseY:
                def metodo_z(self):
                    pass
                def metodo_w(self):
                    func_x()
            objY = ClaseY()
            objY.metodo_z()
        """).strip()
        with open("temp_test_file2.py", "w", encoding="utf-8") as f:
            f.write(source)
        
        ei.eliminar_innecesarios("temp_test_file2.py", "temp_test_file2.py")
        
        with open("temp_test_file2.py", "r", encoding="utf-8") as f:
            new_source = f.read().strip()
            
        os.remove("temp_test_file2.py")
        self.assertEqual(normalize_code(new_source), normalize_code(source))