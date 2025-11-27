import ast
import sys
from typing import Set, Dict

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.calls: Dict[str, Set[str]] = {}
        self.used: Set[str] = set()
        self.current_context = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        name = node.name
        self.functions.add(name)
        self.calls[name] = set()
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_ClassDef(self, node: ast.ClassDef):
        name = node.name
        self.classes.add(name)
        self.calls[name] = set()
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.used.add(func_name)

            if self.current_context:
                self.calls[self.current_context].add(func_name)

        self.generic_visit(node)

    def visit(self, node: ast.AST):
        super().visit(node)
        return self


def expand_used_symbols(used: Set[str], calls: Dict[str, Set[str]]):
    changed = True
    while changed:
        changed = False
        for symbol in list(used):
            for callee in calls.get(symbol, []):
                if callee not in used:
                    used.add(callee)
                    changed = True
    return used


def keep_nodes(tree: ast.Module, used_syms: Set[str]):
    """Mantiene solo funciones y clases usadas."""
    new_body: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in used_syms:
                new_body.append(node)

        elif isinstance(node, ast.ClassDef):
            if node.name in used_syms:
                new_body.append(node)

        else:
            # Código suelto siempre se conserva
            new_body.append(node)

    tree.body = new_body
    return tree


def eliminar_innecesarios(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    analyzer = CodeAnalyzer().visit(tree)

    # Símbolos usados directamente o en el código principal
    used = set(analyzer.used)

    # Expandir por llamadas internas
    used = expand_used_symbols(used, analyzer.calls)

    # Mantener solo funciones/clases necesarias
    new_tree = keep_nodes(tree, used)

    # Código limpio
    cleaned = ast.unparse(new_tree)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Archivo generado sin código muerto: {output_path}")


# Ejemplo de uso:
# eliminar_innecesarios("entrada.py", "salida_limpia.py")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        eliminar_innecesarios(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python eliminar_innecesarios.py archivo_entrada.py archivo_salida.py")
