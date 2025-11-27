import ast
import sys
from typing import Set, Dict

class CodeAnalyzer(ast.NodeVisitor):
    """Analiza el código para encontrar funciones, clases y llamadas.
    
    Atributos:
    -----------
    - functions (Set[str]): Conjunto de nombres de funciones definidas.
    - classes (Set[str]): Conjunto de nombres de clases definidas.
    - calls (Dict[str, Set[str]]): Mapa de funciones/clases a las que llaman.
    - used (Set[str]): Conjunto de funciones/clases usadas directamente.
    - current_context (str | None): Contexto actual (función o clase) durante la visita.
    
    Constructor:
    --------------
    __init__(): Inicializa los conjuntos y el contexto actual. No recibe parámetros.
    
    Métodos:
    --------------
    - visit_FunctionDef(node): Registra una definición de función.
    - visit_ClassDef(node): Registra una definición de clase.
    - visit_Call(node): Registra una llamada a función o clase.
    """
    def __init__(self):
        """Inicializa los conjuntos como conjuntos vacíos y el contexto actual como None. No recibe parámetros."""
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.calls: Dict[str, Set[str]] = {}
        self.used: Set[str] = set()
        self.current_context = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Registra una definición de función.
        
        :param ast.FunctionDef node: Nodo AST que representa la definición de la función.
        """
        name = node.name
        self.functions.add(name)
        self.calls[name] = set()
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Registra una definición de clase.
        
        :param ast.ClassDef node: Nodo AST que representa la definición de la clase.
        """
        name = node.name
        self.classes.add(name)
        self.calls[name] = set()
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_Call(self, node: ast.Call) -> None:
        """Registra una llamada a función o clase.
        
        :param ast.Call node: Nodo AST que representa la llamada.
        """
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.used.add(func_name)
            if self.current_context:
                if self.current_context not in self.calls:
                    self.calls[self.current_context] = set()
                self.calls[self.current_context].add(func_name)
        self.generic_visit(node)

def expand_used_symbols(used: Set[str], calls: Dict[str, Set[str]]) -> Set[str]:
    """Expande el conjunto de símbolos usados siguiendo las llamadas internas.
    
    :param Set[str] used: Conjunto inicial de símbolos usados.
    :param Dict[str, Set[str]] calls: Mapa de funciones/clases a las que llaman.
    
    :return: Conjunto expandido de símbolos usados.
    """
    changed = True
    while changed:
        changed = False
        for symbol in list(used):
            for callee in calls.get(symbol, []):
                if callee not in used:
                    used.add(callee)
                    changed = True
    return used


def keep_nodes(tree: ast.Module, used_syms: Set[str]) -> ast.Module:
    """Mantiene solo funciones y clases usadas.
    
    :param ast.Module tree: Árbol AST del módulo.
    :param Set[str] used_syms: Conjunto de símbolos usados.
    :return: Nuevo árbol AST con solo los nodos necesarios.
    """
    new_body: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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


def eliminar_innecesarios(input_path: str, output_path: str) -> None:
    """Elimina funciones y clases innecesarias de un archivo Python.
    
    :param str input_path: Ruta del archivo de entrada.
    :param str output_path: Ruta del archivo de salida.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

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
