import ast
import sys
from typing import Set, Dict
import asttokens

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

    def visit_Name(self, node: ast.Name) -> None:
        """Registra el uso de un nombre (variable, función, clase, import).
        
        :param ast.Name node: Nodo AST que representa el nombre.
        """
        if isinstance(node.ctx, ast.Load):
            name = node.id
            if self.current_context:
                if self.current_context not in self.calls:
                    self.calls[self.current_context] = set()
                self.calls[self.current_context].add(name)
            else:
                self.used.add(name)
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
    """Elimina funciones, clases e imports innecesarios de un archivo Python.
    
    :param str input_path: Ruta del archivo de entrada.
    :param str output_path: Ruta del archivo de salida.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()

    atok = asttokens.ASTTokens(source, parse=True)
    tree = atok.tree
    analyzer = CodeAnalyzer()
    analyzer.visit(tree) # type: ignore

    # Símbolos usados directamente o en el código principal
    used = set(analyzer.used)

    # Expandir por llamadas internas
    used = expand_used_symbols(used, analyzer.calls)

    # Reconstruir el código manteniendo solo lo necesario, preservando comentarios
    new_body: list[str] = []
    last_category = None  # 'import', 'def', 'other'

    for node in tree.body: # type: ignore
        content = None
        category = None

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in used:
                content = atok.get_text(node) # type: ignore
                category = 'def'
        elif isinstance(node, ast.ClassDef):
            if node.name in used:
                content = atok.get_text(node) # type: ignore
                category = 'def'
        elif isinstance(node, ast.Import):
            new_names = [alias for alias in node.names if (alias.asname or alias.name) in used]
            if new_names:
                node.names = new_names
                content = ast.unparse(node)
                category = 'import'
        elif isinstance(node, ast.ImportFrom):
            new_names = [alias for alias in node.names if (alias.asname or alias.name) in used]
            if new_names:
                node.names = new_names
                content = ast.unparse(node)
                category = 'import'
        else:
            # Código suelto siempre se conserva
            content = atok.get_text(node) # type: ignore
            category = 'other'

        if content:
            prefix = ""
            # Separar funciones y clases con un salto de línea extra
            if category == 'def':
                prefix = "\n"
            # Separar el código principal (main) de los imports o definiciones anteriores
            elif category == 'other' and last_category in ('import', 'def'):
                prefix = "\n"
            
            new_body.append(prefix + content)
            last_category = category

    # Código limpio
    cleaned = '\n'.join(new_body).strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Archivo limpio guardado en: {output_path}")

# Ejemplo de uso:
# eliminar_innecesarios("entrada.py", "salida_limpia.py")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        eliminar_innecesarios(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python eliminar_innecesarios.py archivo_entrada.py archivo_salida.py")
