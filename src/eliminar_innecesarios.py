import ast
import sys
from typing import Set, Dict
import asttokens

class CodeAnalyzer(ast.NodeVisitor):
    """Analiza el código para encontrar funciones, clases y llamadas.

    Atributos:
    - functions (Set[str]): Conjunto de nombres de funciones definidas.
    - classes (Set[str]): Conjunto de nombres de clases definidas.
    - calls (Dict[str, Set[str]]): Mapa de funciones/clases a las que llaman.
    - used (Set[str]): Conjunto de funciones/clases usadas directamente.
    - current_context (str | None): Contexto actual (función o clase) durante la visita.
    - methods (Dict[str, Set[str]]): Mapa de clases a sus métodos.
    - method_to_class (Dict[str, str]): Mapa de método -> clase para evitar colisiones de nombres.
    """
    def __init__(self):
        self.functions: Set[str] = set()
        self.classes: Set[str] = set()
        self.calls: Dict[str, Set[str]] = {}
        self.used: Set[str] = set()
        self.current_context = None
        self.methods: Dict[str, Set[str]] = {}
        self.method_to_class: Dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        name = node.name
        self.functions.add(name)
        # inicializar conjunto de llamadas para la función/método
        if name not in self.calls:
            self.calls[name] = set()
        # si estamos dentro de una clase, registrar método
        if self.current_context and self.current_context in self.classes:
            self.methods[self.current_context].add(name)
            self.method_to_class[name] = self.current_context
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # tratar igual que FunctionDef
        self.visit_FunctionDef(node)  # type: ignore

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        name = node.name
        self.classes.add(name)
        # inicializar estructura para la clase
        if name not in self.calls:
            self.calls[name] = set()
        self.methods[name] = set()
        prev = self.current_context
        self.current_context = name
        self.generic_visit(node)
        self.current_context = prev

    def visit_Call(self, node: ast.Call) -> None:
        # Llamada a método (obj.method())
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # marcar que se usa ese método (por nombre)
            self.used.add(method_name)
        # Llamada al constructor/clase C(...)
        elif isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.classes:
                # marcar uso de la clase (constructor)
                self.used.add(name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            name = node.id
            if self.current_context:
                if self.current_context not in self.calls:
                    self.calls[self.current_context] = set()
                self.calls[self.current_context].add(name)
            else:
                self.used.add(name)
        self.generic_visit(node)

    def finalize(self) -> None:
        """Post-procesa la información recolectada para que:
        - Llamar a cualquier método de una clase incluya las llamadas hechas por todos los métodos
          de esa misma clase.
        - Llamar al constructor/usar la clase incluya las llamadas hechas por todos sus métodos.
        Esto evita perder dependencias internas de la clase cuando se llama a solo un método.
        """
        # Para cada clase, obtener la unión de las llamadas de todos sus métodos
        for cls, methods in self.methods.items():
            union_calls: Set[str] = set()
            for m in methods:
                union_calls |= self.calls.get(m, set())
            # la clase como símbolo también "llama" a esa unión (útil si se instancia la clase)
            self.calls[cls] = self.calls.get(cls, set()) | union_calls
            # además propagar esa unión a cada método concreto de la clase
            for m in methods:
                if m not in self.calls:
                    self.calls[m] = set()
                self.calls[m] |= union_calls

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
    analyzer.finalize()  # Agregar esta línea para aplicar el cambio de forma integral

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
