import ast
import inspect
import importlib
import sys
import types
import os
from typing import Dict, Union, List

def is_standard_module(module_name: str) -> bool:
    """Verifica si un módulo es estándar de manera robusta.
    
    :param module_name: Nombre del módulo a verificar.
    :return: True si es un módulo estándar, False en caso contrario."""
    # Python 3.10+ tiene una lista oficial de nombres de módulos estándar
    if sys.version_info >= (3, 10):
        return module_name in sys.stdlib_module_names
    
    # Fallback para versiones anteriores
    try:
        module = importlib.import_module(module_name)
        filepath = getattr(module, '__file__', None)
        
        # Si no tiene archivo, suele ser builtin (sys, time, etc.)
        if filepath is None:
            return True
            
        # Verificar si está en la ruta de librerías del sistema pero NO en site-packages
        # Y asegurarnos de que no es local (cwd)
        import os
        stdlib_path = os.path.dirname(os.__file__) # Ruta base de la librería estándar
        return filepath.startswith(stdlib_path) and 'site-packages' not in filepath
    except Exception:
        return False
    
def extract_local_functions(module: types.ModuleType) -> Dict[str, str]:
    """Devuelve un diccionario con funciones definidas por el usuario en un módulo local.
    
    :param module: Módulo del cual extraer funciones.
    :return: Diccionario con nombres de funciones como claves y su código fuente como valores."""
    functions: Dict[str, str] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        src_file = inspect.getsourcefile(obj)
        mod_file = getattr(module, '__file__', None)
        # solo funciones propias: ambos archivos deben existir y coincidir (comparamos rutas absolutas)
        if src_file is not None and mod_file is not None and os.path.abspath(src_file) == os.path.abspath(mod_file):
            functions[name] = inspect.getsource(obj)
    return functions

# Transform the AST to remove prefixes for the unprefixed names
class PrefixRemover(ast.NodeTransformer):
    def __init__(self, names_to_unprefix: set[str]):
        self.names_to_unprefix = names_to_unprefix

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.names_to_unprefix:
            # Replace with Name of the attr
            return ast.Name(id=node.attr, ctx=node.ctx)
        self.generic_visit(node)
        return node

def is_constant_name(name: str) -> bool:
    # Consideramos constantes las variables en MAYÚSCULAS (permitiendo guiones bajos y dígitos).
    # Excluimos nombres "dunder" (__X__).
    return (
        name.isupper() and
        not (name.startswith("__") and name.endswith("__"))
    )

def extract_local_constants(module: types.ModuleType) -> Dict[str, str]:
    """Devuelve un diccionario solo con constantes (nombres en MAYÚSCULAS) definidas por el usuario en un módulo local.
    
    :param module: Módulo del cual extraer constantes.
    :return: Diccionario con nombres de constantes como claves y su código fuente como valores."""
    mod_file = getattr(module, '__file__', None)
    if not mod_file:
        return {}
    
    with open(mod_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    tree = ast.parse(code)
    constants: Dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if is_constant_name(name):
                    source_segment = ast.get_source_segment(code, node)
                    if source_segment:
                        constants[name] = source_segment + "\n"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if is_constant_name(name):
                    source_segment = ast.get_source_segment(code, node)
                    if source_segment:
                        constants[name] = source_segment + "\n"
    return constants

def expand_imports(script_path: Union[str, os.PathLike[str]], output_path: Union[str, os.PathLike[str]]) -> bool:
    """Expande las importaciones locales en un script Python y genera un nuevo archivo con el código original
    y las funciones, clases y constantes importadas, removiendo las importaciones de librerías no estándar del código original.
        
    :param script_path: Ruta al script Python original.
    :param output_path: Ruta al archivo de salida con las importaciones expandidas.
    :return: True si quedan más librerías locales por expandir, False si no.
    """
    script_path = str(script_path)
    script_dir = os.path.abspath(os.path.dirname(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = ast.parse(code)

    # Recopilar importaciones estándar del script original
    original_standard_imports: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_standard_module(alias.name):
                    original_standard_imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and is_standard_module(node.module):
                names = ', '.join(alias.name for alias in node.names)
                level = '.' * node.level if node.level else ''
                original_standard_imports.append(f"from {level}{node.module} import {names}")

    # Recopilar constantes del script original (top-level Assign y AnnAssign que sean constantes)
    original_constants: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if is_constant_name(name):
                    source_segment = ast.get_source_segment(code, node)
                    if source_segment:
                        original_constants[name] = source_segment + "\n"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if is_constant_name(name):
                    source_segment = ast.get_source_segment(code, node)
                    if source_segment:
                        original_constants[name] = source_segment + "\n"

    # Recopilar clases y funciones del script original (top-level)
    original_classes: Dict[str, str] = {}
    original_functions: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            source_segment = ast.get_source_segment(code, node)
            if source_segment:
                original_classes[node.name] = source_segment + "\n\n"
        elif isinstance(node, ast.FunctionDef):
            source_segment = ast.get_source_segment(code, node)
            if source_segment:
                original_functions[node.name] = source_segment + "\n\n"

    # Recopilar importaciones con sus nombres (None significa "import X")
    import_specs: List[tuple[str | None, Union[None, List[str]], int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_specs.append((alias.name, None, 0))
        elif isinstance(node, ast.ImportFrom):
            import_specs.append((node.module, [alias.name for alias in node.names], node.level))

    collected_classes: Dict[str, str] = {}
    collected_functions: Dict[str, str] = {}
    collected_constants: Dict[str, str] = {}
    collected_imports: set[str] = set()

    # Collect names to unprefix (from non-standard 'import X')
    names_to_unprefix: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_standard_module(alias.name):
                    names_to_unprefix.add(alias.asname if alias.asname else alias.name)

    processed_modules: set[str] = set()  # Para evitar procesar el mismo módulo múltiples veces
    for mod, names, _ in import_specs:
        if mod is None:
            continue

        # Construir candidatos a importar:
        candidates: List[str] = []
        if names is None:
            candidates.append(mod)
        else:
            # Para "from mod import a, b" intentamos importar mod.a (submódulo) primero,
            # si falla, importamos mod y extraemos los nombres solicitados desde su archivo.
            for n in names:
                if n == '*':
                    candidates.append(mod)
                else:
                    candidates.append(f"{mod}.{n}")
            candidates.append(mod)

        imported_module = None
        for candidate in candidates:
            if candidate in processed_modules:
                continue  # Ya procesado, saltar
            try:
                imported_module = importlib.import_module(candidate)
                # si importó, lo usamos
                processed_modules.add(candidate)  # Marcar como procesado
                break
            except Exception:
                imported_module = None
                continue

        if imported_module is None:
            continue

        # Evitar módulos estándar/externos
        top_pkg = imported_module.__name__.split('.')[0]
        try:
            if is_standard_module(top_pkg):
                continue
        except Exception:
            pass

        mod_file = getattr(imported_module, '__file__', None)
        if not mod_file:
            continue

        classes_in_file, functions_in_file = extract_local_definitions(imported_module)
        # Siempre recopilar todas las definiciones locales del módulo importado
        collected_classes.update(classes_in_file)
        collected_functions.update(functions_in_file)

        # Extraer constantes locales del módulo importado
        constants_in_file = extract_local_constants(imported_module)
        collected_constants.update(constants_in_file)

        # Extraer imports locales del módulo importado
        imports_in_file = extract_local_imports(imported_module, "" if '.' not in imported_module.__name__ else "lib")
        collected_imports.update(imports_in_file)

    # Filtrar el código original para remover importaciones no estándar, constantes, clases y funciones (quedar solo con el resto)
    filtered_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Ya manejamos imports por separado
            continue
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if is_constant_name(name):
                    continue  # Ya recolectado
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                if is_constant_name(name):
                    continue  # Ya recolectado
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue  # Ya recolectados
        else:
            filtered_body.append(node)
    
    tree.body = filtered_body
    transformer = PrefixRemover(names_to_unprefix)
    tree = transformer.visit(tree)
    remaining_code = ast.unparse(tree)

    # Combinar imports: recolectados primero, luego estándar del original
    all_imports = sorted(collected_imports) + original_standard_imports
    # Combinar constantes: recolectadas primero, luego del original
    all_constants = {**collected_constants, **original_constants}
    # Combinar clases: recolectadas primero, luego del original
    all_classes = {**collected_classes, **original_classes}
    # Combinar funciones: recolectadas primero, luego del original
    all_functions = {**collected_functions, **original_functions}

    # Escribir en orden: imports, constantes, clases, funciones, luego el código restante
    with open(output_path, 'w', encoding='utf-8') as out:
        for imp in all_imports:
            out.write(imp + "\n")
        out.write("\n")
        for const_source in all_constants.values():
            out.write(const_source)
        out.write("\n")
        for _, source in all_classes.items():
            out.write(source)
        for _, source in all_functions.items():
            out.write(source)
        out.write(remaining_code)
        
    # Devolver True si quedan más librerías locales por expandir (si hay imports recolectados)
    def is_import_non_standard(imp: str) -> bool:
        if imp.startswith("import "):
            module = imp.split()[1]
            return not is_standard_module(module)
        elif imp.startswith("from "):
            parts = imp.split()
            module = parts[1]
            return not is_standard_module(module)
        return False
    
    return any(is_import_non_standard(imp) for imp in collected_imports)

def extract_local_imports(module: types.ModuleType, package: str = "") -> List[str]:
    """Devuelve una lista de importaciones locales en un módulo.
    
    :param module: Módulo del cual extraer importaciones.
    :param package: Prefijo de paquete para las importaciones (si es necesario).
    :return: Lista de strings con las importaciones locales."""
    mod_file = getattr(module, '__file__', None)
    if not mod_file:
        return []
    
    with open(mod_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    tree = ast.parse(code)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_standard_module(alias.name):
                    imports.append(f"import {package}{alias.name}")
                else:
                    imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = ', '.join(alias.name for alias in node.names)
                level = '.' * node.level if node.level else ''
                if not is_standard_module(node.module):
                    imports.append(f"from {package}{level}{node.module} import {names}")
                else:
                    imports.append(f"from {level}{node.module} import {names}")
    return imports

def extract_local_definitions(module: types.ModuleType) -> tuple[Dict[str, str], Dict[str, str]]:
    """Devuelve una tupla con dos diccionarios: uno para clases y otro para funciones definidas por el usuario en un módulo local.
    
    :param module: Módulo del cual extraer definiciones.
    :return: Tupla (clases, funciones) donde cada uno es un diccionario con nombres como claves y código fuente como valores."""
    classes: Dict[str, str] = {}
    functions: Dict[str, str] = {}
    for name, obj in inspect.getmembers(module, lambda x: inspect.isfunction(x) or inspect.isclass(x)):
        src_file = inspect.getsourcefile(obj)
        mod_file = getattr(module, '__file__', None)
        # solo definiciones propias: ambos archivos deben existir y coincidir (comparamos rutas absolutas)
        if src_file is not None and mod_file is not None and os.path.abspath(src_file) == os.path.abspath(mod_file):
            try:
                if inspect.isclass(obj):
                    classes[name] = inspect.getsource(obj)
                elif inspect.isfunction(obj):
                    functions[name] = inspect.getsource(obj)
            except Exception:
                pass  # Ignorar si no se puede obtener el source
    return classes, functions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python expand_imports.py <ruta_script> <ruta_salida>")
        sys.exit(1)

    script_path = sys.argv[1]
    output_path = sys.argv[2]
    temp1 = "temp1.py"
    temp2 = "temp2.py"
    quedan_librerias_propias = True
    iteracion = 0
    while quedan_librerias_propias:
        if iteracion == 0:
            quedan_librerias_propias = expand_imports(script_path, temp1)
        elif iteracion % 2 == 1:
            quedan_librerias_propias = expand_imports(temp1, temp2)
        else:
            quedan_librerias_propias = expand_imports(temp2, temp1)
        iteracion += 1
    if iteracion % 2 - 1 == 0:
        os.replace(temp1, output_path)
        os.remove(temp2)
    else:
        os.replace(temp2, output_path)
        os.remove(temp1)