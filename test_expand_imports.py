import importlib
import sys
import os

from pytest import fail
import expand_imports as ei
import unittest

sys.path.append(os.getcwd())

class ExpandImportsTest(unittest.TestCase):
    def test_is_standard_module(self):
        self.assertTrue(ei.is_standard_module("math"))
        self.assertTrue(ei.is_standard_module("sys"))
        self.assertTrue(ei.is_standard_module("typing"))
        self.assertFalse(ei.is_standard_module("lib"))  # assuming 'lib' is a local module
        self.assertFalse(ei.is_standard_module("lib.MatrizFactory"))  # assuming 'lib.MatrizFactory' is a local module

    def test_extract_local_functions(self):
        try:
            modulo = importlib.import_module("lib.MatrizFactory")
        except ImportError:
            fail("No se pudo importar el módulo 'lib.MatrizFactory'. Asegúrese de que el archivo existe y la ruta es correcta.")
        funciones_obtenidas = ei.extract_local_functions(modulo)
        funciones_esperadas = {
            "aleatorios":
            """def aleatorios(filas: int, columnas: int, min_val: float = 0.0, max_val: float = 1.0) -> Matriz[float]:\n
            '''Crea una matriz de números aleatorios con las dimensiones especificadas.\n
            \n
            Parámetros:\n
                filas (int): Número de filas de la matriz.\n
                columnas (int): Número de columnas de la matriz.\n
                min_val (float): Valor mínimo para los números aleatorios.\n
                max_val (float): Valor máximo para los números aleatorios.\n
            \n
            Retorna:\n
                Matriz[float]: Una matriz de números aleatorios con las dimensiones dadas.\n
            '''\n
            return Matriz([[random.uniform(min_val, max_val) for _ in range(columnas)] for _ in range(filas)])\n
        """,
            "ceros":
            """def ceros(filas: int, columnas: int) -> Matriz[float]:\n
            '''Crea una matriz de ceros con las dimensiones especificadas.\n
            \n
            Parámetros:\n
                filas (int): Número de filas de la matriz.\n
                columnas (int): Número de columnas de la matriz.\n
            \n
            Retorna:\n
                Matriz[float]: Una matriz llena de ceros con las dimensiones dadas.\n
            '''\n
            return Matriz([[0.0 for _ in range(columnas)] for _ in range(filas)])\n
        """,
            "identidad":
            """def identidad(tamano: int) -> Matriz[float]:\n
            '''Crea la matriz identidad de tamaño dado.\n
            \n
            Parámetros:\n
            tamano (int): Dimensión de la matriz identidad (n x n).\n
            \n
            Retorna:\n
                Matriz[float]: La matriz identidad de tamaño dado.\n
            '''\n
            return Matriz([[1.0 if i == j else 0.0 for j in range(tamano)] for i in range(tamano)])\n
        """,
            "relleno":
            """def relleno(filas: int, columnas: int, valor: float) -> Matriz[float]:\n
            '''Crea una matriz rellenada con un valor especificado.\n
            \n
            Parámetros:\n
                filas (int): Número de filas de la matriz.\n
                columnas (int): Número de columnas de la matriz.\n                valor (float): Valor con el que se rellenará cada entrada.\n
            \n
            Retorna:\n
                Matriz[float]: Una matriz con todas las entradas igual al valor dado.\n
            '''\n
            return Matriz([[valor for _ in range(columnas)] for _ in range(filas)])\n
        """,
            "unos":
            """def unos(filas: int, columnas: int) -> Matriz[float]:\n
            '''Crea una matriz de unos con las dimensiones especificadas.\n
            \n
            Parámetros:\n
                filas (int): Número de filas de la matriz.\n
                columnas (int): Número de columnas de la matriz.\n
            \n
            Retorna:\n
            Matriz[float]: Una matriz llena de unos con las dimensiones dadas.\n
            '''\n
            return Matriz([[1.0 for _ in range(columnas)] for _ in range(filas)])\n
        """}
        print(funciones_obtenidas)
        self.assertEqual(funciones_obtenidas, funciones_esperadas)
    
    def test_extract_local_definitions1(self):
        modulo = importlib.import_module("lib.MatrizFactory")
        definiciones_obtenidas = ei.extract_local_definitions(modulo)
        definiciones_esperadas = {
            "aleatorios": ei.extract_local_functions(modulo)["aleatorios"],
            "ceros": ei.extract_local_functions(modulo)["ceros"],
            "identidad": ei.extract_local_functions(modulo)["identidad"],
            "relleno": ei.extract_local_functions(modulo)["relleno"],
            "unos": ei.extract_local_functions(modulo)["unos"],
        }
        self.assertEqual(definiciones_obtenidas, definiciones_esperadas)
    
    def test_extract_local_constants1(self):
        modulo = importlib.import_module("lib.Matriz")
        constantes_obtenidas = ei.extract_local_constants(modulo)
        constantes_esperadas = {
            "T": "T = TypeVar('T')\n",
        }
        self.assertEqual(constantes_obtenidas, constantes_esperadas)
    
    def test_extract_local_constants2(self):
        modulo = importlib.import_module("experimentos")
        constantes_obtenidas = ei.extract_local_constants(modulo)
        constantes_esperadas = {}
        self.assertEqual(constantes_obtenidas, constantes_esperadas)
    
    def test_expand_imports(self):
        self.assertTrue(ei.expand_imports("experimentos.py", "salida_de_prueba.py"))
        self.assertFalse(ei.expand_imports("salida_de_prueba.py", "salida_de_prueba_2.py"))
    
    def test_extract_local_imports(self):
        modulo = importlib.import_module("experimentos")
        imports_obtenidos = ei.extract_local_imports(modulo)
        imports_esperados = [
            "import lib.MatrizFactory",
            "import lib.Unidad4"
        ]
        self.assertEqual(imports_obtenidos, imports_esperados)

    def test_integracion(self):
        codigo_esperado: str
        codigo_obtenido: str
        with open("salida_de_prueba_2.py", "r", encoding="utf-8") as f:
            codigo_esperado = f.read()
        with open("salida_resultante.py", "r", encoding="utf-8") as f:
            codigo_obtenido = f.read()
        self.assertEqual(codigo_obtenido, codigo_esperado)