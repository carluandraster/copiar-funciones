from typing import TypeVar, Generic
import math
import random

T = TypeVar('T')

class Matriz(Generic[T]):
    """
    Clase que representa una matriz genérica.
    
    Métodos:
    -------------
    - <b>__init__(self, valores: list[list[T]]):</b> Constructor de la clase Matriz.
    - <b>cantidadFilas(self) -> int:</b> Devuelve la cantidad de filas de la matriz.
    - <b>cantidadColumnas(self) -> int:</b> Devuelve la cantidad de columnas de la matriz.
    - <b>inversa(self) -> 'Matriz[float]':</b> Devuelve la matriz inversa si es posible.
    - <b>traspuesta(self) -> 'Matriz[T]':</b> Devuelve la matriz transpuesta.
    - <b>normalizarColumnas(self) -> None:</b> Normaliza las columnas de la matriz.
    
    También soporta las siguientes operaciones:
    -------------
    - Acceso y modificación de elementos mediante índices.
    - Comparación de matrices (==, !=).
    - Suma y resta de matrices (+, -, +=, -=).
    - Multiplicación de matrices y por escalares (*, *=).
    - Representación en cadena (__str__, __repr__).
    """
    __cantFilas: int
    __cantColumnas: int
    __matriz: list[list[T]]

    def __init__(self, valores: list[list[T]]):
        """
        Constructor de la clase Matriz.

        Parámetros:
            - valores (list[list[T]]): Una lista de listas que representa los valores iniciales de la matriz.
        
        Contrato:
            - len(valores) > 0
            - all(len(fila) == len(valores[0]) for fila in valores)
            - Contruye una matriz con las dimensiones dadas y llena con los valores proporcionados.
        """
        self.__cantFilas = len(valores)
        self.__cantColumnas = len(valores[0])
        for i in range(1, self.__cantFilas):
            if len(valores[i]) != self.__cantColumnas:
                raise ValueError("Todas las filas deben tener la misma cantidad de columnas.")
        self.__matriz = [fila[:] for fila in valores]

    @property
    def cantidadFilas(self) -> int:
        return self.__cantFilas
    @property
    def cantidadColumnas(self) -> int:
        return self.__cantColumnas

    def __getitem__(self, indice: int):
        return self.__matriz[indice]
    
    def __setitem__(self, key: int, value: list[T]) -> None:
        self.__matriz[key] = value
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Matriz):
            return NotImplemented
        return (self.__cantFilas, self.__cantColumnas, self.__matriz) == (value.__cantFilas, value.__cantColumnas, value.__matriz) # type: ignore

    def __ne__(self, otro: object) -> bool:
        if not isinstance(otro, Matriz):
            return NotImplemented
        return not self == otro
    
    def __add__(self, other: object) -> 'Matriz[T]':
        resultado: list[list[T]]
        if not isinstance(other, Matriz):
            return NotImplemented
        if self.__cantFilas != other.__cantFilas or self.__cantColumnas != other.__cantColumnas:
            raise ValueError("Las matrices deben tener las mismas dimensiones para sumarse.")
        resultado = [[self.__matriz[i][j] + other.__matriz[i][j] for j in range(self.__cantColumnas)] for i in range(self.__cantFilas)] # type: ignore
        return Matriz(resultado)
    
    def __iadd__(self, other: object) -> 'Matriz[T]':
        if not isinstance(other, Matriz):
            return NotImplemented
        if self.__cantFilas != other.__cantFilas or self.__cantColumnas != other.__cantColumnas:
            raise ValueError("Las matrices deben tener las mismas dimensiones para sumarse.")
        
        for i in range(self.__cantFilas):
            for j in range(self.__cantColumnas):
                self.__matriz[i][j] += other.__matriz[i][j] # type: ignore
        return self
    
    def __sub__(self, other: 'Matriz[T]') -> 'Matriz[T]':
        resultado: list[list[T]]
        if self.__cantFilas != other.__cantFilas or self.__cantColumnas != other.__cantColumnas:
            raise ValueError("Las matrices deben tener las mismas dimensiones para restarse.")
        
        resultado = [[self.__matriz[i][j] - other.__matriz[i][j] for j in range(self.__cantColumnas)] for i in range(self.__cantFilas)] # type: ignore
        return Matriz(resultado)
    
    def __isub__(self, other: 'Matriz[T]') -> 'Matriz[T]':
        if self.__cantFilas != other.__cantFilas or self.__cantColumnas != other.__cantColumnas:
            raise ValueError("Las matrices deben tener las mismas dimensiones para restarse.")
        
        for i in range(self.__cantFilas):
            for j in range(self.__cantColumnas):
                self.__matriz[i][j] -= other.__matriz[i][j] # type: ignore
        return self
    
    def __mul__(self, other: object) -> 'Matriz[T]':
        resultado: list[list[T]]
        if isinstance(other, Matriz):
            # Multiplicación de matrices
            if self.__cantColumnas != other.__cantFilas:
                raise ValueError("El número de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz para multiplicarse.")
            resultado = [[sum(self.__matriz[i][k] * other.__matriz[k][j] for k in range(self.__cantColumnas)) for j in range(other.__cantColumnas)] for i in range(self.__cantFilas)] # type: ignore
            return Matriz(resultado)
        else:
            # Multiplicación por escalar
            resultado = [[self.__matriz[i][j] * other for j in range(self.__cantColumnas)] for i in range(self.__cantFilas)] # type: ignore
            return Matriz(resultado)
    
    def __imul__(self, otro: object) -> 'Matriz[T]':
        self = self * otro
        return self

    def __str__(self) -> str:
        return '\n'.join(['\t'.join([str(self.__matriz[i][j]) if self.__matriz[i][j] is not None else 'None' for j in range(self.__cantColumnas)]) for i in range(self.__cantFilas)])
    
    def __repr__(self) -> str:
        return f"Matriz(filas={self.__cantFilas}, columnas={self.__cantColumnas}, valores={self.__matriz})"
    
    @property
    def inversa(self) -> 'Matriz[float]':
        """
        Calcula la matriz inversa utilizando el método de eliminación de Gauss-Jordan.

        Retorna:
            Matriz[float]: La matriz inversa si es invertible.
        
        Lanza:
            ValueError: Si la matriz no es cuadrada o no es invertible.
        """
        if self.__cantFilas != self.__cantColumnas:
            raise ValueError("La matriz debe ser cuadrada para calcular su inversa.")
        
        n = self.__cantFilas
        identidad = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        matriz_extendida = [self.__matriz[i] + identidad[i] for i in range(n)]
        
        for i in range(n):
            if matriz_extendida[i][i] == 0:
                for j in range(i + 1, n):
                    if matriz_extendida[j][i] != 0:
                        matriz_extendida[i], matriz_extendida[j] = matriz_extendida[j], matriz_extendida[i]
                        break
            
            divisor = matriz_extendida[i][i]
            if divisor == 0:
                raise ValueError("La matriz no es invertible.")
            
            for j in range(2 * n):
                matriz_extendida[i][j] /= divisor # type: ignore
            
            for k in range(n):
                if k != i:
                    factor = matriz_extendida[k][i]
                    for j in range(2 * n):
                        matriz_extendida[k][j] -= factor * matriz_extendida[i][j] # type: ignore
        
        inversa = [fila[n:] for fila in matriz_extendida]
        return Matriz(inversa) # type: ignore
    
    def getFila(self, fila: int) -> list[T]:
        """
        Devuelve la fila especificada de la matriz.
        Parámetros:
            fila (int): El índice de la fila a obtener (0-indexado).
        Lanza:
            IndexError: Si el índice de fila está fuera de rango.
        """
        if 0 <= fila < self.__cantFilas:
            return self.__matriz[fila][:]
        raise IndexError("Índice de fila fuera de rango.")
    
    def getColumna(self, columna: int) -> list[T]:
        """
        Devuelve la columna especificada de la matriz.
        Parámetros:
            columna (int): El índice de la columna a obtener (0-indexado).
        Lanza:
            IndexError: Si el índice de columna está fuera de rango.
        """
        if 0 <= columna < self.__cantColumnas:
            return [self.__matriz[i][columna] for i in range(self.__cantFilas)]
        raise IndexError("Índice de columna fuera de rango.")
    
    def __len__(self) -> int:
        return self.__cantFilas * self.__cantColumnas
    
    def agregarFila(self, fila: list[T]) -> None:
        """
        Agrega una nueva fila a la matriz.

        Parámetros:
            fila (list[T]): La fila a agregar. Debe tener el mismo número de columnas que la matriz.

        Lanza:
            ValueError: Si la longitud de la fila no coincide con el número de columnas.
        """
        if len(fila) != self.__cantColumnas:
            raise ValueError("La fila debe tener el mismo número de columnas que la matriz.")
        self.__matriz.append(fila)
        self.__cantFilas += 1
    
    def agregarColumna(self, columna: list[T]) -> None:
        """
        Agrega una nueva columna a la matriz.

        Parámetros:
            columna (list[T]): La columna a agregar. Debe tener el mismo número de filas que la matriz.

        Lanza:
            ValueError: Si la longitud de la columna no coincide con el número de filas.
        """
        if len(columna) != self.__cantFilas:
            raise ValueError("La columna debe tener el mismo número de filas que la matriz.")
        for i in range(self.__cantFilas):
            self.__matriz[i].append(columna[i])
        self.__cantColumnas += 1
    
    def insertar_fila(self, indice: int, fila: list[T]) -> None:
        """
        Inserta una fila en la posición especificada.

        Parámetros:
            indice (int): La posición donde se insertará la fila (0-indexado).
            fila (list[T]): La fila a insertar. Debe tener el mismo número de columnas que la matriz.

        Lanza:
            ValueError: Si la longitud de la fila no coincide con el número de columnas.
            IndexError: Si el índice está fuera de rango.
        """
        if len(fila) != self.__cantColumnas:
            raise ValueError("La fila debe tener el mismo número de columnas que la matriz.")
        if not (0 <= indice <= self.__cantFilas):
            raise IndexError("Índice fuera de rango.")
        self.__matriz.insert(indice, fila)
        self.__cantFilas += 1
    
    def insertar_columna(self, indice: int, columna: list[T]) -> None:
        """
        Inserta una columna en la posición especificada.

        Parámetros:
            indice (int): La posición donde se insertará la columna (0-indexado).
            columna (list[T]): La columna a insertar. Debe tener el mismo número de filas que la matriz.

        Lanza:
            ValueError: Si la longitud de la columna no coincide con el número de filas.
            IndexError: Si el índice está fuera de rango.
        """
        if len(columna) != self.__cantFilas:
            raise ValueError("La columna debe tener el mismo número de filas que la matriz.")
        if not (0 <= indice <= self.__cantColumnas):
            raise IndexError("Índice fuera de rango.")
        for i in range(self.__cantFilas):
            self.__matriz[i].insert(indice, columna[i])
        self.__cantColumnas += 1

    def normalizar(self, axis: bool = False) -> None:
        """
        Normaliza la matriz a lo largo del eje especificado.
        Parámetros:
            axis (bool): Si es True, normaliza por filas. Si es False, normaliza por columnas.
        """
        total: float
        if axis:
            for i in range(self.__cantFilas):
                total = float(sum(self.__matriz[i])) # type: ignore
                if total > 0:
                    self.__matriz[i] = [x / total for x in self.__matriz[i]] # type: ignore
        else:
            for j in range(self.__cantColumnas):
                total = float(sum([self.__matriz[i][j] for i in range(self.__cantFilas)])) # type: ignore
                if total > 0:
                    for i in range(self.__cantFilas):
                        self.__matriz[i][j] /= total # type: ignore
    
    @property
    def traspuesta(self) -> 'Matriz[T]':
        """
        Devuelve la matriz transpuesta.
        Retorna:
            Matriz[T]: La matriz transpuesta.
        """
        resultado = [[self.__matriz[j][i] for j in range(self.__cantFilas)] for i in range(self.__cantColumnas)]
        return Matriz(resultado)
    
    @property
    def clone(self) -> 'Matriz[T]':
        """
        Devuelve una copia de la matriz.
        Retorna:
            Matriz[T]: Una copia de la matriz actual.
        """
        valores_copiados = [[self.__matriz[i][j] for j in range(self.__cantColumnas)] for i in range(self.__cantFilas)]
        return Matriz(valores_copiados)
    
    def redondear(self, decimales: int) -> 'Matriz[T]':
        """
        Redondea los elementos de la matriz a un número específico de decimales.
        
        Args:
            decimales (int): El número de decimales a los que redondear los elementos de la matriz.

        Returns:
            Matriz[T]: Una nueva matriz con los elementos redondeados.
        """
        matriz_redondeada = self.clone
        for i in range(matriz_redondeada.__cantFilas):
            for j in range(matriz_redondeada.__cantColumnas):
                matriz_redondeada.__matriz[i][j] = round(matriz_redondeada.__matriz[i][j], decimales) # type: ignore
        return matriz_redondeada
class Decodificador:
    """Clase que permite decodificar mensajes codificados o codificar mensajes
    utilizando un alfabeto fuente y una codificacion dada.
    
    Métodos:
        - decodificar(mensaje_codificado: str) -> str: Decodifica un mensaje codificado.
        - codificar(mensaje: str) -> str: Codifica un mensaje utilizando la codificacion dada.
    """
    __alfabeto_fuente: str
    __codificacion: list[str]

    def __init__(self, alfabeto_fuente: str, codificacion: list[str]):
        """Constructor de la clase Decodificador.
        
        Parametros:
            - alfabeto_fuente: lista de simbolos del alfabeto fuente.
            - codificacion: lista de codigos correspondientes a cada simbolo del alfabeto fuente.
        
        Precondiciones:
            - len(alfabeto_fuente) == len(codificacion)
            - Todos los elementos de alfabeto_fuente son distintos.
            - Tanto alfabeto_fuente como codificacion no estan vacios y son distintos de None.
            - La codificacion es binaria (solo contiene '0' y '1').
        """
        self.__alfabeto_fuente = alfabeto_fuente
        self.__codificacion = codificacion

    def decodificar(self, mensaje_codificado: bytearray) -> str:
        """Decodifica un mensaje codificado utilizando la codificacion dada.
        
        Parametros:
            - mensaje_codificado: mensaje codificado a decodificar.
            
        Retorna: mensaje decodificado.
        
        Arroja ValueError si el mensaje no se puede decodificar completamente.
        
        Precoindiciones:
            - mensaje_codificado no es None ni una cadena vacia.
        """
        mensaje_decodificado = ""
        aux = ""
        residuo = mensaje_codificado[0] >> 5 & 0x07  # Obtener los 3 bits mas significativos
        bits = format(mensaje_codificado[0], '08b')
        for bit in bits[3:]:
            aux += bit
            if aux in self.__codificacion:
                mensaje_decodificado += self.__alfabeto_fuente[self.__codificacion.index(aux)]
                aux = ""
        for byte in mensaje_codificado[1:-1]:
            bits = format(byte, '08b')  # Convertir byte a una cadena de 8 unos y ceros
            for bit in bits:
                aux += bit
                if aux in self.__codificacion:
                    mensaje_decodificado += self.__alfabeto_fuente[self.__codificacion.index(aux)]
                    aux = ""
        bits = format(mensaje_codificado[-1], '08b')
        for bit in bits[:8 - residuo]:
            aux += bit
            if aux in self.__codificacion:
                mensaje_decodificado += self.__alfabeto_fuente[self.__codificacion.index(aux)]
                aux = ""
        return mensaje_decodificado
    
    def codificar(self, mensaje: str) -> bytearray:
        """Codifica un mensaje utilizando la codificacion dada.
        Guarda en los primeros 3 bits del primer byte la cantidad de bits de relleno
        que se agregaron al final del mensaje para completar el ultimo byte.
        
        Parametros:
            - mensaje: mensaje a codificar.
        
        Retorna: mensaje codificado.
        
        Arroja ValueError si el mensaje contiene simbolos que no estan en el alfabeto fuente.
        
        Precondiciones:
            - mensaje no es None ni una cadena vacia.
        """
        mensaje_codificado = bytearray()
        aux = "000"
        for simbolo in mensaje:
            if simbolo in self.__alfabeto_fuente:
                aux += self.__codificacion[self.__alfabeto_fuente.index(simbolo)]
                while len(aux) >= 8:
                    mensaje_codificado.append(int(aux[:8], 2))
                    aux = aux[8:]
            else:
                raise ValueError(f"El símbolo '{simbolo}' no está en el alfabeto fuente.")
        residuo = 8 - len(aux)
        mensaje_codificado.append(int(aux.ljust(8, '0'), 2))  # Rellenar con ceros a la derecha si es necesario
        mensaje_codificado[0] = residuo << 5 | mensaje_codificado[0]
        return mensaje_codificado

class MatrizBinaria(Matriz[bool]):
    """Clase que representa una matriz binaria (de valores booleanos).
    La primera fila y la última columna son bits de paridad.
    
    :extends Matriz[bool]: Clase base Matriz parametrizada con booleanos."""
    
    def __init__(self, binarios: list[str]):
        super().__init__(self._convertir_a_matriz(binarios))

    def _convertir_a_matriz(self, binarios: list[str]) -> list[list[bool]]:
        return [[c == '1' for c in fila] for fila in binarios]
    
    def _convertir_a_bytearray(self, lista: list[bool]) -> bytearray:
        byte = 0
        vector_de_bytes = bytearray()
        if len(lista) > 8:
            for i, bit in enumerate(lista):
                if bit:
                    byte |= (1 << (7 - i % 8))
                if i % 8 == 7:
                    vector_de_bytes.append(byte)
                    byte = 0
            if len(lista) % 8 != 0:
                vector_de_bytes.append(byte)
        else:
            for i, bit in enumerate(lista):
                if bit:
                    byte |= (1 << (len(lista) - 1 - i))
            vector_de_bytes.append(byte)
        return vector_de_bytes
    
    def getColumnaEnBytearray(self, columna: int) -> bytearray:
        return self._convertir_a_bytearray(super().getColumna(columna))

    def getFilaEnBytearray(self, fila: int) -> bytearray:
        return self._convertir_a_bytearray(super().getFila(fila))
    
    def get_distancia_de_hamming(self) -> int:
        """Calcula la distancia de Hamming mínima entre las filas de la matriz,
        excluyendo la fila de paridad.
        
        :return: La distancia de Hamming mínima entre las filas."""
        min_distancia = int('inf')
        for i in range(1, self.cantidadFilas - 1):
            for j in range(i + 1, self.cantidadFilas - 1):
                fila_i = self.getFila(i)
                fila_j = self.getFila(j)
                distancia = 0
                for k in range(min(len(fila_i), len(fila_j))):
                    distancia += bin(fila_i[k] ^ fila_j[k]).count('1')
                if distancia < min_distancia:
                    min_distancia = distancia
        return min_distancia if min_distancia != int('inf') else 0
    
    def get_errores_detectables(self) -> int:
        """Calcula la cantidad máxima de errores detectables por la matriz.
        
        :return: La cantidad máxima de errores detectables."""
        d = self.get_distancia_de_hamming()
        return d - 1 if d > 0 else 0
    
    def get_errores_corregibles(self) -> int:
        """Calcula la cantidad máxima de errores corregibles por la matriz.

        :return: La cantidad máxima de errores corregibles."""
        d = self.get_distancia_de_hamming()
        return (d - 1) // 2 if d > 0 else 0
    
    def get_errores(self) -> list[tuple[int, int]]:
        """Devuelve una lista de tuplas que representan las posiciones de los errores.
        
        :return: Una lista de tuplas (fila, columna) indicando las posiciones de los errores.
        Si un error en columna no está apareado con un error en fila, la fila será -1 y viceversa.
        """
        errores_en_columnas: list[int] = []
        errores_en_filas: list[int] = []
        
        # Recorrer columnas
        for i in range(self.cantidadColumnas):
            columna = self.getColumna(i)
            columna_str = ''.join([bin(byte) for byte in columna])
            if columna_str[1:].count('1') % 2 != columna[0] & 0x80:
                errores_en_columnas.append(i)

        # Recorrer filas
        for i in range(self.cantidadFilas):
            fila = self.getFila(i)
            fila_str = ''.join([bin(byte) for byte in fila])
            if fila_str[:-1].count('1') % 2 != fila[-1] & 0x1:
                errores_en_filas.append(i)

        # Emparejar errores
        errores: list[tuple[int, int]] = []
        for i, (columna, fila) in enumerate(zip(errores_en_columnas, errores_en_filas)):
            errores.append((fila, columna))
        if len(errores_en_columnas) > len(errores_en_filas):
            for columna in errores_en_columnas[len(errores_en_filas):]:
                errores.append((-1, columna))
        elif len(errores_en_filas) > len(errores_en_columnas):
            for fila in errores_en_filas[len(errores_en_columnas):]:
                errores.append((fila, -1))

        return errores
    
    def corregir(self) -> bool:
        """Intenta corregir los errores en la matriz si es posible.
        
        :return: True si se pudieron corregir los errores, False en caso contrario.
        """
        errores = self.get_errores()
        if len(errores) == 0:
            return True
        if len(errores) > 1 or errores[0][0] == 0 or errores[0][1] == len(self[0]) - 1:
            return False
        
        fila, columna = errores[0]

        self[fila][columna] = not self[fila][columna]

        return True
    
    def __str__(self):
        string = ""
        for fila in range(self.cantidadFilas):
            for columna in range(self.cantidadColumnas):
                string += '1' if self[fila][columna] else '0'
            string += '\n'
        return string

def cantidadInformacion(p: float, r: int=2) -> float:
    """Dada una probabilidad, calcula la cantidad de información.

    Parámetros:	
        p (float): probabilidad del evento (0 < p <= 1).
        r (int): base del logaritmo (default 2).
    
    Retorna:
        float: la cantidad de información en bits (si r=2).
    """
    if p <= 0 or p > 1:
        resultado = 0
    else:
        resultado = math.log(1/p, r)
    return resultado
def entropia(probabilidades: list[float], r: int=2) -> float:
    """Calcula la entropía de una lista de probabilidades.

    Parámetros:
        probabilidades (list[float]): lista de probabilidades de los eventos.
        r (int): base del logaritmo (default 2).
    
    Retorna:
        float: la entropía en bits (si r=2).
    """
    h: float = 0.0
    for p in probabilidades:
        h += p * cantidadInformacion(p, r)
    return h
def entropia_de_la_fuente(codigos: list[str], probabilidades: list[float]) -> float:
    """Calcula la entropía de una fuente de información dada su distribución de probabilidades y su abecedario.

    Parámetros:
        probabilidades (list[float]): Lista de probabilidades de los símbolos de la fuente.
        codigos (list[str]): Lista de palabras código.
    
    Retorna:
        float: un valor que representa la entropía de la fuente.
    
    Contrato:
        - sum(probabilidades) == 1
        - all(p >= 0 and p <= 1 for p in probabilidades)
        - r > 1 (base del logaritmo)
        - len(probabilidades) > 0
        - len(codigos) > 0
    """
    return entropia(probabilidades, len(get_alfabeto_codigo(codigos)))
def generarConExtension(alfabeto: list[str], probabilidades: list[float], N: int) -> tuple[list[str], list[float]]:
    """Genera el alfabeto y la distribución de probabilidades
    de una fuente con extensión de orden N.
    
    Parámetros:
        alfabeto (list): lista de elementos del alfabeto.
        probabilidades (list[float]): lista de probabilidades de cada elemento del alfabeto.
        N (int): orden de la extensión.
    
    Retorna:
        tuple: (nuevas_letras, nuevas_probabilidades) donde nuevas_letras es una lista de combinaciones
        de longitud N y nuevas_probabilidades es una lista de probabilidades correspondientes.
    """
    if N <= 0:
        return [], []

    # Generar las combinaciones de longitud N
    combinaciones = generar_combinaciones(alfabeto, N)

    # Calcular las probabilidades de cada combinación
    nuevas_probabilidades: list[float] = []
    for combinacion in combinaciones:
        probabilidad = 1.0
        for letra in combinacion:
            indice = alfabeto.index(letra)
            probabilidad *= probabilidades[indice]
        nuevas_probabilidades.append(probabilidad)

    # Convertir las combinaciones de listas de letras a cadenas
    nuevas_letras = ["".join(combinacion) for combinacion in combinaciones]

    return nuevas_letras, nuevas_probabilidades
def generar_combinaciones(alfabeto: list[str], N: int) -> list[list[str]]:
    """Genera todas las combinaciones posibles de longitud N
    a partir del alfabeto dado.
    
    Parámetros:
        alfabeto (list): lista de elementos del alfabeto.
        N (int): longitud de las combinaciones a generar.
    
    Retorna:
        list: lista de combinaciones generadas.
    """
    if N == 1:
        return [[letra] for letra in alfabeto]
    else:
        combinaciones_previas = generar_combinaciones(alfabeto, N - 1)
        nuevas_combinaciones: list[list[str]] = []
        for combinacion in combinaciones_previas:
            for letra in alfabeto:
                nuevas_combinaciones.append(combinacion + [letra])
        return nuevas_combinaciones
def get_alfabeto_codigo(C: list[str])->str:
    """
    Dada una lista que contiene las palabras código de una codificación, obtiene una cadena de caracteres con el alfabeto código.

    Parámetros:
        - codigos: list[str] - Lista de palabras código.

    Retorna:
        - str - Cadena de caracteres con el alfabeto código.
    
    Contrato:
        - Precondición: codigos debe ser una lista de cadenas no vacías y distina de None.
        - Postcondición: El resultado es una cadena que contiene todos los caracteres únicos presentes en las palabras código.
    """
    x = ""
    for codigo in C:
        for caracter in codigo:
            if caracter not in x:
                x += caracter
    return x
def get_frecuencias(texto: str)-> dict[str, int]:
    """
    Esta función recibe un texto y devuelve un diccionario con la frecuencia de cada carácter en el texto.
    
    :param str texto: El texto del cual se quieren obtener las frecuencias de los caracteres.
    :return dict[str, int]: Un diccionario donde las claves son los caracteres y los valores son sus frecuencias.
    """
    frecuencias: dict[str, int] = {}
    for char in texto:
        if char in frecuencias:
            frecuencias[char] += 1
        else:
            frecuencias[char] = 1
    return frecuencias
def get_frecuencias_relativas(texto: str) -> dict[str, float]:
    """
    Esta función recibe un texto y devuelve un diccionario con la frecuencia relativa de cada carácter en el texto.
    
    :param str texto: El texto del cual se quieren obtener las frecuencias relativas de los caracteres.
    :return dict[str, float]: Un diccionario donde las claves son los caracteres y los valores son sus frecuencias relativas.
    """
    frecuencias = get_frecuencias(texto)
    total_caracteres = len(texto)
    frecuencias_relativas = {char: freq / total_caracteres for char, freq in frecuencias.items()}
    return frecuencias_relativas
def get_longitud_media(palabras_codigo: list[str], probabilidades: list[float])->float:
    """
    Dada 2 listas, una de palabras código y otra de probabilidades, devuelve la longitud media del código.

    Parámetros:
        - palabras_codigo (list[str]): Lista de palabras código.
        - probabilidades (list[float]): Lista de probabilidades de los símbolos de la fuente.
    
    Retorna: un float que representa la longitud media del código.

    Contrato:
        - len(palabras_codigo) == len(probabilidades)
        - sum(probabilidades) == 1
        - all(p >= 0 and p<=1 for p in probabilidades)
    """
    longitudes = get_longitudes(palabras_codigo)
    longitud_media = 0
    for p_i, l_i in zip(probabilidades, longitudes):
        longitud_media += p_i * l_i
    return longitud_media
def get_longitudes(codigos: list[str])->list[int]:
    """
    Dada una lista con palabras código, obtiene una lista con las longitudes de cada palabra código.

    Parámetros:
        - codigos: list[str] - Lista de palabras código.
    
    Retorna:
        - list[int] - Lista con las longitudes de cada palabra código.
    
    Contrato:
        - Precondición: codigos debe ser una lista de cadenas no vacías y distinta de None.
        - Postcondición: El resultado es una lista de enteros donde cada entero representa la longitud de la palabra código correspondiente en la lista de entrada.
    """
    return [len(codigo) for codigo in codigos]
def aleatorios(filas: int, columnas: int, min_val: float = 0.0, max_val: float = 1.0) -> Matriz[float]:
    """
    Crea una matriz de números aleatorios con las dimensiones especificadas.

    Parámetros:
        filas (int): Número de filas de la matriz.
        columnas (int): Número de columnas de la matriz.
        min_val (float): Valor mínimo para los números aleatorios.
        max_val (float): Valor máximo para los números aleatorios.
    Retorna:
        Matriz[float]: Una matriz de números aleatorios con las dimensiones dadas.
    """
    return Matriz([[random.uniform(min_val, max_val) for _ in range(columnas)] for _ in range(filas)])

def ceros(filas: int, columnas: int) -> Matriz[float]:
    """
    Crea una matriz de ceros con las dimensiones especificadas.
    
    Parámetros:
        filas (int): Número de filas de la matriz.
        columnas (int): Número de columnas de la matriz.
    Retorna:
        Matriz[float]: Una matriz de ceros con las dimensiones dadas.
    """
    return Matriz([[0.0 for _ in range(columnas)] for _ in range(filas)])

def identidad(tamano: int) -> Matriz[int]:
    """
    Crea una matriz identidad de tamaño especificado.

    Parámetros:
        tamano (int): Tamaño de la matriz identidad.

    Retorna:
        Matriz[float]: Una matriz identidad de tamaño x tamano.
    """
    return Matriz([[1 if i == j else 0 for j in range(tamano)] for i in range(tamano)])

def relleno(filas: int, columnas: int, valor: float) -> Matriz[float]:
    """
    Crea una matriz rellenada con el valor especificado.

    Parámetros:
        filas (int): Número de filas de la matriz.
        columnas (int): Número de columnas de la matriz.
        valor (float): Valor con el que rellenar la matriz.
    Retorna:
        Matriz[float]: Una matriz rellenada con el valor especificado.
    """
    return Matriz([[valor for _ in range(columnas)] for _ in range(filas)])

def unos(filas: int, columnas: int) -> Matriz[float]:
    """
    Crea una matriz de unos con las dimensiones especificadas.
    
    Parámetros:
        filas (int): Número de filas de la matriz.
        columnas (int): Número de columnas de la matriz.
    Retorna:
        Matriz[float]: Una matriz de unos con las dimensiones dadas.
    """
    return Matriz([[1.0 for _ in range(columnas)] for _ in range(filas)])

def _shannon_fano_rec(probabilidades: list[tuple[int, float]], codigos: list[str], prefijo: str=''):
    N = len(probabilidades)
    if N == 1:
        codigos[probabilidades[0][0]] = prefijo
    else:
        total = sum(p for _, p in probabilidades)
        acumulado = 0
        i = 0
        while i < N and acumulado < total / 2:
            acumulado += probabilidades[i][1]
            i += 1
        if i > 0 and i < N:
            if total/2 - (acumulado - probabilidades[i - 1][1]) < acumulado - total/2:
                i -= 1
        if i > 0:
            _shannon_fano_rec(probabilidades[:i], codigos, prefijo + '1')
        if i < N:
            _shannon_fano_rec(probabilidades[i:], codigos, prefijo + '0')

def comprimir_con_rlc(mensaje: str) -> bytearray:
    """Comprime un mensaje utilizando codificación por longitud de carrera (RLC).

    :param mensaje: El mensaje a comprimir.
    :return: El mensaje comprimido en formato bytearray, donde cada par de bytes representa (caracter, longitud).
    """
    if not mensaje:
        return bytearray()

    comprimido = bytearray()
    contador = 1
    caracter_anterior = mensaje[0]

    for caracter in mensaje[1:]:
        if caracter == caracter_anterior:
            contador += 1
        else:
            comprimido.append(ord(caracter_anterior))
            comprimido.append(contador)
            caracter_anterior = caracter
            contador = 1

    # Añadir el último grupo
    comprimido.append(ord(caracter_anterior))
    comprimido.append(contador)

    return comprimido

def es_correcto(byte: int) -> bool:
    """Dado un byte cuyo bit de paridad es el menos significativo, indica si el byte es correcto o no

    Args:
        byte (bytes): byte con la información y su bit de paridad

    Returns:
        bool: True si el byte es correcto, False en caso contrario
    """
    return sum([1 for c in bin(byte & 0b11111110) if c == '1']) % 2 == (byte & 0b00000001)

def from_caracter_to_byte(caracter: str) -> int:
    """
    Convierte un caracter a ASCII y al bit menos significativo lo usa para almacenar la paridad del código

    :param caracter: Caracter a convertir

    :return: Byte con el caracter y su bit de paridad
    """
    if len(caracter) != 1:
        raise ValueError("La entrada debe ser un simple caracter.")
    ascii = bytes(caracter, 'ascii')
    return ascii[0] << 1 | (bin(ascii[0]).count('1') % 2)

def generar_bytearray(x: str) -> bytearray:
    """Dada una cadena de caracteres, genera una secuencia de bytes (bytearray) que
    contiene su representación con código ASCII y sus bits de paridad vertical,
    longitudinal y cruzada.

    :param str x: cadena de caracteres a convertir.
    
    :return: bytearray que representa la cadena con sus bits de paridad.
"""
    byte = 0
    vector_de_bytes = bytearray()
    vector_de_bytes.append(0)
    for caracter in x:
        byte = from_caracter_to_byte(caracter)
        vector_de_bytes.append(byte)
        vector_de_bytes[0] ^= byte
    return vector_de_bytes

def get_distancia_de_hamming(codigo: list[str]) -> tuple[int, int , int]:
    """Dado un código, devuelve la distancia de Hamming y las cantidades de errores detectables y corregibles.
    
    :param codigo: lista de cadenas que representan las palabras del código. Todas las cadenas deben tener la misma longitud.
    
    :return: una tupla con la distancia de Hamming, la cantidad de errores detectables y la cantidad de errores corregibles en ese orden
    """
    distancia_de_haming = int('inf')
    for cod1 in codigo:
        for cod2 in codigo:
            aux = 0
            if cod1 != cod2:
                for i in range(len(cod1)):
                    if cod1[i] != cod2[i]:
                        aux += 1
                if aux < distancia_de_haming:
                    distancia_de_haming = aux
    
    return distancia_de_haming, distancia_de_haming - 1, (distancia_de_haming - 1) // 2

def get_mensaje_original(x: bytearray) -> str:
    """Dada una secuencia de bytes, devuelve el mensaje original o una cadena de caracteres vacía
    si no se pueden corregir los errores.

    :param x: bytearray que representa la cadena con sus bits de paridad.
    
    :return: cadena de caracteres original.
    """
    matriz = MatrizBinaria([format(byte, '08b') for byte in x])
    exito = matriz.corregir()
    if exito:
        mensaje = ''.join(chr(matriz.getFila(i)[0] >> 1 & 0x7f) for i in range(1, matriz.cantidadFilas))
        return mensaje
    else:
        return ''

def get_rendimiento_y_redundancia(probabilidades: list[float], codificacion: list[str]) -> tuple[float, float]:
    """
    Calcula el rendimiento y la redundancia de una codificación.
    
    Parámetros:
        - probabilidades (list[float]): Lista de probabilidades de los símbolos.
        - codificacion (list[str]): Lista de longitudes de los códigos para cada símbolo.
    
    Retorna: rendimiento (float), redundancia (float)
    """
    H = entropia_de_la_fuente(codificacion, probabilidades)
    L = get_longitud_media(codificacion, probabilidades)
    RENDIMIENTO = H / L
    return RENDIMIENTO, 1-RENDIMIENTO

def get_tasa_de_compresion(mensaje: str, mensaje_comprimido: bytearray) -> float:
    """
    Calcula la tasa de compresión entre un mensaje original y su versión comprimida.

    Args:
        mensaje (str): El mensaje original.
        mensaje_comprimido (bytearray): El mensaje comprimido.

    Returns:
        float: La tasa de compresión, definida como la división entre el tamaño original y el tamaño comprimido.
    """
    tamanio_mensaje = len(mensaje)
    tamanio_mensaje_comprimido = len(mensaje_comprimido)
    tasa_de_compresion = tamanio_mensaje / tamanio_mensaje_comprimido if tamanio_mensaje_comprimido else 0
    return tasa_de_compresion

def huffman(probabilidades: list[float]) -> list[str]:
    """
    Construye un codigo compacto de Huffman a partir de una lista de probabilidades.

    Parámetros:
        - probabilidades (list[float]): Lista de probabilidades de los símbolos de la fuente.
    
    Retorna: lista paralela a 'probabilidades' con las palabras código generadas.
    """
    items: list[tuple[float, list[int]]] # En cada item: (probabilidad, [indices de los simbolos])
    menor1: tuple[float, list[int]]
    menor2: tuple[float, list[int]]
    items = [(p, [i]) for i, p in enumerate(probabilidades)]
    codigos = [''] * len(probabilidades)
    while len(items) > 1:
        items = sorted(items, key=lambda x: x[0])
        menor1 = items.pop(0)
        menor2 = items.pop(0)
        for i in menor1[1]:
            codigos[i] = '0' + codigos[i]
        for i in menor2[1]:
            codigos[i] = '1' + codigos[i]
        items.append((menor1[0] + menor2[0], menor1[1] + menor2[1]))
    return codigos

def primer_teorema_shannon(probabilidades: list[float], palabras_codigo: list[str], n: int) -> bool:
    """
    Verifica el primer teorema de Shannon para una fuente de información.

    Parámetros:
        - probabilidades (list[float]): lista de probabilidades de cada símbolo en la fuente.
        - palabras_codigo (list[str]): lista de palabras del código asociado a la fuente.
        - n (int): longitud fija de las palabras del código.

    Retorna: true si se cumple el teorema, false en caso contrario.
    """
    s_n, probabilidades_n = generarConExtension(palabras_codigo, probabilidades, n)
    longitud_media_n = get_longitud_media(s_n, probabilidades_n)
    entropia = entropia_de_la_fuente(palabras_codigo, probabilidades)
    return entropia <= longitud_media_n/n and longitud_media_n/n <= entropia + 1/n

def shannon_fano(probabilidades: list[float]) -> list[str]:
    """
    Construye un codigo compacto de Shannon-Fano a partir de una lista de probabilidades.

    Parámetros:
        - probabilidades (list[float]): Lista de probabilidades de los símbolos de la fuente.
    
    Retorna: lista paralela a 'probabilidades' con las palabras código generadas.
    """
    indexed_probabilidades = list(enumerate(probabilidades))
    indexed_probabilidades.sort(key=lambda x: x[1], reverse=True)
    codigos = [''] * len(probabilidades)
    _shannon_fano_rec(indexed_probabilidades, codigos)
    return codigos

if __name__ == '__main__':
    matriz_de_ceros = ceros(3, 3)
    print(matriz_de_ceros)
    print('Rendimiento: ', get_rendimiento_y_redundancia([0.5, 0.5], ['0', '1'])[0])