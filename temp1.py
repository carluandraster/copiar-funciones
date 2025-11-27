from lib.Matriz import Matriz
from lib.Utils import generarConExtension, entropia_de_la_fuente, get_longitud_media
import random


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