import lib.MatrizFactory as mf
import lib.Unidad4 as u4

if __name__ == "__main__":
    matriz_de_ceros = mf.ceros(3,3)
    print(matriz_de_ceros)
    print("Rendimiento: ", u4.get_rendimiento_y_redundancia([0.5, 0.5], ["0", "1"])[0])