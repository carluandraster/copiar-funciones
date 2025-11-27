@echo off
python expand_imports.py "experimentos.py" "salida_resultante.py"
python eliminar_innecesarios.py "salida_resultante.py" "salida_final.py"
exit