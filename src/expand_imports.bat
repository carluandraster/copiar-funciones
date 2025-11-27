@echo off
python expand_imports.py "experimentos.py" "temporal.py"
python eliminar_innecesarios.py "temporal.py" "salida_final.py"
del temporal.py
exit