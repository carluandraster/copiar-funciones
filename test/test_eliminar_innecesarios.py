import src.eliminar_innecesarios as ei
import unittest


class EliminarInnecesariosTest(unittest.TestCase):
    def testInitCodeAnalyzer(self):
        analyzer = ei.CodeAnalyzer()
        self.assertEqual(analyzer.functions, set())
        self.assertEqual(analyzer.classes, set())
        self.assertEqual(analyzer.calls, {})
        self.assertEqual(analyzer.used, set())
        self.assertIsNone(analyzer.current_context)