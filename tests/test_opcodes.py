import unittest
import csv

from io import StringIO

from z80asm import Z80AsmParser, Z80AsmLayouter, Z80AsmCompiler


DOC_FILE = 'tests/opcodes.csv'
UNDOC_FILE = 'tests/undoc_opcodes.csv'


def compile_asm(source: str, undoc: bool) -> tuple[int, ...]:
    parser = Z80AsmParser(undoc_instructions=undoc)

    stream = StringIO(source)
    program = parser.parse_stream(stream)

    ltr = Z80AsmLayouter()
    ltr.layout_program(program)

    compiler = Z80AsmCompiler()
    compiler.compile_program(program)

    return tuple(compiler.encode())


def tuptohexstr(tup: tuple[int, ...]) -> str:
    return " ".join(f"{i:02X}" for i in tup)


class TestAsm(unittest.TestCase):

    def open_test_file(self, file: str, undoc: bool):
        with open(file, "r") as fin:
            reader = csv.DictReader(fin, delimiter=";", quotechar="|",
                                    skipinitialspace=True)
            for lineno, row in enumerate(reader, 2):
                inst = row["instruction"]
                clue_bytes = tuple(int(i, 16) for i in row["opcode"].split())
                try:
                    out_bytes = compile_asm(inst, undoc=undoc)
                except Exception as e:
                    msg = f"{file}:{lineno}: instruction {inst}"
                    raise AssertionError(msg) from e
                try:
                    self.assertEqual(clue_bytes, out_bytes)
                except AssertionError:
                    msg = (f"{file}:{lineno}: instruction {inst}: "
                           f"expected {tuptohexstr(clue_bytes)}, "
                           f"got {tuptohexstr(out_bytes)}")
                    raise AssertionError(msg) from None

    def test_doc_instruction(self):
        self.open_test_file(DOC_FILE, undoc=False)

    def test_undoc_instruction(self):
        self.open_test_file(UNDOC_FILE, undoc=True)
