import unittest
import csv

from io import StringIO

from z80asm import Z80AsmParser, Z80AsmLayouter, Z80AsmCompiler


INSTRUCTIONS_FILE = 'tests/opcodes.csv'


def compile_asm(source: str) -> tuple[int, ...]:
    parser = Z80AsmParser()

    stream = StringIO(source)
    parser.parse_stream(stream)

    ltr = Z80AsmLayouter(parser.instructions)
    ltr.layout_program()

    compiler = Z80AsmCompiler(parser.instructions)
    compiler.compile_program()

    return tuple(compiler.encode())


def tuptohexstr(tup: tuple[int, ...]) -> str:
    return " ".join(f"{i:02X}" for i in tup)


class TestAsm(unittest.TestCase):

    def test_instruction(self):
        with open(INSTRUCTIONS_FILE) as fin:
            reader = csv.DictReader(fin, delimiter=";", quotechar="|",
                                    skipinitialspace=True)
            for lineno, row in enumerate(reader, 2):
                inst = row["instruction"]
                clue_bytes = tuple(int(i, 16) for i in row["opcode"].split())
                try:
                    out_bytes = compile_asm(inst)
                except Exception as e:
                    msg = f"{INSTRUCTIONS_FILE}:{lineno}: instruction {inst}"
                    raise AssertionError(msg) from e
                try:
                    self.assertEqual(clue_bytes, out_bytes)
                except AssertionError:
                    msg = (f"{INSTRUCTIONS_FILE}:{lineno}: instruction {inst}: "
                           f"expected {tuptohexstr(clue_bytes)}, "
                           f"got {tuptohexstr(out_bytes)}")
                    raise AssertionError(msg) from None
