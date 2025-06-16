import unittest
import difflib

import yaml

from io import StringIO
from typing import IO, Text, Optional

from z80asm import Z80AsmParser, Z80AsmLayouter, Z80AsmCompiler, Z80AsmPrinter, Z80Error


FILE = "tests/programs.yaml"


def compile_format(source: str) -> str:
    istream, ostream = StringIO(source), StringIO()

    parser = Z80AsmParser()
    program = parser.parse_stream(istream)

    ltr = Z80AsmLayouter()
    ltr.layout_program(program)

    compiler = Z80AsmCompiler()
    compiler.compile_program(program)

    printer = Z80AsmPrinter(ostream, replace_names=True)
    printer.print_program(program)

    return ostream.getvalue()


def try_get_lineno(file: IO[Text], content: str) -> Optional[int]:
    if file.seekable():
        file.seek(0)
        for no, line in enumerate(file, 1):
            if content in line:
                return no
    return None


class TestPrograms(unittest.TestCase):

    def test_programs(self):
        with open(FILE, "r") as fin:
            data = yaml.load(fin, Loader=yaml.Loader)
            for test in data["tests"]:
                desc = test["desc"]
                expected = test["expect"].splitlines(keepends=True)
                try:
                    encoded = compile_format(test["source"]).splitlines(keepends=True)
                except Z80Error as e:
                    lineno = try_get_lineno(fin, desc)
                    lineno = f"{lineno}:" if lineno is not None else ""
                    raise AssertionError(f"{FILE}:{lineno}: Test failed: {desc}") from e
                if expected != encoded:
                    diff = "".join(difflib.ndiff(expected, encoded))
                    print(diff)

                    lineno = try_get_lineno(fin, desc)
                    lineno = f"{lineno}:" if lineno is not None else ""
                    self.fail(f"{FILE}:{lineno} Test failed: {desc}")
