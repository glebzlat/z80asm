#!/usr/bin/env python3
from __future__ import annotations

import re
import sys

from functools import wraps
from typing import Optional, Callable, Any, TextIO, BinaryIO, Generator
from io import StringIO, BytesIO
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
from functools import lru_cache
from itertools import repeat


class OperandKind(Enum):
    Int8 = auto()
    Int16 = auto()
    Reg = auto()
    RegPair = auto()
    IX = auto()
    IY = auto()
    Addr = auto()
    IXDAddr = auto()
    IYDAddr = auto()
    Const = auto()
    Flag = auto()

    # Label references. Absolute label reference is 16-bit, relative is 8-bit.
    AbsLabel = auto()
    RelLabel = auto()

    # Page 0 Memory Location
    MemLoc = auto()
    Char = auto()
    String = auto()


class Opcode(Enum):
    LD = auto()
    PUSH = auto()
    POP = auto()
    EX = auto()
    EXX = auto()
    LDI = auto()
    LDIR = auto()
    LDD = auto()
    LDDR = auto()
    CPI = auto()
    CPIR = auto()
    CPD = auto()
    CPDR = auto()
    ADD = auto()
    ADC = auto()
    SUB = auto()
    SBC = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    CP = auto()
    INC = auto()
    DEC = auto()
    DAA = auto()
    CPL = auto()
    NEG = auto()
    CCF = auto()
    SCF = auto()
    NOP = auto()
    HALT = auto()
    DI = auto()
    EI = auto()
    IM = auto()
    RLCA = auto()
    RLA = auto()
    RRCA = auto()
    RRA = auto()
    RLC = auto()
    RL = auto()
    RRC = auto()
    RR = auto()
    SLA = auto()
    SRA = auto()
    SRL = auto()
    RLD = auto()
    RRD = auto()
    BIT = auto()
    SET = auto()
    RES = auto()
    JP = auto()
    JR = auto()
    DJNZ = auto()
    CALL = auto()
    RET = auto()
    RETI = auto()
    RETN = auto()
    RST = auto()
    IN = auto()
    INI = auto()
    INIR = auto()
    IND = auto()
    INDR = auto()
    OUT = auto()
    OUTI = auto()
    OTIR = auto()
    OUTD = auto()
    OTDR = auto()


class DirectiveKind(Enum):
    org = auto()
    equ = auto()
    db = auto()


@dataclass
class ParseInfo:
    lineno: Optional[int] = field(default=None, repr=False, kw_only=True)
    column: Optional[int] = field(default=None, repr=False, kw_only=True)
    line: Optional[str] = field(default=None, repr=False, kw_only=True)
    filename: Optional[str] = field(default=None, repr=False, kw_only=True)


@dataclass
class Operand(ParseInfo):
    """Class representing an instruction operand

    Name is an optional field valid only for Label and Const operands
    after the layout resolved their values.
    """
    kind: OperandKind
    value: Optional[int | str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        if self.value:
            return f"{self.kind.name}:{self.value}"
        return self.kind.name


@dataclass
class Instruction(ParseInfo):
    opcode: Opcode
    operands: list[Operand] = field(default_factory=list)
    op_bytes: Optional[Callable | tuple[int, ...]] = field(default=None, repr=False)
    length: Optional[int] = field(default=None, repr=False)
    addr: Optional[int] = None
    encoded: Optional[tuple[int, ...]] = None

    def __str__(self) -> str:
        operands = ", ".join(str(op) for op in self.operands)
        addr = f"{self.addr:04X} " if self.addr is not None else ""
        return f"{addr}Instruction:{self.opcode.name} {operands}"


@dataclass
class Directive(ParseInfo):
    kind: DirectiveKind
    operands: list[Operand] = field(default_factory=list)
    addr: Optional[int] = None
    length: Optional[int] = None
    encoded: Optional[tuple[int, ...]] = None

    def __str__(self) -> str:
        operands = ", ".join(str(op) for op in self.operands)
        return f"Directive:{self.kind.name} {operands}"


@dataclass
class Label(ParseInfo):
    name: str
    addr: Optional[int] = None

    def __str__(self) -> str:
        addr = f" ({self.addr:04X})" if self.addr is not None else ""
        return f"Label:{self.name}{addr}"


Statement = Instruction | Directive | Label


def ceil_pow2(v: int) -> int:
    """Ceil the value to the nearest power of 2"""
    p = 1
    while v > p:
        p = p << 1
    return p


def isiterable(obj: object) -> bool:
    return getattr(obj, "__iter__", None) is not None


def memoize(fn: Callable):
    """Packrat parser memoize wrapper"""

    @wraps(fn)
    def wrapper(self, *args):
        pos = self.mark()
        key = (fn, args, pos)
        memo = self.memos.get(key)
        if memo is None:
            result = fn(self, *args)
            self.memos[key] = memo = (result, self.mark())
        else:
            self.reset(memo[1])
        return memo[0]

    return wrapper


class Z80Error(Exception):

    def __str__(self) -> str:
        stream = StringIO()
        for err in self.args:
            print(err, file=stream)
        return stream.getvalue().rstrip()


class Z80AsmParser:

    @dataclass
    class InstructionData:
        """Helper class to store the instruction byte length and convertion function

        Convertion function takes instruction arguments as integers and returns a complete
        instruction code.
        """
        instruction_bytes: int
        op_bytes: Callable

    def definitions(self):
        """Instruction and directive definitions"""

        # Shortcuts
        _ = lambda name: Opcode[name].name
        S = lambda s: lambda: self.parse_int8_literal(s)
        D = self.InstructionData

        REG = self.parse_register
        I8 = self.parse_i8_op
        CH = self.parse_char
        I16 = self.parse_i16_op
        BC = self.parse_bc
        DE = self.parse_de
        HL = self.parse_hl
        SP = self.parse_sp
        AF = self.parse_af
        ABC = lambda: self.parse_addr_combine(self.parse_bc)
        ADE = lambda: self.parse_addr_combine(self.parse_de)
        AHL = lambda: self.parse_addr_combine(self.parse_hl)
        ASP = lambda: self.parse_addr_combine(self.parse_sp)
        ADD = lambda: self.parse_addr_combine(self.parse_i16_op)
        IX = self.parse_ix
        IY = self.parse_iy
        IXD = self.parse_ixd_addr
        IYD = self.parse_iyd_addr
        AR = lambda: self.parse_register_name("a")
        IR = lambda: self.parse_register_name("i")
        RR = lambda: self.parse_register_name("r")
        REP = self.parse_regpair
        BIT = self.parse_bit_pos

        LBLA = self.parse_abs_label_ref
        LBLR = self.parse_rel_label_ref
        CFF = self.parse_cond_flag

        CF = self.parse_carry_flag
        NC = self.parse_unset_carry_flag
        ZF = self.parse_zero_flag
        NZ = self.parse_unset_zero_flag

        PML = self.parse_page0_mem_loc

        IOA = lambda: self.parse_addr_combine(self.parse_i8_op)
        IOAC = lambda: self.parse_addr_combine(self.parse_register_name("c"))

        # Parselet   Expression   Convertion
        # IXD        (ix+<int>)   int
        # IYD        (iy+<int>)   int
        # ADD        (<int>)      int
        # REG        <reg-name>   int
        # REGP       <reg-pair>   int
        # BIT        [0-7]        int
        # I8         <int>        int
        # I16        <int>        (lsb, msb)
        # LBL        <id>         (lsb, msb)
        # CONST      <id>         int
        #
        # Parselets not mentioned in the above table are not converted and
        # a _ placeholder should be used.

        self.mnemonics: dict[Opcode, dict[tuple[Callable, ...], self.InstructionData | tuple[int, ...]]] = {
            _("LD"): {
                # 8-bit load group
                (REG, REG): D(1, lambda d, s: (0x40 | (d << 3) | s,)),
                (REG, I8): D(2, lambda d, s: (0x06 | (d << 3), s)),
                (REG, CH): D(2, lambda d, s: (0x06 | (d << 3), s)),
                (REG, AHL): D(1, lambda d, _: (0x46 | (d << 3),)),
                (REG, IXD): D(3, lambda d, n: (0xdd, 0x46 | (d << 3), n)),
                (REG, IYD): D(3, lambda d, n: (0xfd, 0x46 | (d << 3), n)),
                (AHL, REG): D(1, lambda _, r: (0x70 | r,)),
                (IXD, REG): D(3, lambda n, r: (0xdd, 0x70 | r, n)),
                (IYD, REG): D(3, lambda n, r: (0xfd, 0x70 | r, n)),
                (AHL, I8): D(2, lambda _, n: (0x36, n)),
                (IXD, I8): D(4, lambda d, n: (0xdd, 0x36, d, n)),
                (IYD, I8): D(4, lambda d, n: (0xfd, 0x36, d, n)),
                (AR, ABC): (0x0a,),
                (AR, ADE): (0x1a,),
                (AR, ADD): D(3, lambda _, n: (0x3a, n[0], n[1])),
                (ABC, AR): (0x02,),
                (ADE, AR): (0x12,),
                (ADD, AR): D(3, lambda n, _: (0x32, n[0], n[1])),
                (AR, IR): (0xed, 0x57),
                (AR, RR): (0xed, 0x5f),
                (IR, AR): (0xed, 0x47),
                (RR, AR): (0xed, 0x4f),

                # 16-bit load group
                (REP, I16): D(3, lambda d, n: (0x01 | (d << 4), n[0], n[1])),
                (IX, I16): D(4, lambda _, n: (0xdd, 0x21, n[0], n[1])),
                (IY, I16): D(4, lambda _, n: (0xfd, 0x21, n[0], n[1])),
                (HL, ADD): D(3, lambda _, n: (0x2a, n[0], n[1])),
                (REP, ADD): D(4, lambda d, n: (0xed, 0x4b | (d << 4), n[0], n[1])),
                (IX, ADD): D(4, lambda _, n: (0xdd, 0x2a, n[0], n[1])),
                (IY, ADD): D(4, lambda _, n: (0xfd, 0x2a, n[0], n[1])),
                (ADD, HL): D(3, lambda n, _: (0x22, n[0], n[1])),
                (ADD, REP): D(4, lambda n, r: (0xed, 0x43 | (r << 4), n[0], n[1])),
                (ADD, IX): D(4, lambda n, _: (0xdd, 0x22, n[0], n[1])),
                (ADD, IY): D(4, lambda n, _: (0xfd, 0x22, n[0], n[1])),
                (SP, HL): (0xf9,),
                (SP, IX): (0xdd, 0xf9),
                (SP, IY): (0xfd, 0xf9),

                (REP, LBLA): D(3, lambda d, n: (0x01 | (d << 4), n[0], n[1])),
                (IX, LBLA): D(4, lambda _, n: (0xdd, 0x21, n[0], n[1])),
                (IY, LBLA): D(4, lambda _, n: (0xfd, 0x21, n[0], n[1])),
                (HL, LBLA): D(3, lambda _, n: (0x2a, n[0], n[1])),
            },
            _("PUSH"): {
                (BC,): (0xc5,),
                (DE,): (0xd5,),
                (HL,): (0xe5,),
                (AF,): (0xf5,),
                (IX,): (0xdd, 0xe5),
                (IY,): (0xfd, 0xe5),
            },
            _("POP"): {
                (BC,): (0xc1,),
                (DE,): (0xd1,),
                (HL,): (0xe1,),
                (AF,): (0xf1,),
                (IX,): (0xdd, 0xe1),
                (IY,): (0xfd, 0xe1),
            },
            _("EX"): {
                (DE, HL): (0xeb,),
                (AF,): (0x08,),     # ex af, af'
                (ASP, HL): (0xe3,),
                (ASP, IX): (0xdd, 0xe3),
                (ASP, IY): (0xfd, 0xe3),
            },
            _("EXX"): {
                (): (0xd9,),
            },
            _("LDI"): {
                (): (0xed, 0xa0),
            },
            _("LDIR"): {
                (): (0xed, 0xb0),
            },
            _("LDD"): {
                (): (0xed, 0xa8),
            },
            _("LDDR"): {
                (): (0xed, 0xb8),
            },
            _("CPI"): {
                (): (0xed, 0xa1),
            },
            _("CPIR"): {
                (): (0xed, 0xb1),
            },
            _("CPD"): {
                (): (0xed, 0xa9)
            },
            _("CPDR"): {
                (): (0xed, 0xb9)
            },
            _("ADD"): {
                (AR, REG): D(1, lambda _, r: (0x80 | r,)),
                (AR, I8): D(2, lambda _, n: (0xc6, n)),
                (AR, AHL): (0x86,),
                (AR, IXD): D(3, lambda _, d: (0xdd, 0x86, d)),
                (AR, IYD): D(3, lambda _, d: (0xfd, 0x86, d)),

                # 16-bit
                (HL, REP): D(1, lambda _, rp: (0x09 | (rp << 4),)),
                (IX, BC): (0xdd, 0x09),
                (IX, DE): (0xdd, 0x19),
                (IX, IX): (0xdd, 0x29),
                (IX, SP): (0xdd, 0x39),
                (IY, BC): (0xfd, 0x09),
                (IY, DE): (0xfd, 0x19),
                (IY, IY): (0xfd, 0x29),
                (IY, SP): (0xfd, 0x39),
            },
            _("ADC"): {
                (AR, REG): D(1, lambda _, r: (0x88 | r,)),
                (AR, I8): D(2, lambda _, n: (0xce, n)),
                (AR, AHL): (0x8e,),
                (AR, IXD): D(3, lambda _, d: (0xdd, 0x8e, d)),
                (AR, IYD): D(3, lambda _, d: (0xfd, 0x8e, d)),

                # 16-bit
                (HL, REP): D(2, lambda _, rp: (0xed, 0x4a | (rp << 4))),
            },
            _("SUB"): {
                (REG,): D(1, lambda r: (0x90 | r,)),
                (I8,): D(2, lambda n: (0xd6, n)),
                (AHL,): (0x96,),
                (IXD,): D(3, lambda d: (0xdd, 0x96, d)),
                (IYD,): D(3, lambda d: (0xfd, 0x96, d)),
            },
            _("SBC"): {
                (AR, REG): D(1, lambda _, r: (0x98 | r,)),
                (AR, I8): D(2, lambda _, n: (0xde, n)),
                (AR, AHL): (0x9e,),
                (AR, IXD): D(3, lambda _, d: (0xdd, 0x9e, d)),
                (AR, IYD): D(3, lambda _, d: (0xfd, 0x9e, d)),

                # 16-bit
                (HL, REP): D(2, lambda _, rp: (0xed, 0x42 | (rp << 4))),
            },
            _("AND"): {
                (REG,): D(1, lambda r: (0xa0 | r,)),
                (I8,): D(2, lambda n: (0xe6, n)),
                (AHL,): (0xa6,),
                (IXD,): D(3, lambda d: (0xdd, 0xa6, d)),
                (IYD,): D(3, lambda d: (0xfd, 0xa6, d)),
            },
            _("OR"): {
                (REG,): D(1, lambda r: (0xb0 | r,)),
                (I8,): D(2, lambda n: (0xf6, n)),
                (AHL,): (0xb6,),
                (IXD,): D(3, lambda d: (0xdd, 0xb6, d)),
                (IYD,): D(3, lambda d: (0xfd, 0xb6, d)),
            },
            _("XOR"): {
                (REG,): D(1, lambda r: (0xa8 | r,)),
                (I8,): D(2, lambda n: (0xee, n)),
                (AHL,): (0xae,),
                (IXD,): D(3, lambda d: (0xdd, 0xae, d)),
                (IYD,): D(3, lambda d: (0xfd, 0xae, d)),
            },
            _("CP"): {
                (REG,): D(1, lambda r: (0xb8 | r,)),
                (I8,): D(2, lambda n: (0xfe, n)),
                (CH,): D(2, lambda n: (0xfe, n)),
                (AHL,): (0xbe,),
                (IXD,): D(3, lambda d: (0xdd, 0xbe, d)),
                (IYD,): D(3, lambda d: (0xfd, 0xbe, d)),
            },
            _("INC"): {
                (REG,): D(1, lambda r: (0x04 | (r << 3),)),
                (AHL,): (0x34,),
                (IXD,): D(3, lambda d: (0xdd, 0x34, d)),
                (IYD,): D(3, lambda d: (0xfd, 0x34, d)),

                # 16-bit
                (REP,): D(1, lambda rp: (0x03 | (rp << 4),)),
                (IX,): (0xdd, 0x23),
                (IY,): (0xfd, 0x23),
            },
            _("DEC"): {
                (REG,): D(1, lambda r: (0x05 | (r << 3),)),
                (AHL,): (0x35,),
                (IXD,): D(3, lambda d: (0xdd, 0x35, d)),
                (IYD,): D(3, lambda d: (0xfd, 0x35, d)),

                # 16-bit
                (REP,): D(1, lambda rp: (0x0b | (rp << 4),)),
                (IX,): (0xdd, 0x2b),
                (IY,): (0xfd, 0x2b),
            },
            _("DAA"): {
                (): (0x27,),
            },
            _("CPL"): {
                (): (0x2f,),
            },
            _("NEG"): {
                (): (0xed, 0x44),
            },
            _("CCF"): {
                (): (0x3f,),
            },
            _("SCF"): {
                (): (0x37,),
            },
            _("NOP"): {
                (): (0x00,),
            },
            _("HALT"): {
                (): (0x76,),
            },
            _("DI"): {
                (): (0xf3,),
            },
            _("EI"): {
                (): (0xfb,),
            },
            _("IM"): {
                (S("0"),): (0xed, 0x46),
                (S("1"),): (0xed, 0x56),
                (S("2"),): (0xed, 0x5e),
            },
            _("RLCA"): {
                (): (0x07,),
            },
            _("RLA"): {
                (): (0x17,),
            },
            _("RRCA"): {
                (): (0x0f,),
            },
            _("RRA"): {
                (): (0x1f,),
            },
            _("RLC"): {
                (REG,): D(2, lambda r: (0xcb, 0x00 | r)),
                (AHL,): (0xcb, 0x06),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x06)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x06)),
            },
            _("RL"): {
                (REG,): D(2, lambda r: (0xcb, 0x10 | r)),
                (AHL,): (0xcb, 0x16),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x16)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x16)),
            },
            _("RRC"): {
                (REG,): D(2, lambda r: (0xcb, 0x08 | r)),
                (AHL,): (0xcb, 0x0e),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x0e)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x0e)),
            },
            _("RR"): {
                # XXX: RR reg and RRC reg opcodes are the same??
                (REG,): D(2, lambda r: (0xcb, 0x08 | r)),
                (AHL,): (0xcb, 0x1e),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x1e)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x1e)),
            },
            _("SLA"): {
                (REG,): D(2, lambda r: (0xcb, 0x20 | r)),
                (AHL,): (0xcb, 0x26),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x26)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x26)),
            },
            _("SRA"): {
                (REG,): D(2, lambda r: (0xcb, 0x28 | r)),
                (AHL,): (0xcb, 0x2e),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x2e)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x2e)),
            },
            _("SRL"): {
                (REG,): D(2, lambda r: (0xcb, 0x38 | r)),
                (AHL,): (0xcb, 0x3e),
                (IXD,): D(4, lambda d: (0xdd, 0xcb, d, 0x3e)),
                (IYD,): D(4, lambda d: (0xfd, 0xcb, d, 0x3e)),
            },
            _("RLD"): {
                (): (0xed, 0x6f),
            },
            _("RRD"): {
                (): (0xed, 0x67),
            },
            _("BIT"): {
                (BIT, REG): D(2, lambda b, r: (0xcb, 0x40 | (b << 3) | r)),
                (BIT, AHL): D(2, lambda b, _: (0xcb, 0x46 | (b << 3))),
                (BIT, IXD): D(4, lambda b, d: (0xdd, 0xcb, d, 0x46 | (b << 3))),
                (BIT, IYD): D(4, lambda b, d: (0xfd, 0xcb, d, 0x46 | (b << 3))),
            },
            _("SET"): {
                (BIT, REG): D(2, lambda b, r: (0xcb, 0xc0 | (b << 3) | r)),
                (BIT, AHL): D(2, lambda b, _: (0xcb, 0xc6 | (b << 3))),
                (BIT, IXD): D(4, lambda b, d: (0xdd, 0xcb, d, 0xc6 | (b << 3))),
                (BIT, IYD): D(4, lambda b, d: (0xfd, 0xcb, d, 0xc6 | (b << 3))),
            },
            _("RES"): {
                (BIT, REG): D(2, lambda b, r: (0xcb, 0x80 | (b << 3) | r)),
                (BIT, AHL): D(2, lambda b, _: (0xcb, 0x86 | (b << 3))),
                (BIT, IXD): D(4, lambda b, d: (0xdd, 0xcb, d, 0x86 | (b << 3))),
                (BIT, IYD): D(4, lambda b, d: (0xfd, 0xcb, d, 0x86 | (b << 3))),
            },
            _("JP"): {
                (I16,): D(3, lambda n: (0xc3, n[0], n[1])),
                (CFF, I16): D(3, lambda f, n: (0xc2 | (f << 3), n[0], n[1])),
                (CFF, LBLA): D(3, lambda f, n: (0xc2 | (f << 3), n[0], n[1])),

                (HL,): (0xe9,),
                (IX,): (0xdd, 0xe9),
                (IY,): (0xfd, 0xe9),
            },
            _("JR"): {
                (I8,): D(2, lambda n: (0x18, n - 2)),
                (CF, I8): D(2, lambda _, n: (0x38, n - 2)),
                (NC, I8): D(2, lambda _, n: (0x30, n - 2)),
                (ZF, I8): D(2, lambda _, n: (0x28, n - 2)),
                (NZ, I8): D(2, lambda _, n: (0x20, n - 2)),
                (CF, LBLR): D(2, lambda _, n: (0x38, n - 2)),
                (NC, LBLR): D(2, lambda _, n: (0x30, n - 2)),
                (ZF, LBLR): D(2, lambda _, n: (0x28, n - 2)),
                (NZ, LBLR): D(2, lambda _, n: (0x20, n - 2)),
                (LBLR,): D(2, lambda n: (0x18, n - 2)),
            },
            _("DJNZ"): {
                (I8,): D(2, lambda n: (0x10, n - 2)),
                (LBLR,): D(2, lambda n: (0x10, n - 2)),
            },
            _("CALL"): {
                (I16,): D(3, lambda n: (0xcd, n[0], n[1])),
                (CFF, I16): D(3, lambda f, n: (0xc4 | (f << 3), n[0], n[1]))
            },
            _("RET"): {
                (): (0xc9,),
                (CFF,): D(1, lambda f: (0xc0 | (f << 3),)),
            },
            _("RETI"): {
                (): (0xed, 0x4d),
            },
            _("RETN"): {
                (): (0xed, 0x45),
            },
            _("RST"): {
                (PML,): D(1, lambda n: (0xc7 | (n << 3),))
            },
            _("IN"): {
                (AR, IOA): D(2, lambda _, n: (0xdb, n)),
                (REG, IOAC): D(2, lambda r, _: (0xed, 0x40 | (r << 3)))
            },
            _("INI"): {
                (): (0xed, 0xa2),
            },
            _("INIR"): {
                (): (0xed, 0xb2),
            },
            _("IND"): {
                (): (0xed, 0xaa),
            },
            _("INDR"): {
                (): (0xed, 0xba),
            },
            _("OUT"): {
                (IOA, AR): D(2, lambda n, _: (0xd3, n)),
                (IOAC, AR): D(2, lambda _, r: (0xed, 0x41 | (r << 3)))
            },
            _("OUTI"): {
                (): (0xed, 0xa3),
            },
            _("OTIR"): {
                (): (0xed, 0xb3),
            },
            _("OUTD"): {
                (): (0xed, 0xab),
            },
            _("OTDR"): {
                (): (0xed, 0xbb)
            }
        }

        self.directives = {
            "org": [self.parse_i16_op],
            "equ": [self.parse_const, self.parse_i8_op],
            "db": [self.parse_byte_sequence]
        }

        # Used by parse_char
        self.escape_chars = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "0": "\0"
        }

    def __init__(self):
        self.definitions()

        # Memo table used for backtracking.
        self.memos = {}

        # Dictionary column -> list of failed expects. This
        # is used by error and error_from_last_expect functions
        # to provide a position where parsing failed along with
        # user-friendly messages.
        self.expects = {}

        self.current_line: str
        self.lineno: int = 0
        self.pos: int = 0
        self.current_filename: str

        self.instructions: list[Statement] = []
        self.errors = []

    def parse_file(self, file: str):
        # breakpoint()
        self.current_filename = file
        with open(file, "r", encoding="UTF-8") as fin:
            self.parse_stream(fin)

    def parse_stream(self, stream: TextIO):
        if getattr(self, "current_filename", None) is None:
            self.current_filename = "<stream>"
        for line in stream:
            self.current_line = line.rstrip()
            self.lineno += 1
            self.skip()
            if self.eol():
                # Nothing to parse, skip the line.
                self.pos = 0
                continue
            try:
                # parse_directive and parse_instruction are two main functions
                # and they behave different than other parse_ functions: they
                # do not return actual parsing product, but only a boolean indicating
                # whether the parsing was successful.
                if not self.parse_directive() and not self.parse_instruction():
                    self.error("syntax error")
            except Z80Error as e:
                self.errors.append(e.args[0])
                continue
            finally:
                # Memo table and expects are cleared for each line. This
                # levels out one of the main packrat parsing's drawbacks:
                # linear memory consumption.
                self.pos = 0
                self.memos.clear()
                self.expects.clear()
        if self.errors:
            raise Z80Error(*self.errors)

    def print_instructions(self, stream: TextIO):
        for i in self.instructions:
            print(i, file=stream)

    @memoize
    def parse_instruction(self) -> bool:
        """Parse `[<identifier>:] [<identifier> [<arg1> [<arg2>]]]` """
        parse = False
        if label := self.parse_label():
            self.instructions.append(label)
            parse = True

        pos = self.mark()
        if mnemonic := self.expect_identifier():
            parselet_alts = self.mnemonics.get(mnemonic.upper())
            if parselet_alts is None:
                self.reset(pos)
                self.error("unknown mnemonic: {}", mnemonic)
            pos = self.mark()
            for (alt, data) in parselet_alts.items():
                # Instruction may have several operand type alternatives,
                # e.g. `ld <register> <8-bit-int>` and `ld <addr> <register-pair>`,
                # so backtrack until we found a match or there is no alternative left.
                if (args := self.parse_instruction_args(mnemonic, alt)) is not None:
                    if isinstance(data, self.InstructionData):
                        byte_len, op_bytes = data.instruction_bytes, data.op_bytes
                    else:
                        byte_len, op_bytes = len(data), data
                    opcode = Opcode[mnemonic.upper()]
                    instr = Instruction(opcode, args, op_bytes=op_bytes, length=byte_len)
                    self.instructions.append(instr)
                    if not self.eol():
                        # Expect that there is nothing left on the line
                        self.error("unexpected text")
                    parse = True
                    break
                self.reset(pos)
            else:
                self.error_from_last_expect()

        return parse

    def parse_instruction_args(self, mnemonic: str, parselets: list[Callable]) -> Optional[list[Any]]:
        """Try to parse instruction args using this sequence of parselets"""
        last_parselet_idx = len(parselets) - 1
        args = []
        for i, parselet in enumerate(parselets):
            if arg := parselet():
                if i != last_parselet_idx:
                    if self.eol():
                        self.error("instruction {} expects {} args", mnemonic, len(parselets))
                    if not self.expect_comma():
                        # Ensure args are separated by comma
                        self.error_from_last_expect()
                args.append(arg)
            else:
                return None
        return args

    @memoize
    def parse_directive(self) -> bool:
        """Parse `.<identifier> [...]` """
        if self.expect(r"\."):
            pos = self.mark()
            if name := self.expect_identifier():
                parselets = self.directives.get(name)
                if parselets is None:
                    self.reset(pos)
                    self.error("unknown directive: {}", name)

                args = self.parse_directive_args(parselets)

                if not self.eol():
                    # Expect that there is nothing left on the line
                    self.error("unexpected text")

                directive_kind = DirectiveKind[name]
                directive = self.parseinfo(Directive(directive_kind, args), pos)
                self.instructions.append(directive)

                return True
        return False

    def parse_directive_args(self, parselets: list[Callable]) -> list[Any]:
        """Parse arguments expected by the directive"""
        args = []
        last_parselet_idx = len(parselets) - 1
        for i, parselet in enumerate(parselets):
            if arg := parselet():
                if isiterable(arg):
                    args.extend(arg)
                else:
                    args.append(arg)
            else:
                # Directives have no alternatives, so if the parselet failed,
                # throw an error.
                self.error_from_last_expect()
            if i != last_parselet_idx and not self.expect_comma():
                # Ensure arguments are separated by comma
                self.error_from_last_expect()
        return args

    @memoize
    def parse_label(self) -> Optional[Label]:
        """Parse `<identifier>:` """
        pos = self.mark()
        if (name := self.expect_identifier()) and self.expect_colon():
            return self.parseinfo(Label(name), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_register(self) -> Optional[Operand]:
        pos = self.mark()
        if m := self.expect(r"[a-ehl]\b", "register"):
            return self.parseinfo(Operand(OperandKind.Reg, m[0]), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_register_name(self, reg: str) -> Optional[Operand]:
        pos = self.mark()
        if r := self.expect_str(reg):
            return self.parseinfo(Operand(OperandKind.Reg, r), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_bc(self) -> Optional[Operand]:
        """Parse BC register pair"""
        pos = self.mark()
        if rp := self.expect_str("bc"):
            return self.parseinfo(Operand(OperandKind.RegPair, rp), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_de(self) -> Optional[Operand]:
        """Parse DE register pair"""
        pos = self.mark()
        if rp := self.expect_str("de"):
            return self.parseinfo(Operand(OperandKind.RegPair, rp), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_hl(self) -> Optional[Operand]:
        """Parse HL register pair"""
        pos = self.mark()
        if rp := self.expect_str("hl"):
            return self.parseinfo(Operand(OperandKind.RegPair, rp), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_sp(self) -> Optional[Operand]:
        """Parse SP register pair"""
        pos = self.mark()
        if rp := self.expect_str("sp"):
            return self.parseinfo(Operand(OperandKind.RegPair, rp), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_af(self) -> Optional[Operand]:
        """Parse AF register pair"""
        pos = self.mark()
        if rp := self.expect_str("af"):
            return self.parseinfo(Operand(OperandKind.RegPair, rp), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_addr_combine(self, parselet: Callable) -> Optional[Operand]:
        """Parse anything enclosed in braces"""
        pos = self.mark()
        if self.expect(r"\("):
            if op := parselet():
                if self.expect(r"\)"):
                    return self.parseinfo(Operand(OperandKind.Addr, op.value), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_regpair(self) -> Optional[Operand]:
        """Parse register pair: BC, DE, HL, or SP"""
        pos = self.mark()
        if rp := self.parse_bc():
            return rp
        if rp := self.parse_de():
            return rp
        if rp := self.parse_hl():
            return rp
        if rp := self.parse_sp():
            return rp
        self.reset(pos)
        return None

    @memoize
    def parse_ix(self) -> Optional[Operand]:
        """Parse IX index register"""
        pos = self.mark()
        if self.expect_str("ix"):
            return self.parseinfo(Operand(OperandKind.IX), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_iy(self) -> Optional[Operand]:
        """Parse IY index register"""
        pos = self.mark()
        if self.expect_str("iy"):
            return self.parseinfo(Operand(OperandKind.IY), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_ixd_addr(self) -> Optional[Operand]:
        """Parse IX+d address"""
        pos = self.mark()
        if self.expect(r"\("):
            if self.parse_ix():
                # Ensure a sign here, but do not consume it.
                pos1 = self.mark()
                if self.expect(r"[+-]"):
                    self.reset(pos1)
                    if d := self.expect_integer(8):
                        if self.expect(r"\)"):
                            return self.parseinfo(Operand(OperandKind.IXDAddr, d), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_iyd_addr(self) -> Optional[Operand]:
        """Parse IY+d address"""
        pos = self.mark()
        if self.expect(r"\("):
            if self.parse_iy():
                # Ensure a sign here, but do not consume it.
                pos1 = self.mark()
                if self.expect(r"[+-]"):
                    self.reset(pos1)
                    if d := self.expect_integer(8):
                        if self.expect(r"\)"):
                            return self.parseinfo(Operand(OperandKind.IYDAddr, d), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_const(self) -> Optional[Operand]:
        pos = self.mark()
        if name := self.expect_identifier():
            return self.parseinfo(Operand(OperandKind.Const, name), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_cond_flag(self) -> Optional[Operand]:
        if op := self.parse_unset_zero_flag():
            return op
        if op := self.parse_zero_flag():
            return op
        if op := self.parse_unset_carry_flag():
            return op
        if op := self.parse_carry_flag():
            return op
        if op := self.parse_unset_pv_flag():
            return op
        if op := self.parse_pv_flag():
            return op
        if op := self.parse_unset_sign_flag():
            return op
        if op := self.parse_sign_flag():
            return op
        return None

    @memoize
    def parse_zero_flag(self) -> Optional[Operand]:
        return self.parse_flag("z")

    @memoize
    def parse_unset_zero_flag(self) -> Optional[Operand]:
        return self.parse_flag("nz")

    @memoize
    def parse_carry_flag(self) -> Optional[Operand]:
        return self.parse_flag("c")

    @memoize
    def parse_unset_carry_flag(self) -> Optional[Operand]:
        return self.parse_flag("nc")

    @memoize
    def parse_sign_flag(self) -> Optional[Operand]:
        return self.parse_flag("m")

    @memoize
    def parse_unset_sign_flag(self) -> Optional[Operand]:
        return self.parse_flag("p")

    @memoize
    def parse_pv_flag(self) -> Optional[Operand]:
        return self.parse_flag("pe")

    @memoize
    def parse_unset_pv_flag(self) -> Optional[Operand]:
        return self.parse_flag("po")

    @memoize
    def parse_flag(self, s: str) -> Optional[Operand]:
        pos = self.mark()
        if f := self.expect_str(s):
            return self.parseinfo(Operand(OperandKind.Flag, f), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_abs_label_ref(self) -> Optional[Operand]:
        pos = self.mark()
        if name := self.expect_identifier():
            return self.parseinfo(Operand(OperandKind.AbsLabel, name), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_int8_literal(self, int_repr: str) -> Optional[Operand]:
        pos = self.mark()
        if s := self.expect_str(int_repr):
            return self.parseinfo(Operand(OperandKind.Int8, int(s)), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_rel_label_ref(self) -> Optional[Operand]:
        pos = self.mark()
        if name := self.expect_identifier():
            return self.parseinfo(Operand(OperandKind.RelLabel, name), pos)
        self.reset(pos)
        return None

    @memoize
    def expect_str(self, s: str) -> Optional[str]:
        pos = self.mark()
        if m := self.expect(fr"{s}\b"):
            return m[0]
        self.reset(pos)
        return None

    @memoize
    def expect_identifier(self) -> Optional[str]:
        pos = self.mark()
        if m := self.expect(r"[a-z_][a-z0-9_]*\b", "identifier"):
            return m[0]
        self.reset(pos)
        return None

    @memoize
    def parse_i8_op(self) -> Optional[Operand]:
        """Parse 8 bit integer into an Int Operand"""
        pos = self.mark()
        if i := self.expect_integer(8):
            return self.parseinfo(Operand(OperandKind.Int8, i), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_i16_op(self) -> Optional[Operand]:
        """Parse 16 bit integer into an Int Operand"""
        pos = self.mark()
        if i := self.expect_integer(16):
            return self.parseinfo(Operand(OperandKind.Int16, i), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_bit_pos(self) -> Optional[Operand]:
        """Parse bit position integer (from 0 to 7 inclusively)"""
        pos = self.mark()
        if i := self.expect_integer(8):
            if i < 0 or i > 7:
                self.error("bit position must be in range [0, 7], got {}", i)
            return self.parseinfo(Operand(OperandKind.Int8, i), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_page0_mem_loc(self) -> Optional[Operand]:
        """Parse page0 memory location integer"""
        pos = self.mark()
        if i := self.expect_integer(8):
            return self.parseinfo(Operand(OperandKind.MemLoc, i), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_char(self) -> Optional[Operand]:
        """Parse an 8-bit character literal"""
        pos = self.mark()
        if self.expect("'", None, False):
            if self.expect(r"\\", None, False):
                if ch := self.expect(r"[rnt0]"):
                    val = self.escape_chars[ch[0]]
                else:
                    self.error("invalid escape sequence")
            elif ch := self.expect(r"[^']", None, False):
                val = ch[0]
            if not self.expect(r"'"):
                self.error_from_last_expect()
            if (bit_len := ceil_pow2(ord(val).bit_length())) > 8:
                self.reset(pos)
                self.error("char must be 8-bit, got {}-bit", bit_len)
            return self.parseinfo(Operand(OperandKind.Char, val), pos)
        self.reset(pos)
        return None

    @memoize
    def parse_string(self) -> Optional[Operand]:
        """Parse a sequence of 8-bit characters"""
        pos = self.mark()
        chars = []
        if self.expect('"', None, False):
            while not self.expect('"', None, False):
                ch_pos = self.mark()
                if self.eol():
                    self.error_from_last_expect()
                if self.expect(r"\\", None, False):
                    if ch := self.expect(r"[rnt0]", None, False):
                        chars.append(self.escape_chars[ch[0]])
                    else:
                        self.error("invalid escape sequence")
                elif ch := self.expect(r'[^"]', None, False):
                    val = ch[0]
                    if (bit_len := ceil_pow2(ord(val).bit_length())) > 8:
                        self.reset(ch_pos)
                        self.error("char must be 8-bit, got {}-bit", bit_len)
                    chars.append(val)
            s = "".join(chars)
            return self.parseinfo(Operand(OperandKind.String, s), pos)
        self.reset(pos)
        return None

    def parse_byte_sequence(self):
        """Parse sequence of strings, chars, and 8-bit integers"""
        args = []
        while (
            (arg := self.parse_string()) is not None or
            (arg := self.parse_char()) is not None or
            (arg := self.parse_i8_op()) is not None
        ):
            if not self.eol():
                if not self.expect_comma():
                    self.error_from_last_expect()
            args.append(arg)
        if not args:
            self.error("expected at least one string literal, char literal, or 8-bit integer")
        return args

    @memoize
    def expect_integer(self, bits: int = 8) -> Optional[int]:
        """Parses n-bit integer in decimal, hexadecimal, octal, or binary format"""
        pos = self.mark()
        negative = False
        pattern, base = r"[1-9][0-9]*", 10
        if m := self.expect(r"[+-]"):
            negative = m[0] == "-"
        if m := self.expect(r"0x"):
            pattern, base = r"[0-9a-fA-F]+", 16
        elif m := self.expect(r"0b"):
            pattern, base = r"[01_]+", 2
        elif m := self.expect(r"0o"):
            pattern, base = r"[0-7]+", 8

        if m := self.expect(pattern, "integer"):
            val = int(m[0], base) * (-1 if negative else 1)
            if val.bit_length() > bits:
                self.reset(pos)
                pow2bits = ceil_pow2(val.bit_length())
                self.error(f"expected {bits}-bit integer, got {pow2bits}-bit instead")
            return val

        self.reset(pos)
        return None

    @memoize
    def expect_colon(self) -> Optional[re.Match]:
        pos = self.mark()
        if m := self.expect(":", "colon"):
            return m
        self.reset(pos)
        return None

    @memoize
    def expect_comma(self) -> Optional[re.Match]:
        pos = self.mark()
        if m := self.expect(",", "comma"):
            return m
        self.reset(pos)
        return None

    @memoize
    def expect(self, pattern: str, descr: Optional[str] = None, skip: bool = True) -> Optional[re.Match]:
        """Try to match a pattern and optionally skip whitespace

        If pattern matching failed, then adds an expectation record for the current
        parsing position, which can be used to construct an error later. Description
        is an optional string describing the pattern.

        Caches parsing result.
        """
        pos = self.mark()
        if m := self.consume(pattern, skip=skip):
            return m
        column = self.mark()
        expects_list = self.expects.get(column)
        if expects_list is None:
            expects_list = self.expects[column] = []
        expects_list.append(descr or pattern)
        self.reset(pos)
        return None

    def skip(self):
        """Skip over whitespace and comments"""
        while True:
            if self.consume(r"\s+", skip=False):
                continue
            if self.consume(r"(?s);.*", skip=False):
                continue
            break

    def consume(self, pattern: str, skip: bool = True) -> Optional[re.Match]:
        """Try to match a pattern and optionally skip whitespace"""
        p = re.compile(pattern)
        if m := p.match(self.current_line, pos=self.pos):
            self.pos += len(m[0])
            if skip:
                self.skip()
            return m
        return None

    def error(self, message: str, *args):
        """Raise an error"""
        col = self.mark()
        stream = StringIO()
        print(f"error: {message.format(*args)}", file=stream)
        print(f"at: {self.current_filename}:{self.lineno}:{col}", file=stream)
        print(self.current_line, file=stream)
        print(f"{' ' * col}^", file=stream)
        raise Z80Error(stream.getvalue())

    def error_from_last_expect(self):
        """Raise an error for the farthest parsed position"""
        max_col = max(self.expects)
        expect = self.expects[max_col][-1]
        stream = StringIO()
        print(f"error: expected {expect}", file=stream)
        print(f"at: {self.current_filename}:{self.lineno}:{max_col}", file=stream)
        print(self.current_line, file=stream)
        print(f"{' ' * max_col}^", file=stream)
        raise Z80Error(stream.getvalue())

    def mark(self) -> int:
        """Get current parsing position"""
        return self.pos

    def reset(self, pos: int):
        """Backtrack"""
        self.pos = pos

    def eol(self) -> int:
        """Is there an End Of a Line?"""
        return self.pos == len(self.current_line)

    def parseinfo(self, obj: Statement, pos: int) -> Statement:
        """Augment a statement with the parse info"""
        obj.lineno = self.lineno
        obj.column = pos
        obj.line = self.current_line
        obj.filename = self.current_filename
        return obj


class Z80AsmLayouter:

    def __init__(self, program: list[Statement]):
        self.program = program
        self.errors: list[Z80Error] = []

        self.addr = 0
        self.labels: dict[int, Label] = {}
        self.label_refs: list[tuple[Operand, Instruction]] = []
        self.consts: dict[str, int] = {}
        self.const_refs: list[Operand] = []

    def layout_program(self):
        for i, inst in enumerate(self.program):
            self.layout_instruction(i, inst)
        self.assign_label_addrs()
        self.resolve_labels()
        self.resolve_constants()
        if self.errors:
            raise Z80Error(*self.errors)

    def layout_instruction(self, i: int, inst):
        if isinstance(inst, Directive):
            if inst.kind == DirectiveKind.org:
                addr = inst.operands[0].value
                if self.addr > addr:
                    self.error(f"org directive address 0x{addr:04X} is behind current address 0x{self.addr:04X}", inst)
                self.addr = inst.operands[0].value
                inst.addr = self.addr
            elif inst.kind == DirectiveKind.equ:
                self.add_const(inst)
                inst.addr = self.addr
            elif inst.kind == DirectiveKind.db:
                inst.length = self.db_length(inst)
                inst.addr = self.addr
                self.addr += inst.length
        elif isinstance(inst, Label):
            self.labels[i] = inst
        elif isinstance(inst, Instruction):
            inst.addr = self.addr
            self.addr += inst.length

            for op in inst.operands:
                if op.kind in (OperandKind.AbsLabel, OperandKind.RelLabel):
                    self.label_refs.append((op, inst))
                elif op.kind == OperandKind.Const:
                    self.const_refs.append(op)

    def db_length(self, directive: Directive) -> int:
        """Calculate the length of a Define Byte directive"""
        length = 0
        for op in directive.operands:
            if isinstance(op.value, str):
                length += len(op.value)
            elif isinstance(op.value, int):
                length += 1
        return length

    def assign_label_addrs(self):
        for idx, label in self.labels.items():
            if (addr := self.get_next_addr(idx, self.program)) is not None:
                label.addr = addr

    def get_next_addr(self, idx: int, program: list) -> int:
        """Get the next address after the previous instruction"""
        for i in range(idx - 1, -1, -1):
            inst = program[i]
            if isinstance(inst, Instruction):
                return inst.addr + inst.length
            if isinstance(inst, Directive):
                return inst.addr
        return 0

    def resolve_labels(self):
        labels = {label.name: label.addr for label in self.labels.values()}
        for op, inst in self.label_refs:
            if (addr := labels.get(op.value)) is not None:
                if op.kind == OperandKind.RelLabel:
                    addr -= inst.addr
                    if addr > 0:
                        # Next byte offset
                        addr += 1
                    if addr > 129 or addr < -126:
                        self.error("label outside relative jump range")
                op.name = op.value
                op.value = addr
            else:
                self.error("reference to an undefined label {}", op, op.value)

    def resolve_constants(self):
        for op in self.const_refs:
            if (value := self.consts[op.value]) is not None:
                op.name = op.value
                op.value = value
                break
            else:
                self.error("undefined name {}", op, op.value)

    def add_const(self, obj: Directive):
        assert obj.kind == DirectiveKind.equ
        name, value = obj.operands
        if name.value in self.consts:
            self.error("name {} is redefined", name)
        self.consts[name.value] = value

    def error(self, message: str, stmt: Optional[Statement], *args):
        stream = StringIO()
        print(f"error: {message.format(*args)}", file=stream)
        if stmt is None:
            return
        print(f"{stmt.filename}:{stmt.lineno}:{stmt.column}", file=stream)
        print(stmt.line, file=stream)
        print(f"{' ' * stmt.column}^", file=stream)
        self.errors.append(Z80Error(stream.getvalue()))


class Z80AsmPrinter:
    """Pretty print an assembly program

    If `replace_names` is True, replaces Label and Const references by their
    values.
    """

    ESCCHARS = {
        "\0": r"\0",
        "\r": r"\r",
        "\n": r"\n",
        "\t": r"\t"
    }

    def __init__(
        self,
        file: TextIO,
        replace_names: bool = False,
        interpret_literals: bool = False
    ):
        self.file = file
        self.replace_names = replace_names
        self.interpret_literals = interpret_literals
        self.addr: int = 0
        self.new_line = True

    def print_program(self, instructions: list[Statement]):
        for inst in instructions:
            self.print_statement(inst)

    def print_statement(self, stmt: Statement):
        range_start, range_end = 0, 4
        cont_addr = 0
        with self.line():
            self.addr = stmt.addr
            self.put(f"{self.addr:04X}")
            if not isinstance(stmt, Label) and stmt.encoded is not None:
                encoded = " ".join(f"{i:02X}" for i in stmt.encoded[range_start:range_end])
                self.put(f"{encoded:<12}")
                cont_addr = self.addr + range_end
            else:
                self.put(" " * 12)
            if isinstance(stmt, Label):
                self.print_label(stmt)
            elif isinstance(stmt, Instruction):
                self.print_instruction(stmt)
            elif isinstance(stmt, Directive):
                self.print_directive(stmt)
        if not isinstance(stmt, Label) and stmt.encoded is not None:
            if range_end < len(stmt.encoded):
                range_start, range_end = range_end, range_end + 4
                while chunk := stmt.encoded[range_start:range_end]:
                    with self.line():
                        self.put(f"{cont_addr:04X}")
                        encoded = " ".join(f"{i:02X}" for i in chunk)
                        self.put(encoded)
                        range_start, range_end = range_end, range_end + 4
                        cont_addr += len(chunk)

    def print_label(self, lbl: Label):
        self.put(f"{lbl.name}:")

    def print_instruction(self, inst: Instruction):
        self.put(f"    {inst.opcode.name.lower():<6}")
        self.print_operands(inst.operands)

    def print_directive(self, d: Directive):
        self.put(f".{d.kind.name}")
        self.print_operands(d.operands)
        if d.kind == DirectiveKind.db:
            # +1 next byte offset
            self.addr += d.length + 1

    def print_operands(self, ops: list[Operand]):
        last_el_idx = len(ops) - 1
        for i, op in enumerate(ops):
            self.print_operand(op)
            if i != last_el_idx:
                self.put(",", separate=False)

    def print_operand(self, obj: Operand):
        dispatch_dict = self._construct_print_operand_dispatch_dict()
        if (handler := dispatch_dict.get(obj.kind)) is not None:
            handler(obj)
        else:
            self.put(obj.value)

    @lru_cache(maxsize=1)
    def _construct_print_operand_dispatch_dict(self):

        def handle_addr_op(op):
            if isinstance(op.value, str):
                self.put(f"({op.value})")
            else:
                self.put(f"(0x{op.value:04X})")

        def handle_label_op(op):
            if self.replace_names:
                self.put(f"0x{op.value:04X}")
            else:
                self.put(op.name)

        def handle_rel_label_op(op):
            if self.replace_names:
                self.put(f"{op.value:+}")
            else:
                self.put(op.name)

        def handle_char_op(op):
            if self.interpret_literals:
                self.put(f"0x{ord(op.value):02X}")
            else:
                self.put(f"'{self.esc_char(op.value)}'")

        def handle_string_op(op):
            if self.interpret_literals:
                self.put(" ".join(f"0x{ord(c):02X}" for c in op.value))
            else:
                s = "".join(self.esc_char(c) for c in op.value)
                self.put(f'"{s}"')

        dct = {
            OperandKind.Int8: lambda op: self.put(f"0x{op.value:02X}"),
            OperandKind.Int16: lambda op: self.put(f"0x{op.value:04X}"),
            OperandKind.IX: lambda op: self.put("ix"),
            OperandKind.IY: lambda op: self.put("iy"),
            OperandKind.Addr: handle_addr_op,
            OperandKind.IXDAddr: lambda op: self.put(f"(ix{op.value:+})"),
            OperandKind.IYDAddr: lambda op: self.put(f"(iy{op.value:+})"),
            OperandKind.AbsLabel: handle_label_op,
            OperandKind.RelLabel: handle_rel_label_op,
            OperandKind.Const: handle_label_op,
            OperandKind.Char: handle_char_op,
            OperandKind.String: handle_string_op,
        }

        return dct

    @contextmanager
    def line(self):
        """Put the subsequent prints on one line"""
        try:
            yield
        finally:
            self.put("\n", separate=False)
            self.new_line = True

    def put(self, *args, separate: bool = True):
        if separate and not self.new_line:
            print(" ", end="", file=self.file)
        print(*args, end="", file=self.file)
        self.new_line = False

    def esc_char(self, ch: str):
        """Escape a character or return it as is"""
        return self.ESCCHARS.get(ch, ch)


class Z80AsmCompiler:

    REGISTERS = {
        "a": 0b111,
        "b": 0b000,
        "c": 0b001,
        "d": 0b010,
        "e": 0b011,
        "h": 0b100,
        "l": 0b101
    }

    REGPAIRS = {
        "bc": 0b00,
        "de": 0b01,
        "hl": 0b10,
        "sp": 0b11
    }

    FLAGS = {
        "nz": 0b000,
        "z": 0b001,
        "nc": 0b010,
        "c": 0b011,
        "po": 0b100,
        "pe": 0b101,
        "p": 0b110,
        "m": 0b111
    }

    MEMLOCS = {
        0x00: 0b000,
        0x08: 0b001,
        0x10: 0b010,
        0x18: 0b011,
        0x20: 0b100,
        0x28: 0b101,
        0x30: 0b110,
        0x38: 0b111
    }

    def __init__(self, program: list[Statement]):
        self.program = program
        self.errors: list[Z80Error] = []

    def compile_program(self):
        for stmt in self.program:
            self.compile_statement(stmt)
        if self.errors:
            raise Z80Error(*self.errors)

    def compile_statement(self, stmt: Statement):
        """Compile statements into sequences of bytes"""
        if isinstance(stmt, Directive):
            self.compile_directive(stmt)
            return
        if isinstance(stmt, Label):
            return
        if isinstance(stmt, Instruction):
            self.compile_instruction(stmt)

    def compile_directive(self, d: Directive):
        if d.kind == DirectiveKind.db:
            op_bytes = []
            for op in d.operands:
                if isinstance(op.value, str):
                    op_bytes.extend(ord(i) for i in op.value)
                elif isinstance(op.value, int):
                    op_bytes.append(op.value)
            d.encoded = tuple(op_bytes)

    @lru_cache(maxsize=1)
    def _construct_operand_dispatch_dict(self):

        def handle_memloc_op(op):
            if (val := self.MEMLOCS.get(op.value)) is not None:
                return val
            self.error(op, "invalid Page 0 Memory Location")

        dct = {
            OperandKind.Int16: lambda op: self.i16top(op.value),
            OperandKind.AbsLabel: lambda op: self.i16top(op.value),
            OperandKind.RelLabel: lambda op: op.value,
            OperandKind.Reg: lambda op: self.regtoi(op.value),
            OperandKind.RegPair: lambda op: self.regptoi(op.value),
            OperandKind.Addr: lambda op: self.i16top(op.value) if isinstance(op.value, int) else None,
            OperandKind.Flag: lambda op: self.FLAGS[op.value],
            OperandKind.MemLoc: handle_memloc_op,
            OperandKind.Char: lambda op: ord(op.value)
        }

        return dct

    def compile_instruction(self, inst: Instruction):
        if isinstance(inst.op_bytes, tuple):
            op_bytes = inst.op_bytes
        else:
            dispatch_dict = self._construct_operand_dispatch_dict()
            args = []
            for op in inst.operands:
                if (handler := dispatch_dict.get(op.kind)) is not None:
                    args.append(handler(op))
                else:
                    assert isinstance(op.value, int) or op.value is None
                    args.append(op.value)
            op_bytes = inst.op_bytes(*args)
            assert isinstance(op_bytes, tuple)
            assert all(b.bit_length() <= 8 for b in op_bytes)
        inst.encoded = tuple((i & 0xff) for i in op_bytes)

    def encode(self) -> Generator[int, None, None]:
        """Emit compiled program as a stream of integers"""
        prev_addr, prev_len = 0, 0
        for inst in self.program:
            if isinstance(inst, Label):
                continue
            if inst.addr - prev_addr > prev_len:
                # There is a gap between instructions, fill it up with 0x00
                yield from repeat(0, inst.addr - prev_addr - prev_len)
            if inst.encoded is not None:
                yield from inst.encoded
                prev_len = len(inst.encoded)
            prev_addr = inst.addr

    def emit_bytes(self, file: BinaryIO):
        """Emit compiled program as bytes"""
        for i in self.encode():
            file.write(i.to_bytes(1))
        file.flush()

    def i16top(self, i: int) -> tuple[int, int]:
        """Convert 16-bit integer to pair (lsb, msb)"""
        return (i & 0xff, (i >> 8) & 0xff)

    def regtoi(self, reg: str) -> int:
        """Convert register name to its integer representation"""
        return self.REGISTERS[reg]

    def regptoi(self, regp: str) -> int:
        """Convert register pair to its integer representation"""
        return self.REGPAIRS[regp]

    def error(self, stmt: Optional[Statement], message: str, *args):
        stream = StringIO()
        print(f"error: {message.format(*args)}", file=stream)
        if stmt is None:
            return
        print(f"{stmt.filename}:{stmt.lineno}:{stmt.column}", file=stream)
        print(stmt.line, file=stream)
        print(f"{' ' * stmt.column}^", file=stream)
        self.errors.append(Z80Error(stream.getvalue()))


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("inputs", metavar="INPUT", nargs="*")

    ns = argparser.parse_args()

    asm = Z80AsmParser()
    ltr = Z80AsmLayouter(asm.instructions)
    compiler = Z80AsmCompiler(program=asm.instructions)
    printer = Z80AsmPrinter(file=sys.stdout)
    for i in ns.inputs:
        try:
            asm.parse_file(i)
            ltr.layout_program()
            compiler.compile_program()
            printer.print_program(asm.instructions)

            bstream = BytesIO()
            compiler.emit_bytes(bstream)
            print([f"{b:02X}" for b in bstream.getvalue()])
        except Z80Error as e:
            print(e)
