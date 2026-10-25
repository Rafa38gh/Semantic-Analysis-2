from __future__ import annotations

from collections.abc import Sequence

from Lexer import Token, TokenKind
from ast_nodes import (
    Block, Expr, FunctionDecl, Node, Parameter, PrintItem, Program, 
    SourceSpan, Stmt, StringLiteral, TypeName,
    VarDecl, Assignment, CallStmt, IfStmt, WhileStmt, ReturnStmt, 
    PrintStmt, BinaryExpr, UnaryExpr, CallExpr, IdentifierExpr, 
    IntLiteral, BoolLiteral, BinaryOperator, UnaryOperator, dataclasses
)


TYPE_START = {TokenKind.KW_INT, TokenKind.KW_BOOL, TokenKind.KW_VOID}
EXPRESSION_START = {
    TokenKind.IDENTIFIER,
    TokenKind.INT_LITERAL,
    TokenKind.KW_FALSE,
    TokenKind.KW_TRUE,
    TokenKind.LEFT_PAREN,
    TokenKind.LOGICAL_NOT,
    TokenKind.MINUS,
}
STATEMENT_START = TYPE_START | {
    TokenKind.IDENTIFIER,
    TokenKind.KW_IF,
    TokenKind.KW_WHILE,
    TokenKind.KW_RETURN,
    TokenKind.KW_PRINT,
    TokenKind.LEFT_BRACE,
}


TYPE_BY_TOKEN = {
    TokenKind.KW_INT: TypeName.INT,
    TokenKind.KW_BOOL: TypeName.BOOL,
    TokenKind.KW_VOID: TypeName.VOID,
}


class ParserError(Exception):
    def __init__(self, token: Token, expected: set[TokenKind]):
        self.token = token
        self.expected = frozenset(expected)
        super().__init__()

    @property
    def line(self) -> int:
        return self.token.line

    @property
    def column(self) -> int:
        return self.token.column

    def __str__(self) -> str:
        names = ", ".join(kind.name for kind in sorted(
            self.expected,
            key=lambda kind: kind.value,
        ))
        return (
            f"erro sintático em {self.line}:{self.column}: esperado {{{names}}}, "
            f"encontrado {self.token.kind.name} ({self.token.lexeme!r})"
        )


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        if not self.tokens:
            raise ValueError("a sequência de tokens deve terminar em EOF")
        if self.tokens[-1].kind is not TokenKind.EOF:
            raise ValueError("o último token deve ser EOF")
        if any(token.kind is TokenKind.EOF for token in self.tokens[:-1]):
            raise ValueError("EOF deve aparecer uma única vez, no final")
        self.current = 0

    def peek(self, offset: int = 0) -> Token:
        index = min(self.current + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def check(self, kind: TokenKind) -> bool:
        return self.peek().kind is kind

    def advance(self) -> Token:
        token = self.peek()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kinds: TokenKind | set[TokenKind]) -> Token:
        expected = kinds if isinstance(kinds, set) else {kinds}
        token = self.peek()
        if token.kind not in expected:
            raise ParserError(token, set(expected))
        return self.advance()

    @staticmethod
    def _token_span(token: Token) -> SourceSpan:
        return SourceSpan(
            token.line,
            token.column,
            token.line,
            token.column + len(token.lexeme),
        )

    @staticmethod
    def _start(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.start_line, value.span.start_column
        return value.line, value.column

    @staticmethod
    def _end(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.end_line, value.span.end_column
        return value.line, value.column + len(value.lexeme)

    @classmethod
    def _span(cls, first: Token | Node, last: Token | Node) -> SourceSpan:
        start_line, start_column = cls._start(first)
        end_line, end_column = cls._end(last)
        return SourceSpan(start_line, start_column, end_line, end_column)

    def parse(self) -> Program:
        return self.parse_program()

    # program ::= function* EOF
    def parse_program(self) -> Program:
        start = self.peek()
        functions: list[FunctionDecl] = []
        while self.peek().kind in TYPE_START:
            functions.append(self.parse_function())
        eof = self.expect(TokenKind.EOF)
        return Program(functions, span=self._span(start, eof))

    # function ::= type IDENTIFIER ... block
    def parse_function(self) -> FunctionDecl:
        start = self.peek()
        return_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        self.expect(TokenKind.LEFT_PAREN)
        parameters = (
            self.parse_parameter_list()
            if self.peek().kind in TYPE_START
            else []
        )
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return FunctionDecl(
            return_type,
            name.lexeme,
            parameters,
            body,
            span=self._span(start, body),
        )

    # type ::= KW_INT | KW_BOOL | KW_VOID
    def parse_type(self) -> TypeName:
        token = self.expect(TYPE_START)
        return TYPE_BY_TOKEN[token.kind]

    def parse_parameter_list(self) -> list[Parameter]:
        params = [self.parse_parameter()]
        while self.match(TokenKind.COMMA):
            params.append(self.parse_parameter())
        return params

    def parse_parameter(self) -> Parameter:
        start = self.peek()
        param_type = self.parse_type()
        name_token = self.expect(TokenKind.IDENTIFIER)
        return Parameter(param_type, name_token.lexeme, span=self._span(start, name_token))

    def parse_block(self) -> Block:
        start = self.expect(TokenKind.LEFT_BRACE)
        statements = []
        #le os statements ate encontrar a chave fechando o fim do arquivo
        while not self.check(TokenKind.RIGHT_BRACE) and not self.check(TokenKind.EOF):
            statements.append(self.parse_statement())
        end = self.expect(TokenKind.RIGHT_BRACE)
        return Block(statements, span=self._span(start, end))

    def parse_statement(self) -> Stmt:
        token = self.peek()
        if token.kind in TYPE_START:
            return self.parse_declaration()
        elif token.kind == TokenKind.KW_IF:
            return self.parse_if_statement()
        elif token.kind == TokenKind.KW_WHILE:
            return self.parse_while_statement()
        elif token.kind == TokenKind.KW_RETURN:
            return self.parse_return_statement()
        elif token.kind == TokenKind.KW_PRINT:
            return self.parse_print_statement()
        elif token.kind == TokenKind.LEFT_BRACE:
            return self.parse_block()
        elif token.kind == TokenKind.IDENTIFIER:
            return self.parse_id_or_call_statement()
        else:
            raise ParserError(token, STATEMENT_START)

    def parse_id_or_call_statement(self) -> Stmt:
        name_token = self.expect(TokenKind.IDENTIFIER)
        if self.match(TokenKind.ASSIGN):

            target = IdentifierExpr( name_token.lexeme, span=self._token_span(name_token),)
            value = self.parse_expression();
            end = self.expect(TokenKind.SEMICOLON)

            return Assignment(target, value, span=self._span(name_token, end))

        self.expect(TokenKind.LEFT_PAREN)

        arguments = self.parse_arguments()

        close_paren = self.expect(TokenKind.RIGHT_PAREN)

        call = CallExpr(name_token.lexeme, arguments, span=self._span(name_token, close_paren),)
        end = self.expect(TokenKind.SEMICOLON)
        return CallStmt(call, span=self._span(name_token, end))


    def parse_declaration(self) -> Stmt:
        start = self.peek();
        var_type = self.parse_type()
        name_token = self.expect(TokenKind.IDENTIFIER)

        if self.match(TokenKind.ASSIGN) :
            initializer = self.parse_expression()
        else :
            initializer = None

        end = self.expect(TokenKind.SEMICOLON)

        return VarDecl (var_type,name_token.lexeme, initializer, span=self._span(start, end))

    def parse_if_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_IF)

        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)

        then_block = self.parse_block();
        if self.match(TokenKind.KW_ELSE):
            else_block = self.parse_block()
        else :
            else_block = None

        if else_block is not None :
            end = else_block
        else:
            end = then_block
        return IfStmt(condition, then_block, else_block, span=self._span(start, end))

    def parse_while_statement(self) -> Stmt:
        # while_statement ::= KW_WHILE LEFT_PAREN expression RIGHT_PAREN block
        start = self.expect(TokenKind.KW_WHILE)

        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()

        return WhileStmt(condition, body, span=self._span(start, body))

    def parse_return_statement(self) -> Stmt:
        # return_statement ::= KW_RETURN expression? SEMICOLON
        start = self.expect(TokenKind.KW_RETURN)

        if self.peek().kind in EXPRESSION_START:
            value = self.parse_expression()
        else:
            value = None
        end = self.expect(TokenKind.SEMICOLON) # ; obrigatório

        return ReturnStmt(value, span=self._span(start, end))

    def parse_print_statement(self) -> Stmt:
        # print_statement := KW_PRINT LEFT_PAREN print_item (COMMA print_item)* RIGHT_PAREN SEMICOLON

        start = self.expect(TokenKind.KW_PRINT)
        self.expect(TokenKind.LEFT_PAREN)

        items: list[PrintItem] = [self.parse_print_item()]
        while self.match(TokenKind.COMMA):
            items.append(self.parse_print_item())

        self.expect(TokenKind.RIGHT_PAREN)
        end = self.expect(TokenKind.SEMICOLON)

        return PrintStmt(items, span=self._span(start, end))
        
    def parse_print_item(self) -> PrintItem:
        # print_item ::= string_literals | expression
        if self.check(TokenKind.STRING_LITERAL):
            return self.parse_string_literals()
        elif self.peek().kind in EXPRESSION_START:
            return self.parse_expression()
        else:
            expected = {TokenKind.STRING_LITERAL} | EXPRESSION_START
            raise ParserError(self.peek(), expected)

    def parse_string_literals(self) -> StringLiteral:
        # string_literals ::= STRING_LITERAL+
        first = self.expect(TokenKind.STRING_LITERAL)
        last = first
        values = [first.value]

        while self.check(TokenKind.STRING_LITERAL):
            last = self.advance()
            values.append(last.value)

        final_text = "".join(values)
        return StringLiteral(final_text, span=self._span(first, last))

    def parse_expression(self) -> Expr:
      # expression ::= logical_or
      return self.parse_logical_or()

    def parse_logical_or(self) -> Expr:
        # logical_or ::= logical_and(LOGICAL_OR logical_and)*
        left = self.parse_logical_and()
        while self.match(TokenKind.LOGICAL_OR):
            right = self.parse_logical_and()
            left = BinaryExpr(BinaryOperator.LOGICAL_OR, left, right, span=self._span(left, right))
        return left

    def parse_logical_and(self) -> Expr:
        # logical_and ::= equality(LOGICAL_AND equality)*
        left = self.parse_equality()
        while self.match(TokenKind.LOGICAL_AND):
            right = self.parse_equality()
            left = BinaryExpr(BinaryOperator.LOGICAL_AND, left, right, span=self._span(left, right))

        return left

    def parse_equality(self) -> Expr:
        # equality ::= relational((EQUAL | NOT_EQUAL) relational)*
        left = self.parse_relational()
        while True:
            if self.match(TokenKind.EQUAL_EQUAL):
                right = self.parse_relational()
                left = BinaryExpr(BinaryOperator.EQUAL, left, right, span=self._span(left, right))
            elif self.match(TokenKind.NOT_EQUAL):
                right = self.parse_relational()
                left = BinaryExpr(BinaryOperator.NOT_EQUAL, left, right, span=self._span(left, right))
            else:
                break
        return left

    def parse_relational(self) -> Expr:
        # relational ::= additive((LESS | LESS_EQUAL | GREATER | GREATER_EQUAL) additive)*
        left = self.parse_additive()
        op_map = {
            TokenKind.LESS: BinaryOperator.LESS,
            TokenKind.LESS_EQUAL: BinaryOperator.LESS_EQUAL,
            TokenKind.GREATER: BinaryOperator.GREATER,
            TokenKind.GREATER_EQUAL: BinaryOperator.GREATER_EQUAL,
        }

        while self.peek().kind in op_map:
            token = self.advance()
            right = self.parse_additive()
            left = BinaryExpr(op_map[token.kind], left, right, span=self._span(left, right))

        return left

    def parse_additive(self) -> Expr:
        # additive ::= multiplicative((PLUS | MINUS) multiplicative)*
        left = self.parse_multiplicative()
        while True:
            if self.match(TokenKind.PLUS):
                right = self.parse_multiplicative()
                left = BinaryExpr(BinaryOperator.ADD, left, right, span=self._span(left, right))
            elif self.match(TokenKind.MINUS):
                right = self.parse_multiplicative()
                left = BinaryExpr(BinaryOperator.SUBTRACT, left, right, span=self._span(left, right))
            else:
                break

        return left

    def parse_multiplicative(self) -> Expr:
        # multiplicative ::= unary((STAR | SLASH | PERCENT) unary)*
        left = self.parse_unary()
        op_map = {
            TokenKind.STAR: BinaryOperator.MULTIPLY,
            TokenKind.SLASH: BinaryOperator.DIVIDE,
            TokenKind.PERCENT: BinaryOperator.REMAINDER,
        }
        while self.peek().kind in op_map:
            token = self.advance()
            right = self.parse_unary()
            left = BinaryExpr(op_map[token.kind], left, right, span=self._span(left, right))

        return left

    def parse_unary(self) -> Expr:
        # unary ::= (LOGICAL_NOT | MINUS) unary | primary
        start = self.peek()
        if self.match(TokenKind.LOGICAL_NOT):
            operand = self.parse_unary()
            return UnaryExpr(UnaryOperator.NOT, operand, span=self._span(start, operand))
        elif self.match(TokenKind.MINUS):
            operand = self.parse_unary()
            return UnaryExpr(UnaryOperator.NEGATE, operand, span=self._span(start, operand))

        return self.parse_primary()
    
    def parse_primary(self) -> Expr:
        token = self.peek()

        if self.match(TokenKind.INT_LITERAL):
            return IntLiteral(int(token.value), span=self._token_span(token))

        if self.match(TokenKind.KW_TRUE):
            return BoolLiteral(True, span=self._token_span(token))

        if self.match(TokenKind.KW_FALSE):
            return BoolLiteral(False, span=self._token_span(token))

        if self.match(TokenKind.LEFT_PAREN):
            start = token
            expr = self.parse_expression()
            end = self.expect(TokenKind.RIGHT_PAREN)
            return dataclasses.replace(expr, span=self._span(start, end))

        if token.kind is TokenKind.IDENTIFIER:
            name_token = self.advance()
            if self.match(TokenKind.LEFT_PAREN):
                args = self.parse_arguments()
                close_paren = self.expect(TokenKind.RIGHT_PAREN)
                return CallExpr(name_token.lexeme, args, span=self._span(name_token, close_paren))
            return IdentifierExpr(name_token.lexeme, span=self._token_span(name_token))

        raise ParserError(token, EXPRESSION_START)
        

    def parse_arguments(self) -> list[Expr]:
        if self.peek().kind in EXPRESSION_START:
            args = [self.parse_expression()]
            while self.match(TokenKind.COMMA):
                args.append(self.parse_expression())
            return args
        return []