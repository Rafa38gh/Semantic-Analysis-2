from __future__ import annotations

import ast_nodes as ast
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind


class _Unknown:
    """Expressão de tipo desconhecido."""

    def __repr__(self) -> str:
        return "<desconhecido>"

UNKNOWN = _Unknown()

StaticType = ast.TypeName | _Unknown

INT_LITERAL_MAX = 2**63 - 1

VALUE_TYPES = {ast.TypeName.INT, ast.TypeName.BOOL}

ARITHMETIC_OPERATORS = {
    ast.BinaryOperator.ADD,
    ast.BinaryOperator.SUBTRACT,
    ast.BinaryOperator.MULTIPLY,
    ast.BinaryOperator.DIVIDE,
    ast.BinaryOperator.REMAINDER,
}
RELATIONAL_OPERATORS = {
    ast.BinaryOperator.LESS,
    ast.BinaryOperator.LESS_EQUAL,
    ast.BinaryOperator.GREATER,
    ast.BinaryOperator.GREATER_EQUAL,
}
EQUALITY_OPERATORS = {
    ast.BinaryOperator.EQUAL,
    ast.BinaryOperator.NOT_EQUAL,
}
LOGICAL_OPERATORS = {
    ast.BinaryOperator.LOGICAL_AND,
    ast.BinaryOperator.LOGICAL_OR,
}


def check_types(program: ast.Program) -> None:
    """Determine tipos de expressões e valide seus contextos."""

    # 1. Use os símbolos anexados pela resolução de nomes.
    # 2. Determine cada expressão de baixo para cima.
    # 3. Valide operadores, chamadas, comandos e declarações.
    # 4. Anote expressões válidas e acumule os diagnósticos da passagem.
    TypeChecker().check_program(program)


class TypeChecker:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = [] # Lista de erros
        self.current_return_type: ast.TypeName | None = None

    def error(self, kind: SemanticErrorKind, message: str, span: ast.SourceSpan) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, span))

    def check_program(self, program: ast.Program) -> None:
        for function in program.functions:
            self.check_function(function)

        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    def check_function(self, function: ast.FunctionDecl) -> None:
        for parameter in function.parameters:
            if parameter.type is ast.TypeName.VOID:
                self.error(SemanticErrorKind.VOID_PARAMETER, f"parâmetro '{parameter.name}' não pode ser void", parameter.span)

        self.current_return_type = function.return_type
        self.check_block(function.body)
        self.current_return_type = None

    def check_block(self, block: ast.Block) -> None:
        for statement in block.statements:
            self.check_statement(statement)

    def check_statement(self, statement: ast.Stmt) -> None:
        match statement:
            case ast.Block():
                self.check_block(statement)
            case ast.VarDecl():
                self.check_var_decl(statement)
            case ast.Assignment():
                self.check_assignment(statement)
            case ast.CallStmt():
                self.check_expr(statement.call)
            case ast.IfStmt():
                self.check_condition(statement.condition)
                self.check_block(statement.then_block)
                if statement.else_block is not None:
                    self.check_block(statement.else_block)
            case ast.WhileStmt():
                self.check_condition(statement.condition)
                self.check_block(statement.body)
            case ast.ReturnStmt():
                self.check_return(statement)
            case ast.PrintStmt():
                self.check_print(statement)
            case _:
                raise TypeError(f"comando desconhecido: {type(statement).__name__}")

    def check_var_decl(self, declaration: ast.VarDecl) -> None:
        if declaration.type is ast.TypeName.VOID:
            self.error(
                SemanticErrorKind.VOID_VARIABLE,
                f"variável '{declaration.name}' não pode ser void",
                declaration.span,
            )
            expected: StaticType = UNKNOWN
        else:
            expected = declaration.type

        if declaration.initializer is None:
            return

        actual = self.check_value(declaration.initializer)
        if self.is_mismatch(expected, actual):
            self.error(
                SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                f"inicializador do tipo {self.show(actual)}, "
                f"esperado {self.show(expected)}",
                declaration.initializer.span,
            )

    def check_assignment(self, statement: ast.Assignment) -> None:
        expected = self.check_expr(statement.target)
        actual = self.check_value(statement.value)
        if self.is_mismatch(expected, actual):
            self.error(
                SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                f"atribuição de {self.show(actual)} a "
                f"'{statement.target.name}' do tipo {self.show(expected)}",
                statement.value.span,
            )

    def check_condition(self, condition: ast.Expr) -> None:
        actual = self.check_value(condition)
        if self.is_mismatch(ast.TypeName.BOOL, actual):
            self.error(
                SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                f"condição deve ser bool, recebeu {self.show(actual)}",
                condition.span,
            )
    
    def check_return(self, statement: ast.ReturnStmt) -> None:
        expected = self.current_return_type

        if statement.value is None:
            if expected is not ast.TypeName.VOID:
                self.error(
                    SemanticErrorKind.RETURN_MISMATCH,
                    f"'return;' em função que retorna {self.show(expected)}",
                    statement.span,
                )
            return

        if expected is ast.TypeName.VOID:
            self.check_expr(statement.value)
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                "função void não pode retornar um valor",
                statement.value.span,
            )
            return

        actual = self.check_value(statement.value)
        if self.is_mismatch(expected, actual):
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                f"retorno do tipo {self.show(actual)}, "
                f"esperado {self.show(expected)}",
                statement.value.span,
            )

    def check_print(self, statement: ast.PrintStmt) -> None:
        for item in statement.items:
            if isinstance(item, ast.StringLiteral):
                continue
            self.check_value(item)

    def check_expr(self, expr: ast.Expr) -> StaticType:
        match expr:
            case ast.IntLiteral():
                return self.check_int_literal(expr)
            case ast.BoolLiteral():
                return self.annotate(expr, ast.TypeName.BOOL)
            case ast.IdentifierExpr():
                return self.check_identifier(expr)
            case ast.UnaryExpr():
                return self.check_unary(expr)
            case ast.BinaryExpr():
                return self.check_binary(expr)
            case ast.CallExpr():
                return self.check_call(expr)
            case _:
                raise TypeError(f"expressão desconhecida: {type(expr).__name__}")

    def check_value(self, expr: ast.Expr) -> StaticType:
        type_ = self.check_expr(expr)
        if type_ is ast.TypeName.VOID:
            self.error(SemanticErrorKind.VOID_VALUE_USED, "valor de tipo void não pode ser usado", expr.span)
            return UNKNOWN

        return type_

    def check_int_literal(self, expr: ast.IntLiteral) -> StaticType:
        if not 0 <= expr.value <= INT_LITERAL_MAX:
            self.error(
                SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                f"literal inteiro {expr.value} fora do intervalo [0, 2^63-1]",
                expr.span,
            )
        return self.annotate(expr, ast.TypeName.INT)

    def check_identifier(self, expr: ast.IdentifierExpr) -> StaticType:
        symbol = expr.metadata["symbol"]
        if symbol.type not in VALUE_TYPES:
            return UNKNOWN
        return self.annotate(expr, symbol.type)

    def check_unary(self, expr: ast.UnaryExpr) -> StaticType:
        operand = self.check_value(expr.operand)

        if expr.operator is ast.UnaryOperator.NEGATE:
            required = ast.TypeName.INT
        else:  # UnaryOperator.NOT
            required = ast.TypeName.BOOL

        if operand is UNKNOWN:
            return UNKNOWN
        if operand is not required:
            self.error(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                f"'{expr.operator.value}' exige {self.show(required)}, "
                f"recebeu {self.show(operand)}",
                expr.span,
            )
            return UNKNOWN
        return self.annotate(expr, required)

    def check_binary(self, expr: ast.BinaryExpr) -> StaticType:
        left = self.check_value(expr.left)
        right = self.check_value(expr.right)
        op = expr.operator

        if op in ARITHMETIC_OPERATORS:
            return self.check_fixed_operands(expr, left, right, ast.TypeName.INT, ast.TypeName.INT)
        if op in RELATIONAL_OPERATORS:
            return self.check_fixed_operands(expr, left, right, ast.TypeName.INT, ast.TypeName.BOOL)
        if op in LOGICAL_OPERATORS:
            return self.check_fixed_operands(expr, left, right, ast.TypeName.BOOL, ast.TypeName.BOOL)

        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        if left is not right:
            self.invalid_binary(expr, left, right)
            return UNKNOWN
        return self.annotate(expr, ast.TypeName.BOOL)

    def check_fixed_operands(
        self,
        expr: ast.BinaryExpr,
        left: StaticType,
        right: StaticType,
        operand_type: ast.TypeName,
        result_type: ast.TypeName,
    ) -> StaticType:
        """Operadores cujos dois operandos têm um tipo fixo exigido."""
        wrong = any(
            side is not UNKNOWN and side is not operand_type
            for side in (left, right)
        )
        if wrong:
            self.invalid_binary(expr, left, right)
            return UNKNOWN
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        return self.annotate(expr, result_type)

    def invalid_binary(self, expr: ast.BinaryExpr, left: StaticType, right: StaticType) -> None:
        self.error(
            SemanticErrorKind.INVALID_BINARY_OPERANDS,
            f"operandos inválidos para '{expr.operator.value}': "
            f"{self.show(left)} e {self.show(right)}",
            expr.span,
        )

    def check_call(self, call: ast.CallExpr) -> StaticType:
        arg_types = [self.check_value(arg) for arg in call.arguments]

        symbol = call.metadata["symbol"]
        parameter_types = symbol.parameter_types
        valid = True

        if len(arg_types) != len(parameter_types):
            self.error(
                SemanticErrorKind.ARITY_MISMATCH,
                f"'{symbol.name}' espera {len(parameter_types)} argumento(s), "
                f"recebeu {len(arg_types)}",
                call.span,
            )
            valid = False

        for arg, actual, expected in zip(call.arguments, arg_types, parameter_types):
            if expected not in VALUE_TYPES:
                valid = False
            elif self.is_mismatch(expected, actual):
                self.error(
                    SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                    f"argumento do tipo {self.show(actual)}, "
                    f"esperado {self.show(expected)}",
                    arg.span,
                )
                valid = False
            elif actual is UNKNOWN:
                valid = False

        if valid:
            self.annotate(call, symbol.type)
        return symbol.type


    @staticmethod
    def annotate(expr: ast.Expr, type_: StaticType) -> StaticType:
        if isinstance(type_, ast.TypeName):
            expr.metadata["type"] = type_
        return type_

    @staticmethod
    def is_mismatch(expected: StaticType, actual: StaticType) -> bool:
        return (
            expected is not UNKNOWN
            and actual is not UNKNOWN
            and expected is not actual
        )

    @staticmethod
    def show(type_: StaticType | None) -> str:
        if isinstance(type_, ast.TypeName):
            return type_.value
        return "<desconhecido>"