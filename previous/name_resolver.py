from __future__ import annotations

import ast_nodes as ast
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Scope, Symbol, SymbolKind


def declare(scope: Scope, symbol: Symbol) -> bool:
    """Declara um símbolo no escopo.
    Retorna False se o nome já existe no escopo"""
    if symbol.name in scope.symbols:
        return False
    scope.symbols[symbol.name] = symbol
    return True


def lookup(scope: Scope | None, name: str) -> Symbol | None:
    """Procura o nome do escopo mais interno"""
    while scope is not None:
        symbol = scope.symbols.get(name)

        if symbol is not None:
            return symbol
        scope = scope.parent

    return None


def resolve_names(program: ast.Program) -> None:
    """Construa escopos, símbolos e vínculos entre usos e declarações."""

    # 1. Colete todas as assinaturas de função.
    # 2. Valide a existência e a assinatura de main.
    # 3. Percorra os corpos em ordem, criando um escopo para cada bloco.
    # 4. Anote declarações, usos e blocos na AST.
    # 5. Acumule os diagnósticos desta passagem antes de lançar SemanticError.
    NameResolver().resolve_program(program)


class NameResolver:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = [] #Lista de erros
        self.functions: dict[str, FunctionSymbol] = {}      # Tabela de funções
        self.scope: Scope | None = None

    def error(self, kind: SemanticErrorKind, message: str, span: ast.SourceSpan) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, span))

    def resolve_program(self, program: ast.Program) -> None:
        for function in program.functions:
            self.collect_function(function)

        self.check_main(program)

        for function in program.functions:
            self.resolve_function(function)
        
        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    def collect_function(self, function: ast.FunctionDecl) -> None:
        symbol = FunctionSymbol(
            name=function.name,
            kind = SymbolKind.FUNCTION,
            type = function.return_type,
            declaration=function,
            parameter_types=tuple(p.type for p in function.parameters),
        )
        function.metadata['symbol'] = symbol

        if function.name in self.functions:
            self.error(SemanticErrorKind.DUPLICATE_FUNCTION, f"Função {function.name} já declarada", function.span)
            return

        self.functions[function.name] = symbol

    def check_main(self, program: ast.Program) -> None:
        main = self.functions.get("main")

        if main is None:
            self.error(SemanticErrorKind.INVALID_MAIN, "programa não declarou 'int main()'", program.span)
            return

        if main.type is not ast.TypeName.INT or main.parameter_types:
           self.error(SemanticErrorKind.INVALID_MAIN, "a assinatura da main deve ser 'int main()'", main.declaration.span)

    def resolve_function(self, function: ast.FunctionDecl) -> None:
        scope = Scope(parent=None)
        function.body.metadata["scope"] = scope

        for parameter in function.parameters:
            symbol = Symbol(
                name=parameter.name,
                kind=SymbolKind.PARAMETER,
                type=parameter.type,
                declaration=parameter,
            )
            parameter.metadata["symbol"] = symbol
            if not declare(scope, symbol):
                self.error(
                    SemanticErrorKind.DUPLICATE_DECLARATION,
                    f"parâmetro '{parameter.name}' já declarado",
                    parameter.span,
                )

        self.scope = scope
        for statement in function.body.statements:
            self.resolve_statement(statement)
        self.scope = None

    def resolve_block(self, block: ast.Block) -> None:
        """Bloco interno: escopo novo, filho do corrente."""
        scope = Scope(parent=self.scope)
        block.metadata["scope"] = scope

        previous = self.scope
        self.scope = scope
        for statement in block.statements:
            self.resolve_statement(statement)
        self.scope = previous

    def resolve_statement(self, statement: ast.Stmt) -> None:
        match statement:
            case ast.Block():
                self.resolve_block(statement)
            case ast.VarDecl():
                self.resolve_var_decl(statement)
            case ast.Assignment():
                self.resolve_identifier(statement.target)
                self.resolve_expression(statement.value)
            case ast.CallStmt():
                self.resolve_expression(statement.call)
            case ast.IfStmt():
                self.resolve_expression(statement.condition)
                self.resolve_block(statement.then_block)
                if statement.else_block is not None:
                    self.resolve_block(statement.else_block)
            case ast.WhileStmt():
                self.resolve_expression(statement.condition)
                self.resolve_block(statement.body)
            case ast.ReturnStmt():
                if statement.value is not None:
                    self.resolve_expression(statement.value)
            case ast.PrintStmt():
                for item in statement.items:
                    if isinstance(item, ast.Expr):
                        self.resolve_expression(item)
            case _:
                raise TypeError(f"comando desconhecido: {type(statement).__name__}")

    def resolve_var_decl(self, declaration: ast.VarDecl) -> None:
        symbol = Symbol(
            name=declaration.name,
            kind=SymbolKind.VARIABLE,
            type=declaration.type,
            declaration=declaration,
        )
        declaration.metadata["symbol"] = symbol

        if not declare(self.scope, symbol):
            self.error(
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"variável '{declaration.name}' já declarada neste escopo",
                declaration.span,
            )

        if declaration.initializer is not None:
            self.resolve_expression(declaration.initializer)

    def resolve_identifier(self, identifier: ast.IdentifierExpr) -> None:
        symbol = lookup(self.scope, identifier.name)
        if symbol is None:
            self.error(
                SemanticErrorKind.UNDECLARED_VARIABLE,
                f"variável '{identifier.name}' não declarada",
                identifier.span,
            )
            return
        identifier.metadata["symbol"] = symbol

    def resolve_expression(self, expression: ast.Expr) -> None:
        match expression:
            case ast.IdentifierExpr():
                self.resolve_identifier(expression)
            case ast.CallExpr():
                symbol = self.functions.get(expression.name)
                if symbol is None:
                    self.error(
                        SemanticErrorKind.UNDECLARED_FUNCTION,
                        f"função '{expression.name}' não declarada",
                        expression.span,
                    )
                else:
                    expression.metadata["symbol"] = symbol
                # Todos os argumentos são visitados, mesmo com erro.
                for argument in expression.arguments:
                    self.resolve_expression(argument)
            case ast.UnaryExpr():
                self.resolve_expression(expression.operand)
            case ast.BinaryExpr():
                self.resolve_expression(expression.left)
                self.resolve_expression(expression.right)
            case ast.IntLiteral() | ast.BoolLiteral() | ast.StringLiteral():
                pass
            case _:
                raise TypeError(f"expressão desconhecida: {type(expression).__name__}")