"""Construa um CFG auxiliar por função, preservando os nós da AST."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from ast_nodes import Block, FunctionDecl, IfStmt, Stmt, WhileStmt, ReturnStmt, BoolLiteral
from cfg import BlockId, CFG, Jump, Branch, Return, ScopeExit


class CFGBuilder:
    def build(self, function: FunctionDecl) -> CFG:
        self.cfg = CFG(function)
        body_entry = self.cfg.new_block()
        self.cfg.terminate(self.cfg.entry, Jump(body_entry.id))
        continuation = self._build_block(function.body, body_entry.id)
        if continuation is not None:
            self.cfg.terminate(continuation, Jump(self.cfg.fallthrough_exit))
        return self.cfg

    def _build_block(self, block: Block, current: BlockId) -> BlockId | None:
        for statement in block.statements:
            current = self._build_statement(statement, current)
            if current is None:
                break

        if current is not None:
            scope = block.metadata["scope"]
            self.cfg.blocks[current].operations.append(ScopeExit(scope))

        return current


    def _build_statement(self, statement: Stmt, current: BlockId) -> BlockId | None:
        if isinstance(statement, Block):
            return self._build_block(statement, current)
        if isinstance(statement, IfStmt):
            return self._build_if(statement, current)
        if isinstance(statement, WhileStmt):
            return self._build_while(statement, current)
            
        
        
        if isinstance(statement, ReturnStmt):
            self.cfg.terminate(current, Return(statement))
            return None

        self.cfg.blocks[current].operations.append(statement)
            
        return current
    

    def _build_if(self, statement: IfStmt, current: BlockId) -> BlockId | None:
        then_entry = self.cfg.new_block()
        else_entry = self.cfg.new_block() if statement.else_block else None
        merge_block = self.cfg.new_block()
        
        false_target = else_entry.id if else_entry else merge_block.id
        self.cfg.terminate(current, Branch(statement.condition, then_entry.id, false_target))
        
        then_exit = self._build_block(statement.then_block, then_entry.id)
        if then_exit is not None:
            self.cfg.terminate(then_exit, Jump(merge_block.id))
            
        if statement.else_block and else_entry is not None:
            else_exit = self._build_block(statement.else_block, else_entry.id)
            if else_exit is not None:
                self.cfg.terminate(else_exit, Jump(merge_block.id))
                
        if then_exit is None and (not statement.else_block or else_exit is None):
            if not statement.else_block:
                return merge_block.id
            return None
            
        return merge_block.id
    

    def _build_while(self, statement: WhileStmt, current: BlockId) -> BlockId | None:
        header = self.cfg.new_block()
        body_entry = self.cfg.new_block()
        exit_block = self.cfg.new_block()
        
        self.cfg.terminate(current, Jump(header.id))
        condition = statement.condition
        if isinstance(condition, BoolLiteral) and condition.value is True:
            self.cfg.terminate(header.id, Jump(body_entry.id))
            infinite = True
        else:
            self.cfg.terminate(header.id, Branch(condition, body_entry.id, exit_block.id))
            infinite = False
 
        body_exit = self._build_block(statement.body, body_entry.id)
        if body_exit is not None:
            self.cfg.terminate(body_exit, Jump(header.id))
 
        return None if infinite else exit_block.id
