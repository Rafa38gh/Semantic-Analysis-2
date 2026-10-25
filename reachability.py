"""Alcance no CFG. O percurso da AST já foi realizado pelo builder."""

from __future__ import annotations

from cfg import BlockId, CFG


def compute_reachable(cfg: CFG) -> set[BlockId]:
    reachable: set[BlockId] = set()
    pending = [cfg.entry]
    
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
            
        reachable.add(current)
        
        block = cfg.blocks[current]
        for successor in block.successors:
            if successor not in reachable:
                pending.append(successor)
                
    return reachable
