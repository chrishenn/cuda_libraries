# Edge Prune
### edge_prune
###

For each edge in 'edges' [rowid0, rowid1], search through edges to drop 'drop_edges' [rowid0, rowid1]. If the edge in edges appears in drop_edges, indicate it should be dropped with a mask value 0 (otherwise 1). 