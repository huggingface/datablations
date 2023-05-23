"""
Script for searching through logs to look for failing nodes.
"""
import sys
import re


NODE_RANK_RE = re.compile(r'Launching on (\S+) \((\d+)/(\d+)\)')

ERROR_STRINGS = [
    'Segmentation fault',
    'Failed to initialize RSMI device mutex',
     
'ERROR:torch.distributed.elastic.multiprocessing.api:failed',
]

    
if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} STDERR-LOG STDOUT-LOG')
    sys.exit(1)


rank_to_node, node_count = {}, None
with open(sys.argv[2]) as f:
    for line in f:
        m = NODE_RANK_RE.search(line)
        if not m:
            continue
        node, rank, count = m.groups()
        rank, count = int(rank), int(count)
        if node_count is None:
            node_count = count
        else:
            assert node_count == count
        assert rank not in rank_to_node
        rank_to_node[rank] = node


with open(sys.argv[1]) as f:
    for line in f:
        if any(e in line for e in ERROR_STRINGS):
            line = line.rstrip('\n')
            try:
                rank = int(line.split(':')[0])
            except:
                print(f'failed to parse rank: {line}', file=sys.stderr)
                continue
            print(f'{rank_to_node[rank]}\t{line}')
