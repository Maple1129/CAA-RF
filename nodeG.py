from node_auto import Blockchain
lists_all = {'A': ('127.0.0.1', 1), 'B': ('127.0.0.1', 2), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4),
             'E': ('127.0.0.1', 5), 'F': ('127.0.0.1', 6), 'G': ('127.0.0.1', 7), 'H': ('127.0.0.1', 8),
             'I': ('127.0.0.1', 9), 'J': ('127.0.0.1', 10), 'K': ('127.0.0.1', 11), 'L': ('127.0.0.1', 12)}
lists_my = lists_all.copy()
ONESELF = 'G'

lists_my.pop(ONESELF)
MAX_THREADS = len(lists_my)
lists_all = {k: lists_all[k] for k in sorted(lists_all)}
list_keys = list(lists_my.keys())

blockchain = Blockchain(oneself=ONESELF, interval=5, all_node=lists_all)
blockchain.node()
blockchain.menu()

