def get_kth_neighbors(parents, start_node, k):
    def build_adjacency_list(parents):
        from collections import defaultdict
        
        adjacency_list = defaultdict(list)
        for child, parent in enumerate(parents):
            if parent != -1:  # -1 表示根节点
                adjacency_list[parent].append(child)
                adjacency_list[child].append(parent)
        return adjacency_list

    def find_kth_neighbors(adjacency_list, start_node, k):
        from collections import deque
        
        queue = deque([(start_node, 0)])
        visited = set([start_node])
        
        while queue:
            current_node, current_distance = queue.popleft()
            
            if current_distance == k:
                return [node for node, distance in queue if distance == k] + [current_node]
            
            for neighbor in adjacency_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_distance + 1))
        
        return []
    
    adjacency_list = build_adjacency_list(parents)
    return find_kth_neighbors(adjacency_list, start_node, k)

def get_subgraph_single(sentence, dep, word, hop):
    adj_node = get_kth_neighbors(dep, word, hop)
    return ", ".join([sentence[i] for i in adj_node if sentence[i] not in [',', '.', '-', ]])

def get_subgraph_span(sentence, dep, span1, span2, hop):
    span = sentence[span1:span2+1]
    adj_nodes = []
    for i in range(span1, span2+1):
        adj_node = get_kth_neighbors(dep, i, hop)
        adj_nodes.extend(adj_node)
    adj_nodes = [item for item in adj_nodes if sentence[item] not in span]
    return ", ".join([sentence[i] for i in adj_nodes if sentence[i] not in [',', '.', '-', ]])

def get_subgraph(input, clean_sent, target_seq, dep_tree):
    all_subgraph_a, all_subgraph_o = [], []
    target_seq.sort(key=lambda x: (x[0][-1], x[3][-1]))
    for tri in target_seq:
        if tri[0][0] == 10000:
            sub_graph_a = ['nothing']
        elif len(tri[0]) == 1:
            begin = find_corresponding_index(input, clean_sent, tri[0][0], 1)
            sub_graph_a_1hop = get_subgraph_single(clean_sent, dep_tree, begin, 1)
            sub_graph_a_2hop = get_subgraph_single(clean_sent, dep_tree, begin, 2)
            sub_graph_a = [sub_graph_a_1hop, sub_graph_a_2hop]
        else:
            begin, end = find_corresponding_index(input, clean_sent, tri[0][0], 1), find_corresponding_index(input, clean_sent, tri[0][-1], 0)
            sub_graph_a_1hop = get_subgraph_span(clean_sent, dep_tree, begin, end, 1)
            sub_graph_a_2hop = get_subgraph_span(clean_sent, dep_tree, begin, end, 2)
            sub_graph_a = [sub_graph_a_1hop, sub_graph_a_2hop]
        if tri[3][0] == 10000:
            sub_graph_o = ['nothing']
        elif len(tri[3]) == 1:
            begin = find_corresponding_index(input, clean_sent, tri[3][0], 1)
            sub_graph_o_1hop = get_subgraph_single(clean_sent, dep_tree, begin, 1)
            sub_graph_o_2hop = get_subgraph_single(clean_sent, dep_tree, begin, 2)
            sub_graph_o = [sub_graph_o_1hop, sub_graph_o_2hop]
        else:
            begin, end = find_corresponding_index(input, clean_sent, tri[3][0], 1), find_corresponding_index(input, clean_sent, tri[3][-1], 0)
            sub_graph_o_1hop = get_subgraph_span(clean_sent, dep_tree, begin, end, 1)
            sub_graph_o_2hop = get_subgraph_span(clean_sent, dep_tree, begin, end, 2)
            sub_graph_o = [sub_graph_o_1hop, sub_graph_o_2hop]
        all_subgraph_a.append(sub_graph_a)
        all_subgraph_o.append(sub_graph_o)
    return {'all_subgraph_a': all_subgraph_a,
            'all_subgraph_o': all_subgraph_o}

def find_corresponding_index(l1, l2, a, begin):
    # 查找 l1[a] 删除标点符号后的版本在 l2 中的索引
    # print(l1, l2, a)
    target = l1[a]
    try:
        # 查找 l2 中与 l1[a] 的删除标点符号后的版本相匹配的索引
        return l2.index(target)
    except ValueError:
        if begin:
            return find_corresponding_index(l1, l2, a+1, begin)
        else:
            return find_corresponding_index(l1, l2, a-1, begin)