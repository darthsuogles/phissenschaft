''' Get rebuild order of a set of docker images with parents
'''

def serialize_build_order(relations):
    if not relations: return []
    from collections import defaultdict

    ''' Auxiliary data '''
    nodes = set()
    build_targets = set()
    in_degree = defaultdict(int)
    children = defaultdict(list)

    # Add nodes and edges
    for img, parent in relations:
        build_targets.add(img)
        nodes.add(img)
        nodes.add(parent)
        in_degree[img] += 1
        children[parent].append(img)

    # Remove in_degree and children count for external images
    extern_nodes = nodes - build_targets
    for img in extern_nodes:
        for dep in children[img]:
            in_degree[dep] -= 1        
        del children[img]

    del extern_nodes
    del nodes

    queue = [img for img in build_targets if 0 == in_degree[img]]
    build_order = []
    while queue:
        curr = queue.pop(0)
        build_order.append(curr)
        for dep in children[curr]:
            in_deg = in_degree[dep] - 1
            if 0 == in_deg:
                queue.append(dep)
            in_degree[dep] = in_deg

    return build_order


relations = [
    ('img_a', 'img_root'),
    ('img_a', 'img_extern_dep1'),
    ('img_a', 'img_extern_dep2'),
    ('img_c', 'img_a'),
    ('img_b', 'img_a'),
    ('img_b', 'img_extern_dep2'),
    ('img_c', 'img_a'),
    ('img_c', 'img_b'),
    ('img_b', 'img_c')
]

res = serialize_build_order(relations)
