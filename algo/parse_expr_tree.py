''' Apply distributed rule

     *                      +
  +      +      =>   *    *    *    *
a   b  c   d        a c  a d  b c  b d
'''

class ExprTreeNode(object):
    def __init__(self, ch=None):
        self.op = ch
        self.children = []

    def __str__(self):
        if self.op not in ['+', '*']:
            return self.op

        sep = ' {} '.format(self.op)
        sub_strs = map(lambda r: ('(' + str(r) + ')') if '+' == r.op else str(r),
                       self.children)
        return sep.join(list(sub_strs))
        
    @classmethod
    def constr(cls, s):
        ''' Parse a given string into a tree expression
        '''
        if not s: return None
        
        curr = ExprTreeNode()
        nodes = []
        last_op = None
        mult_oprands = []
        n = len(s)
        i = 0
        while i < n:
            ch = s[i]

            if '(' == ch:
                j = i + 1
                num_paren = 1
                while j < n:
                    if s[j] == '(':
                        num_paren += 1
                    elif s[j] == ')':
                        num_paren -= 1
                    if 0 == num_paren:
                        sub_node = cls.constr(s[(i+1):j])
                        if last_op == '*':
                            mult_oprands.append(sub_node)
                        else:
                            nodes.append(sub_node)
                        break
                    j += 1

                i = j + 1
                continue

            i += 1

            if ch == ' ':
                continue

            # Parse non parentheses chars
            if ch == '+':
                if last_op == '*':
                    mult_node = ExprTreeNode('*')
                    mult_node.children = mult_oprands
                    mult_oprands = []
                    nodes.append(mult_node)
                last_op = '+'
                continue

            if ch == '*':
                if last_op != '*':
                    assert(nodes)
                    mult_oprands = [nodes[-1]]
                    nodes = nodes[:-1]
                last_op = '*'
                continue

            if last_op == '*':
                mult_oprands.append(ExprTreeNode(ch))
                continue

            nodes.append(ExprTreeNode(ch))

        if last_op == '*':
            mult_node = ExprTreeNode('*')
            mult_node.children = mult_oprands
            if not nodes:
                return mult_node

            nodes.append(mult_node)
            last_op = '+'

        root = ExprTreeNode(last_op)
        root.children = nodes
        return root


def simplify(root):
    if not root: return root
    if not root.children: return root
    
    if root.op != '*' and root.op != '+':
        return root

    children = list(map(simplify, root.children))
    if '+' == root.op:
        root = ExprTreeNode('+')        
        for r in children:
            if '+' == r.op:
                root.children += r.children
            else:
                root.children.append(r)
        return root

    # Unwind the nodes into lists of nodes
    # Distribute the operations if possible
    def prod(node_grps):
        if not node_grps: return '', []

        node, node_grps = node_grps[0], node_grps[1:]
        next_op, next_nodes = prod(node_grps)

        curr_nodes = []
        if node.op == '*':
            if next_op == '*':
                # If both ops are "*", we simply combine the nodes
                return '*', node.children + next_nodes
            # If only the left node is "*", 
            curr_nodes = [node]
        elif node.op != '+':
            # For "const" or "*" nodes, use a single expression
            curr_nodes = [node]
        else:
            # This is a "+" node
            curr_nodes = node.children

        if not next_nodes:
            return node.op, curr_nodes

        nodes = []
        for n1 in curr_nodes:
            for n2 in next_nodes:
                r = ExprTreeNode('*')
                r.children = [n1, n2]
                nodes.append(r)                

        return '+', nodes

    _op, _nodes = prod(children)
    root = ExprTreeNode(_op)
    root.children = _nodes
    return root


def TEST(str_expr):
    print('-----------------------------')
    root = ExprTreeNode.constr(str_expr)
    print('  original:', str(root))
    root_flatten = simplify(root)
    print('simplified:', str(root_flatten))


TEST('(a + b) + c * (d + e) + (0 + 1) * 3 * 7')
