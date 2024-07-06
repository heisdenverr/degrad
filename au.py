#! C:\Users\HP\Documents\demo\.venv\Scripts\python.exe

class Node:
    
    def __init__(self, data, parents=[], _op=""):
        self.data = data
        self.parents = parents
        self._op = ""
        self.grad = 0
        self.grad_fn = None
        

    def __repr__(self) -> str:
        return f"Node object: ({self.data},)"
    
    def _backward(self, grad=1):
        self.grad = grad
        
        if self.grad_fn:
            self.grad_fn()
            
    def backward(loss_node):
        loss_node._backward(grad=1)

    def zero_gradients(node):
        node.grad = 0
        for parent in node.parents: 
            zero_gradients(parent) # type: ignore
        
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = self.data + other.data
        new_node = Node(data=out, parents=[self, other], _op="+")
        
        
    
        def add_grad_fn(node=new_node):
            self, other = node.parents
            
            self.grad += node.grad
            other.grad += node.grad

        new_node.grad_fn = lambda: add_grad_fn(new_node)

        return new_node
    
    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = self.data - other.data
        new_node = Node(data=out, parents=[self, other], _op="-")

        def sub_grad_fn(node=new_node):
            self, other = node.parents
            self.grad += node.grad
            other.grad -= node.grad


        new_node.grad_fn = lambda: sub_grad_fn(new_node)

        return new_node


    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = self.data * other.data
        new_node = Node(data=out, parents=[self, other], _op="*")
        
    
        def mul_grad_fn(node=new_node):
            self, other = node.parents
            self.grad = node.grad * other.data
            other.grad = node.grad * self.data

        new_node.grad_fn = lambda: mul_grad_fn(new_node)

        return new_node
    
    def __div__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = self.data * (other.data**-1)
        new_node = Node(data=out, parents=[self, other], _op="/")

        def div_grad_fn(node=new_node):
            self, other = node.parents
            self.grad += node.grad / other.data
            other.grad -= (node.grad * self.data) / (other.data ** 2)

        new_node.grad_fn = lambda: div_grad_fn(new_node)

        return new_node
    
    def __call__(self):
        return Node()
