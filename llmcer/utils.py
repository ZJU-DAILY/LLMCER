
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                self.size[rootX] += self.size[rootY]
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
                self.size[rootY] += self.size[rootX]
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
                self.size[rootX] += self.size[rootY]

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def get_size(self, x):
        rootX = self.find(x)
        return self.size[rootX]

import random

def pick_elements(list1, list2, n=2):
    combined_length = len(list1) + len(list2)
    
    if combined_length <= n:
        return list1 + list2
    
    # Ensure at least one element from each list
    if len(list1) == 0 or len(list2) == 0:
        raise ValueError("Both lists must contain at least one element")
    result = [random.choice(list1), random.choice(list2)]
    
    remaining_slots = n - len(result)
    combined_list = list1 + list2
    combined_list.remove(result[0])
    combined_list.remove(result[1])
    
    # Randomly choose the remaining elements
    result += random.sample(combined_list, remaining_slots)
    
    return result
