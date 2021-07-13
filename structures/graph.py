from collections import defaultdict, deque
from DSA.structures.stacks import Stack
from DSA.structures.queues import PriorityQueue


# class Graph:
#
#     def __init__(self, vertex):
#         self.v = vertex
#         self.graph = defaultdict(set)  # Changed to set factory
#
#     def add_edge(self, s, d):
#         self.graph[s].add(d)
#
#     def dfs(self, d, visited_vertex):
#         visited_vertex[d] = True
#         print(d, end=' ')
#         for i in self.graph[d]:
#             if not visited_vertex[i]:
#                 self.dfs(i, visited_vertex)
#
#     def fill_order(self, d, visited_vertex, stack):
#         visited_vertex[d] = True
#         for i in self.graph[d]:
#             if not visited_vertex[i]:
#                 self.fill_order(i, visited_vertex, stack)
#         stack.push(d)
#
#     def transpose(self):
#         g = Graph(self.v)
#
#         for i in self.graph:
#             for j in self.graph[i]:
#                 g.add_edge(j, i)
#         return g
#
#     def print_scc(self):
#         """ Kosaraju's Algorithm.
#         O(V+E)
#         """
#         stack = Stack()
#         visited_vertex = [False for _ in range(self.v)]
#
#         for i in range(self.v):
#             if not visited_vertex[i]:
#                 self.fill_order(i, visited_vertex, stack)
#
#         gr = self.transpose()
#
#         visited_vertex = [False for _ in range(self.v)]
#
#         while stack:
#             i = stack.pop()
#             if not visited_vertex[i]:
#                 gr.dfs(i, visited_vertex)
#                 print('')


class GraphMatrix:

    def __init__(self, size, vertices_dict, directed=False):
        self.vertices = vertices_dict
        self.directed = directed
        self.adjacents = {number: [] for number in range(size)}
        self.matrix = [[0 for _ in range(size)] for _ in range(size)]
        self.size = size

    def add_edge(self, start, finish, cost=1):
        if start == finish:
            print(f'Same vertex {start} and {finish}')
        else:
            self.matrix[start][finish] = cost
            self.adjacents[start].append(finish)
            if not self.directed:
                self.matrix[finish][start] = cost
                self.adjacents[finish].append(start)

    def remove_edge(self, v1, v2):
        if self.matrix[v1][v2] == 0:
            print(f'No edge between {v1} and {v2}')
            return
        self.matrix[v1][v2], self.matrix[v2][v1] = 0, 0

    def __len__(self):
        return self.size

    def __str__(self):
        f = f'\t'
        for i in range(self.size):
            f += f'{self.vertices[i]}\t'
        f += '\n'
        for i in range(self.size):
            f += f'{self.vertices[i]}\t'
            for val in self.matrix[i]:
                f += f'{val}\t'
            f += '\n'
        return f

    @classmethod
    def _transpose(cls, size, adjacents, matrix, directed, vertices):
        g = cls(size=size, vertices_dict=vertices, directed=directed)
        for vertex in range(size):
            for neighbour in adjacents[vertex]:
                g.add_edge(neighbour, vertex, cost=matrix[vertex][neighbour])
        return g

    def dfs(self, d, visited_vertex, res=None):
        res = res or []
        visited_vertex[d] = True
        res.append(f'{self.vertices[d]} ({d}) ')
        # print(f'{self.vertices[d]} ({d}) - ', end=' ')
        for i in self.adjacents[d]:
            if not visited_vertex[i]:
                self.dfs(i, visited_vertex, res)
        return res

    def _fill_order(self, d, visited_vertex, stack):
        visited_vertex[d] = True
        for i in self.adjacents[d]:
            if not visited_vertex[i]:
                self._fill_order(i, visited_vertex, stack)
        stack.push(d)

    def kosaraju(self):
        print('Strongly connected components are:')
        stack = Stack()
        visited = [False for _ in range(self.size)]

        for i in range(self.size):
            if not visited[i]:
                self._fill_order(i, visited_vertex=visited, stack=stack)

        gr = GraphMatrix._transpose(size=self.size, adjacents=self.adjacents, matrix=self.matrix,
                                    directed=self.directed, vertices=self.vertices)
        visited = [False for _ in range(self.size)]

        while stack:
            i = stack.pop()
            if not visited[i]:
                result = gr.dfs(i, visited_vertex=visited)
                result.reverse()
                print(' - '.join(result))

    def floyd_warshall(self):
        distance = [[i if i else float('inf') for i in row] for row in self.matrix]
        for k in range(self.size):
            for i in range(self.size):
                for j in range(self.size):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

        _a = '\t'.join((self.vertices[i] for i in range(self.size)))
        print(f"\t{_a}")

        for i, row in enumerate(distance):
            print(self.vertices[i], *row, sep='\t')

    def prim(self):
        selected = [0 for _ in range(self.size)]
        edges_number = 0
        # the number of egde in minimum spanning tree will be
        # always less than(V - 1), where V is number of vertices in
        # graph
        # choose 0th vertex and make it true
        selected[0] = True
        print("Edge : Weight\n")
        while edges_number < self.size - 1:
            # For every vertex in the set S, find the all adjacent vertices
            # , calculate the distance from the vertex selected at step 1.
            # if the vertex is already in the set S, discard it otherwise
            # choose another vertex nearest to selected vertex  at step 1.
            minimum = float('inf')
            x = 0
            y = 0
            for i in range(self.size):
                if selected[i]:
                    for j in range(self.size):
                        if not selected[j] and self.matrix[i][j]:
                            # not in selected and there is an edge
                            if minimum > self.matrix[i][j]:
                                minimum = self.matrix[i][j]
                                x = i
                                y = j
            print(f'{self.vertices[x]} - {self.vertices[y]} : {self.matrix[x][y]}')
            selected[y] = True
            edges_number += 1

    def _find(self, parent, i):
        return i if parent[i] == i else self._find(parent, parent[i])

    def _apply_union(self, parent, rank, x, y):
        xroot = self._find(parent, x)
        yroot = self._find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        result = []

        structure = []
        for i in range(self.size):
            for j in self.adjacents[i]:
                structure.append((i, j, self.matrix[i][j]))
        graph = sorted(structure, key=lambda item: item[2])
        # print(graph)
        parent = []
        rank = []
        for node in range(self.size):
            parent.append(node)
            rank.append(0)
        i, e = 0, 0
        while e < self.size - 1:
            u, v, w = graph[i]
            i += 1
            x = self._find(parent, u)
            y = self._find(parent, v)
            if x != y:
                e += 1
                result.append((u, v, w))
                self._apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print(f'{u} ({self.vertices[u]}) - {v} ({self.vertices[v]}): {weight}')

    def bellman_ford(self, start):
        # Step 1: fill the distance array and predecessor array
        distances = [float("inf") for _ in range(self.size)]
        # Mark the source vertex
        distances[start] = 0
        backtrack = {start: None}

        # Step 2: relax edges |V| - 1 times
        for _ in range(self.size - 1):
            for vertex in range(self.size):
                for neighbor in self.adjacents[vertex]:
                    weight = self.matrix[vertex][neighbor]
                    if distances[vertex] != float('-inf') and distances[vertex] + weight < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight
                        backtrack[neighbor] = vertex

        # Step 3: detect negative cycle
        # if value changes then we have a negative cycle in the graph
        # and we cannot find the shortest distances
        for vertex in range(self.size):
            for neighbor in self.adjacents[vertex]:
                weight = self.matrix[vertex][neighbor]
                if distances[vertex] != float('-inf') and distances[vertex] + weight < distances[neighbor]:
                    print(f'Graph has negative weight cycle at vertices {self.vertices[vertex]} '
                          f'and {self.vertices[neighbor]}')
                    return
        # No negative weight cycle found!
        # Print the distance and predecessor array
        print("Vertex Distance from Source")
        # print(list(backtrack.items()))
        for i in range(self.size):
            print(f'{self.vertices[i]}, ({i}):\t\t{distances[i]}', end='\t|\t')
            while i:
                print(f'{i} <-', end='')
                i = backtrack[i]
            print(i)


class Vertex:
    def __init__(self, name):
        self.name = name
        self.connections = {}
        self.distance = float('inf')
        self.predecessor = None
        self.color = 'white'
        self.discoveryTime = 0
        self.finishTime = 0

    def __repr__(self):
        s = f'Vertex {self.name}: disc {self.discoveryTime}; fin {self.finishTime}; ' \
            f'dist {self.distance}; pred\t[{self.predecessor}]'
        return s

    def __str__(self):
        s = f'Vertex {self.name}'
        return s


class GraphList:
    def __init__(self, directed=False):
        self.vertices = {}
        self.time = 0
        self.directed = directed

    def __iter__(self):
        return iter(self.vertices.items())

    def clean_color_predecessor_distance(self):
        for i in self.vertices:
            self[i].color = 'white'
            self[i].predecessor = None
            self[i].distance = float('inf')

    def __getitem__(self, item):
        return self.vertices.get(item, None)

    def __contains__(self, item):
        return item in self.vertices

    def __str__(self):
        f = ''
        for vertex, value in self.vertices.items():
            f += f'Vertex {vertex} (predecessor {value.predecessor}):\t-->'
            for connected in value.connections:
                f += f'\t{connected} (cost {value.connections[connected]})\t'
                f += '|'
            f += f'\n'
        return f

    # def addVertex(self, key):
    #     newVertex = Vertex(key)
    #     self.vertices[key] = newVertex
    #     return newVertex

    def addEdge(self, start, finish, cost=1):
        if start not in self.vertices:
            startVertex = Vertex(start)
            self.vertices[start] = startVertex
        if finish not in self.vertices:
            finishVertex = Vertex(finish)
            self.vertices[finish] = finishVertex
        self.vertices[start].connections[finish] = cost
        if not self.directed:
            self.vertices[finish].connections[start] = cost

    def dfs(self):
        self.clean_color_predecessor_distance()
        for aVertex in self:
            if self[aVertex].color == 'white':
                self.dfsvisit(aVertex)

    def dfsvisit(self, startVertex_key):
        self[startVertex_key].color = 'grey'
        self.time += 1
        self[startVertex_key].discoveryTime = self.time
        for nextVertex in self[startVertex_key].connections:
            if self[nextVertex].color == 'white':
                self[nextVertex].predecessor = startVertex_key
                self.dfsvisit(nextVertex)
        self[startVertex_key].color = 'black'
        self.time += 1
        self[startVertex_key].finish = self.time

    def bfs(self, start):
        """
                O(V)
                """
        self.clean_color_predecessor_distance()
        self[start].distance = 0
        self[start].predecessor = None
        queue = deque([start])
        while queue:
            currentVertex_key = queue.popleft()
            for neighbor_key in self[currentVertex_key].connections:
                neighbor = self[neighbor_key]
                if neighbor.color == 'white':
                    neighbor.color = 'gray'
                    neighbor.distance = self[currentVertex_key].distance + 1
                    neighbor.predecessor = currentVertex_key
                    queue.append(neighbor_key)
            self[currentVertex_key].color = 'black'

    def traverse(self, toVertex):
        x = self[toVertex]
        while x.predecessor:
            print(x.name)
            x = self[x.predecessor]
        print(x.name)

    def dijkstra(self, start):
        """
                Finally, let us look at the running time of Dijkstra’s algorithm.
                We first note that building the priority queue takes O(V) time since we initially
                add every vertex in the graph to the priority queue. Once the queue is constructed the
                while loop is executed once for every vertex since vertices are all added at the
                beginning and only removed after that. Within that loop each call to delMin, takes
                O(logV) time. Taken together that part of the loop and the calls to delMin take O(Vlog(V)).
                The for loop is executed once for each edge in the graph, and within the for loop the call
                to decreaseKey takes time O(Elog(V)). So the combined running time is O((V+E)log(V)).
                """
        pq = PriorityQueue()
        self.clean_color_predecessor_distance()
        self[start].distance = 0
        pq.buildHeap([(vertex.distance, vertex_key) for vertex_key, vertex in self])
        while not pq.isEmpty():
            current_vertex = self[pq.delMin()]
            for nex_vert_key in current_vertex.connections:
                new_distance = current_vertex.distance + current_vertex.connections[nex_vert_key]
                if new_distance < self[nex_vert_key].distance:
                    self[nex_vert_key].distance = new_distance
                    self[nex_vert_key].predecessor = current_vertex.name
                    pq.decreaseKey(nex_vert_key, new_distance)


def build_graphMatrix(number_of_vertices, vertex_list, vertices_dict, directed=False):
    """
    Vertex list - (start, finish, weight)
    """
    g = GraphMatrix(number_of_vertices, vertices_dict=vertices_dict, directed=directed)
    for start, finish, weight in vertex_list:
        g.add_edge(start, finish, weight)
    return g


def build_graphList(dic, costs=None, directed=False):
    g = GraphList(directed=directed)

    if costs:
        for vertex in dic:
            neighbours = dic[vertex]
            while neighbours:
                n = neighbours.pop()
                g.addEdge(vertex, n, cost=costs[vertex][n])  # ?????
    else:
        for vertex in dic:
            neighbours = dic[vertex]
            while neighbours:
                n = neighbours.pop()
                g.addEdge(vertex, n)
    return g


if __name__ == '__main__':
    j = {
        'A': {'B', 'C'},
        'B': {'A', 'D', 'E'},
        'C': {'A', 'F'},
        'D': {'B'},
        'E': {'B', 'F'},
        'F': {'C', 'E'}
    }
    weights = {
        'A': {'B': 2, 'C': 1},
        'B': {'A': 2, 'D': 1, 'E': 3},
        'C': {'A': 1, 'F': 1},
        'D': {'B': 1},
        'E': {'B': 3, 'F': 1},
        'F': {'C': 1, 'E': 1}
    }

    graph = build_graphList(j, costs=weights)
    print(graph)
    graph.dfsvisit('A')
    print(graph)
    graph.bfs('A')
    print(graph)
    graph.traverse('C')

    print(graph['A'] == graph[graph['C'].predecessor])

    graph.dijkstra('A')

    for v, vertex in graph:
        print(v, vertex.distance, end=' ')
        while vertex.predecessor:
            print(vertex, end=' - ')
            vertex = graph[vertex.predecessor]
        print(vertex)

    h = build_graphMatrix(4, ((0, 1, 5), (0, 2, 4), (1, 3, 3), (2, 1, 6), (3, 2, 2)),
                          vertices_dict={0: 'A', 1: 'B', 2: 'C', 3: 'D'}, directed=True)
    print(h)
    print(h.vertices)
    print(h.adjacents)
    h.bellman_ford(0)
    h.kruskal()
    h.floyd_warshall()
    h.kosaraju()

    h = {n: l for l, n in zip('ABCDEFGH', range(8))}
    g = GraphMatrix(8, vertices_dict=h, directed=True)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 0)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 4)
    g.add_edge(6, 7)
    print(g)
    g.kosaraju()
