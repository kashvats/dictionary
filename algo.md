# All 25 Algorithms

# Searching Algorithms

# 1. Linear Search
# Linear search is a simple search algorithm that checks each element in the list sequentially.
def linear_search(arr, target):
    """
    This function performs a linear search to find the index of 'target' in the 'arr'.
    Returns the index if found, otherwise returns -1.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


# 2. Binary Search
# Binary search works by repeatedly dividing the search interval in half.
def binary_search(arr, target):
    """
    This function performs a binary search to find the index of 'target' in the 'arr'.
    The input array must be sorted.
    Returns the index if found, otherwise returns -1.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# 3. Depth First Search (DFS)
# DFS is a graph traversal algorithm that explores as far as possible along each branch before backtracking.
def dfs(graph, start, visited=None):
    """
    This function performs a Depth First Search on a graph.
    The 'graph' is a dictionary where keys are nodes and values are lists of adjacent nodes.
    """
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited


# 4. Breadth First Search (BFS)
# BFS is a graph traversal algorithm that explores all the nodes at the present depth level before moving on to nodes at the next depth level.
from collections import deque

def bfs(graph, start):
    """
    This function performs a Breadth First Search on a graph.
    It uses a queue to explore nodes level by level.
    """
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


# Sorting Algorithms

# 1. Insertion Sort
# Insertion sort builds the final sorted array one item at a time by inserting elements into the correct position.
def insertion_sort(arr):
    """
    This function sorts an array using the insertion sort algorithm.
    It iterates through the list and places each element in its correct position.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# 2. Heap Sort
# Heap sort works by building a heap data structure and then repeatedly removing the largest element.
import heapq

def heap_sort(arr):
    """
    This function sorts an array using heap sort.
    It first converts the list into a heap and then pops the smallest element to form a sorted list.
    """
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]


# 3. Selection Sort
# Selection sort repeatedly selects the smallest (or largest) element from the unsorted part of the array and swaps it with the first unsorted element.
def selection_sort(arr):
    """
    This function sorts an array using the selection sort algorithm.
    It selects the smallest element and swaps it with the first unsorted element.
    """
    for i in range(len(arr)):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]


# 4. Merge Sort
# Merge sort is a divide-and-conquer algorithm that divides the array into halves, sorts them, and merges them.
def merge_sort(arr):
    """
    This function sorts an array using merge sort.
    It recursively splits the list in half and merges the sorted halves back together.
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


# 5. Quick Sort
# Quick sort is a divide-and-conquer algorithm that partitions the array into two sub-arrays and recursively sorts them.
def quick_sort(arr):
    """
    This function sorts an array using quick sort.
    It divides the array into two parts and recursively sorts them.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# 6. Counting Sort
# Counting sort works by counting the number of occurrences of each element and using this information to place each element in the correct position.
def counting_sort(arr):
    """
    This function sorts an array using counting sort.
    It counts occurrences of elements and places them in sorted order.
    """
    max_val = max(arr)
    min_val = min(arr)
    range_of_elements = max_val - min_val + 1

    count = [0] * range_of_elements
    output = [0] * len(arr)

    for i in range(len(arr)):
        count[arr[i] - min_val] += 1

    for i in range(1, range_of_elements):
        count[i] += count[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    return output


# Graph Algorithms

# 1. Kruskal's Algorithm
# Kruskal's algorithm finds the minimum spanning tree of a graph by adding the least weight edges.
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def kruskal(n, edges):
    """
    This function implements Kruskal's algorithm to find the minimum spanning tree.
    It uses a disjoint set (union-find) to manage connected components.
    """
    disjoint_set = DisjointSet(n)
    mst = []

    edges.sort(key=lambda x: x[2])  # Sort edges by weight

    for u, v, w in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, w))

    return mst


# 2. Dijkstra's Algorithm
# Dijkstra's algorithm finds the shortest path from a source node to all other nodes in a weighted graph.
import heapq

def dijkstra(graph, start):
    """
    This function implements Dijkstra's algorithm to find the shortest path from 'start' node.
    It uses a priority queue (min-heap) for efficient selection of the next node to visit.
    """
    pq = [(0, start)]  # (distance, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances


# 3. Bellman-Ford Algorithm
# The Bellman-Ford algorithm finds the shortest path in a graph with negative weight edges.
def bellman_ford(graph, start):
    """
    This function implements Bellman-Ford's algorithm to find the shortest path from 'start' node.
    It can handle negative edge weights.
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
    return distances


# 4. Floyd-Warshall Algorithm
# The Floyd-Warshall algorithm finds the shortest paths between all pairs of nodes in a graph.
def floyd_warshall(graph):
    """
    This function implements Floyd-Warshall's algorithm for finding the shortest paths between all pairs of nodes.
    """
    dist = {node: {other: float('inf') for other in graph} for node in graph}
    for node in graph:
        dist[node][node] = 0
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            dist[node][neighbor] = weight

    for k in graph:
        for i in graph:
            for j in graph:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


# 5. Topological Sort Algorithm
# Topological sort orders the vertices of a directed graph such that for every directed edge u -> v, vertex u comes before v.
def topological_sort(graph):
    """
    This function implements topological sort using Kahn's Algorithm (BFS-based approach).
    """
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = [node for node in graph if in_degree[node] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) == len(graph):
        return result
    else:
        return "Graph contains a cycle"


# 6. Flood Fill Algorithm
# Flood fill is used to find connected components in a graph or to fill areas in a grid.
def flood_fill(grid, start, target_value, new_value):
    """
    This function implements flood fill for a 2D grid, replacing all target_value with new_value starting from the 'start'.
    """
    rows, cols = len(grid), len(grid[0])
    def fill(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != target_value:
            return
        grid[r][c] = new_value
        fill(r + 1, c)
        fill(r - 1, c)
        fill(r, c + 1)
        fill(r, c - 1)
    fill(start[0], start[1])


# 7. Lee Algorithm (Shortest Path in Grid)
# Lee's algorithm is used for finding the shortest path in a grid with obstacles.
from collections import deque

def lee_algorithm(grid, start, end):
    """
    This function implements Lee's algorithm to find the shortest path in a grid.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, 0)])  # (current_position, distance)
    visited = set()
    visited.add(start)

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 4 directions

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == end:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return -1  # No path found


# Arrays Algorithms

# 1. Kadane's Algorithm
# Kadane's algorithm finds the maximum sum subarray in an array.
def kadane(arr):
    """
    This function implements Kadane's Algorithm to find the maximum sum subarray.
    """
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


# 2. Floyd's Cycle Detection Algorithm
# Floyd's cycle detection algorithm is used to find cycles in a linked list.
def floyds_cycle_detection(head):
    """
    This function implements Floyd's cycle detection algorithm (Tortoise and Hare) to detect cycles in a linked list.
    """
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


# 3. KMP Algorithm (Knuth-Morris-Pratt)
# KMP is an efficient string searching algorithm that searches for a substring in a string.
def kmp_search(text, pattern):
    """
    This function implements the KMP algorithm for pattern searching.
    """
    lps = [0] * len(pattern)
    j = 0

    # Preprocess pattern to create longest prefix suffix (LPS) array
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
            i += 1
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                lps[i] = 0
                i += 1

    # Search for the pattern in the text
    i = 0
    j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j  # Pattern found at index i-j
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


# 4. Quick Select Algorithm
# Quick Select is a selection algorithm to find the kth smallest element in an unordered list.
def quick_select(arr, k):
    """
    This function implements Quick Select to find the kth smallest element in an array.
    """
    if len(arr) == 1:
        return arr[0]
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    if k < len(left):
        return quick_select(left, k)
    elif k < len(left) + len([pivot]):
        return pivot
    else:
        return quick_select(right, k - len(left) - 1)


# 5. Boyer-Moore Majority Vote Algorithm
# Boyer-Moore is used to find the majority element in an array.
def boyer_moore_majority_vote(arr):
    """
    This function implements the Boyer-Moore Majority Vote algorithm to find the majority element in the array.
    """
    count = 0
    candidate = None

    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)

    return candidate if arr.count(candidate) > len(arr) // 2 else None


# Basics Algorithms

# 1. Huffman Coding Compression Algorithm
# Huffman coding is a lossless data compression algorithm.
import heapq

def huffman_coding(text):
    """
    This function implements Huffman Coding algorithm to compress text.
    """
    frequency = {}
    for char in text:
        frequency[char] = frequency.get(char, 0) + 1

    heap = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff = sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))
    return {item[0]: item[1] for item in huff}


# 2. Euclid's Algorithm
# Euclid's algorithm computes the greatest common divisor (GCD) of two numbers.
def euclid_gcd(a, b):
    """
    This function implements Euclid's algorithm to compute the greatest common divisor (GCD).
    """
    while b:
        a, b = b, a % b
    return a


# 3. Union Find Algorithm
# Union-Find is a data structure used to manage disjoint sets and supports union and find operations.
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


# Recap of the Algorithms
print("All 25 algorithms are implemented above, including common searching, sorting, graph, and other algorithmic techniques.")