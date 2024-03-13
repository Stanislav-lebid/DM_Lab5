import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


def to_adjacency_list(graph):
    n = len(graph)
    adj_list = {}
    for node in range(n):
        adj_list[node] = []
        for neighbour in range(n):
            if graph[node][neighbour] == 1:
                adj_list[node].append(neighbour)
    return adj_list


def spectrum(matrix):
    # calc eigenvalues of matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # count values and sort spectrum
    spectrum = [[], []]
    for value in eigenvalues:
        value = round(value, 2)
        if not (value in spectrum[0]):
            spectrum[0].append(value)
            spectrum[0].sort()
            spectrum[1].insert(spectrum[0].index(value), 1)
        else:
            spectrum[1][spectrum[0].index(value)] += 1

    return spectrum


def spectrum_method(graph1, graph2):
    # calc spectrum
    spectrum_1 = spectrum(graph1)
    spectrum_2 = spectrum(graph2)

    spectrum_len = len(spectrum_1[0])

    # compare
    if spectrum_len != len(spectrum_2[0]):
        return False
    else:
        for i in range(spectrum_len):
            if spectrum_1[0][i] != spectrum_2[0][i]:
                return False
            if spectrum_1[1][i] != spectrum_2[1][i]:
                return False
    return True


def find_cycles(graph):
    """
    Знаходить унікальні цикли в графі, використовуючи пошук в глибину.
    """
    def dfs(node, visited, path):
        visited[node] = True
        path.append(node)

        for neighbor in range(len(graph)):
            if graph[node][neighbor] == 1:
                if neighbor not in path:  # Якщо сусід не вже входить у поточний шлях, рекурсивно відвідуємо його
                    dfs(neighbor, visited, path)
                else:
                    # Знайдено цикл, виводимо його
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    # Додавання унікального циклу до множини
                    cycles.add(len(cycle))

        # Після обробки всіх сусідів видаляємо поточний вузол зі шляху
        path.pop()

    cycles = set()
    num_nodes = len(graph)
    visited = [False] * num_nodes

    for start_node in range(num_nodes):
        dfs(start_node, visited, [])

    return cycles


def wl_coloring(adj_matrix):
    """
    Perform one iteration of the Weisfeiler-Leman graph coloring algorithm.
    """
    num_nodes = adj_matrix.shape[0]
    labels = np.zeros(num_nodes, dtype=int)
    new_labels = np.zeros(num_nodes, dtype=int)
    label_dict = {}

    # Initial coloring based on the node degrees
    for i in range(num_nodes):
        label = tuple(sorted([labels[j]
                      for j in range(num_nodes) if adj_matrix[i][j]]))

        if label not in label_dict:
            label_dict[label] = len(label_dict) + 1
        new_labels[i] = label_dict[label]

    return new_labels


def are_graphs_isomorphic(adj_matrix1, adj_matrix2, max_iterations=10):
    """
    Check if two graphs represented by adjacency matrices are isomorphic using Weisfeiler-Leman algorithm.
    """
    for _ in range(max_iterations):
        labels1 = wl_coloring(adj_matrix1)
        labels2 = wl_coloring(adj_matrix2)

        if np.array_equal(labels1, labels2):
            return True  # Graphs are isomorphic
        else:
            adj_matrix1 = np.dot(adj_matrix1, adj_matrix1.T)
            adj_matrix2 = np.dot(adj_matrix2, adj_matrix2.T)

    return False  # Reached maximum iterations without finding isomorphism


def isomorphic(graph1, graph2, iterations=1):
    """
    Check if two graphs represented by their adjacency matrices are isomorphic using the Weisfeiler-Lehman test.
    """
    isIsomorphicWithRefinement = are_graphs_isomorphic(
        np.array(graph1), np.array(graph2), iterations)

    c1 = find_cycles(graph1)
    c2 = find_cycles(graph2)

    for cycle in c1:
        if cycle not in c2:
            print("Граф 1 має цикл довжини %d, на відміну від графу 2" % cycle)
            return False

    for cycle in c2:
        if cycle not in c1:
            print("Граф 2 має цикл довжини %d, на відміну від графу 1" % cycle)
            return False

    isIsomorphicWithSpectralPartition = spectrum_method(graph1, graph2)

    return isIsomorphicWithRefinement and isIsomorphicWithSpectralPartition


def read_graph(file_name):
    f = open(file_name, "r")
    n = int(f.readline())

    graph = []
    for i in range(0, n):
        s = [int(j) for j in f.readline().strip().split(" ")]

        graph.append(s)

    return graph


def build_graph(graph):
    G = nx.Graph()
    for i in range(len(graph)):
        G.add_node(i + 1)

    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edge(i + 1, j + 1)

    return G


def draw_graph(graph):
    plt.figure(figsize=(6, 6))

    pos = nx.circular_layout(graph)

    # Assign different colors to nodes and edges
    node_colors = range(len(graph))
    edge_colors = range(len(graph.edges()))

    nx.draw(graph, pos=pos, with_labels=True, arrows=True,
            node_color=node_colors, edge_color=edge_colors, cmap='rainbow')

    ax = plt.gca()
    ax.collections[0].set_edgecolor("#fff")

    plt.show()


# read all file names from tests folder
test_cases = os.listdir("./tests")

for (i, case) in enumerate(test_cases):
    # Example usage:
    graph1 = read_graph("tests/%s/g1.txt" % case)
    graph2 = read_graph("tests/%s/g2.txt" % case)

    G1 = build_graph(graph1)
    G2 = build_graph(graph2)

    print("Перевірка на ізоморфізм бібліотечним методом: %s" %
          ("ізоморфні" if nx.is_isomorphic(G1, G2) else "не ізоморфні"))

    # Draw graphs
    draw_graph(G1)
    draw_graph(G2)

    if isomorphic(graph1, graph2):
        print("Графи (%s) є ізоморфними." % case)
    else:
        print("Графи (%s) не є ізоморфними." % case)

    if i == len(test_cases) - 1:
        break

    prompt = input("Press Enter to continue...")

    if prompt == "exit":
        break
