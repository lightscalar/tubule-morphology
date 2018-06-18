import numpy as np
from mathtools.utils import Vessel
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import io, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import seaborn as sns
from ipdb import set_trace as debug
from tqdm import tqdm
from skimage.morphology import skeletonize, remove_small_objects, binary_erosion
from mahotas.morph import hitmiss as hit_or_miss
from mahotas.labeled import label
import mahotas as mh


class Struct:
    pass


def power_iteration(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[0])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def find_branch_points(skel):
    X = []
    # cross X
    X0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    X1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    # T like
    T = []
    # T0 contains X0
    T0 = np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])

    T1 = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])  # contains X1

    T2 = np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])

    T3 = np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])

    T4 = np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])

    T5 = np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])

    T6 = np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])

    T7 = np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    # Y like
    Y = []
    Y0 = np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])

    Y1 = np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]])

    Y2 = np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])

    Y2 = np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])

    Y3 = np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])

    Y4 = np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + hit_or_miss(skel, x)
    for y in Y:
        bp = bp + hit_or_miss(skel, y)
    for t in T:
        bp = bp + hit_or_miss(skel, t)

    return bp


def find_end_points(skel):
    endpoint1 = np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]])

    endpoint2 = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]])

    endpoint3 = np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]])

    endpoint4 = np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]])

    endpoint5 = np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]])

    endpoint6 = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]])

    endpoint7 = np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]])

    endpoint8 = np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])

    ep1 = hit_or_miss(skel, endpoint1)
    ep2 = hit_or_miss(skel, endpoint2)
    ep3 = hit_or_miss(skel, endpoint3)
    ep4 = hit_or_miss(skel, endpoint4)
    ep5 = hit_or_miss(skel, endpoint5)
    ep6 = hit_or_miss(skel, endpoint6)
    ep7 = hit_or_miss(skel, endpoint7)
    ep8 = hit_or_miss(skel, endpoint8)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    return ep


def borderless_image(image, cmap="hot", fignum=100, filename=None):
    """Make a nice borderless image."""
    plt.figure(fignum)
    plt.imshow(image, cmap=cmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.grid("off")
    plt.show()


def process_image(img):
    """Process the image using Hessian filters."""
    # img_orig = color.rgb2gray(img_orig)
    img = color.rgb2gray(img_orig)

    # Image is largest eigenvalue of Hessian.
    hxx, hxy, hyy = hessian_matrix(img, sigma=3, order="rc")
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    Z = i1
    border = 12
    Z = Z[border:-border, border:-border]

    # Binarize the image.
    thresh = threshold_otsu(Z)
    alpha = 1
    Z = np.abs(Z)
    img_ = img[border:-border, border:-border]

    thresh = threshold_otsu(Z)
    Z_ = Z > thresh
    return Z, Z_


def second_order_neighborhood(row, col):
    """Returns the eight-neighborhod associated with a given row & column."""
    r, c = row, col
    return [
        (r, c),
        (r, c + 1),
        (r, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
        (r + 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r - 1, c + 1),
    ]


def make_graph(img):
    """Creates a graph from raw video frame."""
    # Use hessian filters to find ridges; binarize the image.
    Z, Z_binary = process_image(img_orig)

    # Create a skeletonized version of binary image; remove noise.
    Z_skel = skeletonize(Z_binary)
    Z_skel = remove_small_objects(Z_skel, 25, connectivity=2)

    # Plot a nice picture of the skeleton.
    borderless_image(Z_skel)

    # Find all branch points and endpoints in the image.
    branch_pts = find_branch_points(Z_skel)
    end_pts = find_end_points(Z_skel)
    interesting_pts = branch_pts + 0 * end_pts

    # Create a kill points strawman image.
    kill_points = 1.0 * Z_skel

    # Identify and extract nodes.
    nodes = []
    r, c = np.where(interesting_pts > 0)
    for point in zip(r, c):
        # Cut out the neighborhood of the vertices and endpoints.
        r, c = point
        idxs = set(tuple(second_order_neighborhood(r, c)))
        for r, c in idxs:
            kill_points[r, c] = 0
        nodes.append(idxs)
    nodes = tuple(nodes)

    # Find and label all connected components.
    labeled, nb_edges = mh.label(kill_points, np.ones((3, 3), bool))
    labeled_ = labeled > 0

    r_, c_ = np.where(labeled > 0)
    label_map = {(r_, c_): labeled[r_, c_] for r_, c_ in zip(r_, c_)}

    # Find the endpoints in the edge image.
    edge_endpoints = find_end_points(labeled_)
    r, c = np.where(edge_endpoints > 0)

    # Now replace the endpoints with their local eight-neighborhoods.
    edge_endpoints = {}
    edge_labels = {}
    for ri, ci in zip(r, c):
        key = labeled[ri, ci]
        if key not in edge_endpoints.keys():
            # First endpoint!
            edge_endpoints[key] = []
            edge_labels[key] = labeled[ri, ci]
            neighbors = second_order_neighborhood(ri, ci)
            edge_endpoints[key] += neighbors
        else:
            # Second endpoint. And we're done.
            neighbors = second_order_neighborhood(ri, ci)
            edge_endpoints[key] += neighbors

    # Convert edge endpoints to set for efficient comparison.
    for key in edge_endpoints.keys():
        edge_endpoints[key] = set(edge_endpoints[key])

    # Attach edges to nodes; this is inefficient, could be better.
    unique_labels = np.unique(labeled)
    nb_labels = len(unique_labels)
    M = np.zeros((len(nodes), len(unique_labels)))
    for node_nb, node in tqdm(enumerate(nodes)):
        for edge_nb in range(1, nb_labels):
            if edge_nb in edge_endpoints.keys():
                edge = edge_endpoints[edge_nb]
                if len(node.intersection(edge)) > 0:
                    M[node_nb, edge_nb] += 1

    # Now construct the adjacency matrix.
    nb_nodes = len(nodes)
    A = np.zeros((nb_nodes, nb_nodes))
    for node_i in tqdm(range(nb_nodes)):
        edges_i, = np.where(M[node_i, :] > 0)
        edges_i = set(edges_i)
        for node_j in range(node_i + 1, nb_nodes):
            edges_j, = np.where(M[node_j, :] > 0)
            edges_j = set(edges_j)
            common_edges = edges_i.intersection(edges_j)
            if len(common_edges) > 0:
                # Add the edge label to the adjacency matrix:
                A[node_i, node_j] = list(common_edges)[0]
                A[node_j, node_i] = list(common_edges)[0]

    # Compute degree centrality of the graph.
    A_ = A > 0
    centrality = A_.sum(1)
    pdf = centrality / centrality.sum()

    # Compute the entropy of the graph.
    entropy = -np.sum(pdf[pdf > 0] * np.log(pdf[pdf > 0]))

    data = Struct()
    data.Z = Z
    data.skeleton = Z_skel
    data.branch_pts = branch_pts
    data.end_pts = end_pts
    data.A = A
    data.A_ = A
    data.pdf = pdf
    data.entropy = entropy
    data.edge_labels = edge_labels
    data.labeled = labeled
    return data


if __name__ == "__main__":
    plt.close("all")
    frame_number = 583
    img_orig = io.imread("images/img.{:04d}.png".format(frame_number))

    # Use hessian filters to find ridges; binarize the image.
    Z, Z_binary = process_image(img_orig)

    # Create a skeletonized version of binary image; remove noise.
    Z_skel = skeletonize(Z_binary)
    Z_skel = remove_small_objects(Z_skel, 25, connectivity=2)

    # Plot a nice picture of the skeleton.
    borderless_image(Z_skel)

    # Find all branch points and endpoints in the image.
    branch_pts = find_branch_points(Z_skel)
    end_pts = find_end_points(Z_skel)
    interesting_pts = branch_pts + 0 * end_pts

    # Plot these "interesting points" on the image.
    r, c = np.where(interesting_pts > 0)
    plt.plot(c, r, "r.")

    # Create a kill points strawman image.
    kill_points = 1.0 * Z_skel

    nodes = []
    for point in zip(r, c):
        # Cut out the neighborhood of the vertices and endpoints.
        r, c = point
        idxs = set(tuple(second_order_neighborhood(r, c)))
        for r, c in idxs:
            kill_points[r, c] = 0
        nodes.append(idxs)
    nodes = tuple(nodes)

    # Find and label all connected components.
    labeled, nb_edges = mh.label(kill_points, np.ones((3, 3), bool))
    labeled_ = labeled > 0

    r_, c_ = np.where(labeled > 0)
    label_map = {(r_, c_): labeled[r_, c_] for r_, c_ in zip(r_, c_)}

    # Plot the connected components.
    # plt.figure(200)
    # plt.imshow(labeled_)
    # plt.show()

    # Find the endpoints of the edge_image.
    edge_endpoints = find_end_points(labeled_)
    r, c = np.where(edge_endpoints > 0)
    # plt.plot(c,r, 'r.')

    # Now replace the endpoints with their local eight-neighborhoods.
    edge_endpoints = {}
    edge_labels = {}
    for ri, ci in zip(r, c):
        key = labeled[ri, ci]
        if key not in edge_endpoints.keys():
            # First endpoint!
            edge_endpoints[key] = []
            edge_labels[key] = labeled[ri, ci]
            neighbors = second_order_neighborhood(ri, ci)
            edge_endpoints[key] += neighbors
        else:
            # Second endpoint. And we're done.
            neighbors = second_order_neighborhood(ri, ci)
            edge_endpoints[key] += neighbors

    # Convert edge endpoints to set for efficient comparison.
    for key in edge_endpoints.keys():
        edge_endpoints[key] = set(edge_endpoints[key])

    # Attach edges to nodes; this is inefficient, could be better.
    unique_labels = np.unique(labeled)
    nb_labels = len(unique_labels)
    M = np.zeros((len(nodes), len(unique_labels)))
    for node_nb, node in tqdm(enumerate(nodes)):
        for edge_nb in range(1, nb_labels):
            if edge_nb in edge_endpoints.keys():
                edge = edge_endpoints[edge_nb]
                if len(node.intersection(edge)) > 0:
                    M[node_nb, edge_nb] += 1

    # Now construct the adjacency matrix.
    nb_nodes = len(nodes)
    A = np.zeros((nb_nodes, nb_nodes))
    for node_i in tqdm(range(nb_nodes)):
        edges_i, = np.where(M[node_i, :] > 0)
        edges_i = set(edges_i)
        for node_j in range(node_i + 1, nb_nodes):
            edges_j, = np.where(M[node_j, :] > 0)
            edges_j = set(edges_j)
            common_edges = edges_i.intersection(edges_j)
            if len(common_edges) > 0:
                # Add the edge label to the adjacency matrix:
                A[node_i, node_j] = list(common_edges)[0]
                A[node_j, node_i] = list(common_edges)[0]

    # Measure the eigenvector centrality of graph vertices.
    A_ = A > 0
    # centrality = power_iteration(A_, 5000)
    centrality = A_.sum(1)
    # w, v = np.linalg.eigh(A)
    # centrality = v[:,-1]
    pdf = centrality / centrality.sum()

    # Compute the entropy of the graph.
    entropy = -np.sum(pdf[pdf > 0] * np.log(pdf[pdf > 0]))
    print("Graph Entropy is {}".format(entropy))

    # plt.figure(300)
    # plt.bar(range(len(centrality)), centrality)
    # plt.show()

    img_bw = color.rgb2gray(img_orig)
    border = 12
    img_bw = img_bw[border:-border, border:-border]
    borderless_image(img_bw, fignum=400)

    most_connected = np.argsort(pdf)[-1:0:-1]
    # plt.figure(100)
    # for k in range(100):
    #     w = most_connected[k]
    #     node = list(nodes[w])
    #     r,c = node[3]
    #     plt.plot(c, r, 'bo')
