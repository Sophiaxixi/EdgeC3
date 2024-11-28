import numpy as np
import pickle, struct, socket, math

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def get_even_odd_from_one_hot_label(label): # 偶数为1，奇数为-1
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return i


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot # 返回 array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


def  get_indices_each_node_case(n_nodes, maxCase, label_list):
    indices_each_node_case = [] # 二维列表，对应4种case x 5个节点

    for i in range(0, maxCase): # 有4种case,把data分配给nodes的不同的分配方式
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1

    for i in range(0, len(label_list)): # 48000个数据分给5个node
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i) # 随意的 各个node是uniform

        # case 2 # 同一个node有相同的label
        tmp_target_node = int((label_list[i] - min_label) % n_nodes) # 2 0
        if n_nodes > num_labels: # 5 < 10
            tmp_min_index = 0
            tmp_min_val = math.inf # 下界
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)
        # #-------- 先把n_nodes改成5 HAR ---------
        # tmp_target_node = int((label_list[i] - min_label) % 5) # 2 0
        # if 5 > num_labels: # 5 < 10
        #     tmp_min_index = 0
        #     tmp_min_val = math.inf # 下界
        #     for n in range(0, n_nodes):
        #         if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
        #             tmp_min_val = len(indices_each_node_case[1][n])
        #             tmp_min_index = n
        #     tmp_target_node = tmp_min_index
        # indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        for n in range(0, n_nodes): # 每个节点都有全部的数据集 full information
            indices_each_node_case[2][n].append(i)

        # case 4 # 一半case1 一半case2
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2)) # 向上取整
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp # case1
        elif n_nodes > 1: # case2 相同label 且是大的一半的label
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[3][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[3][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[3][tmp_target_node].append(i)

        # case 5 # 5个node各自有6种labels
        # num = int(label_list[i])
        # if num == 0:
        #     indices_each_node_case[4][0].append(i)
        # elif num == 1:
        #     for j in range(0,2):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 2:
        #     for j in range(0,3):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 3:
        #     for j in range(0,4):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 4:
        #     for j in range(0,5):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 5:
        #     for j in range(0, 5):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 6:
        #     for j in range(1, 5):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 7:
        #     for j in range(2, 5):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 8:
        #     for j in range(3, 5):
        #         indices_each_node_case[4][j].append(i)
        # elif num == 9:
        #     for j in range(4, 5):
        #         indices_each_node_case[4][j].append(i)

        # case 6 # 5个node各自有6种labels


    return indices_each_node_case