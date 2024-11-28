import numpy as np
from numpy import linalg
import socket

from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauClient, ControlAlgAdaptiveTauServer
from util.utils import send_msg, recv_msg
from data_reader.data_reader import get_stream_data, get_data
from models.get_model import get_model
from config import *

client_sock_all = []
count = 0
while count < 7:
    sock = socket.socket()
    sock.connect((SERVER_ADDR, SERVER_PORT[count]))
    client_sock_all.append(sock)
    count = count + 1

D = np.random.poisson(lam=lamb, size=T)
for i in range(T):
    if D[i]==0:
        D[i] =1


t = 0
t0 = 0

print('-----------------------connected---------------------------------------------')

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # ipv4协议/tcp协议 通信前要用connect（）建立连接状态
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT[7]))


# Establish connections to each client, up to n_nodes clients
while len(client_sock_all) < n_nodes:
    listening_sock.listen(4)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept() # 得到的是client的地址
    print('Got connection from ', (ip,port)) # ('127.0.0.1', 12051)
    print(client_sock) # <socket.socket fd=2376, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('127.0.0.1', 51000), raddr=('127.0.0.1', 12051)>

    client_sock_all.append(client_sock)

print('All clients connected')

msg = recv_msg(client_sock_all[0], 'MSG_INIT_SERVER_TO_CLIENT')
        # ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, step_size,
#                w_global, tau_config, control_alg, indices_this_node]
model_name = msg[1]
dataset = msg[2]
step_size = msg[3]
w_global = msg[4]
tau = msg[5]
control_alg_server_instance = msg[6]
indices_this_node = msg[7]

model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

w_last_global = w_global

# train_image, train_label, test_image, test_label, train_label_orig = get_stream_data(
#     dataset, total_data, dataset_file_path, int(D[0]))
train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)
# print("total :")
# # print(train_image)
# print(len(train_image))
#
# print("indicates is")
# print(indices_this_node)


np.random.RandomState(seed=1000).shuffle(indices_this_node)# 把seed=1000去掉 否则每次shuffle之后的数都一样 tau没影响
sample_indices = indices_this_node[0:D[0]]  # 数目是D[0]


# print("sample_indices is")
# print(sample_indices)
# stream_train_image = [train_image[i] for i in sample_indices]
# stream_train_label = [train_label[i] for i in sample_indices]
# stream_train_label_orig = [train_label_orig[i] for i in sample_indices]

data_size_h = len(sample_indices)
init_grad = model.gradient(train_image, train_label, w_global, sample_indices) # \nabla F(w(t))
# print("grad的维度是：{}".format(init_grad.shape))
grad_last_global = init_grad  # TODO init_grad的维度为什么不是784/3072

w = w_global - step_size * init_grad # w_u
grad_last_local_last_round = model.gradient(train_image, train_label, w, sample_indices)# \nabla F(w_u(t))
w_last_local_last_round = w
if True in np.isnan(w_global):
    grad_last_local_last_round = init_grad
# try:
#     # Note: This has to follow the gradient computation line above
#     loss_last_global = model.loss_from_prev_gradient_computation()
#     print('*** Loss computed from previous gradient computation')

# Will get an exception if the model does not support computing loss
# from previous gradient computation

loss_last_global = model.loss(train_image, train_label, w_global, sample_indices) # 第一次是0.5
loss_last_local_last_round = model.loss(train_image, train_label, w, sample_indices)
# acc = model.accuracy(test_image,test_label,w) # 计算精度的时候也只计算sample个 可以为none 则计算所有
# print('*** Loss computed from data, loss is {}'.format(loss_last_local_last_round))
# print('*** Loss computed from data, acc is {}'.format(acc))

msg = ['MSG_INIT_GRADIENT_CLIENT_TO_SERVER', init_grad, D[0]]
# print(init_grad)
send_msg(client_sock_all[0], msg)

if isinstance(control_alg_server_instance, ControlAlgAdaptiveTauServer):  # 考虑继承关系，判断两个类型是否相同
    control_alg = ControlAlgAdaptiveTauClient()
else:
    control_alg = None

while True:


    # if control_alg is not None:
    #     control_alg.init_new_round(w_global)

    data_size_h = len(sample_indices) # TODO 上一次的data_size才对，应该用solver解完的 data_size_gen是生成的，data_size是H
    # print("data size is:{}".format(data_size_h))
    # print("sample_indices is")
    # print(sample_indices)

    # if control_alg is not None:
    #     control_alg.update_after_all_local(model, stream_train_image, stream_train_label, train_indices,
    #                                        w, w_last_global, loss_last_global)
        # 计算了arfa beta


    # computation cost and capicity
    # phi = np.random.normal(50, 10, 1) # 均值为200，方差为100的正态分布
    # Phi = np.random.uniform(200000, 250000) # 均匀分布
    #
    # msg = ['MSG_phi_CLIENT_TO_SERVER', phi, Phi]
    # # data_size用的是上次local update的数量，用于估计全局的rho beta，指导本次的卸载决策
    # send_msg(client_sock_all[0], msg)
    # print("发送了啊")

    if(t-t0 == tau):
        t0 = t

        # ---------------计算beta rho-------------------------------------
        # compute beta

        c = grad_last_local_last_round - grad_last_global
        tmp_norm = linalg.norm(w_last_local_last_round - w_last_global)
        if tmp_norm > 1e-10:
            beta_adapt = linalg.norm(c) / tmp_norm
            # print("beta:{}".format(beta_adapt))

        else:
            beta_adapt = 0
            # print("too small")

        # 是新计算的全局loss
        loss_last_global = model.loss(train_image, train_label, w_global, sample_indices)
        if tmp_norm > 1e-10:
            rho_adapt = linalg.norm(loss_last_local_last_round - loss_last_global) / tmp_norm
            # print(tmp_norm)
        else:
            rho_adapt = 0
            # print("too small")

        if beta_adapt < 1e-5 or np.isnan(beta_adapt):
            beta_adapt = 1e-5

        if np.isnan(rho_adapt):
            rho_adapt = 0


        msg = ['MSG_GRADIENT_CLIENT_TO_SERVER', grad_last_local_last_round, w, data_size_h]
        # print(init_grad)
        # print(" loss_last_local_last_round:{}是有降低吧".format(loss_last_local_last_round))
        # print("对应的传过去的w是：{}".format(w))
        send_msg(client_sock_all[0], msg)

        msg = recv_msg(client_sock_all[0], 'MSG_MODEL_TO_CLIENT')
        # ['MSG_MODEL_TAU_TO_CLIENT', w_global, tau_config]
        w_global = msg[1]
        # print("收到的全局的w是：{}".format(w_global))
        grad_last_global = model.gradient(train_image, train_label, w_global, sample_indices)# \nable F(w(t))






        # print("beta is:{}".format(beta_adapt)) # 10.777395238220514
        # print("rho is:{}".format(rho_adapt)) # 1.2313606239517996
        msg = ['MSG_RHO_BETA_CLIENT_TO_SERVER', rho_adapt, beta_adapt, loss_last_global]
        # print("rho is {}, beta is {}".format(rho_adapt, beta_adapt))

        send_msg(client_sock_all[0], msg)
        print("send the loss last global is {}".format(loss_last_global))

        w = w_global

        # acc = model.accuracy(test_image, test_label, w)
        # print('***收到的全局模型，acc不能很低， acc is {}'.format(acc))



        msg = recv_msg(client_sock_all[0], 'MSG_TAU_TO_CLIENT')
        # ['MSG_TAU_TO_CLIENT', tau_config]
        tau = msg[1]

    if(t>0 and intera):
        msg = recv_msg(client_sock_all[0], 'MSG_solver_TO_CLIENT') # row_list[0][n]
        if msg[1] > 1:
            data_size_h = msg[1]
        else:
            data_size_h = 1
        print("data_size_h is {}".format(data_size_h))


    t = t + 1

    # if int(data_size_h) == 0:
    #     continue

    start = 0
    s = [0, 0.2, 0.1, 0.1, 0.02]

    # for n in range(1, n_nodes): # 先不考虑卸载到云
    #     stream_train_image_send = stream_train_image[start:start+int(data_size_gen*s[n])]#向下取整
    #     stream_train_label_send = stream_train_label[start:start+int(data_size_gen*s[n])]
    #     msg = ['MSG_DATA_TO_CLIENT', stream_train_image_send, stream_train_label_send] # 发送初始的model 和 tau
    #     send_msg(client_sock_all[n], msg)
    #     start = start + int(data_size_gen*s[n])
    # stream_train_image = stream_train_image[start:]#剩下的
    # stream_train_label = stream_train_label[start:]
    # data_size = len(stream_train_image)

    # sample_indices = [i for i in range(0, len(stream_train_image))]

    grad_last_local_last_round = model.gradient(train_image, train_label, w, sample_indices)# \nable F_u(w_u(t))
    w_last_local_last_round = w
    w = w - grad_last_local_last_round * step_size #本地更新


    loss_last_local_last_round = model.loss(train_image, train_label, w, sample_indices)# 本地loss
    # acc = model.accuracy(test_image, test_label, w)
    # print('*** Loss computed from data, loss is {}'.format(loss_last_local_last_round))
    # print('*** Loss computed from data, acc is {}'.format(acc))

    if not intera:
        data_size_h = data#D[t]

    # train_image, train_label, test_image, test_label, train_label_orig = get_stream_data(
    #     dataset, total_data, dataset_file_path, int(data_size_h))
    if non_inclu:
        indices_this_node.sort() # 要先排序，不然用shuffle还是打乱之后的基础上继续打乱
    np.random.RandomState(seed=1000).shuffle(indices_this_node)  # 把seed=1000去掉 否则每次shuffle之后的数都一样 tau没影响
    sample_indices = indices_this_node[0:int(round(data_size_h))]  # 数目是D[0]



    # data_size = D[t];
