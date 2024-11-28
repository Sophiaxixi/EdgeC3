import math
import socket
import os


import pandas as pd
import random
import time

import numpy as np
from numpy import linalg
from pandas import Series, DataFrame

import matplotlib.pyplot as plt



from models.get_model import get_model
from data_reader.data_reader import get_stream_data, get_data
from statistic.collect_stat import CollectStatistics
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from util.utils import send_msg, recv_msg, get_indices_each_node_case


# Configurations are in a separate config.py file
from config import *

model = get_model(model_name)
if hasattr(model, 'create_graph'):# 判断有这个方法
    model.create_graph(learning_rate=step_size)

D = np.random.poisson(lam=lamb, size=T)
for i in range(T):
    if D[i]==0:
        D[i] =1





# train_image, train_label, test_image, test_label, train_label_orig = get_stream_data(dataset, total_data, dataset_file_path, D[0])
train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)
# This function takes a long time to complete,
# putting it outside of the sim loop because there is no randomness in the current way of computing the indices
indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

indices = [i for i in range(0,len(train_image))]
np.random.RandomState(seed=1000).shuffle(indices)# 把seed=1000去掉 否则每次shuffle之后的数都一样 tau没影响
sample_indices = indices[0:D[0]]  # 数目是D[0]

# need to test
# gra_distribute = np.random.standard_normal(len(stream_train_image[0])) # 暂时一开始给定，服从标准正态分布 TODO 是否随时间变化？
gra_distribute = np.random.standard_normal(dia) # 暂时一开始给定，服从标准正态分布 TODO 是否随时间变化？

t = 0
t0 = 0
k = 0
# x_k = np.zeros(tau_max)
x_k = np.random.dirichlet(np.ones(tau_max),size=1).tolist()[0] # 随机生成tau_max维度，总和为1
x_k.insert(0,0) # 为了让下标从1开始
# x_k[0] = 1
# for i in range(1,tau_max+1):
#     x_k[i] = 1/tau_max # 初始化的不能随机，要平均
#     print("x_i是{0:.9f}".format(x_k[i]),end=',')
choice_list = [i for i in range(1,tau_max+1)]


# tau_config = random.choices(choice_list, weights=x_k[1:], k=1)# 以对应概率sample


listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # ipv4协议/tcp协议 通信前要用connect（）建立连接状态
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT[0]))
client_sock_all=[]

# Establish connections to each client, up to n_nodes clients
while len(client_sock_all) < n_nodes:
    listening_sock.listen(n_nodes)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept() # 得到的是client的地址
    print('Got connection from ', (ip,port)) # ('127.0.0.1', 12051)
    print(client_sock) # <socket.socket fd=2376, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('127.0.0.1', 51000), raddr=('127.0.0.1', 12051)>

    client_sock_all.append(client_sock)

if single_run: # 画图用的
    stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
else:
    stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

for tau_setup in tau_setup_all:
    stat.init_stat_new_global_round()

    dim_w = model.get_weight_dimension(train_image, train_label) # 一张img有784维
    w_global_init = model.get_init_weight(dim_w, rand_seed=0)
    w_global = w_global_init
    w_global_prev = w_global

    # accuracy = model.accuracy(test_image, test_label, w_global)

    w_global_min_loss = None
    loss_min = np.inf
    prev_loss_is_min = False

    if tau_setup < 0: # 表示adaptive的tau
        is_adapt_local = True
        # tau_config = 1
    else:
        is_adapt_local = False
        # tau_config = tau_setup

    if is_adapt_local: # adaptive 或者需要一直估计rho beta才需要自适应的控制算法
        control_alg = ControlAlgAdaptiveTauServer()
    else:
        control_alg = None

    for n in range(0, n_nodes):
        indices_this_node = indices_each_node_case[case][n]
        msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, step_size,
               w_global, tau_config, control_alg, indices_this_node] # 发送初始的model 和 tau
        send_msg(client_sock_all[n], msg)

    print('All clients connected')

    init_gra_all = []
    data_size_all = []
    gra_global = 0
    data_size_total = 0


    # Wait until all clients complete data preparation and sends a message back to the server
    for n in range(0, n_nodes):
        msg = recv_msg(client_sock_all[n], 'MSG_INIT_GRADIENT_CLIENT_TO_SERVER')
        # ['MSG_INIT_GRADIENT_CLIENT_TO_SERVER', init_grad, D[0]]
        init_gra = msg[1]
        init_gra_all.append(init_gra)
        data_size_local = msg[2]
        data_size_all.append(data_size_local)
        gra_global += init_gra * data_size_local
        data_size_total += data_size_local

    gra_global = gra_global/data_size_total
    Lamb = linalg.norm(gra_distribute - gra_global) # TODO gra_global的维度为什么不是784/3072
    print("gra_distribute的维度是：{}".format(gra_distribute.shape))
    print("gra_global的维度是：{}".format(gra_global.shape))
    aggre_time_list = []
    loss_list = []
    not_loss_list = []
    not_acc = []
    acc = []

    print('Start learning')

    time_global_aggregation_all = None

    total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
    total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
                                # using predefined values when use_fixed_averaging_slots = true
    it_each_local = None
    it_each_global = None

    is_last_round = False
    is_eval_only = False

    tau_new_resume = None




    # with open(solver_para_file_path_u, 'a') as f:
    #     f.write(
    #         'timestep,rho,beta,Lamb,tau\n')
    #     f.close()


    while t<T:
        print('---------------------------------------------------------------------------')
        # if t == 305:
        #     print("没有出来嘛")
        #     break

        print('current tau config:{}, t:{}, t0:{}, Lamb:{}'.format(tau_config,t,t0,Lamb))


        local_data_size_all = []
        beta_adapt = 0
        rho_adapt = 0
        data_size_total = 0

        gra_global = 0
        loss_global = 0
        w_sum = 0

        # for n in range(0, n_nodes):
        #     print("收到了吗")
        #     msg = recv_msg(client_sock_all[n], 'MSG_phi_CLIENT_TO_SERVER')
        #     print("应该")
        #     # ['MSG_phi_CLIENT_TO_SERVER', phi, Phi]
        #
        #     local_phi = msg[1]
        #     local_Phi = msg[2]
        #
        #
        #     local_phi_all.append(local_phi)
        #     local_Phi_all.append(local_Phi)




        if(t-t0 == tau_config):


            for n in range(0, n_nodes):
                msg = recv_msg(client_sock_all[n], 'MSG_GRADIENT_CLIENT_TO_SERVER')
                # ['MSG_GRADIENT_CLIENT_TO_SERVER', grad_last_local_last_round, w, data_size]

                grad_last_local_last_round = msg[1]
                w_local = msg[2]
                local_data_size = msg[3]
                local_data_size_all.append(local_data_size)
                # data_size_total += local_data_size;

                if True not in np.isnan(grad_last_local_last_round):
                    gra_global += grad_last_local_last_round * local_data_size #TODO 这里出现了nan 改正做法：判断某个client有nan就不使用它的
                    w_sum += w_local * local_data_size  #TODO 这里出现了nan
                    data_size_total += local_data_size;



            gra_global /= data_size_total
            w_global = w_sum / data_size_total
            newLamb = linalg.norm(gra_distribute - gra_global)
            # print("newlamb没有吗？".format(newLamb))

            if True in np.isnan(w_global):
                print('*** w_global is NaN, using previous value')
                w_global = w_global_prev  # If current w_global contains NaN value, use previous w_global
            else:
                w_global_prev = w_global # 没有nan的话就可以更新

           # if t >540:
            accuracy = model.accuracy(test_image, test_label, w_global)
            # else:
            #     accuracy=-1

            # global updata
            # w_global = w_global - step_size * gra_global # 这样只用到了本地更新多次的最后一次的梯度

            for n in range(0, n_nodes):
                msg = ['MSG_MODEL_TO_CLIENT', w_global]  # 发送每轮的的model
                send_msg(client_sock_all[n], msg)

            # jasc的对比代码在这里写的 if JASC:

            for n in range(0, n_nodes):
                msg = recv_msg(client_sock_all[n], 'MSG_RHO_BETA_CLIENT_TO_SERVER')
                # ['MSG_RHO_BETA_CLIENT_TO_SERVER', rho_adapt, beta_adapt, loss_last_global ]

                rho_adapt_local = msg[1]
                beta_adapt_local = msg[2]
                loss_last_global = msg[3]
                # local_data_size = msg[3]
                # local_data_size_all.append(local_data_size)

                beta_adapt += local_data_size_all[n] * beta_adapt_local
                rho_adapt += local_data_size_all[n] * rho_adapt_local
                loss_global += loss_last_global * local_data_size_all[n]
                # data_size_total += local_data_size;

            beta_adapt /= data_size_total
            rho_adapt /= data_size_total
            loss_global /= data_size_total
            print('*** Global loss computed from data, loss is {}'.format(loss_global))
            print('*** Global loss computed from data, acc is {}'.format(accuracy))
            print('*** rho:{} and beta:{} have values'.format(rho_adapt,beta_adapt))


            aggre_time_list.append(t)

            loss_list.append(loss_global)
            acc.append(accuracy)

        if (t > 0):
            solver_para_file_path_u = solver_file_path + '/SolverPara_u' + str(t) + '.csv'
            solver_para_file_path_e = solver_file_path + '/SolverPara_e' + str(t) + '.csv'
            if not os.path.exists(os.path.dirname(solver_para_file_path_u)):
                os.makedirs(os.path.dirname(solver_para_file_path_u))
                print("youa")
            if not os.path.exists(os.path.dirname(solver_para_file_path_e)):
                os.makedirs(os.path.dirname(solver_para_file_path_e))
            data_frame = pd.DataFrame({'rho': rho_adapt, 'beta': beta_adapt, 'Lam': Lamb, 'Tau': tau_config},index=[t])
            data_frame.to_csv(solver_para_file_path_u, sep=',')
            data_frame = pd.DataFrame({'rho': 1}, index=[t])
            data_frame.to_csv(solver_para_file_path_e, sep=',')
            Lamb = newLamb # 下一个time才是新的一轮
            # with open(solver_para_file_path_u, 'a') as f: # 最终写入的
            #     f.write(str(t) + ',' + str(rho_adapt) + ',' + str(beta_adapt) + ',' + str(Lamb) + ',' + str(tau_config) + ','
            #             + '\n')
            #     f.close()


            # 读取S
            if intera:
                # solver_result_path_sr = solver_file_path+'/result_sr'+str(t)+'.csv'
                solver_result_path_h = solver_file_path+'/result_h'+str(t)+'.csv'
                solver_result_path_e = solver_file_path + '/result_e' + str(t) + '.csv'
                # while not os.path.exists(solver_result_path_sr):
                #     time.sleep(0.1)#0.1s
                while not os.path.exists(solver_result_path_e): # 检测空文件
                   time.sleep(0.11) # 0.1s
                # 读取
                # time.sleep(0.7)
                h_result = pd.read_csv(solver_result_path_h,header = None,error_bad_lines=False,engine='python')
                row_list = h_result.values.tolist()#二维列表
                print(f"行读取结果：{row_list}")
                if row_list[4][0] == 1: #中断信号
                    print(f"没有到")
                    break;
                # if t > 9000: #中断信号
                #     print(f"时间到")
                #     break;

                #----------------UCB-----------------------------
                print(f"p是：{row_list[3][0]}")
                if t - t0 == 1:
                    p_1= row_list[3][0] # 一个round的开头不计算
                else:
                    u = u + row_list[3][0] #是p obj
                u_list.append(row_list[3][0])
                print(f"u是：{u}")

            # -----没交互也要传tau
            if (t - t0 == tau_config):
                # for data in u_list:
                #     print(data) #打印这轮的p 也就是u
                # u_list = []

                if tau_adaptive:
                    if tau_config == 1:
                        ave_u = p_1; # tau为1不计算
                    else:
                        ave_u = u/(tau_config -1)
                    aveu_list.append(ave_u)
                    # tau_list.append(tau_config)
                    u = 0 # 一个round之后u要清零
                    print(f"这个round的feedback p是{ave_u}")
                    u_i[tau_config] = (u_i[tau_config]*T_i[tau_config] + ave_u)/(T_i[tau_config]+1) #ui(t-1)
                    k = k + 1
                    T_i[tau_config] = T_i[tau_config] + 1


                    #
                    ran = random.randint(2, tau_max) #随机选取一个而不是从1开始增大
                    if T_i[ran] == 0:
                        tau_config = ran
                    else:
                        for i in range(1,tau_max+1):
                            if T_i[i] == 0:
                                score[i] = -1 * math.inf # 暂时还没选到，先不管它 这次不会选
                                # break # 找到了有没被探索过的，这次就用它
                            else:
                                score[i] = -1 * u_i[i] + math.sqrt(2*math.log(k)/T_i[i]) # math.log就是默认以e为底的对数 因为预处理的时候给所有都试了，这里的总次数就变成了k+tau_max
                                print(f"-ui是{-1 * u_i[i]},explore是{math.sqrt(2*math.log(k)/T_i[i])}")
                                #score[i] = -1 * score[i] # 要求最大，而p是最小 所以取相反数
                        for data in score:
                            print(data,end=',')
                        tau_config = np.argmax(np.array(score[2:]))+2 # 不然就是0



                    # 从1开始遍历
                    # for i in range(1,tau_max+1):
                    #     if T_i[i] == 0:
                    #         score[i] = math.inf # 暂时还没选到，先不管它 这次不会选中它就行
                    #         # break # 找到了有没被探索过的，这次就用它
                    #     else:
                    #         score[i] = -1 * u_i[i] + math.sqrt(2*math.log(k)/T_i[i]) # math.log就是默认以e为底的对数 因为预处理的时候给所有都试了，这里的总次数就变成了k+tau_max
                    #         print(f"-ui是{-1 * u_i[i]},explore是{math.sqrt(2*math.log(k)/T_i[i])}")
                    #         #score[i] = -1 * score[i] # 要求最大，而p是最小 所以取相反数
                    # for data in score:
                    #     print(data,end=',')
                    # tau_config = np.argmax(np.array(score[2:]))+2 # 不然就是0 tau为1也不计算


                    print("tau is {}".format(tau_config))

                else:
                    if t > 0:
                        tau_config = 1
                        # if k>=len(tau_list):
                        #     tau_config = 50
                        # else:
                        #     tau_config = tau_list[k]
                        #     k = k + 1
                    index = index +1;
                    # if index == len(tau_list): # TODO 记得去掉
                    #     break;

                for n in range(0, n_nodes):
                    msg = ['MSG_TAU_TO_CLIENT', tau_config]  # 发送每轮的tau
                    send_msg(client_sock_all[n], msg)

                t0 = t;
                #-----------------


            if intera:
                for n in range(0, n_nodes):
                    # msg = ['MSG_solver_TO_CLIENT', row_list[0][n]]  # 发送每轮的的model 和 tau 卸载
                    # msg = ['MSG_solver_TO_CLIENT', row_list[1][n]]  # 发送每轮的的model 和 tau 没卸载
                    msg = ['MSG_solver_TO_CLIENT', row_list[2][n]]  # 发送每轮的的model 和 tau random
                    send_msg(client_sock_all[n], msg)

        t=t+1

    # loss_list = [0.4703591348442263, 0.45747618331359946, 0.4455419397599384, 0.43430390845112304, 0.4240349649138991, 0.41435088246007784, 0.40526838246783964, 0.3967776995295057, 0.3888566311032097, 0.38145442290051573, 0.37452864505662586, 0.367875868481574, 0.36193334269846866, 0.3562085495880183, 0.350862835633347, 0.3458382347269678, 0.3410668517346356, 0.33655901889951256, 0.33235233668039676]
    # loss_list = [i for i in range(1,T)]
    not_loss_list = [0.428868928188244, 0.41266966107363584, 0.3935811654227178, 0.39225344262254885, 0.380469612216685, 0.3654763302128119, 0.35123616196082685, 0.3391710616270343, 0.32607648983942816, 0.3200175531140228, 0.3099655459697424, 0.3070845296486551, 0.29293503843148677, 0.28266122247444136, 0.2760989757612523, 0.2733177760384602, 0.26833478426706403, 0.2640003012289906, 0.2561007232807124, 0.25257824102811666, 0.24251541327051493, 0.24321955965895475, 0.23627694773947347, 0.23263193373376045, 0.227276065097285, 0.22418625671077405, 0.21902980160652108, 0.21593537608084257, 0.211849446873814, 0.2091980483917152, 0.20503895622555782, 0.20553080371811244, 0.2009384400416922, 0.19859322170383947, 0.19671797654487516, 0.1945912147054486, 0.18970037742011592, 0.18839994316574613, 0.188911052009432, 0.18676001811746595, 0.18060086332128755, 0.18083496001380572, 0.179926852253439, 0.18287856770705968, 0.17290938737417538, 0.17310918614476808, 0.1735920705239301, 0.1718665448288478, 0.16774749964736357]
    loss_list =[0.4596618856300934, 0.44310733442371636, 0.4247694914768732, 0.412974277695137, 0.40138525044963475,
     0.3885084358076705, 0.3764724137334339, 0.3634606528687894, 0.35575192393467603, 0.3472716788799476,
     0.3429917804530219, 0.3318386038448181, 0.3205317567324384, 0.3105866383262584, 0.3086838326607578,
     0.2998787423130626, 0.2864118869860313, 0.28022576461355625, 0.27850311596770927, 0.28372010940970926,
     0.2652013310651268, 0.2644975448617915, 0.263562646478478, 0.26393137095342717, 0.2574901325260229,
     0.2550919108900776, 0.2460776878791019, 0.23261390265725615, 0.2486244437031654, 0.2367021411046645,
     0.23759013851262514, 0.22811723451200822, 0.22528047908713641, 0.22558671496180574, 0.22624294105714804,
     0.21816440238374668, 0.2208912243641529, 0.2204731879152773, 0.20433904206330566, 0.22264771089854757,
     0.19960397940052177, 0.20748308275475724, 0.20726225999542522, 0.1969958997911414, 0.19908502600842531,
     0.20292909627202974, 0.2026847864430735, 0.20342329511591423, 0.20609484776016332]
    # not_acc = [ 0.5073, 0.5092, 0.5208, 0.5205, 0.522, 0.5448, 0.553, 0.5714, 0.6106, 0.6153, 0.6051, 0.5952, 0.6002, 0.6095, 0.6046, 0.6113, 0.6302, 0.6378, 0.6411, 0.6384, 0.6508, 0.6484, 0.6453, 0.651, 0.6584, 0.6623, 0.6644, 0.6633, 0.664, 0.6632, 0.6578, 0.656, 0.6593, 0.6676, 0.6665, 0.6662, 0.6669, 0.6787, 0.6745, 0.6673, 0.6675, 0.6637, 0.6608, 0.6694, 0.6699, 0.6652, 0.6683, 0.6604, 0.6675, 0.6729, 0.6715, 0.6683, 0.6686, 0.6606, 0.6631, 0.6646, 0.6656, 0.6665, 0.6596, 0.6623, 0.6633, 0.6626, 0.6633, 0.6646, 0.6623, 0.6611, 0.6536, 0.6574, 0.6606, 0.6622, 0.6614, 0.6629, 0.6622, 0.6555, 0.6586, 0.6585, 0.6551, 0.6578, 0.6577, 0.6578, 0.6605, 0.6618, 0.6629, 0.6611, 0.663, 0.6617, 0.6634, 0.6637, 0.6644, 0.6648, 0.6672, 0.6609, 0.6631, 0.6591, 0.6598, 0.6619, 0.6642, 0.6631, 0.6644, 0.6654, 0.6661, 0.6648, 0.6652, 0.6646, 0.6651, 0.6656, 0.6661, 0.6665, 0.6663, 0.6679, 0.6661, 0.6666, 0.6651, 0.664, 0.6651, 0.6637, 0.6631, 0.6599, 0.6601, 0.6605, 0.6621, 0.6587, 0.6604, 0.6604, 0.6605, 0.6621, 0.6617, 0.6624, 0.6629, 0.6618, 0.6611, 0.663, 0.6593, 0.6593, 0.6607, 0.6603, 0.6619, 0.6621, 0.6623, 0.6646, 0.6647, 0.6646, 0.6592, 0.6607, 0.6617, 0.6626, 0.6615, 0.6629, 0.6613]
    # acc = [0.5073, 0.5076, 0.5134, 0.5349, 0.5443, 0.5634, 0.5635, 0.6061, 0.6372, 0.6442, 0.6301, 0.6638, 0.6638, 0.6515, 0.6827, 0.6894, 0.699, 0.6926, 0.6978, 0.709, 0.7065, 0.7082, 0.7191, 0.7181, 0.727, 0.7218, 0.7249, 0.72, 0.7177, 0.7156, 0.7172, 0.7171, 0.7196, 0.7333, 0.7284, 0.734, 0.7438, 0.7345, 0.7293, 0.7224, 0.7381, 0.7466, 0.7377, 0.7486, 0.7398, 0.7442, 0.7402, 0.7391, 0.753, 0.7468, 0.7417, 0.7518, 0.7592, 0.7586, 0.7648, 0.7671, 0.7575, 0.7629, 0.7561, 0.7523, 0.7593, 0.7547, 0.7515, 0.7509, 0.7581, 0.7608, 0.7627, 0.7668, 0.7596, 0.7562, 0.7588, 0.7552, 0.7527, 0.759, 0.7593, 0.7534, 0.7581, 0.7509, 0.7511, 0.7477, 0.7554, 0.7567, 0.7548, 0.7557, 0.7567, 0.7537, 0.7573, 0.7609, 0.7569, 0.7569, 0.7552, 0.7496, 0.7498, 0.7549, 0.754, 0.7541, 0.7514, 0.7504, 0.7536, 0.7522, 0.7493, 0.7481, 0.7487, 0.7489, 0.7488, 0.7475, 0.7482, 0.7484, 0.7488, 0.7562, 0.7532, 0.7518, 0.7581, 0.7541, 0.7595, 0.7676, 0.7669, 0.7706, 0.7655, 0.7691, 0.7638, 0.7685, 0.7636, 0.7683, 0.7678, 0.7634, 0.7642, 0.7669, 0.7666, 0.7628, 0.7587, 0.7644, 0.7572, 0.7562, 0.7553, 0.7552, 0.7546, 0.7582, 0.7553, 0.7547, 0.7542, 0.7532, 0.7611, 0.7676, 0.764, 0.7648, 0.7676, 0.7713, 0.7682]
    plt.axis([0, T, 0, 1])
    print("time is:{}".format(aggre_time_list))
    # print("not_acc is:{}".format(not_acc))
    print("acc is:{}".format(acc))
    # for data in u_list:
    #     print(data) #打印所有的p 也就是u
    #
    # print("接下去是平均值")

    for data,tau in zip(aveu_list,tau_list):
        print("tau is {},ave_p is {}".format(tau,data))

    # with open(result_acc_file_path, 'a') as f: # 最终写入的
    #     print("打开了啊")
    #     print(result_acc_file_path)
    #     for data in acc:
    #         f.write(str(data) + ', ')
    #         f.flush()
    #     f.close()
    # plt.title('Offloading VS not offloading')
    # plt.plot(aggre_time_list, loss_list, 'r--',aggre_time_list, not_loss_list, 'b--')
    plt.plot(aggre_time_list, acc, 'r--',aggre_time_list, not_acc, 'b--')
    plt.xlabel('Time step')
    plt.ylabel('Loss function Value (on Training Data)')
    plt.legend(['not offloading', 'offloading'], loc=2)
    plt.show()

    # print("loss is:{}".format(loss_list))
    # print("loss is:{}".format(not_loss_list))
























