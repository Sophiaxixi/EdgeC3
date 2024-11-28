import paramiko
import time
import os

def RemoteScp(host_ip, host_port, host_username, host_password, remote_path, local_path):
    print(123)
    scp = paramiko.Transport((host_ip, host_port))
    scp.connect(username=host_username, password=host_password)
    sftp = paramiko.SFTPClient.from_transport(scp)
    T = 90020

    for t in range(1,T):
        file_para =  '/SolverPara_u' + str(t) + '.csv'
        file_para_e = '/SolverPara_e' + str(t) + '.csv'
        file_h = '/result_h' + str(t) + '.csv'
        file_e = '/result_e' + str(t) + '.csv'

        print(remote_path + file_para)
        print(local_path+ file_para)

        while 1:
            try:
                sftp.stat(remote_path + file_para_e)
                break #有文件就退出
            except IOError:
                time.sleep(0.1)
        time.sleep(0.1) #还是会出现拉取空文件 的情况
        sftp.get(remote_path + file_para, local_path+ file_para)
        sftp.get(remote_path + file_para_e, local_path + file_para_e)

        while not os.path.exists(local_path + file_e): # 判断的是第二个文件
            time.sleep(0.01)  # 0.1s

        time.sleep(0.1)
        sftp.put(local_path + file_h, remote_path + file_h)
        sftp.put(local_path + file_e, remote_path + file_e)



    scp.close()

if __name__ == '__main__':
    host_ip = '172.18.232.84'  # 远程服务器IP
    # host_ip = '192.168.26.85'
    host_port = 20222  # 远程服务器端口
    # host_port = 31022
    host_username = 'linsh'  # 远程服务器用户名 linsh abc
    host_password = 'linsh2842'  # 远程服务器密码 linsh2842 19991111lsh
    remote_path = '/home/linsh/experiment/C3_Offloading_ucb_lesstime' \
                  '/results/Solver'  # 这个是远程目录/home/linsh/experiment/C3_Offloading_ucb/results/Solver /config/workspace/C3_Offloading_ucb_lesstime/results/Solver
    local_path = r'D:\science\paper\offload\experient\C3_Offloading_ucb_lesstime\results\Solver\\'  # 这个是本地目录
    RemoteScp(host_ip, host_port, host_username, host_password, remote_path, local_path)  #调用方法

