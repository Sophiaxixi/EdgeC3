import math
import numpy as np
from numpy import linalg
from util.utils import recv_msg, send_msg
from config import tau_max


class ControlAlgAdaptiveTauServer():
    pass


class ControlAlgAdaptiveTauClient:
    def __init__(self):
        self.w_last_local_last_round = None # 本地模型 w_i(t)
        self.grad_last_local_last_round = None # 上一次的全局模型 w(t)
        self.loss_last_local_last_round = None # f(w_i(t))

    def init_new_round(self, w):
        self.control_param_computed = False
        self.beta_adapt = None
        self.rho_adapt = None
        self.grad_last_global = None

    def update_after_each_local(self, iteration_index, w, grad, total_iterations):
        if iteration_index == 0:
            self.grad_last_global = grad

        return False

    def update_after_all_local(self, model, train_image, train_label, train_indices,
                               w, w_last_global, loss_last_global):

        # Only compute beta and rho locally, delta can only be computed globally
        #if (self.w_last_local_last_round is not None) and (self.grad_last_local_last_round is not None) and \
        #        (self.loss_last_local_last_round is not None):

        # compute beta
        c = self.grad_last_local_last_round - self.grad_last_global
        tmp_norm = linalg.norm(self.w_last_local_last_round - w_last_global)
        if tmp_norm > 1e-10:
            self.beta_adapt = linalg.norm(c) / tmp_norm
        else:
            self.beta_adapt = 0

        # Compute rho
        if tmp_norm > 1e-10:
            self.rho_adapt = linalg.norm(self.loss_last_local_last_round - loss_last_global) / tmp_norm
        else:
            self.rho_adapt = 0

        if self.beta_adapt < 1e-5 or np.isnan(self.beta_adapt):
            self.beta_adapt = 1e-5

        if np.isnan(self.rho_adapt):
            self.rho_adapt = 0

        print('betaAdapt =', self.beta_adapt)

        self.control_param_computed = True

        self.grad_last_local_last_round = model.gradient(train_image, train_label, w, train_indices)

        try:
            self.loss_last_local_last_round = model.loss_from_prev_gradient_computation()
        except:  # Will get an exception if the model does not support computing loss from previous gradient computation
            self.loss_last_local_last_round = model.loss(train_image, train_label, w, train_indices)

        self.w_last_local_last_round = w

    def send_to_server(self, sock):

        msg = ['MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER', self.control_param_computed]
        send_msg(sock, msg)

        if self.control_param_computed:
            msg = ['MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER', self.beta_adapt, self.rho_adapt, self.grad_last_global]
            send_msg(sock, msg)