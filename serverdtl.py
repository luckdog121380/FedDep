import copy
import math
import random
import time
import torch
from flcore.clients.clientdtl import clientdtl
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data

class FedDTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientdtl)
        self.current_round = 0
        self.layers_to_aggregate = 1
        self.nns = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models()


            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            print("\nTraining completed. Monitoring results saved.")
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientdtl)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self, start_layer=0, num_layers_to_skip=0):
        assert len(self.clients) > 0, "Client list is empty!"
        total_data_sent = 0  # 初始化通信开销计数

        for client in self.clients:
            start_time = time.time()
            p = 10
            t = self.current_round
            T = self.global_rounds
            pt = p * t / T
            pl = math.floor(pt)

            model_size = sum(param.numel() * param.element_size() for param in self.global_model.parameters())
            total_data_sent += model_size

            for idx, ((client_param_name, client_param), (global_param_name, global_param)) in enumerate(
                    zip(reversed(list(client.model.named_parameters())),
                        reversed(list(self.global_model.named_parameters())))):
                if idx >=pl:
                    client_param.data = global_param.data.clone()  # 更新参数
                else:
                    print(f"Skipping layer: {client_param_name}")  # 打印跳过的层的名称


        client.send_time_cost['num_rounds'] += 1
        client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

        print(f"Total data sent to clients: {total_data_sent / 1e6:.2f} MB")

    def sparse_reconstruction(self, sparsity=0.5):
        for name, param in self.global_model.named_parameters():
            if 'fc' in name or 'linear' in name:
                threshold = torch.quantile(torch.abs(param.data), sparsity)
                param.data = torch.where(torch.abs(param.data) < threshold,
                                         torch.zeros_like(param.data),
                                         param.data)

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


        if self.global_rounds < 200:
            self.sparse_reconstruction(sparsity=0.5)






