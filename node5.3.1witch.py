import collections
import hashlib
import json
import string
from pprint import pprint
from typing import Any, Dict
import time
import socket
import random
import threading
from collections import Counter

'''模拟女巫攻击'''
class Collect:
    def __init__(self):
        # 存储前100个连接的客户端地址
        self.connection_history_flow = []
        # 存储前100个连接错误的客户端地址
        self.connection_error_history_flow = []
        # 存储前2秒内连接的客户端地址
        self.connection_history_time = []
        # 存储前2秒内连接错误的客户端地址
        self.connection_error_history_time = []

    def collecting(self, node, client_address, start_time, encoded_message, error):
        try:
            # 采`集信息
            formatted_time = time.strftime('%d.%H:%M:%S', time.localtime(start_time))
            duration = str(time.time() - start_time)
            # 是否是报错状态
            if error:
                self.connection_error_history_flow.append({'client_address': client_address, 'time_stamp': start_time})
                self.connection_error_history_time.append({'client_address': client_address, 'time_stamp': start_time})
                while len(self.connection_history_flow) > 100:
                    self.connection_history_flow.pop(0)  # 若列表超过100个元素，移除第一个（最早）元素
            else:
                self.connection_history_flow.append({'client_address': client_address, 'time_stamp': start_time})
                self.connection_history_time.append({'client_address': client_address, 'time_stamp': start_time})
                while len(self.connection_history_flow) > 100:
                    self.connection_history_flow.pop(0)  # 若列表超过100个元素，移除第一个（最早）元素
            # print(f"duration:{duration}")
            # print(f"byte_count: {len(encoded_message)}")

            # 当前节点连接次数connection_counter，基于流量,统计出当前client_address的出现次数，记录下来
            history_flow = self.connection_history_flow.copy()
            if len(self.connection_history_flow) > 100:
                self.connection_history_flow.pop(0)  # 若列表超过100个元素，移除第一个（最早）元素
            # 计数此连接节点
            cc_flow = 0
            for historic_address in history_flow:
                if client_address == historic_address['client_address']:
                    cc_flow += 1

            if len(history_flow) == 0:
                r1 = 0.0
            else:
                r1 = cc_flow * 1.0 / len(history_flow)
            # print(f"host_count:{cc_flow}")
            # print(f"host_count:{cc_flow}")
            # print(f"host_rate:{r1}")

            # 当前节点连接错误次数error_counter，基于流量,统计出当前client_address的出现次数，记录下来
            error_flow = self.connection_error_history_flow.copy()
            if len(self.connection_error_history_flow) > 100:
                self.connection_error_history_flow.pop(0)  # 若列表超过100个元素，移除第一个（最早）元素
            ec_flow = 0
            for historic_address in error_flow:
                if client_address == historic_address['client_address']:
                    ec_flow += 1
            if len(error_flow) == 0:
                r2 = 0.0
            else:
                r2 = ec_flow * 1.0 / len(error_flow)
            if ec_flow + cc_flow:
                r3 = 0.0
            else:
                r3 = ec_flow / (ec_flow + cc_flow)
            # print(f"host_error:{ec_flow}")
            # print(f"此节点连接的错误的占全部连接比率:{r2}")             # 此节点连接的错误的占全部连接比率
            # print(f"此节点连接的错误的占自身连接比率:{r3}")             # 此节点连接的错误的占自身连接比率

            # 当前节点连接次数connection_counter，基于时间,统计出当前client_address的出现次数，记录下来
            history_time = self.connection_history_time.copy()
            cc_time = 0
            for connection in history_time:
                if connection['time_stamp'] - start_time > 3:
                    self.connection_history_flow.pop(connection)
                    continue
                if client_address == connection['client_address']:
                    cc_time += 1
            if len(history_time) == 0:
                r4 = 0
            else:
                r4 = cc_time * 1.0 / len(history_time)
            # print(f"host_count:{cc_time}")
            # print(f"host_count_rate:{r4}")

            # 当前节点连接错误次数error_counter，基于时间,统计出当前client_address的出现次数，记录下来
            error_time = self.connection_error_history_time.copy()
            ec_time = 0
            for connection in error_time:
                if connection['time_stamp'] - start_time > 3:
                    self.connection_history_flow.pop(connection)
                    continue
                if client_address == connection['client_address']:
                    ec_time += 1

            if len(error_time) == 0:
                r5 = 0
            else:
                r5 = cc_time * 1.0 / len(error_time)
            if ec_time + cc_time == 0:
                r6 = 0
            else:
                r6 = cc_time / (ec_time + cc_time)
            # print(f"host_error:{ec_time}")
            # print(f"此节点连接的错误的占全部连接比率:{r5}")         # 此节点连接的错误的占全部连接比率
            # print(f"此节点连接的错误的占自身连接比率:{r6}")           # 此节点连接的错误的占自身连接比率
            # with open(f"output{self.oneself}.txt", "a") as output_file:
            # output_file.write(f"此节点连接的错误的占全部连接比率:{r5},此节点连接的错误的占自身连接比率:{r6}", end=',')
            print(f"formatted_time:{formatted_time}", end=',')
            print(f"duration:{duration}", end=',')
            print(f"byte_count: {len(encoded_message)}", end=',')
            print(f"host_count:{cc_flow}", end=',')
            print(f"host_rate:{r1}", end=',')
            print(f"host_error:{ec_flow}", end=',')
            print(f"此节点连接的错误的占全部连接比率:{r2}", end=',')
            print(f"此节点连接的错误的占自身连接比率:{r3}", end=',')
            print(f"host_count:{cc_time}", end=',')
            print(f"host_count_rate:{r4}", end=',')
            print(f"host_error:{ec_time}", end=',')
            print(f"此节点连接的错误的占全部连接比率:{r5}", end=',')
            print(f"此节点连接的错误的占自身连接比率:{r6}")

            print(
                f"{formatted_time},{duration},{len(encoded_message)},{cc_flow},{r1},{ec_flow},{r2}，{r3},{cc_time},{r4},{ec_time},{r5},{r6}")

            with open(f"result{node}.txt", "a") as output_file:
                #     # output_file.write(duration + )
                output_file.write(
                    f"{formatted_time},{duration},{len(encoded_message)},{cc_flow},{r1},{ec_flow},{r2},{r3},{cc_time},{r4},{ec_time},{r5},{r6}\n")
        except Exception as e:
            print(f"采集失败: {e}")
            time.sleep(0.3)


class Blockchain:
    def __init__(self, oneself, interval):
        self.interval = interval
        self.is_mined = None
        self.oneself = oneself
        self.updater = oneself[0]
        MAX_THREADS = len(lists_all) - 1
        self.server_socket = {}
        # 随机数用来产生区块
        self.random_dict = {}
        self.money = {}
        self.balance = {}
        for i, _ in lists_all.items():
            self.money[i] = 10
            self.balance[i] = 1
        for i in ONESELF:
            self.server_socket[i] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket[i].bind(lists_all[i])        # 绑定套接字到主机端口，address为一个元组(‘0.0.0.0’, 12345)
            self.server_socket[i].listen(MAX_THREADS+5)     # 监听客户端连接，参数为最大连接数量

        self.chain = []  # 存块
        self.current_transactions = []  # 交易实体
        self.chain_hash_list = []
        # 创建创世区块
        self.new_block(validator='0', previous_hash='1')
        self.collect = Collect()

    def node(self):
        miner = threading.Thread(target=self.mine)
        miner.start()
        # for active_threads in range(len(lists_my)):
        # 列表中有几个节点，就开几个线程接收数据
        for i in ONESELF:
            send_handler = threading.Thread(target=self.data_recv, args=(self.server_socket[i], self.updater))
            send_handler.start()
            time.sleep(0.5)  # 调整这个时间以适应你的需求

# 菜单中的东西仍然有待测试
#     def menu(self):
#         while True:
#             choice = None
#             print('------请选择------')
#             print('----1:转账交易----')
#             print('----2:发送消息----')
#             print('----3:查看链条----')
#             print('----4:更改押币----')
#             print('----5:查询押币----')
#             choice = input('----输入数字编号----:\n')
#             if choice == '1':
#                 self.transfer(self.oneself[0])
#
#             elif choice == '2':  # 发送消息
#                 self.send_message()
#
#             elif choice == '3':  # 查看链条
#                 self.show_chain()
#
#             elif choice == '4':
#                 self.change()
#
#             elif choice == '5':
#                 self.show_balance()
#             else:
#                 print('编号无效,请检查你的输入')

    def menu(self):
        while True:
            time.sleep(2)
            casual = random.randint(0, 100000)
            if (casual % 3 == 1):
                transfer = threading.Thread(target=self.transfer, args=[self.oneself])
                transfer.start()
                # self.transfer(self.oneself)
            elif (casual % 3 == 2):
                send_message = threading.Thread(target=self.send_message)
                send_message.start()
                # self.send_message()
            elif (casual % 3 == 3):
                change = threading.Thread(target=self.change)
                change.start()
                # self.change()

    def transfer(self, oneself):
        ID_self = self.random_item(oneself)
        to_node = self.random_item(lists_my)
        message = self.new_transaction(time.time(), ID_self, to_node, 0)
        self.data_send_all(message)

    def mine(self):
        while True:
            # 大概每二十秒挖一次
            tim = int(time.time())
            time.sleep(0.3)
            if tim % self.interval == 0:
                time.sleep(1)
                self.chain_hash_list.clear()
                validator = self.select_validator_all()  # 人数大于一半
                if validator in self.oneself:
                    # 发送者为 "0" 表明是新挖出的币,为矿工提供奖励
                    self.new_transaction(
                        time_stamp=time.time(),
                        sender="0",
                        recipient=validator,
                        amount=1,
                    )
                    # 打包旧区块，生成一个新块
                    block = self.new_block(validator, None)
                    response = {
                        'message': "打包成功，新区块已生成！",
                        'index': block['index'],
                        'transactions': block['transactions'],
                        'validator': block['validator'],
                        'previous_hash': block['previous_hash'],
                        'time_stamp': block['time_stamp']
                    }
                    pprint(response, indent=4)
                    message = {'identifier': 50, 'block': block, 'sender': validator}
                    self.data_send_all(message,validator)
                    print("-----sending block-----")
                    print('打包成功，新链已生成并发送给其他节点！')
                    time.sleep(1)
                self.random_dict = {}
            elif tim % self.interval == 0.75 * self.interval and len(self.chain_hash_list) != 0:
                # 求出出现频率最高的,先提取出来hash
                print(self.chain_hash_list)
                chain_hash_list = []
                for hash in self.chain_hash_list:
                    chain_hash_list.append(hash['chain_hash'])
                print(chain_hash_list)
                counter = Counter(chain_hash_list)
                most_common_element, occurrence_count = counter.most_common(1)[0]
                # print(f'most_common_element:{most_common_element}')
                print(f'most_common_element:')
                for hash in self.chain_hash_list:
                    if hash['chain_hash'] == most_common_element:
                        message = {
                            'identifier': 53,
                            'sender': self.oneself[0],
                            'message': '我需要你的整体链',
                            'recipient': hash['sender']
                        }
                        self.ready_send(message, message['recipient'])
                        break
                # self.chain_hash_list.clear()  写在前面的if中
    def select_validator_all(self):
        for node_name in self.oneself:
            select = threading.Thread(target=self.select_validator, args=(node_name,))
            select.start()

        print("开始挑选节点")
        # 等待一会儿，收集信息
        # 出块后清空 self.random_dict = {}
        while True:
            tim = int(time.time())
            time.sleep(0.3)
            if (tim % self.interval >= 0.25 * self.interval and len(self.random_dict) * 2 > len(lists_all)):
                break
            if len(self.random_dict) == len(lists_all):
                break
        selected_node = None
        accumulated_stake = 0.0  # 根据随机数和股权产生node
        total_stake = 0.0
        for node in self.random_dict.keys():
            print(node, self.balance[node])
            total_stake += self.balance[node]

        if len(self.random_dict) == 0:
            random_average = 0.0
        else:
            random_average = sum([self.random_dict[node] for node in self.random_dict]) / len(self.random_dict)
        print(f'入选人数:{len(self.random_dict)},total_stake: {total_stake}')

        # 这个平均数并不能保证每个人的概率一样，靠近0.5的概率高,所以在进行一次处理
        random_average = self.hash_to_decimal(random_average)
        print(f'random_average:{random_average}')

        # 排序是为了保持各节点统一
        random_sorted = collections.OrderedDict(sorted(self.random_dict.items()))
        for node in random_sorted:
            if total_stake == 0.0:
                break
            accumulated_stake += self.balance[node] / total_stake
            if accumulated_stake >= random_average:
                selected_node = node
                break

        # 给出块人利息
        self.money[selected_node] += 1
        print(f'selected_node: {selected_node},给出块人{selected_node}利息:1')
        return selected_node

    def select_validator(self, node_name):
        # 按照质押金额比例随机选择验证者
        if self.balance[node_name] > 0:
            message = {'identifier': 30,
                       'time_stamp': time.time(),
                       'random_number': random.randint(0, 99999),
                       'sender': node_name
                       }
            self.random_dict[node_name] = message['random_number']
            self.data_send_all(message, identity_name=node_name)
            print(f'{node_name}的数字为{message["random_number"]}')
        else:
            print(f"----{node_name}无质押币----")

    def new_block(self, validator, previous_hash=None):  # 新建区块
        if previous_hash is not None:
            block = {
                'index': len(self.chain) + 1,
            }
        else:
            block = {
                'index': len(self.chain) + 1,
                'time_stamp': time.time(),
                'transactions': self.current_transactions,
                'transactions_root': self.merkle_root(self.current_transactions),  # Merkle根
                'validator': validator,
                'validator_stake': self.balance[validator],  # 验证者的质押金额
                'previous_hash': self.hash(self.last_block),
            }
        self.current_transactions = []  # 新建区块打包后重置当前交易信息
        while True:
            try:
                self.chain[block["index"] - 1] = block
                print(f'添加成功{block["index"]}')
                break
            except Exception as e:
                print("添加区块")
                self.chain.append('xxxx')
                continue
        # self.chain.append(block)  # 把新建的区块加入链
        return block

    def send_message(self):
        ID_self = self.random_item(ONESELF)
        to_node = self.random_item(lists_my)
        message = {'identifier': 20, "time_stamp": time.time(), 'message': '', 'sender': ID_self,
                   'recipient': to_node}
        message['message'] = self.generate_random_string(random.randint(1, 554))
        self.data_send_all(message)

    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def show_chain(self):
        response = {
            'length': len(self.chain),
            'chain': self.chain,
        }
        # print(f'length: {len(self.chain)}')
        pprint(self.chain, indent=4)

    def change(self):
        ID_self = self.random_item(ONESELF)
        message = {'identifier': 40, 'sender': ID_self, 'balance': self.balance, 'money': self.money,
                   "time_stamp": time.time()}
        self.data_send_all(message)

    def show_balance(self):
        pprint(self.balance, indent=4)

    def ready_send_chain(self, node, message):
        '''发送链'''
        pass

    # 与正常节点有所不同
    def data_send_all(self, message, identity_name=None):
        if identity_name is not None:
            for ID, _ in lists_my.items():
                ready_send = threading.Thread(target=self.ready_send, args=(message, ID, identity_name))
                ready_send.start()
        else:
            for ID, _ in lists_my.items():
                ready_send = threading.Thread(target=self.ready_send, args=(message, ID))
                ready_send.start()

    def ready_send(self, message, tonode, identity=None):
        '''noed:"A","B","C"'''
        message['is_correct'] = 1
        if identity is not None:
            message['sender'] = identity
        message = json.dumps(message)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                # 连接
                client_socket.connect(lists_my[tonode])

                connected = True
                # 发送
                while True:
                    try:
                        encoded_message = message.encode('utf-8')
                        client_socket.sendall(encoded_message)
                        print(f'已发送')
                        client_socket.close()
                        break
                    except OSError as e:
                        if e.errno == 10038:
                            print(f"发送失败：{e}")
                            break
                        else:
                            print(f"发送失败：{e}")
            except Exception as e:
                if connected:
                    break
                if int(time.time()) % self.interval >= 0.9 * self.interval:
                    break
                time.sleep(0.2)

    # 只设置一个节点可以更新区块,
    def data_recv(self, server_socket, node_name):
        while True:
            try:
                start_time = time.time()
                client_socket, client_address = server_socket.accept()  # 接受连接，返回套接字和地址
                # print(client_socket, client_address)
            except Exception as e:
                print(f"接收失败1: {e}")
                time.sleep(0.3)
                continue
            encoded_message = None
            try:
            # if True:
                encoded_message = client_socket.recv(4096 * 8)
                decoded_message = encoded_message.decode('utf-8')
                received_data = json.loads(decoded_message)
                if 'sender' not in received_data.keys():
                    received_data['sender'] = ('unknown', 'unknown')
                if 'identifier' not in received_data:
                    received_data['identifier'] = 0
                # self.collect.collecting(self.oneself, client_address=lists_all[received_data['sender']],start_time=start_time, encoded_message=encoded_message, error=False)
                pprint(received_data, indent=4)

                # 交易阶段
                if 'identifier' in received_data and received_data['identifier'] == 10:
                    print('running deal_10')
                    self.deal_10(received_data,node_name)

                # 接收消息
                elif 'identifier' in received_data and received_data['identifier'] == 20:
                    print('running deal_20')
                    self.deal_20(received_data,node_name)

                # 平均数随机选择
                elif 'identifier' in received_data and received_data['identifier'] == 30:
                    print('running deal_30')
                    self.deal_30(received_data, node_name)

                # 通知其他区块修改质押币
                elif 'identifier' in received_data and received_data['identifier'] == 40:
                    print('running deal_40')
                    self.deal_40(received_data, node_name)

                # 校验，接收区块
                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 50:
                    print('running deal_50')
                    self.consensus(received_data["block"], self.updater)

                # 给请求的人发链哈希
                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 511:
                    print('running deal_511,发送链，被用于统计')
                    self.deal_511(received_data,self.updater)

                #
                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 512:
                    print('running deal_512，请求链，用于统计')
                    self.deal_511(received_data,self.updater)

                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 52:
                    print('running deal_52，收集链，用于统计')
                    self.deal_52(received_data,self.updater)

                # 发送链
                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 53:
                    print('running deal_53，发送链，被用于替换')
                    self.deal_53(received_data, self.updater)

                # 接收链
                elif node_name == self.updater and 'identifier' in received_data and received_data["identifier"] == 54:
                    print('running deal_54,接收链')
                    self.deal_54(received_data, self.updater)

                client_socket.close()
            except Exception as e:
                # 然后在使用前检查
                if encoded_message is not None:
                    # 安全地使用变量
                    self.collect.collecting(self.oneself, client_address, start_time, encoded_message, error=True)
                else:
                    self.collect.collecting(self.oneself, client_address, start_time, 'Reading error', error=True)
                print(f"接收失败2: {e}")
                time.sleep(0.3)

    def deal_10(self, message, node_name):
        message = self.new_transaction(message['time_stamp'], message['sender'],
                                       message['recipient'], message['amount'])
        self.money[message['sender']] -= message['amount']
        self.money[message['recipient']] += message['amount']
        print(f'{message["sender"]}转账给{message["recipient"]}{message["amount"]}')

    def deal_20(self, message,node_name):
        print(f'{message["sender"]}发送给{message["recipient"]}消息{message["message"]}')

    def deal_30(self, message,node_name):
        self.random_dict[message['sender']] = message['random_number']
        print(f"{message['sender']}的数字为{message['random_number']}")

    def deal_40(self, message,node_name):
        self.balance = message['balance']
        self.money = message['money']
        sender = message['sender']
        print(f"{sender}修改质押币为{message['balance'][sender]},余下{message['money'][sender]}币")

    def deal_511(self, message, node_name):
        '''给请求的人发链哈希'''
        print(f"收到{message['sender']}请求链hash")
        chain_hash = {}
        for block in self.chain[2:]:
            chain_hash[block['index'] - 1] = block['previous_hash']
        chain_hash[self.chain[-1]['index']] = self.hash(self.chain[-1])
        chain_hash = json.dumps(chain_hash)
        chain_hash = hash(chain_hash)
        message = {
            'identifier': 52,
            'chain_hash': chain_hash,
            # 默认第一个
            'sender': node_name,
            'recipient': message['sender']
        }
        self.ready_send(message, message['recipient'], identity=node_name)

    def deal_512(self, message, node_name):
        message = {'identifier': 511,
                   # 默认第一个
                   'sender': node_name,
                   'message': "请求你的链哈希"}
        time.sleep(1)
        self.data_send_all(message)

    def deal_52(self, message,node_name):
        # 先收集，留到mine函数统计
        self.chain_hash_list.append(message)
        print('已收集:', len(self.chain_hash_list))

    def deal_53(self, message,node_name):
        print(f"向{message['sender']}发送整条链")
        recipient = message['sender']
        for i in range(1, len(self.chain)):
            message = {
                'identifier': 54,
                'index': self.chain[i + 1]['index'],
                'block': self.chain[i + 1],
                'sender': message['recipient'],
                'recipient': recipient
            }
            self.ready_send(message, recipient, message['recipient'])

    def deal_54(self, message,node_name):
        while True:
            try:
                print(f'接收来自{message["sender"]}的链块{message["index"]}')
                self.chain[message["index"] - 1] = message['block']
                print(f'替换成功{message["index"]}')
                break
            except Exception as e:
                print(f"54接收块时发送错误:{e}\n修复错误")
                self.chain.append('xxxx')
                continue

    def consensus(self, block, node_name):
        try:
            print('running consensus')
            add = self.resolve_conflicts(block, node_name)

            if add:
                # response = {'message': '有最新链生成，已替换', 'new_chain': self.chain}
                print('有最新块生成，已添加,len:', len(self.chain))
            else:
                response = {'message': '当前链符合要求', 'chain': self.chain}
                pprint(f'当前链符合要求,len: {len(self.chain)}', indent=4)
        except Exception as e:
            print(f"consensus: {e}")

    def resolve_conflicts(self, block,node_name):
        """共识算法，解决冲突，以最长且有效的链为主"""
        try:
            if block in self.chain:
                return True
            length = block['index']
            last_block = self.chain[-1]  # 获取当前链长度
            if last_block['index'] + 1 < block['index']:
                # 自己链短，请求其他节点给自己发新的
                message = {'identifier': 511,
                           'sender': node_name,
                           'message': "请求你的链哈希"}
                time.sleep(1)
                self.data_send_all(message)
                return False
            if last_block['index'] + 1 < block['index']:
                # 自己链长，给出块人发消息，告诉他它的链有问题
                message = {'identifier': 512,
                           'sender': node_name,
                           'message': '你的链短，有问题'}
                self.data_send_all(message)
                return False
            if self.vaild_chain(block):
                self.chain.append(block)
                return True
            else:
                return False
        except Exception as e:
            print(f"resolve_conflicts: {e}")

    def vaild_chain(self, block):
        """验证链是否合理：最长且有效
            index过大，说明本节点缺少，通知这个区块的出块人给我发来上一个区块
            index过小，说明出块人的链长短，那么把我自己的发给"""
        try:
            last_block = self.chain[-1]  # 只验证最后一个区块是否可以接续

            # 如果当前区块的前哈希和前一个计算出来的哈希值不同则是无效链
            if block['previous_hash'] != self.hash(last_block) and last_block['index'] != 1:
                print(f'----前区块哈希值有误,出块人{block["validator"]}质押币减一{self.balance[block["validator"]]}-->',
                      end=' ')
                self.balance[block['validator']] = self.balance[block["validator"]] - 1
                print(self.balance[block['validator']])
                return False

            if block['transactions_root'] != self.merkle_root(block['transactions']):
                print(
                    f'----前区块merkle_root有误,出块人{block["validator"]}质押币减一{self.balance[block["validator"]]}-->',
                    end=' ')
                self.balance[block['validator']] = self.balance[block["validator"]] - 1
                print(self.balance[block['validator']])
                return False

            print('----验证链确实合理----')
            return True
        except Exception as e:
            print(f"vaild_chain: {e}")

    def new_transaction(self, time_stamp, sender, recipient, amount):
        self.current_transactions.append({
            'time_stamp': time_stamp,
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        })
        # identifier为任务编号，用来识别是什么任务
        message = {
            'identifier': 10,
            "time_stamp": time_stamp,
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }
        return message

    def hash(self, block):
        """计算哈希值,返回哈希后的摘要信息"""
        block_string = json.dumps(block, sort_keys=True)
        block_string = block_string.encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

    @property  # 只读属性
    def last_block(self) -> Dict[str, Any]:  # 获取当前链中最后一个区块
        return self.chain[-1]

    def hash_to_decimal(self, input_number, digits=10):
        # 将数字转换为字符串，然后进行哈希计算
        hashed_str = hashlib.md5(str(input_number).encode()).hexdigest()

        # 取哈希结果的前10位
        hashed_truncated = int(hashed_str, 16)
        hashed_str = str(hashed_truncated)[-digits:]
        decimal_value = float(hashed_str)
        # 将这个整数转化为小数
        decimal_value = decimal_value / (10 ** digits)
        return decimal_value

    def random_item(self, dic_list):
        if dic_list and isinstance(dic_list, dict):
            return random.choice(list(dic_list.keys()))
        if dic_list and isinstance(dic_list, list):
            return random.choice(dic_list)
        raise ValueError("不能为空")

    def merkle_root(self, transactions):
        print('merkle_root is running')
        # 如果没有交易，则返回空字符串
        if not transactions:
            return ""

        hash_list = []
        # 对交易进行排序，确保每次生成相同的Merkle根
        transactions = sorted(transactions, key=lambda transaction: transaction['time_stamp'])
        for transaction in transactions:
            # 将交易内容按固定顺序排序并拼接成字符串
            serialized_string = json.dumps(transaction)

            # 计算哈希值
            tx_hash = hashlib.sha256(serialized_string.encode()).hexdigest()

            # 将哈希值添加到列表中
            hash_list.append(tx_hash)

        # 将交易哈希列表转换为字节形式的列表，以便进行哈希运算
        transaction_hashes_bytes = [bytes.fromhex(hash) for hash in hash_list]

        # 初始化Merkle树的层级
        current_layer = transaction_hashes_bytes

        while len(current_layer) > 1:
            # 创建下一层级
            next_layer = []

            # 对于当前层级中的每一对哈希（包括处理奇数个的情况）
            for i in range(0, len(current_layer), 2):
                # 如果是奇数个，则复制最后一个哈希与自己配对
                if i + 1 == len(current_layer):
                    combined_hash = hashlib.sha256(transaction_hashes_bytes[i] + transaction_hashes_bytes[i]).digest()
                else:
                    combined_hash = hashlib.sha256(
                        transaction_hashes_bytes[i] + transaction_hashes_bytes[i + 1]).digest()

                # 将新的哈希添加到下一层级（以字节形式）
                next_layer.append(combined_hash)

            # 更新当前层级为下一层级
            current_layer = next_layer

        # 最后一层只剩下一个元素时，将其转换为十六进制字符串形式作为Merkle树根
        merkle_root_hex = current_layer[0].hex()

        return merkle_root_hex


# lists_all = {'A': ('127.0.0.1', 1), 'B': ('127.0.0.1', 2), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4),
#              'E': ('127.0.0.1', 5), 'F': ('127.0.0.1', 6), 'G': ('127.0.0.1', 7), 'H': ('127.0.0.1', 8),
#              'I': ('127.0.0.1', 9), 'J': ('127.0.0.1', 10), 'K': ('127.0.0.1', 11), 'L': ('127.0.0.1', 12),
#              'M': ('127.0.0.1', 13), 'N': ('127.0.0.1', 14), 'P': ('127.0.0.1', 15)}
lists_all = {'A': ('127.0.0.1', 1), 'B': ('127.0.0.1', 2), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4),
             'E': ('127.0.0.1', 5), 'F': ('127.0.0.1', 6), 'G': ('127.0.0.1', 7), 'H': ('127.0.0.1', 8),
             'I': ('127.0.0.1', 9), 'J': ('127.0.0.1', 10), 'K': ('127.0.0.1', 11), 'L': ('127.0.0.1', 12)}
# 这表关系到可以给谁发信息，与接受信息无关
# lists_all = {'A': ('127.0.0.1', 12), 'B': ('127.0.0.1', 42), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4), 'E': ('127.0.0.1', 5),'P': ('127.0.0.1', 15), 'Q': ('127.0.0', 16)}
lists_my = lists_all.copy()

ONESELF = ['H', 'I', 'J', 'K', 'L']
for i in range(len(ONESELF)):
    lists_my.pop(ONESELF[i])
MAX_THREADS = len(lists_my)
lists_all = {k: lists_all[k] for k in sorted(lists_all)}  # 排序
list_keys = list(lists_my.keys())

blockchain = Blockchain(oneself=ONESELF, interval=5)
blockchain.node()
blockchain.menu()


# # 打开或创建文件（以写入模式 'w'，会覆盖原有内容；或追加模式 'a'，在文件末尾添加内容）
# with open("output.txt", "w") as output_file:
#     # 写入数据
#     output_file.write(data_to_write)
#
# # 如果是多行数据，可以使用换行符 '\n' 分隔
# multiline_data = ["Line 1", "Line 2", "Line 3"]
# with open("output.txt", "w") as output_file:
#     for line in multiline_data:
#         output_file.write(line + "\n")