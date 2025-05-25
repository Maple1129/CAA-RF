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



class Blockchain:
    def __init__(self, oneself, interval, all_node):
        lists_my = all_node.copy()
        ONESELF = oneself
        lists_my.pop(ONESELF)
        self.lists_my = lists_my
        self.all_node = all_node
        self.interval = interval
        self.is_mined = None
        self.oneself = oneself
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        MAX_THREADS = len(all_node) - 1
        self.server_socket.bind(self.all_node[oneself])
        self.server_socket.listen(MAX_THREADS)

        self.chain = []
        self.current_transactions = []
        self.chain_hash_list = []
        self.random_dict = {}
        self.money = {}
        self.balance = {}
        for i in self.all_node:
            self.money[i] = 10
            self.balance[i] = 1

        # 创建创世区块
        self.new_block(validator='0', previous_hash='1')

    def node(self):
        miner = threading.Thread(target=self.mine)
        miner.start()
        send_handler = threading.Thread(target=self.data_recv)
        send_handler.start()
        time.sleep(0.5)  # 调整这个时间以适应你的需求

    def menu(self):
        while True:
            # 这个地方可以调整一下攻击的间隔时间
            time.sleep(0.0001)
            casual = random.randint(0, 100000)
            if(casual % 3 ==1):
                transfer = threading.Thread(target=self.transfer, args=self.oneself)
                transfer.start()
                # self.transfer(self.oneself)
            elif(casual %3 == 2):
                send_message = threading.Thread(target=self.send_message)
                send_message.start()
                # self.send_message()
            elif(casual % 3 == 3):
                change = threading.Thread(target=self.change)
                change.start()
                # self.change()

    def mine(self):
        while True:
            # 大概每二十秒挖一次
            tim = int(time.time())
            time.sleep(0.3)
            if tim % self.interval == 0:
                time.sleep(1)
                self.chain_hash_list.clear()
                validator = self.select_validator()  # 人数大于一半
                if validator == self.oneself:
                    # 发送者为 "0" 表明是新挖出的币,为矿工提供奖励
                    self.new_transaction(
                        timestamp=time.time(),
                        sender="0",
                        recipient=self.oneself,
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
                        'timestamp': block['timestamp']
                    }
                    pprint(response, indent=4)
                    message = {'sender': self.oneself,'identifier': 50, 'block': block}
                    self.data_send_all(message)
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
                print(f'most_common_element:')
                for hash in self.chain_hash_list:
                    if hash['chain_hash'] == most_common_element:
                        message = {
                            'identifier': 53,
                            'sender': self.oneself,
                            'message': '我需要你的整体链',
                            'recipient': hash['sender']
                        }
                        self.ready_send(message, message['recipient'])
                        break

    def select_validator(self):
        # 按照质押金额比例随机选择验证者
        if self.balance[self.oneself] > 0:
            message = {'identifier': 30,
                       'random_number': random.randint(0, 99999),
                       'sender': self.oneself
                       }
            self.random_dict[self.oneself] = message['random_number']
            self.data_send_all(message)
            print(f'{self.oneself}的数字为{message["random_number"]}')
        else:
            print("----无质押币----")

        print("开始挑选节点")
        while True:
            tim = int(time.time())
            time.sleep(0.3)
            if (tim % self.interval >= 0.25  * self.interval and len(self.random_dict) * 2 > len(self.all_node)):
                break
            if len(self.random_dict) == len(self.all_node):
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

        # 二次处理
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


    def new_block(self, validator, previous_hash=None):  # 新建区块
        if previous_hash is not None:
            block = {
                'index': len(self.chain) + 1,
            }
        else:
            block = {
                'index': len(self.chain) + 1,
                'timestamp': time.time(),
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
        return block

    def transfer(self, oneself):
        to_node = self.random_item(self.lists_my)
        message = self.new_transaction(time.time(), oneself, to_node, 0)
        # 把发送节点“sender”伪装一下
        message['sender'] = self.random_item(self.all_node)
        self.data_send_all(message)

    def send_message(self):
        to_node = self.random_item(self.lists_my)
        message = {'identifier': 20, 'message': '解析错误,无法显示', 'sender': self.oneself, 'recipient': to_node, "timestamp": time.time()}
        # 把发送节点“sender”伪装一下
        message['sender'] = self.random_item(self.all_node)
        message['message'] = self.generate_random_string(random.randint(1, 554))
        self.data_send_all(message)

    def change(self):
        message = {'identifier': 40, 'sender': self.oneself, 'balance': self.balance, 'money': self.money, "timestamp": time.time()}
        # 把发送节点“sender”伪装一下
        message['sender'] = self.random_item(self.all_node)
        self.data_send_all(message)

    def show_chain(self):
        response = {
            'length': len(self.chain),
            'chain': self.chain,
        }
        pprint(self.chain, indent=4)

    def show_balance(self):
        pprint(self.balance, indent=4)



    def data_send_all(self, message):
        for ID, _ in self.lists_my.items():
            ready_send = threading.Thread(target=self.ready_send, args=(message, ID))
            ready_send.start()

    def ready_send(self, message, to_node):
        message['is_correct'] = 1
        message = json.dumps(message)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                client_socket.connect(self.lists_my[to_node])
                print(f'已连通{to_node}')
                connected = True
                while True:
                    try:
                        encoded_message = message.encode('utf-8')
                        client_socket.sendall(encoded_message)
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
        client_socket.close()

    def data_recv(self):
        while True:
            try:
                start_time = time.time()
                client_socket, client_address = self.server_socket.accept()  # 接受连接，返回套接字和地址
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
                pprint(received_data, indent=4)

                # 交易阶段
                if 'identifier' in received_data and received_data['identifier'] == 10:
                    print('running deal_10')
                    self.deal_10(received_data)

                # 接收消息
                elif 'identifier' in received_data and received_data['identifier'] == 20:
                    print('running deal_20')
                    self.deal_20(received_data)

                # 平均数随机选择
                elif 'identifier' in received_data and received_data['identifier'] == 30:
                    print('running deal_30')
                    self.deal_30(received_data)

                # 通知其他区块修改质押币
                elif 'identifier' in received_data and received_data['identifier'] == 40:
                    print('running deal_40')
                    self.deal_40(received_data)

                # 校验，接收区块
                elif 'identifier' in received_data and received_data["identifier"] == 50:
                    print('running deal_50')
                    self.consensus(received_data["block"])

                # 给请求的人发链哈希
                elif 'identifier' in received_data and received_data["identifier"] == 511:
                    print('running deal_511,发送链，被用于统计')
                    self.deal_511(received_data)

                #
                elif 'identifier' in received_data and received_data["identifier"] == 512:
                    print('running deal_512，请求链，用于统计')
                    self.deal_511(received_data)

                elif 'identifier' in received_data and received_data["identifier"] == 52:
                    print('running deal_52，收集链，用于统计')
                    self.deal_52(received_data)

                # 发送链
                elif 'identifier' in received_data and received_data["identifier"] == 53:
                    print('running deal_53，发送链，被用于替换')
                    self.deal_53(received_data)

                # 接收链
                elif 'identifier' in received_data and received_data["identifier"] == 54:
                    print('running deal_54,接收链')
                    self.deal_54(received_data)

                client_socket.close()
            except Exception as e:
                print(f"接收失败2: {e}")
                time.sleep(0.3)

    def deal_10(self, message):
        message = self.new_transaction(message['timestamp'], message['sender'],
                                       message['recipient'], message['amount'])
        self.money[message['sender']] -= message['amount']
        self.money[message['recipient']] += message['amount']
        print(f'{message["sender"]}转账给{message["recipient"]}{message["amount"]}')

    def deal_20(self, message):
        print(f'{message["sender"]}发送给{message["recipient"]}消息{message["message"]}')

    def deal_30(self, message):
        self.random_dict[message['sender']] = message['random_number']
        print(f"{message['sender']}的数字为{message['random_number']}")

    def deal_40(self, message):
        self.balance = message['balance']
        self.money = message['money']
        sender = message['sender']
        print(f"{sender}修改质押币为{message['balance'][sender]},余下{message['money'][sender]}币")

    def deal_511(self, message):
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
            'sender': self.oneself,
            'recipient': message['sender']
        }
        self.ready_send(message, message['recipient'])

    def deal_512(self, message):
        message = {'identifier': 511,
                   'sender': self.oneself,
                   'message': "请求你的链哈希"}
        time.sleep(1)
        self.data_send_all(message)

    def deal_52(self, message):
        # 先收集，留到mine函数统计
        self.chain_hash_list.append(message)
        print('已收集:', len(self.chain_hash_list))

    def deal_53(self, message):
        print(f"向{message['sender']}发送整条链")
        recipient = message['sender']
        for i in range(1, len(self.chain)):
            message = {
                'identifier': 54,
                'index': self.chain[i + 1]['index'],
                'block': self.chain[i + 1],
                'sender': self.oneself,
                'recipient': recipient
            }
            self.ready_send(message, recipient)

    def deal_54(self, message):
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

    def consensus(self, block):
        try:
            print('running consensus')
            add = self.resolve_conflicts(block)

            if add:
                # response = {'message': '有最新链生成，已替换', 'new_chain': self.chain}
                print('有最新块生成，已添加,len:', len(self.chain))
            else:
                response = {'message': '当前链符合要求', 'chain': self.chain}
                pprint(f'当前链符合要求,len: {len(self.chain)}', indent=4)
        except Exception as e:
            print(f"consensus: {e}")

    def resolve_conflicts(self, block):
        """共识算法，解决冲突，以最长且有效的链为主"""
        try:
            length = block['index']
            last_block = self.chain[-1]  # 获取当前链长度
            if last_block['index'] + 1 < block['index']:
                # 自己链短，请求其他节点给自己发新的
                message = {'identifier': 511,
                           'sender': self.oneself,
                           'message': "请求你的链哈希"}
                time.sleep(1)
                self.data_send_all(message)
                return False
            if last_block['index'] + 1 < block['index']:
                # 自己链长，给出块人发消息，告诉他它的链有问题
                message = {'identifier': 512,
                           'sender': self.oneself,
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

    def merkle_root(self, transactions):
        print('merkle_root is running')
        # 如果没有交易，则返回空字符串
        if not transactions:
            return ""

        hash_list = []
        # 对交易进行排序，确保每次生成相同的Merkle根
        transactions = sorted(transactions, key=lambda transaction: transaction['timestamp'])
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

    def new_transaction(self, timestamp, sender, recipient, amount):
        self.current_transactions.append({
            'timestamp': timestamp,
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        })
        # identifier为任务编号，用来识别是什么任务
        message = {
            'identifier': 10,
            "timestamp": timestamp,
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
        hashed_str = hashlib.md5(str(input_number).encode()).hexdigest()
        hashed_truncated = int(hashed_str, 16)
        hashed_str = str(hashed_truncated)[-digits:]
        decimal_value = float(hashed_str)
        decimal_value = decimal_value / (10 ** digits)
        return decimal_value

    def random_item(self, dic_list):
        if dic_list and isinstance(dic_list, dict):
            return random.choice(list(dic_list.keys()))
        if dic_list and isinstance(dic_list, list):
            return random.choice(dic_list)
        raise ValueError("不能为空")

    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))






# lists_all = {'A': ('127.0.0.1', 1), 'B': ('127.0.0.1', 2), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4),
#              'E': ('127.0.0.1', 5), 'F': ('127.0.0.1', 6), 'G': ('127.0.0.1', 7), 'H': ('127.0.0.1', 8),
#              'I': ('127.0.0.1', 9), 'J': ('127.0.0.1', 10), 'K': ('127.0.0.1', 11), 'L': ('127.0.0.1', 12),
#              'M': ('127.0.0.1', 13), 'N': ('127.0.0.1', 14), 'P': ('127.0.0.1', 15), 'Q': ('127.0.0', 16)}
lists_all = {'A': ('127.0.0.1', 1), 'B': ('127.0.0.1', 2), 'C': ('127.0.0.1', 3), 'D': ('127.0.0.1', 4),
             'E': ('127.0.0.1', 5), 'F': ('127.0.0.1', 6), 'G': ('127.0.0.1', 7)}
lists_my = lists_all.copy()

ONESELF = 'G'
lists_my.pop(ONESELF)
MAX_THREADS = len(lists_my)
lists_all = {k: lists_all[k] for k in sorted(lists_all)}
list_keys = list(lists_my.keys())

blockchain = Blockchain(oneself=ONESELF, interval=5, all_node=lists_all)
blockchain.node()
blockchain.menu()