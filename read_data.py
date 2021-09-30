import scipy.io
import torch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import re
import itertools
from graph_handler import *
from tqdm import tqdm
import os
import sys
import numpy
import csv
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import matplotlib.cm as cm

def open_data(path):
    # open file
    similarity_matrix = scipy.io.loadmat(path)
    similarity_matrix = similarity_matrix["corr"]
    size = len(similarity_matrix)
    # print(similarity_matrix)

    # round to 1e-10 (clean the data)
    for i in range(size):
        for j in range(size):
            similarity_matrix[i][j] = round(similarity_matrix[i][j], 10)

    # print(similarity_matrix[0][0])

    # convert into dissimilarity matrix so that we can use MDS
    dissimilarity_matrix = []
    for i in range(size):
        dissimilarity_matrix.append([])
        for j in range(size):
            dissimilarity_matrix[i].append([])
    # print(dissimilarity_matrix)

    for i in range(size):
        for j in range(size):
            dissimilarity_matrix[i][j] = np.sqrt(
                similarity_matrix[i][i] + similarity_matrix[j][j] - 2 * similarity_matrix[i][j])
            # dissimilarity_matrix[i][j] = 1/similarity_matrix[i][j]

    # print(dissimilarity_matrix)
    mapping = MDS()
    result = mapping.fit_transform(similarity_matrix)
    # print(result)

    return size, result

'''
def read_sensor_data(ajr, cuda_flag, train=False, scale=False):
    problem_list = ['0', '1', '2', '3', '4']
    size_list = []
    opt_value = []
    test_g = []
    scales = []
    print('read sensor data...')
    for problem_num in tqdm(range(len(problem_list))):
        name = problem_list[problem_num]
        if train:
            path_x = os.path.abspath('.') + '/result/ColocationSensors/' + "train_sensor_" + name + '.csv'
            G = GraphGenerator(k=10, m=4, ajr=ajr)
        else:
            path_x = os.path.abspath('.') + '/result/ColocationSensors/' + "test_sensor_" + name + '.csv'
            G = GraphGenerator(k=10, m=4, ajr=ajr)
        size, x = ajr+1, []
        with open(path_x, 'rU') as p:
            x = list(list(map(float,rec)) for rec in csv.reader(p, delimiter=','))
        x = torch.FloatTensor(x)
        size_list.append(size)

        if scale:
            scales.append(torch.abs(x).max().numpy())
            test_g.append(G.generate_graph(x=0.5 + (x / (2 * scales[-1])), cuda_flag=cuda_flag))
        else:
            test_g.append(G.generate_graph(x=x, cuda_flag=cuda_flag))

    return test_g, problem_list, opt_value, size_list, scales
'''

"""
def read_sensor_data(problem_list, ajr, cuda_flag, train=False, scale=False):
    size_list = []
    x = []
    test_index, train_index = [], []
    print('read sensor data...')
    for problem_num in tqdm(range(len(problem_list))):
        problem_name = problem_list[problem_num]
        path_x = os.path.abspath('.') + '/result/ColocationSensors/' + "train_sensor_" + problem_name + '.csv'
        size = ajr+1
        test_group = random.randint(0,4)
        size_list.append(size)

        with open(path_x, 'rU') as p:
            csv_reader = csv.reader(p, delimiter=',')
            for i, line in enumerate(csv_reader):
                if i == test_group:
                    for item in line:
                        test_index.append(int(item))
                else:
                    for item in line:
                        train_index.append(int(item))
                if i == 4:
                    break

    return train_index, test_index
"""

def read_sensor_data(problem_list):
    size_list = []
    test_x, train_x = [], []
    print('read sensor data...')
    for problem_num in tqdm(range(len(problem_list))):
        test_x.append([])
        train_x.append([])
        problem_name = problem_list[problem_num]
        path_x = os.path.abspath('.') + '/result/ColocationSensors/' + "sensor_" + problem_name + '.csv'
        # size = ajr+1
        test_group = random.sample(range(0,40), 10)
        # print(test_group)
        # size_list.append(size)

        with open(path_x, 'rU') as p:
            csv_reader = csv.reader(p, delimiter=',')
            for i, line in enumerate(csv_reader):
                if i//4 in test_group:
                    # test_x[problem_num].append(list(map(int,item) for item in line))
                    test_x[problem_num].append(list(map(float, line)))
                else:
                    # train_x[problem_num].append(list(map(int,item) for item in line))
                    train_x[problem_num].append(list(map(float, line)))

    return train_x, test_x

"""
def gen_graphs(problem_list, x, k, m, ajr, cuda_flag, train=False, scale=False):
    # shuffle and return graph, we need to do this for every episode we train
    g = []
    for num, problem in enumerate(problem_list):
        path_x = os.path.abspath('.') + '/result/ColocationSensors/' + "train_sensor_" + problem + ".csv"
        scales = []
        x_data = []
        G = GraphGenerator(k=k, m=m, ajr=ajr)
        # print(x)
        if train:
            random.shuffle(x[num])
            x[num] = x[num][:10]
        # print(x)
        with open(path_x, 'rU') as p:
            csv_reader = csv.reader(p, delimiter=',')
            for i, line in enumerate(csv_reader):
                # print(i)
                if (i-5)//4 in x[num]:
                    # print(i)
                    x_data.append(list(map(float, line)))
        # print(x_data)
        x_data = torch.FloatTensor(x_data)
        if scale:
            scales.append(torch.abs(x_data).max().numpy())
            g.append(G.generate_graph(x=0.5 + (x_data / (2 * scales[-1])), cuda_flag=cuda_flag))
        else:
            g.append(G.generate_graph(x=x_data, cuda_flag=cuda_flag))
    return g
    
"""

def gen_graphs(x, k, m, ajr, cuda_flag, graph_generator, batch_size=1, train=False, scale=True):
    """
    shuffle and return graph, we need to do this for every episode we train
    """
    g = []
    G = graph_generator
    scales = []
    for num, problem in enumerate(x):
        x_data = []
        if train:
            train_group = random.sample(range(0,k*m), 10 * batch_size)
            # print(train_group)
            for index, item in enumerate(x[num]):
                    if index//4 in train_group:
                        x_data.append(item)
        else:
            for index, item in enumerate(x[num]):
                x_data.append(item)
        x_data = torch.FloatTensor(x_data)
        # print(x_data.shape)

        if scale:
            scales.append(torch.abs(x_data).max().numpy())
            g.append(G.generate_graph(x=0.5 + (x_data / (2 * scales[-1])), batch_size=batch_size, cuda_flag=cuda_flag))
        else:
            g.append(G.generate_graph(x=x_data, batch_size=batch_size, cuda_flag=cuda_flag))
    return g


def read_corr_data(ajr, cuda_flag, scale=False):
    problem_list = ['0','1','2','3','4', 'test']
    # should not use opt_value before we determine it
    opt_value = []

    # size_list = [int(x) for x in itertools.chain(*[re.findall(r"\d+\.?\d*", name) for name in problem_list])]
    # print(size_list)

    size_list = []

    test_g = []
    scales = []
    print('read corr data...')
    for problem_num in tqdm(range(len(problem_list))):
        name = problem_list[problem_num]
        path_x = os.path.abspath('.') + '/result/RelationalInferenceOutput/' + "corr_" + name + '.mat'
        size, x = open_data(path_x)
        x = torch.from_numpy(x).float()
        # print(x)
        size_list.append(size)
        G = GraphGenerator(k=10, m=4, ajr=ajr, style="cluster")

        if scale:
            scales.append(torch.abs(x).max().numpy())
            test_g.append(G.generate_graph(x=0.5 + (x / (2 * scales[-1])), cuda_flag=cuda_flag))
            data = 0.5 + (x / (2 * scales[-1]))
        else:
            test_g.append(G.generate_graph(x=x, cuda_flag=cuda_flag))
            data = x

        x_coor, y_coor = data.T
        plt.scatter(x_coor, y_coor)
        # plt.title("similarity")
        plt.show()

    return test_g, problem_list, opt_value, size_list, scales

def write_result(batch_size, k, m, test):
    # handling room accuracy
    avr_accuracy = 0
    avr_S = 0
    avr_initial_value = 0
    room_result = []
    correct_room = 0
    for j in range(k * batch_size):
        room_result.append([])
    # print(len(list_result))
    list_result = test.bg.ndata['label'].tolist()
    # list_result = [
    #     [0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],
    #     [1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
    #     [0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],
    #     [0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],
    #     [0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
    #     [0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],
    #     [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0],
    #     [0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,0],
    #     [0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1]]
    for j in range(batch_size):
        for l in range(k*m):
            curr_point = k*m*j+l
            room_number = list_result[curr_point].index(1.0)
            room_result[room_number+k*j].append(l)
    print(room_result)

    for j in range(len(room_result)):
        correct = True
        if room_result[j][0] % 4 != 0:
            correct = False
        else:
            for l in range(len(room_result[j])):
                if room_result[j][l] != room_result[j][0] + l:
                    correct = False
        if correct:
            correct_room += 1
    avr_initial_value += torch.mean(test.S).item()
    avr_accuracy = correct_room*100/(batch_size*k)
    return avr_accuracy, avr_initial_value

def save_fig(dataset):
    print_x = []
    print_y = []

    for i in range(len(dataset)//4):
        print_x.append([])
        print_y.append([])

    for index, item in enumerate(dataset):
        # print(index)
        print_x[index//4].append(float(item[0]))
        print_y[index//4].append(float(item[1]))

    colors = cm.rainbow(np.linspace(0, 1, len(print_y)))
    for x, y, c in zip(print_x, print_y, colors):
        plt.scatter(x, y, color=c)
        # plt.plot(x, y, color=c)

    plt.savefig("./result/output.png")


def save_fig_with_result(k, m, test_episode, x, test, n_iter):
    x_list = []
    y_list = []
    print_x = []
    print_y = []

    for j in range(k * test_episode):
        x_list.append([])
        y_list.append([])
        print_x.append([])
        print_y.append([])

    # x = MDS().fit_transform(x)
    # print(len(list_result))

    list_result = test.bg.ndata['label'].tolist()
    for index, item in enumerate(x):
        # print(index)
        print_x[index//4].append(float(item[0]))
        print_y[index//4].append(float(item[1]))
    # print("-----")
    # print(print_x)
        
    for j in range(test_episode):
        for l in range(k*m):
            curr_point = k*m*j+l
            room_number = list_result[curr_point].index(1.0)
            x_list[room_number+k*j].append(x[m*k*j+l][0])
            y_list[room_number+k*j].append(x[m*k*j+l][1])
    # print(x_list)
    # print("------")
    colors = cm.rainbow(np.linspace(0, 1, len(x_list)))

    for x, y, x_original, y_original, c in zip(x_list, y_list, print_x, print_y, colors):
        plt.scatter(x_original, y_original, color=c)
        plt.plot(x, y)

    plt.savefig("./result/output" + str(n_iter) + ".png")
    plt.figure().clear()

'''
Edit the .mat file
'''
# test_matrix = []
# for i in range(40):
#     test_matrix.append([])
#     room_num = i//4
#     for j in range(40):
#         if room_num*4 <= j < (room_num + 1)*4:
#             test_matrix[i].append(5+random.random())
#         else:
#             test_matrix[i].append(0)
# a = scipy.io.loadmat("result/RelationalInferenceOutput/corr_test.mat")
# a['corr'] = test_matrix
# scipy.io.savemat("result/RelationalInferenceOutput/corr_test5.mat", a)

# a = scipy.io.loadmat("result/RelationalInferenceOutput/corr_test.mat")
# print(a['corr'])
# a, b, c, d, e = read_sensor_data(39, True, True, True)
# print()

# print(read_sensor_data(['0']))

def ri_cal_room_acc(best_solution):
    pp, pn, np, nn = 0, 0, 0, 0 #(ground_truth, prediction)
    for i in range(len(best_solution)):
        for j in range(len(best_solution[i]) - 1):
            for k in range(j + 1, len(best_solution[i])):
                if(int(best_solution[i][j] / 4) == int(best_solution[i][k] / 4)):
                    pp += 1
                else:
                    pn += 1
                    np += 1
    nn = (len(best_solution) * len(best_solution[0])) * (len(best_solution) * len(best_solution[0]) - 1) /2 - pp - pn - np
    recall = pp / (pp + pn)
    acc_room = 0
    for i in range(len(best_solution)):
        r_id = int(best_solution[i][0] / 4)
        for j in range(1, 5):
            if j == 4:
                acc_room += 1
                break
            if int(best_solution[i][j] / 4) != r_id:
                break
    room_wise_acc = acc_room / len(best_solution)
    confusion = [[pp, np], [pn, nn]]
    return recall, room_wise_acc

