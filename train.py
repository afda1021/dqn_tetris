import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""") # パーサを作る
    # parser.add_argumentで受け取る引数を追加していく
    # ('--xxx')オプション引数（指定しなくても良い引数）を追加
    parser.add_argument("--width", type=int, default=10, help="The common width for all images") #幅 #コマンドライン引数で指定しなければwidth=10
    parser.add_argument("--height", type=int, default=20, help="The common height for all images") #高さ
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block") #1ブロック(ピクセル)のサイズ
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)  # エポック数
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000, # 最大記録数30000
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args() # 引数を解析
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size) #高さ、幅、1ブロックの大きさを指定
    model = DeepQNetwork()  #インスタンス生成
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()  # 初期状態　tensor([0., 0., 0., 0.])
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)  #maxで30000、
    epoch = 0
    while epoch < opt.num_epochs:  # 指定したエポック数繰り返す
        #1ピース目の取りうる全ての行動に対して、それぞれ状態を計算  {(左から何番目か,何回転か):tensor([,,,]),*n}
        next_steps = env.get_next_states()
        # εグリーディー的なやつ
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random() # 0～1
        random_action = u <= epsilon  # True, False

        next_actions, next_states = zip(*next_steps.items()) #next_stepsのkeyとvalueを取得　#( , )*n
        next_states = torch.stack(next_states) # tensor([[ , , , ],*n])
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]  #DeepQNetworkのforward #tensor([,～,])これはそれぞれの行動に対するQ値のようなもの
        model.train()
        # next_stepsのインデックスをランダムor最適で指定
        if random_action:  # ランダムな行動
            index = randint(0, len(next_steps) - 1)
        else:  # 最適な行動(最大のpredictionsに基づく)
            index = torch.argmax(predictions).item()

        # 行動と次の状態を決定
        next_state = next_states[index, :]  #ある行動を選択したときの次の状態 #tensor([ , , , ])
        action = next_actions[index]  #行動 #(左から何番目か,何回転か)

        reward, done = env.step(action, epoch, render=True) #行動を実行、報酬(スコア)を求める、溢れた場合done=True、描画

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done]) #deque([[tensor([0., 0., 0., 0.]), 1, tensor([0., 0., 2., 4.]), False]],..., maxlen=30000)

        if done:  # 溢れた場合 or 上限100手
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()  # 初期状態　tensor([0., 0., 0., 0.])
            if torch.cuda.is_available():
                state = state.cuda()
        else:  # 溢れてない場合
            state = next_state  # 状態を更新  tensor([0., 1., 2., 5.])とか
            continue  #while epoch～に戻る
        if len(replay_memory) < opt.replay_memory_size / 1000:  #溢れた場合判定(累計ピースが3000以下ならcontinue)
            continue
        
        # 累計ピースが3000に到達した後、溢れる毎に以下を実行
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size)) #replay_memoryからbatch_size個ランダムに取り出す(len(replay_memory) < opt.batch_sizeのときはlen(replay_memory)個取り出す)
        
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))  #tensor([[0., 26., 16., 62.],*batch_size個])
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])  #tensor([[1.],*batch_size個])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))  #tensor([[0., 32., 13., 72.],*batch_size個])

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)  #予測Q値、q_values=tensor([[0.1810],*batch_size個])
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)  #次の状態に対する予測Q値
        model.train()
        # Q値の正解値を更新式で求める
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()  #最適化アルゴリズム
        loss = criterion(q_values, y_batch)  #損失関数はmse、q_values:予測値、y_batch:正解値
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                epoch,
                opt.num_epochs,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))

# python train.pyで実行した場合__name__は"__main__"という文字列になる
# (import trainした場合__name__は"train"という文字列になる)
if __name__ == "__main__":
    opt = get_args()
    #print(opt.width)  # 10
    train(opt)
