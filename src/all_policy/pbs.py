import gym
import math
import numpy as np
import random
from typing import Tuple, List
from dataclasses import dataclass
from collections import deque
from collections import defaultdict

@dataclass
class AgentInfo:
    pos: Tuple[float, float]
    current_start: int
    current_goal: int
    action_history: List[int]
    pos_history: List[Tuple[float,float]]
    step_account: int = 0

@dataclass
class OtherAgentsInfo:
    #各エージェントの位置をstep数分格納
    other_agents_pos: List[List[Tuple[float,float]]]

class PBS:
    def __init__(self, args):
        self.env = gym.make("drp_env:drp-" + str(1) + "agent_" + args.map_name + "-v2",
                             state_repre_flag = "onehot_fov",
                             task_flag = False,
                             use_lare_path = False,
                             use_lare_task = False,
                             pbs_mode = True,
                             )
        self.num_agents = args.agent_num
        self.time_limit = getattr(args, "time_limit", 1000)
        self.schedule_actions = []
        self.goal_rec = []
        self.priority_rec = []
        self.change_rec = [] #ノード上にいるエージェントを記録するリスト（優先度の変更に使う）
        self.no_change_rec = [] #エッジ上にいるエージェントを記録
        self.tmp_goal_rec = [-1 for _ in range(self.num_agents)]


    #優先度を決める（優先度リストを作る）
    #優先度順に最短経路を決定
    #最短経路の求め方
    #環境の初期化（goal_arrayの設定）
    #dequeの先頭のデータをとる(popleft)
    #agentのデータを環境にセットする
    #avail_actionの数だけstepを実行，ゴールしてたなら@へ
    #衝突判定がないものでdataclassのインスタンスを作り，データを更新，dequeへ(append)
    #@action_historyを参照してschedule_actionへ追加，pos_historyを参照してposを更新
    def culc_actions(self, obs ,env):
        #AAAA
        #tmp = 0
        self.schedule_actions = [[] for _ in range(self.num_agents)]
        #今：step × agent_num
        #other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.time_limit - env.step_account)])
        #改：agent_num × step
        
        #tmp_list = [[] for _ in range(self.time_limit+1)]
        #other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.num_agents)])
        other_agents_infos = OtherAgentsInfo(other_agents_pos=[[[] for _ in range(self.time_limit + 1)]
                                                               for _ in range(self.num_agents)])

        #ノード上にいないエージェントの対応
        self.fill_non_nodes_agents_pos_history(env, other_agents_infos)
        
        priority_list = self.get_priority(obs, env)
        self.priority_rec = priority_list.copy()
        index = 0
        recal_count = 0
        #print("priority_list", priority_list)
        while index < len(priority_list):
            i = priority_list[index]
            index += 1

            agent_info = AgentInfo(pos=(env.obs[i][0], env.obs[i][1]), 
                                   current_start=env.current_start[i], 
                                   current_goal=env.current_goal[i], 
                                   action_history=[], 
                                   pos_history=[(env.obs[i][0], env.obs[i][1])],
                                   step_account=0)
            
            near_goal_nodes = self.env.get_near_nodes(env.goal_array[i])
            while self.schedule_actions[i] == []:
                #解がない場合，そのエージェントを最優先にして再計算
                #agent_num回再計算したらどうする？
                #current_goalに固定，Noneならcurrent_start
                if len(near_goal_nodes) == 0:
                    """
                    if i in self.change_rec:
                        self.change_rec.remove(i)
                        self.change_rec.insert(0, i)
                    elif i in self.no_change_rec:
                        self.no_change_rec.remove(i)
                        self.no_change_rec.insert(0,i)
                    
                    priority_list = self.change_rec.copy() + self.no_change_rec.copy()
                    """

                    priority_list.remove(i)
                    priority_list.insert(0,i)
                    self.priority_rec = priority_list.copy()
                    #print("priority_list", priority_list)
                    index = 0
                    recal_count += 1
                    
                    if recal_count >= self.num_agents:
                        #print("再計算回数超過",i)
                        for j in range(self.num_agents):
                            if env.current_goal[j] is not None:
                                self.schedule_actions[j] = [env.current_goal[j]] * self.time_limit
                            else:
                                self.schedule_actions[j] = [env.current_start[j]] * self.time_limit
                    else:
                        self.schedule_actions = [[] for _ in range(self.num_agents)]
                    other_agents_infos = OtherAgentsInfo(other_agents_pos=[[[] for _ in range(self.time_limit + 1)]
                                                               for _ in range(self.num_agents)])
                    self.fill_non_nodes_agents_pos_history(env, other_agents_infos)
                    break
                tmp_goal = near_goal_nodes.pop(0)
                #print("agent", i, "try goal", tmp_goal)

                agent_infos_deque = deque([agent_info])
                current_agent_info = agent_info

                self.env.reset()
                goal_flag = False
                #重複する条件のための処理
                step_account_check = agent_info.step_account
                visitted_states = set()

                while not goal_flag and len(agent_infos_deque) > 0 and current_agent_info.step_account+1 < self.time_limit:
                    current_agent_info = agent_infos_deque.popleft()
                    #AAAA
                    #tmp += 1

                    #重複する条件のための処理
                    if step_account_check != current_agent_info.step_account:
                        visitted_states = set()
                        step_account_check = current_agent_info.step_account

                    self.env.reset()
                    self.set_env_info(current_agent_info, tmp_goal)

                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    for action in avail_actions:

                        self.set_env_info(current_agent_info, tmp_goal)

                        obs, reward, done, info = self.env.step([action])
                        new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                        #重複する条件のための処理
                        new_state = (round(new_pos[0]), round(new_pos[1]), action)
                        if new_state in visitted_states:
                            continue
                        visitted_states.add(new_state)

                        collision_flag = self.collision_detect(new_pos, other_agents_infos.other_agents_pos, current_agent_info.step_account+1, i)
                        
                        if collision_flag == True:#衝突とゴールが同時の場合にバグが起こるためリセット
                            #self.env.reset()
                            pass

                        elif collision_flag == False:
                            new_action_history = current_agent_info.action_history + [action]
                            new_pos_history = current_agent_info.pos_history + [new_pos]
                            new_current_start = self.env.current_start[0]
                            new_current_goal = self.env.current_goal[0]

                            new_agent_info = AgentInfo(pos=new_pos, 
                                                    current_start=new_current_start, 
                                                        current_goal=new_current_goal, 
                                                    action_history=new_action_history, 
                                                    pos_history=new_pos_history,
                                                    step_account=current_agent_info.step_account + 1)
                            
                            if all(done) is True:
                                if info["goal"] == True:
                                    #ゴールした後，そこに止まったときに優先度の高いエージェントと衝突しないか見るのを実装
                                    #衝突する場合，continue
                                    #print("ゴール発見")
                                    if self.check_after_collision(new_agent_info, other_agents_infos.other_agents_pos, i) == True:
                                        #print("ゴール後衝突発見")
                                        continue
                                    else:
                                        #print("ゴール後衝突なし")
                                        pass
                                    """
                                    print("AAAAAA",new_agent_info.step_account)
                                    print(current_agent_info.pos, other_agents_infos.other_agents_pos[current_agent_info.step_account])
                                    print(new_agent_info.pos_history)
                                    print(other_agents_infos.other_agents_pos)
                                    """
                                    #今：各ステップ毎に格納（）
                                    #改：各エージェント毎に格納（i番目に入れるだけ）
                                    self.schedule_actions[i] = new_agent_info.action_history + new_agent_info.action_history[-1:]*(self.time_limit - len(new_agent_info.action_history))

                                    for _ in range(self.time_limit - len(new_agent_info.pos_history)+1):
                                        new_agent_info.pos_history.append(new_agent_info.pos_history[-1])

                                    other_agents_infos.other_agents_pos[i] = new_agent_info.pos_history
                                    
                                    goal_flag = True
                                    
                                    break

                                elif info["collision"] == True:
                                    pass
                                elif info["timeup"] == True:
                                    #print("timeup!!!!!!")
                                    pass

                            else:
                                agent_infos_deque.append(new_agent_info)

            ###
            #print(other_agents_infos.other_agents_pos)
        #AAAA
        #print(other_agents_infos.other_agents_pos)
        self.goal_rec = env.goal_array.copy()
        self.tmp_goal_rec = [self.schedule_actions[i][-1] for i in range(self.num_agents)]
        
    ###############################################
    def fill_non_nodes_agents_pos_history(self, env, other_agents_infos: OtherAgentsInfo) -> None:

        for i in range(self.num_agents):
            # ノード上にいない場合，ノードにつくまでの pos を other_agents_infos に格納
            if env.pos[env.current_start[i]] != [env.obs[i][0], env.obs[i][1]]:
                agent_info = AgentInfo(
                    pos=(env.obs[i][0], env.obs[i][1]),
                    current_start=env.current_start[i],
                    current_goal=env.current_goal[i],
                    action_history=[],
                    pos_history=[(env.obs[i][0], env.obs[i][1])],
                )

                self.env.reset()
                self.set_env_info(agent_info, env.goal_array[i])

                while self.env.pos[self.env.current_start[0]] != [self.env.obs[0][0], self.env.obs[0][1]]:
                    self.set_env_info(agent_info, env.goal_array[i])
                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    action = avail_actions[0]

                    obs, reward, done, info = self.env.step([action])

                    new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                    agent_info = AgentInfo(
                        pos=new_pos,
                        current_start=self.env.current_start[0],
                        current_goal=self.env.current_goal[0],
                        action_history=agent_info.action_history + [action],
                        pos_history=agent_info.pos_history + [new_pos],
                        step_account=0,
                    )

                agent_info.pos_history.extend([agent_info.pos_history[-1]]*1)
                agent_info.pos_history.extend([] for _ in range(self.time_limit - len(agent_info.pos_history) + 1))

                other_agents_infos.other_agents_pos[i] = agent_info.pos_history

    ###############################################
    def get_priority(self, obs, env):
        priority_list = []
        change_list = []#ノード上にいるエージェントを記録するリスト
        no_change_list = []
        """
        for i in range(self.num_agents):
            path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
            priority_list.append((i, path_length))
        #遠いものを優先
        #priority_list.sort(key=lambda x: x[1], reverse=True)
        #近いものを優先
        priority_list.sort(key=lambda x: x[1], reverse=False)
        priority_list = [x[0] for x in priority_list]
        return priority_list
        """
        #ノード上にいるエージェントを最後に追加
        if self.priority_rec == []:
            for i in range(self.num_agents):
                path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
                priority_list.append((i, path_length))
            #遠いものを優先
            #priority_list.sort(key=lambda x: x[1], reverse=True)
            #近いものを優先
            priority_list.sort(key=lambda x: x[1], reverse=False)
            priority_list = [x[0] for x in priority_list]
            return priority_list
        else:
            priority_list = self.priority_rec.copy()
            no_change_list = self.priority_rec.copy()
            for i in range(self.num_agents):
                if env.pos[env.current_start[i]] == [env.obs[i][0], env.obs[i][1]]:
                    priority_list.remove(i)
                    priority_list.append(i)
                    change_list.append(i)
                    no_change_list.remove(i)

            self.change_rec = change_list.copy()
            self.no_change_rec = no_change_list.copy()
            
            return priority_list
        
    
    #エージェントのxy座標(self.obs)環境にセットする
    #obs=tuple(array[],array[]...)
    def set_env_info(self, agent_infos, goal_array):
        self.env.set_1agent_info(pos=agent_infos.pos, 
                                 current_start=agent_infos.current_start, 
                                 current_goal=agent_infos.current_goal, 
                                 goal_array = goal_array,)
        return
    
    #衝突判定
    #step_tのときに，agent_posとother_agents_posが衝突するか
    #1ステップ前後も見ることで，取り替えるエージェントとの衝突を防ぐ
    def collision_detect(self, agent_pos, other_agents_pos, step_t, agent_num):
        speed = 5
        collision_flag = False
        
        for i in range(len(other_agents_pos)):
            if i==agent_num or other_agents_pos[i][step_t] == []:
                continue
            
            distance = math.dist(agent_pos, other_agents_pos[i][step_t])
            distance_before = math.dist(agent_pos, other_agents_pos[i][step_t-1])
            distance_after = speed+1
            if len(other_agents_pos[i])>step_t+1 and other_agents_pos[i][step_t+1] != []:
                distance_after = math.dist(agent_pos, other_agents_pos[i][step_t+1])

            if distance<speed or distance_before<speed or distance_after<speed:
                collision_flag = True

        return collision_flag
    
    #passを見つけた後，その後に衝突する恐れがないか確認
    #2ステップだけ確認
    def check_after_collision(self, agent_info, other_agents_pos, agent_num):
        collision_flag = False
        """
        for t in range(len(agent_info.pos_history)):
            if self.collision_detect(agent_info.pos_history[t], other_agents_pos, t,  agent_num):
                collision_flag = True
                break
        """
        for t in range(len(agent_info.pos_history), len(agent_info.pos_history)+2):
            if self.collision_detect(agent_info.pos_history[-1], other_agents_pos, t,  agent_num):
                collision_flag = True
                break
        return collision_flag

    def policy(self, obs, env):
        actions = []
        #schedule_actionsが変なとき，初期化
        if self.schedule_actions != []:
            for i in range(self.num_agents):
                if self.schedule_actions[i] == []:
                    self.schedule_actions = []
                    break
            
        #だれかのゴールが変わったとき，計算し直す
        if self.goal_rec != env.goal_array:
            self.schedule_actions = []

        #ゴール付近のノードで妥協している場合，いつ再計算をするか
        #案1：tmpで途中のものを指定した時，そこについたら再計算：だれかのcurrent_startがtmp_goalと一致した時
        ##その場に止まるものは除かない（除いたほうがいいかも）
        for i in range(self.num_agents):
            if env.current_start[i] == self.tmp_goal_rec[i]:
                self.schedule_actions = []
                break

        #案2：なくなるまで実行：なにも書かない
        

        if self.schedule_actions == []:
            #print("新たに計算")
            self.culc_actions(obs, env)
            #print("schedule_actions", self.schedule_actions)
            #return [0 for _ in range(self.num_agents)]
            
        for i in range(self.num_agents):
            actions.append(self.schedule_actions[i].pop(0))

        return actions