import copy

class Random():
    def __init__(self):
        pass

    def assign_task(self, env):
        current_tasklist = copy.deepcopy(env.current_tasklist)
        assigned_list = copy.deepcopy(env.assigned_list)
        assigned_tasks = copy.deepcopy(env.assigned_tasks)
        task_assign = []
        #current_tasklist = [current_tasklist[i] for i, task in enumerate(assigned_tasks) if task == -1]
        #print("1:",current_tasklist)
        #print("2:",assigned_list)
        #print("3:",assigned_tasks)

        task_idx = 0
        for i in range(env.agent_num):
            if getattr(env, "use_dynamic_agents", False) and (not env.active[i] or env.pending_off[i]):
                task_assign.append(-1)
                continue
            if assigned_tasks[i] == [] and len(current_tasklist)-task_idx > 0 :
                while task_idx < len(current_tasklist):
                    if assigned_list[task_idx] == -1:
                        task_assign.append(task_idx)
                        task_idx += 1
                        break
                    else:
                        task_idx += 1
            else:
                task_assign.append(-1)

        return task_assign

    """
    def assign_task(self, env, current_tasklist, assigned_tasklist):
        task_assign = []
        current_tasklist = current_tasklist.copy()
        assigned_tasklist = assigned_tasklist.copy()

        for i in range(env.agent_num):
            task = 0
            if len(assigned_tasklist[i]) == 0 and len(current_tasklist)>0 :
                task_assign.append(task)
                current_tasklist.pop(0)
                task += 1
            else:
                task_assign.append(-1)


        return task_assign
    """