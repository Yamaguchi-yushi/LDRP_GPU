import copy

class TP():
    def __init__(self):
        pass

    def assign_task(self, env):
        current_tasklist = copy.deepcopy(env.current_tasklist)
        assigned_list = copy.deepcopy(env.assigned_list)
        assigned_tasks = copy.deepcopy(env.assigned_tasks)
        task_assign = []

        #current_tasklist = [current_tasklist[i] for i, task in enumerate(assigned_tasks) if task == -1]

        for i in range(env.agent_num):
            best_task = -1
            if getattr(env, "use_dynamic_agents", False) and (not env.active[i] or env.pending_off[i]):
                task_assign.append(best_task)
                continue
            if assigned_tasks[i] == [] and len(current_tasklist) > 0:
                shortest_path_length = float('inf')

                for j in range(len(current_tasklist)):
                    if assigned_list[j] != -1:
                        continue

                    path_length = env.get_path_length(env.goal_array[i], current_tasklist[j][0])

                    if shortest_path_length > path_length:
                        shortest_path_length = path_length
                        best_task = j

                assigned_list[best_task] = i

            task_assign.append(best_task)
                
        return task_assign
    
    """
    def assign_task(self, env, current_tasklist, assigned_tasklist):
        current_tasklist = copy.deepcopy(current_tasklist)
        assigned_tasklist = copy.deepcopy(assigned_tasklist)
        task_assign = []

        for i in range(env.agent_num):
            best_task = -1
            if len(assigned_tasklist[i]) == 0 and len(current_tasklist) > 0:
                shortest_path_length = float('inf')

                for j in range(len(current_tasklist)):
                    path_length = env.get_path_length(env.goal_array[i], current_tasklist[j][0])

                    if shortest_path_length > path_length:
                        shortest_path_length = path_length
                        best_task = j

                current_tasklist.pop(best_task)

            task_assign.append(best_task)
                
        return task_assign
    """
    