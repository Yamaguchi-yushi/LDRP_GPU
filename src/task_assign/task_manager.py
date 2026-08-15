from src.task_assign.task_policy.random import Random
from src.task_assign.task_policy.ppo import PPOAgent
from src.task_assign.task_policy.tp import TP

class TaskManager():
    def __init__(self, name, args=None):
        self.name = name
        self.debug = bool(getattr(args, "debug_task_assign", False))
        if name == "fifo":
            #print("call fifo")
            self.task_assigner = Random()
        elif name == "tp":
            #print("call TP")
            self.task_assigner = TP()
        elif name == "ppo":
            #print("call ppo")
            self.task_assigner = PPOAgent(args)
        else:
            raise ValueError(f"Unknown task assignment method: {name}")

    def assign_task(self, env):
        task_assign = self.task_assigner.assign_task(env)
        if self.debug and any(a != -1 for a in task_assign):
            self._print_debug(env, task_assign)
        return task_assign

    def _print_debug(self, env, task_assign):
        step = getattr(env, "step_account", "?")
        agent_num = getattr(env, "agent_num", len(task_assign))
        current_tasklist = getattr(env, "current_tasklist", [])
        assigned_list = getattr(env, "assigned_list", [])
        assigned_tasks = getattr(env, "assigned_tasks", [])
        current_start = getattr(env, "current_start", [None] * agent_num)
        goal_array = getattr(env, "goal_array", [None] * agent_num)

        print(f"[task_assign] step={step} method={self.name}")
        print("  current_tasklist (idx: [start,goal,time]  status):")
        if not current_tasklist:
            print("    (empty)")
        for j, task in enumerate(current_tasklist):
            owner = assigned_list[j] if j < len(assigned_list) else -1
            status = f"assigned_to=a{owner}" if owner != -1 else "free"
            print(f"    {j}: {task}  {status}")

        print("  agents (idx: start->goal  status):")
        for i in range(agent_num):
            busy = assigned_tasks[i] if i < len(assigned_tasks) else []
            status = f"busy(task={busy})" if busy else "free"
            s = current_start[i] if i < len(current_start) else "?"
            g = goal_array[i] if i < len(goal_array) else "?"
            print(f"    a{i}: {s}->{g}  {status}")

        pairs = ", ".join(
            f"a{i}<-t{t}" if t != -1 else f"a{i}<-none"
            for i, t in enumerate(task_assign)
        )
        print(f"  result: {task_assign}  ({pairs})")
