import numpy as np

# GUI 依存 (tk / matplotlib tkagg) を遅延 import 化. headless サーバでも
# `import drp_env` がコケないようにし, GUI を実際に使うときだけエラーにする.
_GUI_IMPORT_ERROR = None
try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('tkagg')
except Exception as _e:  # ImportError も RuntimeError (backend 不在) も拾う
    _GUI_IMPORT_ERROR = _e

class GUI_tasklist():
    def __init__(self):
        if _GUI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "GUI_tasklist は GUI 環境 (tkinter + matplotlib tkagg) が必要ですが, "
                f"この環境では利用できません: {_GUI_IMPORT_ERROR}"
            )
        self.root = tk.Tk()
        self.root.title("Simulation with Task Manager")
        self.init_gui()
        
    def init_gui(self):
        # Create the main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for task management
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right frame for task management
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # GUI for task management in left frame
        self.assigned_task_label = ttk.Label(left_frame, text="Tasks Assigned")
        self.assigned_task_label.pack(padx=10, pady=10)

        self.assigned_task_listbox = tk.Listbox(left_frame, height=20, width=30)
        self.assigned_task_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # GUI for task management in right frame (empty column)
        self.task_label = ttk.Label(right_frame, text="Tasks to Assign")
        self.task_label.pack(padx=10, pady=10)

        self.task_listbox = tk.Listbox(right_frame, height=20, width=30)
        self.task_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def update_gui(self):
        """Updates the GUI elements with the current state."""
        self.populate_assigned_task_listbox()
        self.root.update_idletasks()

    def populate_assigned_task_listbox(self, agent_num, assigned_tasklist):
        # Clear the current content
        self.assigned_task_listbox.delete(0, tk.END)

        # Insert tasks into the assigned_task_listbox
        for i in range(agent_num):
            if len(assigned_tasklist[i])>0:
                task = [assigned_tasklist[i][0], assigned_tasklist[i][1]]
            else:
                task = [-1,-1]

            self.assigned_task_listbox.insert(tk.END, f"agetn{i}: {task[0]} -> {task[1]}")

    def populate_task_listbox(self, current_tasklist, assigned_list):
        self.task_listbox.delete(0, tk.END)
        for i in range(len(current_tasklist)):
            if assigned_list[i] == -1:
                self.task_listbox.insert(tk.END, f"{current_tasklist[i][0]} -> {current_tasklist[i][1]}")
            else:
                self.task_listbox.insert(tk.END, f"{current_tasklist[i][0]} -> {current_tasklist[i][1]} : assigned to agent{assigned_list[i]}")

    def show_tasklist(self, agent_num, assigned_tasklist, current_tasklist, assigned_list):
        self.populate_assigned_task_listbox(agent_num, assigned_tasklist)
        self.populate_task_listbox(current_tasklist, assigned_list)
        self.root.update_idletasks()