import gym
import numpy as np
import sys
import copy
import os


from drp_env.state_repre import REGISTRY
from drp_env.EE_map import MapMake
from drp_env.gui_task import GUI_tasklist

sys.path.append(os.path.join(os.path.dirname(__file__), ''))

class DrpEnv(gym.Env):
	def __init__(self,
			agent_num,
			speed,
			start_ori_array,
			goal_array,
			visu_delay,
			state_repre_flag,
			time_limit,
			collision,
			map_name="map_3x3",
			reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
			task_flag=True,
			task_list = None,
			use_lare_path=True,
			use_lare_path_training=True,
			lare_path_factor_dim=10,
			lare_path_decoder_hidden_dim=64,
			lare_path_decoder_n_layers=3,
			lare_path_use_transformer=False,
			lare_path_transformer_heads=4,
			lare_path_transformer_depth=2,
			lare_path_buffer_capacity=512,
			lare_path_min_buffer=256,
			lare_path_update_freq=3,
			lare_path_batch_size=32,
			lare_path_lr=5e-4,
			use_pretrained_lare_path=False,
			pretrained_lare_path_model_path=None,
			use_finetuning_lare_path=False,
			finetuning_lare_path_model_path=None,
			lare_path_autosave=True,
			lare_path_autosave_path=None,
			lare_path_save_dir=None,
			# --- LaRe-Task (System B) ---
			use_lare_task=False,
			use_lare_task_training=True,
			lare_task_factor_dim=10,
			lare_task_decoder_hidden_dim=64,
			lare_task_decoder_n_layers=2,
			lare_task_buffer_capacity=512,
			lare_task_min_buffer=256,
			lare_task_update_freq=3,
			lare_task_batch_size=32,
			lare_task_lr=5e-4,
			use_pretrained_lare_task=False,
			pretrained_lare_task_model_path=None,
			use_finetuning_lare_task=False,
			finetuning_lare_task_model_path=None,
			lare_task_autosave=False,
			lare_task_autosave_path=None,
			lare_task_save_dir=None,
		  ):
		self.agent_num = agent_num
		self.n_agents = agent_num # for epymarl
		self.state_repre_flag = state_repre_flag
		self.map_name = map_name
		self.speed = speed
		self.visu_delay = visu_delay
		self.start_ori_array = start_ori_array
		self.goal_array = goal_array

		# reward
		self.r_goal = reward_list["goal"]
		self.r_coll = reward_list["collision"]
		self.r_wait = reward_list["wait"]
		self.r_move = reward_list["move"]

		# collision machnism
		self.collision = collision

		self.time_limit = time_limit

		self.colli_distan_value = self.speed
		self.r_flag = 0
		self.flag_indicate = 0
		self.episode_account = 0

		# for tasklist
		self.task_completion = 0

		self.distance_from_start = np.zeros(self.agent_num)

		# create ee_env and pass self.variable
		self.ee_env = MapMake(self.agent_num, self.start_ori_array, self.goal_array, self.map_name)
		self.pos = self.ee_env.pos
		self.start_ori_array = self.ee_env.start_ori_array
		self.goal_array = self.ee_env.goal_array
		self.G = self.ee_env.G
		self.edge_labels = self.ee_env.edge_labels # unused

		self.current_goal  = [ None for i in range(self.agent_num)]

		self.obs_manager = REGISTRY[self.state_repre_flag](self)

		# create gym-like mdp elements
		self.n_nodes = len(self.G.nodes)
		self.n_actions = self.n_nodes
		self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.n_nodes)] * self.agent_num))

		obs_box = self.obs_manager.get_obs_box()
		self.observation_space = gym.spaces.Tuple(tuple([obs_box] * self.agent_num))

		self.log = {}

		# flag for tasklist
		self.is_tasklist = task_flag
		self.current_tasklist=[]
		self.assigned_tasks=[]#エージェントが割り当てられたタスク(未ピックを含む)
		self.assigned_list=[]#未実行のタスクとエージェントの割り当て表
		self.task_num = self.agent_num*2 # for tasklist, each agent can have 2 tasks at most
		self.alltasks = task_list

		if self.is_tasklist:
			self.ee_env.task_flag_on()

		# --- LaRe-Path (System A) ---
		# Mirrors Safe-TSL-DBCT's 4 modes:
		#   (1) use_lare_path=False                                                  -> baseline (no LaRe)
		#   (2) use_lare_path=True (pretrained=False, finetuning=False)              -> train online from scratch
		#   (3) use_lare_path=True, use_pretrained_lare_path=True                    -> load + freeze (inference only)
		#   (4) use_lare_path=True, use_finetuning_lare_path=True                    -> load + continue training
		# Pretrained takes precedence over finetuning if both are set.
		self.use_lare_path = bool(use_lare_path)
		self.use_lare_path_training = bool(use_lare_path_training)
		self.use_pretrained_lare_path = bool(use_pretrained_lare_path)
		self.pretrained_lare_path_model_path = pretrained_lare_path_model_path
		self.use_finetuning_lare_path = bool(use_finetuning_lare_path)
		self.finetuning_lare_path_model_path = finetuning_lare_path_model_path
		self.lare_path_autosave = bool(lare_path_autosave)
		self.lare_path_autosave_path = lare_path_autosave_path
		self.lare_path_save_dir = lare_path_save_dir
		self.lare_path_module = None
		self._lare_prev_onehot_pos = None
		self._lare_current_colliding_pairs = None
		# Cumulative step counter (NOT reset between episodes) — used to derive
		# the "{N.N}M" steps token in saved-model filenames, mirroring Safe-TSL-DBCT.
		self._lare_total_step_account = 0

		# --- LaRe-Task (System B) ---
		self.use_lare_task = bool(use_lare_task)
		self.use_lare_task_training = bool(use_lare_task_training)
		self.use_pretrained_lare_task = bool(use_pretrained_lare_task)
		self.pretrained_lare_task_model_path = pretrained_lare_task_model_path
		self.use_finetuning_lare_task = bool(use_finetuning_lare_task)
		self.finetuning_lare_task_model_path = finetuning_lare_task_model_path
		self.lare_task_autosave = bool(lare_task_autosave)
		self.lare_task_autosave_path = lare_task_autosave_path
		self.lare_task_save_dir = lare_task_save_dir
		self.lare_task_module = None
		# Parallel to current_tasklist: per-task creation step (set when task added).
		self._lare_task_creation_steps = []

		if self.use_lare_path:
			self._init_lare_path(
				factor_dim=lare_path_factor_dim,
				decoder_hidden_dim=lare_path_decoder_hidden_dim,
				decoder_n_layers=lare_path_decoder_n_layers,
				use_transformer=lare_path_use_transformer,
				transformer_heads=lare_path_transformer_heads,
				transformer_depth=lare_path_transformer_depth,
				buffer_capacity=lare_path_buffer_capacity,
				min_buffer=lare_path_min_buffer,
				update_freq=lare_path_update_freq,
				batch_size=lare_path_batch_size,
				learning_rate=lare_path_lr,
			)

		if self.use_lare_task:
			self._init_lare_task(
				factor_dim=lare_task_factor_dim,
				decoder_hidden_dim=lare_task_decoder_hidden_dim,
				decoder_n_layers=lare_task_decoder_n_layers,
				buffer_capacity=lare_task_buffer_capacity,
				min_buffer=lare_task_min_buffer,
				update_freq=lare_task_update_freq,
				batch_size=lare_task_batch_size,
				learning_rate=lare_task_lr,
			)

		#for rendering
		#if self.is_tasklist:
		#	self.taskgui=GUI_tasklist()

	# ---------------- LaRe-Path naming helpers (Safe-TSL-DBCT convention) ----------------
	def _lare_repo_root(self):
		return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

	def _lare_default_save_dir(self):
		return self.lare_path_save_dir or os.path.join(self._lare_repo_root(), "src", "lare", "path", "saved_models")

	def _lare_get_safe_prefix(self):
		"""Return "Safe" if running under SafeEnv (drp_safe-...), else ""."""
		return "Safe" if self.__class__.__name__ == "SafeEnv" else ""

	def _lare_get_algorithm_name(self):
		"""Detect algorithm name from CLI (--config=qmix etc.) — uppercased.

		Falls back to the configured path_planner-like hint, then "UNKNOWN".
		"""
		try:
			argv = list(sys.argv)
			for i, a in enumerate(argv):
				if a == "--config" and i + 1 < len(argv):
					return str(argv[i + 1]).upper()
				if a.startswith("--config="):
					return a.split("=", 1)[1].upper()
		except Exception:
			pass
		return "UNKNOWN"

	def _lare_get_steps_str(self):
		"""Format cumulative step count as "X.YM" (Safe-TSL-DBCT style)."""
		steps_in_millions = self._lare_total_step_account / 1_000_000
		return f"{steps_in_millions:.1f}M"

	def _lare_get_source_base_name(self):
		"""For finetuning: strip directories, ".pth", "_final"/"_checkpoint", and leading "FT_"."""
		path = self.finetuning_lare_path_model_path
		if not path:
			return "unknown_source"
		try:
			name = os.path.basename(str(path))
			if name.endswith(".pth"):
				name = name[:-4]
			name = name.replace("_final", "").replace("_checkpoint", "")
			while name.startswith("FT_"):
				name = name[3:]
			return name or "unknown_source"
		except Exception:
			return "unknown_source"

	def _lare_build_save_filename(self, suffix):
		"""Assemble filename for LaRe-Path checkpoints.

		Scratch:    "{Safe_}{ALGO}_PATH_{map}_{N}agents_{S}M_{suffix}.pth"
		Finetuning: "FT_{Safe_}{source_base}_{map}_{N}agents_{S}M_{suffix}.pth"
		The "Safe_" prefix is omitted entirely for non-Safe envs (no leading "_").
		The PATH token disambiguates path-system models from LaRe-Task ones; folders
		(src/lare/path/saved_models/ vs src/lare/task/saved_models/) provide the
		primary separation.
		"""
		safe = self._lare_get_safe_prefix()
		safe_seg = f"{safe}_" if safe else ""
		algo = self._lare_get_algorithm_name()
		map_name = getattr(self, "map_name", "unknown_map")
		agents = getattr(self, "agent_num", "?")
		steps = self._lare_get_steps_str()
		if self.use_finetuning_lare_path and self.finetuning_lare_path_model_path:
			source = self._lare_get_source_base_name()
			return f"FT_{safe_seg}{source}_{map_name}_{agents}agents_{steps}_{suffix}.pth"
		return f"{safe_seg}{algo}_PATH_{map_name}_{agents}agents_{steps}_{suffix}.pth"

	def _lare_resolve_autosave_path(self):
		"""Decide the autosave path for the current run.

		Priority:
		  1. explicit `lare_path_autosave_path` (used as-is)
		  2. `lare_path_autosave=True` -> auto-generated filename under `lare_path_save_dir`
		  3. None (autosave disabled)
		"""
		if self.lare_path_autosave_path:
			return self.lare_path_autosave_path
		if self.lare_path_autosave:
			fname = self._lare_build_save_filename("checkpoint")
			return os.path.join(self._lare_default_save_dir(), fname)
		return None

	# ---------------- LaRe-Task naming helpers (Safe-TSL-DBCT convention, _LARETASK suffix) ----------------
	def _lare_task_default_save_dir(self):
		return self.lare_task_save_dir or os.path.join(self._lare_repo_root(), "src", "lare", "task", "saved_models")

	def _lare_task_get_source_base_name(self):
		path = self.finetuning_lare_task_model_path
		if not path:
			return "unknown_source"
		try:
			name = os.path.basename(str(path))
			if name.endswith(".pth"):
				name = name[:-4]
			name = name.replace("_final", "").replace("_checkpoint", "")
			while name.startswith("FT_"):
				name = name[3:]
			return name or "unknown_source"
		except Exception:
			return "unknown_source"

	def _lare_task_build_save_filename(self, suffix):
		"""Filename for LaRe-Task models. TASK token + folder separation distinguishes from path models.

		Scratch:    "{Safe_}{ALGO}_TASK_{map}_{N}agents_{S}M_{suffix}.pth"
		Finetuning: "FT_{Safe_}{source_base}_{map}_{N}agents_{S}M_{suffix}.pth"
		"""
		safe = self._lare_get_safe_prefix()
		safe_seg = f"{safe}_" if safe else ""
		algo = self._lare_get_algorithm_name()
		map_name = getattr(self, "map_name", "unknown_map")
		agents = getattr(self, "agent_num", "?")
		steps = self._lare_get_steps_str()
		if self.use_finetuning_lare_task and self.finetuning_lare_task_model_path:
			source = self._lare_task_get_source_base_name()
			return f"FT_{safe_seg}{source}_{map_name}_{agents}agents_{steps}_{suffix}.pth"
		return f"{safe_seg}{algo}_TASK_{map_name}_{agents}agents_{steps}_{suffix}.pth"

	def _lare_task_resolve_autosave_path(self):
		if self.lare_task_autosave_path:
			return self.lare_task_autosave_path
		if self.lare_task_autosave:
			fname = self._lare_task_build_save_filename("checkpoint")
			return os.path.join(self._lare_task_default_save_dir(), fname)
		return None

	def _init_lare_path(self, factor_dim, decoder_hidden_dim, decoder_n_layers,
						use_transformer, transformer_heads, transformer_depth,
						buffer_capacity, min_buffer, update_freq, batch_size, learning_rate):
		"""Initialize LaRe-Path module. Falls back silently to disabled mode on import failure."""
		try:
			# Resolve the LDRP repo root and add it to sys.path so `src.lare.*` imports work.
			repo_root = self._lare_repo_root()
			if repo_root not in sys.path:
				sys.path.append(repo_root)
			from src.lare.path.lare_path_module import LaRePathConfig, LaRePathModule

			# Decide effective training/frozen flags based on the mode.
			pretrained = self.use_pretrained_lare_path and self.pretrained_lare_path_model_path
			finetuning = (
				self.use_finetuning_lare_path
				and self.finetuning_lare_path_model_path
				and not pretrained
			)
			# Pretrained -> frozen; finetuning -> trainable (continues from loaded weights);
			# scratch -> trainable.
			cfg_frozen = bool(pretrained)
			# autosave only when training (scratch or finetune). When the user gives an
			# explicit path, freeze it; otherwise use a callable so the {N.N}M token
			# in the filename reflects the cumulative step count at each save time.
			if cfg_frozen:
				autosave = None
			elif self.lare_path_autosave_path:
				autosave = self.lare_path_autosave_path
			elif self.lare_path_autosave:
				autosave = self._lare_resolve_autosave_path
			else:
				autosave = None

			cfg = LaRePathConfig(
				factor_dim=factor_dim,
				decoder_hidden_dim=decoder_hidden_dim,
				decoder_n_layers=decoder_n_layers,
				use_transformer=use_transformer,
				transformer_heads=transformer_heads,
				transformer_depth=transformer_depth,
				transformer_seq_length=self.time_limit,
				buffer_capacity=buffer_capacity,
				seq_length=self.time_limit,
				min_buffer=min_buffer,
				update_freq=update_freq,
				batch_size=batch_size,
				learning_rate=learning_rate,
				use_lare_training=self.use_lare_path_training,
				frozen=cfg_frozen,
				autosave_path=autosave,
			)
			self.lare_path_module = LaRePathModule(self, cfg)

			# Resolve and load weights for pretrained / finetuning modes.
			if pretrained:
				self._load_lare_path_weights(self.pretrained_lare_path_model_path, freeze=True, label="PRETRAINED")
			elif finetuning:
				self._load_lare_path_weights(self.finetuning_lare_path_model_path, freeze=False, label="FINETUNE")
			else:
				print(
					f"[LaRe-Path] Initialized (mode=scratch, training={self.use_lare_path_training}, "
					f"factors={factor_dim})"
				)
		except Exception as e:
			print(f"[LaRe-Path] Failed to initialize, falling back to env reward: {e}")
			self.use_lare_path = False
			self.lare_path_module = None

	def _load_lare_path_weights(self, model_path, freeze, label):
		"""Try common locations for `model_path`, then call module.load_model(..., freeze=freeze)."""
		repo_root = self._lare_repo_root()
		save_dir = self._lare_default_save_dir()
		candidates = []
		if os.path.isabs(model_path):
			candidates.append(model_path)
		else:
			candidates.append(model_path)
			candidates.append(os.path.join(repo_root, model_path))
			candidates.append(os.path.join(save_dir, model_path))
		# Allow filenames without extension.
		extra = []
		for p in candidates:
			if not p.endswith(".pth"):
				extra.append(p + ".pth")
		candidates += extra

		resolved = next((p for p in candidates if os.path.exists(p)), None)
		if resolved is None:
			print(f"[LaRe-Path][{label}] Model not found in: {candidates}")
			print(f"[LaRe-Path][{label}] Falling back to scratch training.")
			return

		try:
			self.lare_path_module.load_model(resolved, freeze=freeze)
			mode = "frozen (inference only)" if freeze else "trainable (finetuning)"
			print(f"[LaRe-Path][{label}] Loaded {resolved} - {mode}")
		except Exception as e:
			print(f"[LaRe-Path][{label}] Load failed ({e}); falling back to scratch.")

	# ---------------- LaRe-Task initialisation / weight-loading ----------------
	def _init_lare_task(self, factor_dim, decoder_hidden_dim, decoder_n_layers,
						buffer_capacity, min_buffer, update_freq, batch_size, learning_rate):
		try:
			repo_root = self._lare_repo_root()
			if repo_root not in sys.path:
				sys.path.append(repo_root)
			from src.lare.task.lare_task_module import LaReTaskConfig, LaReTaskModule

			pretrained = self.use_pretrained_lare_task and self.pretrained_lare_task_model_path
			finetuning = (
				self.use_finetuning_lare_task
				and self.finetuning_lare_task_model_path
				and not pretrained
			)
			cfg_frozen = bool(pretrained)
			if cfg_frozen:
				autosave = None
			elif self.lare_task_autosave_path:
				autosave = self.lare_task_autosave_path
			elif self.lare_task_autosave:
				autosave = self._lare_task_resolve_autosave_path
			else:
				autosave = None

			cfg = LaReTaskConfig(
				factor_dim=factor_dim,
				decoder_hidden_dim=decoder_hidden_dim,
				decoder_n_layers=decoder_n_layers,
				buffer_capacity=buffer_capacity,
				min_buffer=min_buffer,
				update_freq=update_freq,
				batch_size=batch_size,
				learning_rate=learning_rate,
				use_lare_training=self.use_lare_task_training,
				frozen=cfg_frozen,
				autosave_path=autosave,
			)

			# Reuse LaRe-Path's graph_diameter when available (saves a Dijkstra all-pairs).
			gd = None
			if self.lare_path_module is not None:
				gd = float(getattr(self.lare_path_module, "graph_diameter", None) or 0.0) or None
			self.lare_task_module = LaReTaskModule(self, cfg, graph_diameter=gd)

			if pretrained:
				self._load_lare_task_weights(self.pretrained_lare_task_model_path, freeze=True, label="PRETRAINED")
			elif finetuning:
				self._load_lare_task_weights(self.finetuning_lare_task_model_path, freeze=False, label="FINETUNE")
			else:
				print(
					f"[LaRe-Task] Initialized (mode=scratch, training={self.use_lare_task_training}, "
					f"factors={factor_dim})"
				)
		except Exception as e:
			print(f"[LaRe-Task] Failed to initialize, falling back to env reward: {e}")
			self.use_lare_task = False
			self.lare_task_module = None

	def _load_lare_task_weights(self, model_path, freeze, label):
		repo_root = self._lare_repo_root()
		save_dir = self._lare_task_default_save_dir()
		candidates = []
		if os.path.isabs(model_path):
			candidates.append(model_path)
		else:
			candidates.append(model_path)
			candidates.append(os.path.join(repo_root, model_path))
			candidates.append(os.path.join(save_dir, model_path))
		extra = []
		for p in candidates:
			if not p.endswith(".pth"):
				extra.append(p + ".pth")
		candidates += extra
		resolved = next((p for p in candidates if os.path.exists(p)), None)
		if resolved is None:
			print(f"[LaRe-Task][{label}] Model not found in: {candidates}")
			print(f"[LaRe-Task][{label}] Falling back to scratch training.")
			return
		try:
			self.lare_task_module.load_model(resolved, freeze=freeze)
			mode = "frozen (inference only)" if freeze else "trainable (finetuning)"
			print(f"[LaRe-Task][{label}] Loaded {resolved} - {mode}")
		except Exception as e:
			print(f"[LaRe-Task][{label}] Load failed ({e}); falling back to scratch.")

	def _lare_compute_colliding_pairs(self, obs_prepare):
		"""Mirror MARL4DRP.get_collision_agents() — returns list of pairs [[i, j], ...]."""
		pairs = []
		for i in range(self.agent_num - 1):
			for j in range(i + 1, self.agent_num):
				pi = [obs_prepare[i][0], obs_prepare[i][1]]
				pj = [obs_prepare[j][0], obs_prepare[j][1]]
				import math
				if math.dist(pi, pj) < 5:
					pairs.append([i, j])
		return pairs

	def _lare_capture_prev_onehot_pos(self):
		"""Snapshot the (n_agents, n_nodes) position-onehot before the action is processed."""
		prev = np.zeros((self.agent_num, self.n_nodes), dtype=np.float32)
		for i in range(self.agent_num):
			oh = np.asarray(self.obs_onehot[i]).flatten()
			if oh.size >= self.n_nodes:
				prev[i] = oh[:self.n_nodes]
		return prev

	def get_obs(self):
		return self.obs

	def get_state(self): # unused
		return self.s

	def _get_avail_agent_actions(self, agent_id, n_actions):
		avail_actions = self.ee_env.get_avail_action_fun(self.obs[agent_id], self.current_start[agent_id], self.current_goal[agent_id], self.goal_array[agent_id])
		avail_actions_one_hot = np.zeros(n_actions)
		if avail_actions[0] == None:
			avail_actions[0] = 0
		avail_actions_one_hot[avail_actions] = 1
		return avail_actions_one_hot, avail_actions
	
	def get_avail_agent_actions(self, agent_id, n_actions):
		return self._get_avail_agent_actions(agent_id, n_actions)

	def reset(self):
		# if goal and start are not assigned, randomly generate every episode    
		self.start_ori_array = copy.deepcopy(self.ee_env.input_start_ori_array)
		self.goal_array = copy.deepcopy(self.ee_env.input_goal_array)
		#print("self.start_ori_array", self.start_ori_array)
		if self.start_ori_array == []:
			self.ee_env.random_start()
			self.start_ori_array = self.ee_env.start_ori_array
		if self.goal_array == []:
			self.ee_env.random_goal()
			self.goal_array = self.ee_env.goal_array
		#print("self.start_ori_array after", self.start_ori_array)

		#initialize task list
		if self.is_tasklist:
			self.goal_array = copy.deepcopy(self.start_ori_array)
			self.current_tasklist=[]
			self.assigned_list=[]
			#self.assigned_tasks[i] is a task assigned to agent i
			self.assigned_tasks=[[] for _ in range(self.agent_num)]
			# LaRe-Task: per-task creation step (parallel to current_tasklist).
			self._lare_task_creation_steps = []
			if self.alltasks is None:
				self.alltasks = self.ee_env.create_tasklist(self.time_limit, self.agent_num, 1)

		#initialize obs
		self.obs = tuple(np.array([self.pos[self.start_ori_array[i]][0], self.pos[self.start_ori_array[i]][1], self.start_ori_array[i], self.goal_array[i]]) for i in range(self.agent_num))
		self.obs_current_chache = copy.deepcopy(self.obs)# used for calculating reward
		#initialize obs_one-hot
		self.obs_onehot = np.zeros((self.agent_num, self.n_nodes*2))
		for i in range(self.agent_num):
			self.obs_onehot[i][int(self.start_ori_array[i])] = 1 #current position
			self.obs_onehot[i][int(self.goal_array[i])+self.n_nodes] = 1 #current goal


		self.current_start = self.start_ori_array # [0,1]
		self.current_goal  = [None for _ in range(self.agent_num)]
		self.terminated    = [False for _ in range(self.agent_num)]

		self.distance_from_start = np.zeros(self.agent_num) # info
		self.wait_count = np.zeros(self.agent_num) # info

		self.reach_account = 0
		self.step_account = 0
		self.episode_account += 1

		# for tasklist
		self.task_completion = 0
		#print('Environment reset obs: \n', self.obs)

		obs = self.obs_manager.calc_obs()

		return obs
		

	def _default_task_assign_tp(self):
		"""Built-in nearest-pending-task assigner used when the caller does not
		supply a task action (e.g., epymarl's gymma wrapper passes only a path
		action list). Mirrors src/task_assign/task_policy/tp.py inline so the
		env has no runtime dependency on the task_assign package.

		Returns: list of length agent_num. Each element is -1 (no assignment)
		or an index j into self.current_tasklist (assign task j to agent i).
		"""
		assigned_local = list(self.assigned_list)
		task_assign = [-1] * self.agent_num
		for i in range(self.agent_num):
			if self.assigned_tasks[i] == [] and self.current_tasklist:
				shortest = float("inf")
				best = -1
				for j in range(len(self.current_tasklist)):
					if assigned_local[j] != -1:
						continue
					path_len = self.get_path_length(self.goal_array[i], self.current_tasklist[j][0])
					if path_len < shortest:
						shortest = path_len
						best = j
				if best != -1:
					assigned_local[best] = i
					task_assign[i] = best
		return task_assign

	def step(self, joint_action):

		#print("tasks",self.current_tasklist)

		if isinstance(joint_action, dict):
			task_assign = joint_action.get("task", None)
			joint_action = joint_action.get("pass", joint_action)
		else:
			task_assign = None

		# Fallback: callers that pass only path actions (e.g., epymarl's gymma
		# wrapper, which sends list[int]) don't supply task decisions. When the
		# task system is active, use the built-in TP assigner so MARL training
		# can run with task_flag=True without an external task policy.
		if self.is_tasklist and task_assign is None:
			task_assign = self._default_task_assign_tp()

		# LaRe-Path: snapshot per-agent onehot positions BEFORE movement.
		if self.use_lare_path and self.lare_path_module is not None:
			self._lare_prev_onehot_pos = self._lare_capture_prev_onehot_pos()
		# Advance the cumulative step counter (shared by Path & Task naming).
		if (self.use_lare_path and self.lare_path_module is not None) or \
		   (self.use_lare_task and self.lare_task_module is not None):
			self._lare_total_step_account += 1

		#transite env based on joint_action
		self.step_account += 1
		self.obs_current_chache = copy.deepcopy(self.obs)

		self.obs_prepare = []
		self.obs_onehot_prepare = copy.deepcopy(self.obs_onehot)
		self.current_start_prepare = copy.deepcopy(self.current_start)
		self.current_goal_prepare = copy.deepcopy(self.current_goal)
		# 1) first judge action_i whether available, to output !!!obs_prepare & obs_onehot_prepare!!!
		for i in range(self.agent_num):
			action_i = joint_action[i]  
			# 1) first judge action_i whether available, to output obs_prepare: 
			# if unavailable ⇢ obs_prepare.append( self.obs_old[i])
			#print("Avaible actions",self.get_avail_agent_actions(i, self.n_actions)[1])
			if action_i not in self._get_avail_agent_actions(i, self.n_actions)[1]:
				#print("This is not Avaible",i,action_i,self.get_avail_agent_actions(i, self.n_actions)[1])
				self.obs_prepare.append(self.obs_current_chache[i])
				#self.obs_onehot_prepare[i]= self.obs_onehot[i]

				self.wait_count[i] += 1

			# if action_i is current start node -> stop
			elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
				self.obs_prepare.append(self.obs_current_chache[i])
				self.wait_count[i] += 1
				#pbsのため，その場待機でもcurrent_goalをNoneのままでないように変更
				#従来のdrpは以下の行はなし
				self.current_goal_prepare[i] = action_i
			# if available ⇢ obs_prepare update by obs_i_
			else:
				#self.joint_action_old[i] = joint_action[i]
				self.current_goal_prepare[i] = joint_action[i] #update 行き先ノード when avable action is taken
				obs_i = self.obs[i]
		
				#calculate current distance
				current_goal = list(self.pos[int(action_i)])
				current_x1,current_y1 = obs_i[0], obs_i[1]
				x = current_goal[0] - current_x1
				y = current_goal[1] - current_y1
				dist_to_cgoal = np.sqrt(np.square(x) + np.square(y))# the distance to current goal

				if dist_to_cgoal>self.speed:# move on edge
					current_x1 = round(current_x1+(self.speed*x/dist_to_cgoal), 2)
					current_y1 = round(current_y1+(self.speed*y/dist_to_cgoal), 2)
					obs_i_ = [round(current_x1,2), round(current_y1,2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					x = list(self.pos[self.current_start[i]])[0] - current_x1
					y = list(self.pos[self.current_start[i]])[1] - current_y1
					dist_to_cstart = np.sqrt(np.square(x) + np.square(y))# the distance to current goal
					dist_to_cstart_rate = round(dist_to_cstart/(dist_to_cstart+dist_to_cgoal), 2)
					
					#print("self.obs_onehot_prepare before",self.obs_onehot_prepare )
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.current_start[i])] = 1-dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					#print("self.obs_onehot_prepare after",self.obs_onehot_prepare )
					self.distance_from_start[i] += self.speed
				# arrive at node
				else:
					obs_i_ = [round(self.pos[int(action_i)][0],2), round(self.pos[int(action_i)][1],2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = 1
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					
					# update current_start only when arrive at node
					self.current_start_prepare[i] = int(action_i) #update 出発ノード when　行き先ノード　has been arrived
					self.current_goal_prepare[i] = None #update 行き先ノード when it has been arrived

					self.distance_from_start[i] += dist_to_cgoal

				self.obs_prepare.append(obs_i_)
		
		# 2) !!!obs_prepare & obs_onehot_prepare!!! を持って、
		# second judge whether to !!! obs & obs_onehot !!! according to collision happen
		collision_flag = self.ee_env.collision_detect(self.obs_prepare)
		# LaRe-Path: also compute the explicit list of colliding pairs for the encoder.
		if self.use_lare_path and self.lare_path_module is not None:
			self._lare_current_colliding_pairs = self._lare_compute_colliding_pairs(self.obs_prepare)
		info = {
			"goal": False,
			"collision": False,
			"timeup": False, # for epymarl
			"distance_from_start": None,
			"step": self.step_account,
			"wait": self.wait_count,
			"goal_account": self.reach_account,
			"1agent_goal_account": self.reach_account/self.agent_num,
			"task_completion": self.task_completion,
		}
		# happen
		if collision_flag==1:#collision
			#collision_reward=-1
			collision_reward = self.r_coll*self.speed
			if self.collision == "bounceback":
				self.terminated = [False for _ in range(self.agent_num)]
			else: # default -> self.collision == "terminated"
				self.terminated = [True for _ in range(self.agent_num)]
			info["collision"] = True
			#obs = self.obs_manager.calc_obs()
			ri_array = [collision_reward for _ in range(self.agent_num)]

			self.obs = tuple([np.array(i) for i in self.obs_prepare])
			self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)
			self.current_start = copy.deepcopy(self.current_start_prepare) 
			self.current_goal = copy.deepcopy(self.current_goal_prepare)

			
			# return obs, [collision_reward for _ in range(self.agent_num)], self.terminated, info 
			
		# not happen
		else: #non collision
			self.obs = tuple([np.array(i) for i in self.obs_prepare])
			self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)
			self.current_start = copy.deepcopy(self.current_start_prepare) 
			self.current_goal = copy.deepcopy(self.current_goal_prepare)

			team_reward = 0
			ri_array = []
			for i in range(self.agent_num):
				ri = self.reward(i)
				team_reward += ri
				ri_array.append(ri)
			
			if self.terminated == [True for _ in range(self.agent_num)]: # all reach goal
				#print("!!!all reach goal!!!")
				# info
				info["goal"] = True
			else:
				pass

			#obs = self.obs_manager.calc_obs()

		if self.is_tasklist:
			# add tasks(now, add only one task by step)
			for i in range(len(self.alltasks[self.step_account-1])):
				if len(self.current_tasklist) < self.task_num:
					new_task = self.alltasks[self.step_account-1][i]
					self.current_tasklist.append(new_task)
					self.assigned_list.append(-1) # -1 means unassigned
					# LaRe-Task: track creation step parallel to current_tasklist.
					self._lare_task_creation_steps.append(self.step_account)

			# remove the task from the list if it has been completed
			for i in range(self.agent_num):
				pos_agenti = [self.obs[i][0],self.obs[i][1]]
				if self.assigned_tasks[i] != []:
					if str(pos_agenti)==str(self.pos[self.goal_array[i]]):
						if self.goal_array[i] == self.assigned_tasks[i][1]:
							self.assigned_tasks[i] = [] # remove the task from assigned_tasks
							self.task_completion += 1

			# assign tasks to agents — capture pre-assignment state for LaRe-Task.
			lare_task_decisions = []
			for i in range(self.agent_num):
				if (self.assigned_tasks[i] == [] or i in self.assigned_list) and task_assign[i] != -1:
					r = task_assign[i]
					task_r = self.current_tasklist[r]
					was_idle = (self.assigned_tasks[i] == [])
					prev_goal = None if was_idle else self.goal_array[i]
					creation_step = (
						self._lare_task_creation_steps[r]
						if 0 <= r < len(self._lare_task_creation_steps)
						else self.step_account
					)
					wait_steps = max(0, self.step_account - creation_step)

					self.assigned_tasks[i] = self.current_tasklist[r]
					self.goal_array[i] = self.assigned_tasks[i][0] # update goal to pick node
					self.assigned_list[task_assign[i]] = i # update assigned_list

					lare_task_decisions.append({
						"agent_id": int(i),
						"pickup": int(task_r[0]),
						"dropoff": int(task_r[1]),
						"agent_prev_goal": prev_goal,
						"agent_was_idle": bool(was_idle),
						"wait_steps": int(wait_steps),
					})

			# LaRe-Task: feed the new assignments through the encoder/decoder.
			if (
				self.use_lare_task
				and self.lare_task_module is not None
				and len(lare_task_decisions) > 0
			):
				try:
					loads_after = [
						1 if len(self.assigned_tasks[k]) > 0 else 0
						for k in range(self.agent_num)
					]
					unassigned_after = sum(1 for v in self.assigned_list if v == -1)
					n_step = len(lare_task_decisions)
					full_decisions = [
						{**d,
						 "agent_loads_after": loads_after,
						 "unassigned_after": unassigned_after,
						 "n_assignments_step": n_step}
						for d in lare_task_decisions
					]
					self.lare_task_module.record_step_assignments(self, full_decisions)
				except Exception as e:
					print(f"[LaRe-Task] step hook error (no proxy this step): {e}")

			# update agent's start and goal
			for i in range(self.agent_num):
				pos_agenti = [self.obs[i][0],self.obs[i][1]]
				if len(self.assigned_tasks[i])>0:
					if str(pos_agenti)==str(self.pos[self.goal_array[i]]):
						#when agent i reach the pick node
						if self.goal_array[i]==self.assigned_tasks[i][0]:
							self.start_ori_array[i] = self.goal_array[i]
							self.goal_array[i] = self.assigned_tasks[i][1]
							try:
								idx = self.assigned_list.index(i)
								self.current_tasklist.pop(idx)
								self.assigned_list.pop(idx)
								if 0 <= idx < len(self._lare_task_creation_steps):
									self._lare_task_creation_steps.pop(idx)
							except ValueError:
								print("ValueError: agent ", i, " 's assigned task is not in the current_tasklist")
						#when agent i reach the drop node
						elif self.goal_array[i]==self.assigned_tasks[i][1]:
							self.start_ori_array[i] = self.goal_array[i]
							#self.goal_array[i] = self.assigned_tasks[i][0]
						else:
							print(self.goal_array[i], self.assigned_tasks[i])
							raise ValueError("Error in task execution")
						
				self.obs_prepare[i] = [self.obs[i][0], self.obs[i][1], self.start_ori_array[i], self.goal_array[i]]
				self.obs_onehot[i] = np.zeros((1, len(list(self.G.nodes()))*2))
				self.obs_onehot[i][int(self.current_start[i])] = 1
				self.obs_onehot[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1

			self.obs = tuple([np.array(i) for i in self.obs_prepare])

		obs = self.obs_manager.calc_obs()

		# Check whether time is over
		if self.step_account >= self.time_limit:
			#print("!!!time up!!!")
			info["timeup"]= True
			self.terminated = [True for _ in range(self.agent_num)]

		info["distance_from_start"] = self.distance_from_start

		# LaRe-Path: compute factors, record the step, and (if trained + enabled) swap rewards.
		if self.use_lare_path and self.lare_path_module is not None:
			try:
				factors = self.lare_path_module.compute_factors(
					self._lare_prev_onehot_pos,
					self._lare_current_colliding_pairs,
				)
				env_reward_sum = float(sum(ri_array))
				self.lare_path_module.record_step(factors, env_reward_sum)

				if self.use_lare_path_training and self.lare_path_module.is_trained:
					proxy = self.lare_path_module.proxy_rewards(factors)
					if proxy is not None:
						ri_array = [float(x) for x in proxy]
			except Exception as e:
				print(f"[LaRe-Path] step hook error (falling back to env reward): {e}")

		# LaRe-Task: surface the (possibly trained) proxy reward for this step's assignments
		# so the PPO trainer can pick it up via info[...] without taking a hard dep on the env.
		if self.use_lare_task and self.lare_task_module is not None:
			info["lare_task_proxy_reward"] = self.lare_task_module.consume_step_proxy_reward()
			info["lare_task_is_trained"] = bool(self.lare_task_module.is_trained)

		if all(self.terminated) is True:
			self.update_log(info)
			# LaRe-Path: close out the episode and (when ready) trigger a decoder update.
			if self.use_lare_path and self.lare_path_module is not None:
				try:
					self.lare_path_module.end_episode()
				except Exception as e:
					print(f"[LaRe-Path] end_episode error: {e}")
			# LaRe-Task: episode-level training target is the task completion count.
			if self.use_lare_task and self.lare_task_module is not None:
				try:
					self.lare_task_module.end_episode(self.task_completion)
				except Exception as e:
					print(f"[LaRe-Task] end_episode error: {e}")

		return obs, ri_array, self.terminated, info
	
	def update_log(self, info):
		log_episode = {}

		if info["goal"] is True:
			log_episode["result"] = "goal"
		elif info["collision"] is True:
			log_episode["result"] = "collision"
		elif info["timeup"] is True:
			log_episode["result"] = "timeup"
		else:
			log_episode["result"] = "exception"

		log_episode["termination_time"] = info["step"]
		log_episode["distance_from_start"] = info["distance_from_start"]

		self.log[self.episode_account] = log_episode

	def get_log(self, epi):
		return self.log[epi]

	def reward(self, i):
		pre_pos_agenti = [self.obs_current_chache[i][0],self.obs_current_chache[i][1]]
		pos_agenti = [self.obs[i][0],self.obs[i][1]]

		if self.is_tasklist: #ここから
			if self.start_ori_array[i] == self.goal_array[i]:
				r_i = 0
			else:
				if str(pos_agenti)==str(self.pos[self.goal_array[i]]): # at goal				
					if len(self.assigned_tasks[i])>0 : #first time to reach goal 
						r_i = self.r_goal
						self.reach_account += 1
					else: # stop at goal
						r_i = 0
						# self.distance_from_start[i] -= self.speed
			
				else: #at a general node 
					if pre_pos_agenti==pos_agenti: # stop at a general node 
						r_i = self.r_wait*self.speed
					else: # just move 
						r_i = self.r_move*self.speed

		else:
			if str(pos_agenti)==str(self.pos[self.goal_array[i]]): # at goal				
				if pre_pos_agenti!=pos_agenti : #first time to reach goal 
					r_i = self.r_goal
					self.reach_account += 1
					self.terminated[i] = True
				else: # stop at goal
					r_i = 0   
					# self.distance_from_start[i] -= self.speed
			
			else: #at a general node 
				if pre_pos_agenti==pos_agenti: # stop at a general node 
					r_i = self.r_wait*self.speed
				else: # just move 
					r_i = self.r_move*self.speed
		return r_i

	def render(self, mode='human'):
		self.ee_env.plot_map_dynamic(
			self.visu_delay,self.obs_current_chache,
			self.obs,self.goal_array,
			self.agent_num,
			self.current_goal,
			self.reach_account,
			self.step_account,
			self.episode_account,
			self.current_tasklist,
			self.assigned_tasks,
		) # a must be a angle !!!list!!!

		if self.is_tasklist:
			self.taskgui.show_tasklist(
				self.agent_num, 
				self.assigned_tasks, 
				self.current_tasklist,
				self.assigned_list
				)
		

	def close(self):
		print('Environment CLOSE')
		return None
    
	def get_pos_list(self):
		pos_list = []
		all_onehot_obs = np.array(self.obs_onehot)
		onehot_obs = all_onehot_obs[:, :self.n_nodes]

		# get all agent state and position
		for i, obs_i in enumerate(onehot_obs):
			edge_or_node = tuple([i for i, o in enumerate(obs_i) if o!=0])
			if len(edge_or_node)==1:
				node = edge_or_node[0]
				pos = {"type": "n", "pos": node}
				obs_i = np.array(obs_i)*self.agent_num
			else:
				edge = edge_or_node
				pos = {"type": "e", "pos": edge, "current_goal": self.current_goal[i], "current_start": self.current_start[i], "obs": obs_i}
			pos_list.append(pos)

		return pos_list

	def get_path_length(self, start, goal):
		if start == goal:
			return 0
		else:
			return self.ee_env.get_path_length(start, goal)
		
	def get_near_nodes(self, node_num):
		return self.ee_env.get_near_nodes(node_num)

	def set_1agent_info(self, pos, current_start, current_goal, goal_array):
		self.obs = tuple(np.array([pos[0], pos[1], self.obs[0][2], self.obs[0][3]]) for _ in range(1))
		self.current_start[0] = current_start
		self.current_goal[0] = current_goal
		self.goal_array[0] = goal_array
		self.step_account = 0

		return