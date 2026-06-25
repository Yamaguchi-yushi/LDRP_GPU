from .basic_controller import BasicMAC


class MATMAC(BasicMAC):
    """
    MAT-Dec Policy class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, scheme, groups, args):

        super().__init__(scheme, groups, args)

        self.n_agents = args.n_agents
        self.input_shape = self.agent.input_shape
        self.n_actions = args.n_actions
        self.n_embd = args.n_embd
        self.use_rnn = args.use_rnn

        assert self.agent_output_type == "pi_logits", f"'self.agent_output_type': {self.agent_output_type}"
        assert args.action_selector == "soft_policies", f"'self.action_selector': {self.action_selector}"
        assert self.use_rnn is False, f"'self.use_rnn': {self.use_rnn}"

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        extra_returns = {}

        values, actions, action_log_probs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        values, actions, action_log_probs = values[bs], actions[bs], action_log_probs[bs]

        extra_returns.update({'log_probs': action_log_probs.clone().detach(),
                              'values': values.clone().detach()})

        return actions, extra_returns

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        values, actions, action_log_probs = self.get_actions(ep_batch, t, agent_inputs, avail_actions, test_mode)

        return values, actions, action_log_probs

    def get_actions(self, ep_batch, t, obs, available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param obs: (np.ndarray) local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        """

        obs = obs.reshape(-1, self.n_agents, self.input_shape)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)

        actions, action_log_probs, values = self.agent.get_actions(ep_batch, t, obs, available_actions, deterministic)

        return values, actions, action_log_probs

    def get_values(self, obs, available_actions=None):
        """
        Get value function predictions.
        :param obs: (np.ndarray) local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return values: (torch.Tensor) value function predictions.
        """

        obs = obs.reshape(-1, self.n_agents, self.input_shape)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)

        values = self.agent.get_values(obs, available_actions)

        values = values.view(-1, 1)

        return values

    def evaluate_actions(self, ep_batch, t):
        """
        Get action logprobs / entropy and value function predictions for actor update.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        agent_inputs = self._build_inputs(ep_batch, t)

        actions = ep_batch["actions"][:, t]
        available_actions = ep_batch["avail_actions"][:, t]
        agent_inputs = agent_inputs.reshape(-1, self.n_agents, self.input_shape)
        actions = actions.reshape(-1, self.n_agents, 1)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)

        action_log_probs, values, entropy = self.agent.evaluate_actions(ep_batch,
                                                                        t,
                                                                        agent_inputs,
                                                                        actions,
                                                                        available_actions)

        action_log_probs = action_log_probs.view(-1, 1)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, 1)
        entropy = entropy.mean()

        return values, action_log_probs, entropy

    def init_hidden(self, batch_size):
        """
        Just for compatibility
        """
        return