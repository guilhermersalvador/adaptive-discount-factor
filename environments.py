from enum import Enum
from typing import Optional, Callable
import numpy as np
from gym import Env, Space
from gym.spaces import Box, Discrete
import pygame


class PathSpace(Space):
    def __init__(self, n_paths: int, path_length: Callable):
        self.paths: list[Box] = [
            Box(
                low=np.array([nth_path, 1]),
                high=np.array([nth_path, path_length(nth_path)]),
                dtype=int,
            )
            for nth_path in range(1, n_paths + 1)
        ]
        self.start_state: np.ndarray = np.array([0, 0])
        self.absorbing_state: np.ndarray = np.array([-1, -1])
        self.n = sum([path.high[1] for path in self.paths]) + 1


class Pathworld(Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        paths: int,
        path_length: Callable,
        path_reward: Callable,
        render_mode: Optional[str] = None,
        variable_hazard: bool = True,
        hazard_rate: float = 0,
        hazard_distribution: Callable = lambda: np.random.exponential(0.05),
    ):
        self.paths: int = paths
        self.path_length: Callable = path_length
        self.path_reward: Callable = path_reward
        self.variable_hazard: bool = variable_hazard
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space: PathSpace = PathSpace(paths, path_length)
        self.action_space: Space = Discrete(paths)

        self.window_size: int = 800
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.hazard_rate: float = hazard_rate
        self.hazard_distribution = hazard_distribution
        self._agent_location: np.ndarray | None = None

        self._total_timesteps: int = 0

    def _transition(self, state, action):
        location = self._state_to_location(state)

        if np.random.uniform() < 1 - np.exp(-self.hazard_rate):
            location = self.observation_space.absorbing_state
        elif self._is_end_of_path(location):
            return self._location_to_state(location)
        elif np.array_equal(location, self.observation_space.start_state):
            location = np.array([action + 1, 1])
        elif action + 1 == location[0]:
            location[1] += 1
        elif action + 1 > location[0]:
            location[0] = action + 1
        
        return self._location_to_state(location)


    def _reward(self, action, original_state, result_state):
        current_location = self._state_to_location(original_state)
        next_location = self._state_to_location(result_state)
        if self._is_end_of_path(next_location) and not self._is_end_of_path(current_location):
            return self.path_reward(next_location[0])
        else:
            return 0

    def _location_to_state(self, location):
        if np.array_equal(location, self.observation_space.start_state):
            return 0
        elif np.array_equal(location, self.observation_space.absorbing_state):
            return -1
        else:
            acc = 0
            for path in self.observation_space.paths:
                for depth in range(1, path.high[1] + 1):
                    acc += 1
                    if np.array_equal(location, np.array([path.low[0], depth])):
                        return acc

    def _state_to_location(self, state):
        if state == 0:
            return self.observation_space.start_state
        elif state == -1:
            return self.observation_space.absorbing_state
        else:
            acc = 0
            for path in self.observation_space.paths:
                for depth in range(1, path.high[1] + 1):
                    acc += 1
                    if acc == state:
                        return np.array([path.low[0], depth])

    def _is_end_of_path(self, location):
        return (
            not np.array_equal(location, self.observation_space.start_state)
            and not np.array_equal(location, self.observation_space.absorbing_state)
            and location[1] == self.path_length(location[0])
        )

    def get_model(self):
        rewards = np.zeros((self.observation_space.n, self.action_space.n))
        for path in self.observation_space.paths:
            rewards[self._location_to_state(np.array([path.low[0], path.high[1] - 1])), path.low[0] - 1] = (
                self.path_reward(path.low[0])
            )

        transition_probabilities = np.zeros(
            (self.action_space.n, self.observation_space.n, self.observation_space.n)
        )

        for action in range(self.action_space.n):
            for state in range(self.observation_space.n):
                location = self._state_to_location(state)
                if np.array_equal(location, self.observation_space.start_state):
                    next_location = np.array([action + 1, 1])
                    transition_probabilities[
                        action,
                        state,
                        self._location_to_state(np.array([action + 1, 1])),
                    ] = 1
                else:
                    if self._is_end_of_path(location):
                        transition_probabilities[
                            action,
                            state,
                            state,
                        ] = 1
                    else:
                        if action + 1 == location[0]:
                            next_location = np.array([location[0], location[1] + 1])
                            transition_probabilities[
                                action, state, self._location_to_state(next_location)
                            ] = 1
                        elif action + 1 > location[0]:
                            transition_probabilities[
                                action,
                                state,
                                self._location_to_state([action + 1, location[1]]),
                            ] = 1
                        else:
                            transition_probabilities[
                                action,
                                state,
                                state
                            ] = 1
        return transition_probabilities, rewards

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        square_size = np.array(
            [
                self.window_size / self.paths,
                self.window_size / self.path_length(self.paths),
            ]
        )

        for path in self.observation_space.paths:
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                (
                    square_size[0] * (path.low[0] - 1),
                    self.window_size - square_size[1] * path.high[1],
                    square_size[0],
                    square_size[1],
                ),
            )
            pygame.draw.line(
                canvas,
                0,
                (
                    square_size[0] * (path.low[0] - 1),
                    self.window_size - square_size[1] * path.high[1],
                ),
                (square_size[0] * (path.low[0] - 1), self.window_size),
                width=3,
            )
            for i in range(path.high[1] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (
                        square_size[0] * (path.low[0] - 1),
                        self.window_size - square_size[1] * i,
                    ),
                    (
                        square_size[0] * path.low[0],
                        self.window_size - square_size[1] * i,
                    ),
                    width=3,
                )

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                int(
                    square_size[0] * (self._agent_location[0] - 1) + square_size[0] / 2
                ),
                int(
                    self.window_size
                    - square_size[1] * self._agent_location[1]
                    + square_size[1] / 2
                ),
            ),
            int(min(square_size) / 4),
        )

        if (
            self.render_mode == "human"
            and self.window is not None
            and self.clock is not None
        ):
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.hazard_rate = (
            self.hazard_distribution()
            if self.variable_hazard
            else self.hazard_rate
        )
        self._agent_location = self.observation_space.start_state
        self._total_timesteps = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._location_to_state(self._agent_location), {}

    def step(self, action):
        previous_state = self._location_to_state(self._agent_location)
        self._agent_location = self._state_to_location(
            self._transition(previous_state, action)
        )

        if self.hazard_rate > 0:
            terminated = np.array_equal(
                self._agent_location, self.observation_space.absorbing_state
            )
        else:
            terminated = self._is_end_of_path(self._agent_location)

        info = {"timesteps": self._total_timesteps}

        self._total_timesteps += 1

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._location_to_state(self._agent_location),
            self._reward(action, previous_state, self._location_to_state(self._agent_location)),
            terminated,
            False,
            {} if not terminated else info,
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class GridSpace(Space):
    def __init__(self, n: int):
        self.n = n**2
        self.state_space = Box(low=np.array([0, 0]), high=np.array([n - 1, n - 1]), dtype=int)
        self.absorbing_state = np.array([-1, -1])
    
    def sample(self):
        return self.state_space.sample()

class GridAction(Enum):
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3

class GridEnvironment(Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        n: int,
        render_mode: Optional[str] = None,
        variable_hazard: bool = True,
        hazard_rate: float = 0,
        hazard_distribution: Callable = None,
        seed=None,
        enable_walls: bool = True,
        stocasticity_prob: float = 0.0,
    ):
        self.n = n
        self.observation_space = GridSpace(n)

        self.action_space = Discrete(len(GridAction))

        self._wall_locations = self._place_walls(enable_walls)
        
        self._stochoasticity_prob = stocasticity_prob

        self._reward_prob = 0.3
        self._reward_locations = []
        for i in range(self.n):
            for j in range(self.n):
                if np.random.uniform() < self._reward_prob and not (i, j) in self._wall_locations:
                    self._reward_locations.append(np.array([i, j]))
        np.random.shuffle(self._reward_locations)

        self._agent_state = self.observation_space.sample()
        while tuple(self._agent_state) in self._wall_locations:
            self._agent_state = self.observation_space.sample()
        self._initial_state = np.copy(self._agent_state)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window_size: int = 800
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.variable_hazard: bool = variable_hazard
        self.hazard_rate: float = hazard_rate
        self.hazard_distribution = hazard_distribution

        self.seed = seed
        self.episodes = 0

        self.reset()

    def _place_walls(self, enable_walls):
        line_walls_prob = 0.1
        continuing_wall_prob = 0.6
        walls = set()
        for i in range(self.n):
            for j in range(self.n):
                if enable_walls and np.random.uniform() < line_walls_prob:
                    walls.add((i, j))
                    direction = np.random.choice([0, 1])
                    while np.random.uniform() < continuing_wall_prob:
                        if direction == 0 and j < self.n - 1:
                            walls.add((i, j + 1))
                            j += 1
                        elif direction == 1 and i < self.n - 1:
                            walls.add((i + 1, j))
                            i += 1
                        else:
                            break
        return list(walls)

    def _state_to_id(self, state):
        return state[0] * self.n + state[1]

    def _id_to_state(self, id):
        return np.array([id // self.n, id % self.n])
    
    def _act(self, state, action):
        if action == GridAction.MOVE_UP.value:
            next_state = np.array([state[0], max(0, state[1] - 1)])
        elif action == GridAction.MOVE_RIGHT.value:
            next_state = np.array([min(self.n - 1, state[0] + 1), state[1]])
        elif action == GridAction.MOVE_DOWN.value:
            next_state = np.array([state[0], min(self.n - 1, state[1] + 1)])
        elif action == GridAction.MOVE_LEFT.value:
            next_state = np.array([max(0, state[0] - 1), state[1]])

        for wall in self._wall_locations:
            if np.array_equal(next_state, wall):
                return state
        return next_state
        
    def _transition(self, state, action) -> None:
        if np.random.uniform() < 1 - np.exp(-self.hazard_rate):
            return self.observation_space.absorbing_state
        else:
            if np.random.uniform() < self._stochoasticity_prob/((self.action_space.n - 1)/self.action_space.n):
                action = np.random.choice(self.action_space.n)
            return self._act(state, action)

    def _reward(self, previous_state, action, result_state):
        for i, reward_location in enumerate(self._reward_locations):
            if np.array_equal(result_state, reward_location):
                return i + 1
        else:
            return 0

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for i in range(self.n):
            for j in range(self.n):
                n_rewards = len(self._reward_locations)
                for k, reward_location in enumerate(self._reward_locations):
                    if i == reward_location[0] and j == reward_location[1]:
                        pygame.draw.circle(
                            canvas,
                            (0, 255, 0),
                            (
                                int(self.window_size / self.n * i + self.window_size / self.n / 2),
                                int(self.window_size / self.n * j + self.window_size / self.n / 2),
                            ),
                            int(self.window_size / self.n * k/n_rewards / 2),
                        )
                        break
                else:
                    for wall in self._wall_locations:
                        if i == wall[0] and j == wall[1]:
                            pygame.draw.rect(
                                canvas,
                                (125, 125, 125),
                                (
                                    self.window_size / self.n * i,
                                    self.window_size / self.n * j,
                                    self.window_size / self.n,
                                    self.window_size / self.n,
                                ),
                            )
                            break
                    else:
                        pygame.draw.rect(
                            canvas,
                            (255, 255, 255),
                            (
                                self.window_size / self.n * i,
                                self.window_size / self.n * j,
                                self.window_size / self.n,
                                self.window_size / self.n,
                            ),
                        )
                    
                if i == self._agent_state[0] and j == self._agent_state[1]:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        (
                            self.window_size / self.n * i + self.window_size / self.n / 4,
                            self.window_size / self.n * j + self.window_size / self.n / 4,
                            self.window_size / self.n / 2,
                            self.window_size / self.n / 2,
                        ),
                    )
                
                pygame.draw.line(
                    canvas,
                    0,
                    (self.window_size / self.n * i, 0),
                    (self.window_size / self.n * i, self.window_size),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (0, self.window_size / self.n * j),
                    (self.window_size, self.window_size / self.n * j),
                    width=3,
                )
        if (
            self.render_mode == "human"
            and self.window is not None
            and self.clock is not None
        ):
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def get_model(self):
        rewards = np.zeros((self.observation_space.n, self.action_space.n))
        for state_id in range(self.observation_space.n):
            state = self._id_to_state(state_id)
            for action in range(self.action_space.n):
                rewards[state_id, action] = self._reward(state, action, self._act(state, action))

        transition_probabilities = np.zeros(
            (self.action_space.n, self.observation_space.n, self.observation_space.n)
        )

        for action in range(self.action_space.n):
            for state_id in range(self.observation_space.n):
                if self._stochoasticity_prob > 0:
                    for _ in range(self.action_space.n):
                        transition_probabilities[
                            action,
                            state_id,
                            self._state_to_id(self._transition(self._id_to_state(state_id), action)),
                        ] = self._stochoasticity_prob/(self.action_space.n - 1)

                transition_probabilities[
                    action, 
                    state_id, 
                    self._state_to_id(self._act(self._id_to_state(state_id), action))
                ] = 1 - self._stochoasticity_prob
        return transition_probabilities, rewards

    def reset(self, seed=None, full_reset=False):
        super().reset(seed=seed)

        if self.variable_hazard and self.hazard_distribution is not None:
            self.hazard_rate = self.hazard_distribution()
        elif self.variable_hazard and full_reset:
            self.hazard_rate = 0.1 + 0.09 * np.sin(2 * np.pi * self.episodes / 50)
            self.episodes += 1

        self._agent_state = self._initial_state

        self._total_timesteps = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._state_to_id(self._agent_state), {}

    def step(self, action: GridAction):
        previous_state = self._agent_state.copy()
        self._agent_state = self._transition(previous_state, action)

        terminated = np.array_equal(self._agent_state, self.observation_space.absorbing_state)

        self._total_timesteps += 1
        info = {"timesteps": self._total_timesteps}

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._state_to_id(self._agent_state),
            self._reward(previous_state, action, self._agent_state),
            terminated,
            False,
            {} if not terminated else info,
        )
