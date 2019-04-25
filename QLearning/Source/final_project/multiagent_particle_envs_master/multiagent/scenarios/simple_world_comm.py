import math
import numpy as np
from multiagent_particle_envs_master.multiagent.core import World, Agent, Landmark
from multiagent_particle_envs_master.multiagent.scenario import BaseScenario
from Source.final_project.train import global_env


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        # world.damping = 1
        num_adversaries = global_env.predator
        num_good_agents = global_env.prey
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        num_food = global_env.food
        num_forests = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            landmark.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests
        # world.landmarks += self.set_boundaries(world)  # world boundaries now penalized with negative reward
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.hunger = 5
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (agent1.size + agent2.size) / 2
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, episode_step):
        # Agents are rewarded based on minimum agent distance to each landmark
        if self.outside_boundary(agent):
            return -20

        if agent.adversary:
            return self.adversary_reward(agent, world)
        else:
            return self.agent_reward(agent, world, episode_step)

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 2 or \
                agent.state.p_pos[0] < -2 or \
                agent.state.p_pos[1] > 2 or \
                agent.state.p_pos[1] < -2:
            return True
        else:
            return False

    def agent_reward(self, agent, world, episode_step):
        rew = 0
        agents = global_env.knn[agent.id]

        # todo add hunger
        # Agents do not like colliding with predators
        for a in agents:
            if world.agents[a].adversary:
                if self.is_collision(world.agents[a], agent):
                    rew -= 10
                    agent.state.hunger += 10

        # Agents are rewarded for collision with food
        collided = False
        for food in world.food:
            if self.is_collision(agent, food):
                rew += 200 * agent.state.hunger
                collided = True
                agent.state.hunger -= 10


        # Agents are heavily penalized for getting distance from the closest food
        # todo train with absolute
        if not collided:
            rew -= min([math.hypot(agent.state.p_pos[0] - food.state.p_pos[0], agent.state.p_pos[1] - food.state.p_pos[1]) for food in world.food]) * agent.state.hunger

        return rew

    def adversary_reward(self, agent, world):
        rew = 0
        agents = global_env.knn[agent.id]

        # Agents are rewarded a ton for collision with prey.
        prey = []
        for a in agents:
            if not world.agents[a].adversary:
                prey.append(world.agents[a])
                if self.is_collision(world.agents[a], agent):
                    rew += 10 * agent.state.hunger
                    agent.state.hunger -= 1

        # Agents are penalized for distance from closest prey.
        if prey:
            rew -= min([np.sum(np.square(p.state.p_pos - agent.state.p_pos)) for p in prey])*5 * agent.state.hunger
        return rew

    def observation(self, agent, world):
        other_type = []
        entity_pos = []
        other_pos = []
        other_vel = []
        food_pos = []
        in_forest = []
        comm = [world.agents[0].state.c]

        # set up knn
        if global_env.agents != world.agents:
            global_env.agents = world.agents
            global_env.calculate_knn()

        # setup landmark distance and collisions
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        for forest in world.forests:
            if self.is_collision(agent, forest):
                in_forest.append(np.array([1]))
            else:
                in_forest.append(np.array([-1]))

        # setup food distance and collisions
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)

        # set up knn here
        # entity is distance to landmarks
        # other_pos is other guys pos
        # get agent number
        # figure out what to do with entity
        agent_number = int(str.split(agent.name, ' ')[1])
        for i in global_env.knn[agent_number]:
            other = world.agents[i]
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel - agent.state.p_vel)
            if agent.adversary:
                other_type.append(np.array([1]))
            else:
                other_type.append(np.array([0]))

            if len(other_pos) == global_env.k:
                break

        if not agent.adversary:
            np.concatenate([agent.state.p_vel] +
                           [agent.state.p_pos] +
                           other_pos +
                           other_vel +
                           other_type +
                           food_pos +
                           [np.array([agent.state.hunger])] +
                           comm)
        return np.concatenate([agent.state.p_vel] +
                              [agent.state.p_pos] +
                              other_pos +
                              other_vel +
                              other_type +
                              [np.array([agent.state.hunger])] +
                              comm)
