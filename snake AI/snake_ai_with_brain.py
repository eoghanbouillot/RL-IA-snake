import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
GRID_WIDTH = 20
GRID_HEIGHT = 15
BLOCK_SIZE = 25
GAME_WIDTH = GRID_WIDTH * BLOCK_SIZE
GAME_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 900
FPS = 20

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BLUE = (30, 144, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 215, 0)
ORANGE = (255, 140, 0)
PURPLE = (138, 43, 226)
GRAY = (50, 50, 50)
LIGHT_GRAY = (150, 150, 150)


class SnakeGame:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.reset()
        
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.score = 0
        self.frame_iteration = 0
        self.place_food()
        return self.get_state()
    
    def place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food = (x, y)
            if self.food not in self.snake:
                break
    
    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)
        
        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'
        
        state = [
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1]
        ]
        
        return np.array(state, dtype=int)
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        x, y = self.snake[0]
        
        if self.direction == 'RIGHT':
            x += 1
        elif self.direction == 'LEFT':
            x -= 1
        elif self.direction == 'DOWN':
            y += 1
        elif self.direction == 'UP':
            y -= 1
        
        self.snake.insert(0, (x, y))
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score


class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def forward_with_activations(self, x):
        """Retourne aussi les activations intermédiaires"""
        a1 = torch.relu(self.linear1(x))
        a2 = torch.relu(self.linear2(a1))
        output = self.linear3(a2)
        return output, a1, a2


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)
        self.model = DQNetwork(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
    def get_action_with_activations(self, state):
        """Obtenir l'action et les activations du réseau"""
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction, hidden1, hidden2 = self.model.forward_with_activations(state_tensor)
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = torch.argmax(prediction).item()
        
        final_move[move] = 1
        
        # Retourner l'action et les activations
        return final_move, {
            'input': state,
            'hidden1': hidden1.detach().numpy()[:16],  # Premiers 16 neurones
            'hidden2': hidden2.detach().numpy()[:16],
            'output': prediction.detach().numpy()
        }
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])
    
    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
    
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        
        pred = self.model(states)
        target = pred.clone()
        
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class CombinedUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('🐍 Snake IA + 🧠 Réseau de Neurones en Direct')
        self.clock = pygame.time.Clock()
        
        self.font_huge = pygame.font.Font(None, 56)
        self.font_large = pygame.font.Font(None, 42)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        self.scores_history = []
        
    def draw_game(self, game, x_offset, y_offset):
        """Dessiner le jeu Snake"""
        # Grille
        for gx in range(GRID_WIDTH):
            for gy in range(GRID_HEIGHT):
                rect = pygame.Rect(x_offset + gx * BLOCK_SIZE, y_offset + gy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Serpent
        for i, (x, y) in enumerate(game.snake):
            rect = pygame.Rect(x_offset + x * BLOCK_SIZE + 2, y_offset + y * BLOCK_SIZE + 2, 
                             BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            if i == 0:
                pygame.draw.rect(self.screen, YELLOW, rect, border_radius=6)
                pygame.draw.rect(self.screen, ORANGE, rect, 2, border_radius=6)
            else:
                intensity = max(100, 255 - i * 5)
                pygame.draw.rect(self.screen, (0, intensity, 0), rect, border_radius=4)
        
        # Nourriture
        food_x, food_y = game.food
        food_rect = pygame.Rect(x_offset + food_x * BLOCK_SIZE + 3, y_offset + food_y * BLOCK_SIZE + 3,
                               BLOCK_SIZE - 6, BLOCK_SIZE - 6)
        pygame.draw.circle(self.screen, RED, food_rect.center, BLOCK_SIZE // 2 - 3)
        
        # Bordure
        border_rect = pygame.Rect(x_offset - 2, y_offset - 2, GAME_WIDTH + 4, GAME_HEIGHT + 4)
        pygame.draw.rect(self.screen, CYAN, border_rect, 3)
    
    def draw_neural_network(self, activations, x_offset, y_offset):
        """Dessiner le réseau de neurones simplifié"""
        layer_spacing = 200
        neuron_radius = 8
        
        # Configuration
        layers_config = [
            ('INPUT', activations['input'], CYAN, 11),
            ('HIDDEN 1', activations['hidden1'], BLUE, 16),
            ('HIDDEN 2', activations['hidden2'], PURPLE, 16),
            ('OUTPUT', activations['output'], GREEN, 3)
        ]
        
        all_neurons = []
        
        for layer_idx, (name, values, color, count) in enumerate(layers_config):
            x = x_offset + layer_idx * layer_spacing
            spacing = min(500 / count, 40)
            start_y = y_offset + 250 - (spacing * count) / 2
            
            layer_neurons = []
            
            for i in range(count):
                y = start_y + i * spacing
                
                # Intensité basée sur l'activation
                if i < len(values):
                    activation = float(values[i])
                    intensity = min(1.0, abs(activation))
                    
                    if activation > 0:
                        neuron_color = (0, int(255 * intensity), int(200 * intensity))
                    else:
                        neuron_color = (int(255 * intensity), int(100 * intensity), 0)
                else:
                    neuron_color = GRAY
                
                # Dessiner le neurone
                pygame.draw.circle(self.screen, neuron_color, (int(x), int(y)), neuron_radius)
                pygame.draw.circle(self.screen, WHITE, (int(x), int(y)), neuron_radius, 1)
                
                layer_neurons.append((x, y))
            
            all_neurons.append(layer_neurons)
            
            # Label de couche
            label = self.font_small.render(name, True, color)
            label_rect = label.get_rect(center=(x, y_offset - 30))
            self.screen.blit(label, label_rect)
        
        # Dessiner quelques connexions
        for layer_idx in range(len(all_neurons) - 1):
            for i, (x1, y1) in enumerate(all_neurons[layer_idx]):
                for j, (x2, y2) in enumerate(all_neurons[layer_idx + 1]):
                    if random.random() < 0.1:  # 10% des connexions
                        pygame.draw.line(self.screen, (50, 50, 50), (x1, y1), (x2, y2), 1)
        
        # Labels des outputs
        output_labels = ["TOUT DROIT", "DROITE", "GAUCHE"]
        output_neurons = all_neurons[-1]
        
        for i, (label_text, (x, y)) in enumerate(zip(output_labels, output_neurons)):
            value = activations['output'][i]
            color = GREEN if i == np.argmax(activations['output']) else WHITE
            
            label = self.font_medium.render(f"{label_text}: {value:.2f}", True, color)
            self.screen.blit(label, (x + 20, y - 10))
    
    def draw_stats(self, game, agent, record, avg_score):
        """Dessiner les statistiques"""
        stats_x = 50
        stats_y = GAME_HEIGHT + 100
        
        # Génération
        gen_text = self.font_huge.render(f"GÉN: {agent.n_games}", True, PURPLE)
        self.screen.blit(gen_text, (stats_x, stats_y))
        
        # Score
        score_text = self.font_large.render(f"SCORE: {game.score}", True, YELLOW)
        self.screen.blit(score_text, (stats_x, stats_y + 70))
        
        # Record
        record_text = self.font_large.render(f"RECORD: {record}", True, ORANGE)
        self.screen.blit(record_text, (stats_x, stats_y + 120))
        
        # Moyenne
        avg_text = self.font_medium.render(f"Moyenne: {avg_score:.1f}", True, CYAN)
        self.screen.blit(avg_text, (stats_x, stats_y + 170))
    
    def draw_mini_graph(self):
        """Dessiner un mini graphique"""
        if len(self.scores_history) < 2:
            return
        
        graph_x = 1400
        graph_y = 700
        graph_w = 350
        graph_h = 150
        
        pygame.draw.rect(self.screen, GRAY, (graph_x, graph_y, graph_w, graph_h), 2)
        
        recent = self.scores_history[-50:]
        max_score = max(recent) if recent else 1
        
        points = []
        for i, score in enumerate(recent):
            x = graph_x + (i / len(recent)) * graph_w
            y = graph_y + graph_h - (score / max_score) * graph_h
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, GREEN, False, points, 2)
        
        title = self.font_small.render("PROGRESSION", True, WHITE)
        self.screen.blit(title, (graph_x, graph_y - 25))
    
    def draw(self, game, agent, record, avg_score, activations):
        """Dessiner toute l'interface"""
        self.screen.fill(BLACK)
        
        # Titre
        title = self.font_huge.render("🐍 SNAKE IA + 🧠 CERVEAU EN DIRECT", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Jeu à gauche
        self.draw_game(game, 50, 100)
        
        # Réseau à droite
        self.draw_neural_network(activations, 750, 100)
        
        # Stats en bas à gauche
        self.draw_stats(game, agent, record, avg_score)
        
        # Graphique en bas à droite
        self.draw_mini_graph()
        
        # Instructions
        info = self.font_small.render("ESPACE: Pause | Q: Quitter", True, LIGHT_GRAY)
        self.screen.blit(info, (WINDOW_WIDTH - 250, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()


def train():
    game = SnakeGame()
    agent = Agent()
    ui = CombinedUI()
    
    record = 0
    total_score = 0
    paused = False
    
    print("🐍 + 🧠 SNAKE IA AVEC VISUALISATION DU CERVEAU")
    print("=" * 70)
    print("Regarde le serpent jouer ET son cerveau décider en temps réel!")
    print("=" * 70)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_q:
                    running = False
        
        if not paused:
            state_old = game.get_state()
            final_move, activations = agent.get_action_with_activations(state_old)
            
            reward, done, score = game.step(final_move)
            state_new = game.get_state()
            
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    print(f"🏆 RECORD: {record} (Génération {agent.n_games})")
                
                total_score += score
                ui.scores_history.append(score)
                
                if agent.n_games % 10 == 0:
                    avg_score = total_score / agent.n_games
                    print(f"Gén {agent.n_games:4d} | Score: {score:3d} | Record: {record:3d} | Moy: {avg_score:.2f}")
        
        avg_score = total_score / max(agent.n_games, 1)
        ui.draw(game, agent, record, avg_score, activations if not paused else {
            'input': game.get_state(),
            'hidden1': np.zeros(16),
            'hidden2': np.zeros(16),
            'output': np.zeros(3)
        })
        ui.clock.tick(FPS)
    
    print(f"\n✅ Terminé! Générations: {agent.n_games} | Record: {record}")
    pygame.quit()


if __name__ == '__main__':
    train()
