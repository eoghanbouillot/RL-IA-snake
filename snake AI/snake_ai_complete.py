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
BLOCK_SIZE = 30
WINDOW_WIDTH = GRID_WIDTH * BLOCK_SIZE + 400  # Espace pour les stats à droite
WINDOW_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
FPS = 15  # Vitesse du jeu

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BLUE = (30, 144, 255)
YELLOW = (255, 215, 0)
PURPLE = (138, 43, 226)
GRAY = (50, 50, 50)
LIGHT_GRAY = (180, 180, 180)
ORANGE = (255, 140, 0)

class SnakeGame:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.reset()
        
    def reset(self):
        # Position initiale au centre
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
        """Retourne l'état pour l'IA (11 valeurs)"""
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
            # Danger tout droit
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            # Danger à droite
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            # Danger à gauche
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Direction actuelle
            dir_l, dir_r, dir_u, dir_d,
            
            # Position de la nourriture
            self.food[0] < head[0],  # nourriture à gauche
            self.food[0] > head[0],  # nourriture à droite
            self.food[1] < head[1],  # nourriture en haut
            self.food[1] > head[1]   # nourriture en bas
        ]
        
        return np.array(state, dtype=int)
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        
        # Collision avec les murs
        if point[0] >= self.width or point[0] < 0 or point[1] >= self.height or point[1] < 0:
            return True
        
        # Collision avec soi-même
        if point in self.snake[1:]:
            return True
        
        return False
    
    def step(self, action):
        """
        action: [tout droit, droite, gauche]
        """
        self.frame_iteration += 1
        
        # Nouvelle direction
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
        
        # Déplacer
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
        
        # Vérifier game over
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Manger la nourriture
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


class GameUI:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('🐍 Snake IA - Apprentissage par Renforcement')
        self.clock = pygame.time.Clock()
        
        # Polices
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Pour le graphique
        self.scores_history = []
        self.avg_scores_history = []
        
    def draw_text(self, text, font, color, x, y, align='left'):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if align == 'center':
            text_rect.center = (x, y)
        elif align == 'right':
            text_rect.right = x
        else:
            text_rect.left = x
        
        text_rect.top = y
        self.display.blit(text_surface, text_rect)
    
    def draw_game(self, game, agent, record, avg_score):
        self.display.fill(BLACK)
        
        # Zone de jeu avec bordure
        game_area = pygame.Rect(0, 0, GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE)
        pygame.draw.rect(self.display, GRAY, game_area, 3)
        
        # Dessiner la grille
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, GRAY, rect, 1)
        
        # Dessiner le serpent
        for i, (x, y) in enumerate(game.snake):
            rect = pygame.Rect(x * BLOCK_SIZE + 2, y * BLOCK_SIZE + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            if i == 0:
                # Tête
                pygame.draw.rect(self.display, YELLOW, rect, border_radius=8)
                pygame.draw.rect(self.display, ORANGE, rect, 2, border_radius=8)
                # Yeux
                eye_size = 4
                if game.direction == 'RIGHT':
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 20, y * BLOCK_SIZE + 10), eye_size)
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 20, y * BLOCK_SIZE + 20), eye_size)
                elif game.direction == 'LEFT':
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 10, y * BLOCK_SIZE + 10), eye_size)
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 10, y * BLOCK_SIZE + 20), eye_size)
                elif game.direction == 'UP':
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 10, y * BLOCK_SIZE + 10), eye_size)
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 20, y * BLOCK_SIZE + 10), eye_size)
                else:  # DOWN
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 10, y * BLOCK_SIZE + 20), eye_size)
                    pygame.draw.circle(self.display, BLACK, (x * BLOCK_SIZE + 20, y * BLOCK_SIZE + 20), eye_size)
            else:
                # Corps
                color_intensity = max(100, 255 - i * 5)
                body_color = (0, color_intensity, 0)
                pygame.draw.rect(self.display, body_color, rect, border_radius=5)
                pygame.draw.rect(self.display, DARK_GREEN, rect, 2, border_radius=5)
        
        # Dessiner la nourriture (pomme)
        food_x, food_y = game.food
        food_rect = pygame.Rect(food_x * BLOCK_SIZE + 4, food_y * BLOCK_SIZE + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8)
        pygame.draw.circle(self.display, RED, food_rect.center, BLOCK_SIZE // 2 - 4)
        # Petite feuille
        leaf_rect = pygame.Rect(food_x * BLOCK_SIZE + BLOCK_SIZE - 10, food_y * BLOCK_SIZE + 2, 6, 8)
        pygame.draw.ellipse(self.display, GREEN, leaf_rect)
        
        # PANNEAU DE STATISTIQUES À DROITE
        stats_x = GRID_WIDTH * BLOCK_SIZE + 20
        
        # Titre
        self.draw_text("🧠 IA SNAKE", self.font_large, PURPLE, stats_x, 20)
        
        # Séparateur
        pygame.draw.line(self.display, LIGHT_GRAY, (stats_x, 70), (WINDOW_WIDTH - 20, 70), 2)
        
        # Génération actuelle
        y_pos = 90
        self.draw_text("GÉNÉRATION", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        self.draw_text(f"#{agent.n_games}", self.font_large, WHITE, stats_x, y_pos + 25)
        
        y_pos += 90
        pygame.draw.line(self.display, GRAY, (stats_x, y_pos), (WINDOW_WIDTH - 20, y_pos), 1)
        
        # Score actuel
        y_pos += 15
        self.draw_text("Score Actuel", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        score_color = GREEN if game.score > 10 else YELLOW if game.score > 5 else WHITE
        self.draw_text(f"{game.score}", self.font_large, score_color, stats_x, y_pos + 25)
        
        # Record
        y_pos += 90
        self.draw_text("🏆 Record", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        self.draw_text(f"{record}", self.font_large, ORANGE, stats_x, y_pos + 25)
        
        # Moyenne
        y_pos += 90
        self.draw_text("📊 Moyenne", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        self.draw_text(f"{avg_score:.1f}", self.font_medium, BLUE, stats_x, y_pos + 25)
        
        y_pos += 75
        pygame.draw.line(self.display, GRAY, (stats_x, y_pos), (WINDOW_WIDTH - 20, y_pos), 1)
        
        # Graphique mini
        y_pos += 15
        self.draw_text("Progression", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        self.draw_mini_graph(stats_x, y_pos + 30)
        
        # Instructions en bas
        y_pos = WINDOW_HEIGHT - 60
        self.draw_text("Contrôles:", self.font_small, LIGHT_GRAY, stats_x, y_pos)
        self.draw_text("ESPACE = Pause", self.font_small, WHITE, stats_x, y_pos + 20)
        self.draw_text("Q = Quitter", self.font_small, WHITE, stats_x, y_pos + 40)
        
        pygame.display.flip()
    
    def draw_mini_graph(self, x, y):
        """Dessine un mini graphique de progression"""
        graph_width = 350
        graph_height = 120
        
        # Fond
        graph_rect = pygame.Rect(x, y, graph_width, graph_height)
        pygame.draw.rect(self.display, GRAY, graph_rect, 1)
        
        if len(self.scores_history) < 2:
            return
        
        # Normaliser les données
        max_score = max(max(self.scores_history[-50:]) if len(self.scores_history) > 0 else 1, 1)
        
        # Dessiner les scores
        points_scores = []
        points_avg = []
        
        recent_scores = self.scores_history[-50:]
        recent_avg = self.avg_scores_history[-50:]
        
        for i, score in enumerate(recent_scores):
            px = x + (i / max(len(recent_scores) - 1, 1)) * graph_width
            py = y + graph_height - (score / max_score) * graph_height
            points_scores.append((px, py))
        
        for i, score in enumerate(recent_avg):
            px = x + (i / max(len(recent_avg) - 1, 1)) * graph_width
            py = y + graph_height - (score / max_score) * graph_height
            points_avg.append((px, py))
        
        # Tracer les lignes
        if len(points_scores) > 1:
            pygame.draw.lines(self.display, LIGHT_GRAY, False, points_scores, 1)
        if len(points_avg) > 1:
            pygame.draw.lines(self.display, BLUE, False, points_avg, 2)
        
        # Légende
        self.draw_text(f"Max: {max_score}", self.font_small, WHITE, x + 5, y + 5)
    
    def update_graph(self, score, avg_score):
        self.scores_history.append(score)
        self.avg_scores_history.append(avg_score)


def train():
    """Fonction principale"""
    game = SnakeGame()
    agent = Agent()
    ui = GameUI()
    
    record = 0
    total_score = 0
    paused = False
    
    print("🐍 SNAKE IA - APPRENTISSAGE PAR RENFORCEMENT")
    print("=" * 60)
    print("L'IA va apprendre à jouer toute seule!")
    print("Regarde la génération augmenter et le score s'améliorer!")
    print("=" * 60)
    print("\nÉVOLUTION EN COURS...\n")
    
    running = True
    while running:
        # Événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸️  PAUSE' if paused else '▶️  REPRISE'}")
                if event.key == pygame.K_q:
                    running = False
        
        if not paused:
            # Obtenir état et action
            state_old = game.get_state()
            final_move = agent.get_action(state_old)
            
            # Jouer
            reward, done, score = game.step(final_move)
            state_new = game.get_state()
            
            # Entraîner
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # Partie terminée
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    print(f"🏆 NOUVEAU RECORD: {record} (Génération {agent.n_games})")
                
                total_score += score
                avg_score = total_score / agent.n_games
                
                # Mise à jour graphique
                ui.update_graph(score, avg_score)
                
                # Log tous les 10 générations
                if agent.n_games % 10 == 0:
                    print(f"Génération {agent.n_games:4d} | Score: {score:3d} | Record: {record:3d} | Moyenne: {avg_score:.2f}")
        
        # Affichage
        avg_score = total_score / max(agent.n_games, 1)
        ui.draw_game(game, agent, record, avg_score)
        ui.clock.tick(FPS)
    
    # Fin
    print("\n" + "=" * 60)
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"📊 Total générations: {agent.n_games}")
    print(f"🏆 Record: {record}")
    print(f"📈 Score moyen: {avg_score:.2f}")
    print("=" * 60)
    
    # Sauvegarder
    torch.save(agent.model.state_dict(), '/mnt/user-data/outputs/snake_model.pth')
    print("💾 Modèle sauvegardé: snake_model.pth")
    
    pygame.quit()


if __name__ == '__main__':
    train()
