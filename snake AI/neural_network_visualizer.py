import pygame
import numpy as np
import torch
import torch.nn as nn
import math
import random
from collections import deque

# Configuration
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS = 60

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (30, 144, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 127)
YELLOW = (255, 215, 0)
ORANGE = (255, 140, 0)
RED = (220, 20, 60)
PURPLE = (138, 43, 226)
PINK = (255, 105, 180)
GRAY = (70, 70, 70)
LIGHT_GRAY = (150, 150, 150)

# Chargement du modèle (simplifié pour la démo)
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


class Neuron:
    """Représente un neurone visuel"""
    def __init__(self, x, y, radius=8):
        self.x = x
        self.y = y
        self.radius = radius
        self.activation = 0.0
        self.target_activation = 0.0
        self.pulse_effect = 0.0
        
    def update(self):
        # Smooth transition vers la valeur cible
        self.activation += (self.target_activation - self.activation) * 0.15
        
        # Effet de pulsation
        if abs(self.target_activation) > 0.1:
            self.pulse_effect = min(1.0, self.pulse_effect + 0.1)
        else:
            self.pulse_effect = max(0.0, self.pulse_effect - 0.05)
    
    def draw(self, screen):
        # Couleur basée sur l'activation
        intensity = abs(self.activation)
        if self.activation > 0:
            # Positif = vert/cyan
            color = (
                int(0 + intensity * 100),
                int(255 * intensity),
                int(127 + intensity * 128)
            )
        else:
            # Négatif = rouge/orange
            color = (
                int(255 * intensity),
                int(100 * intensity),
                int(0)
            )
        
        # Effet de glow
        glow_radius = self.radius + int(self.pulse_effect * 10)
        for i in range(3):
            alpha_radius = glow_radius + i * 3
            glow_surface = pygame.Surface((alpha_radius * 2, alpha_radius * 2), pygame.SRCALPHA)
            alpha = int(50 * self.pulse_effect * (1 - i / 3))
            pygame.draw.circle(glow_surface, (*color, alpha), (alpha_radius, alpha_radius), alpha_radius)
            screen.blit(glow_surface, (self.x - alpha_radius, self.y - alpha_radius))
        
        # Neurone principal
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius, 2)


class Connection:
    """Représente une connexion entre neurones"""
    def __init__(self, neuron_from, neuron_to, weight=0.0):
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = weight
        self.particles = deque(maxlen=5)
        self.active = False
        
    def add_particle(self):
        if self.active and random.random() < 0.3:
            self.particles.append({
                'progress': 0.0,
                'life': 1.0
            })
    
    def update(self):
        # Mettre à jour les particules
        for particle in list(self.particles):
            particle['progress'] += 0.05
            particle['life'] -= 0.02
            if particle['progress'] >= 1.0 or particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        # Couleur basée sur le poids
        intensity = min(1.0, abs(self.weight))
        if self.weight > 0:
            color = (0, int(255 * intensity), int(200 * intensity))
        else:
            color = (int(255 * intensity), int(100 * intensity), 0)
        
        # Dessiner la ligne de connexion
        alpha = int(50 + 100 * intensity) if self.active else 30
        line_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(line_surface, (*color, alpha), 
                        (self.neuron_from.x, self.neuron_from.y),
                        (self.neuron_to.x, self.neuron_to.y), 
                        2 if self.active else 1)
        screen.blit(line_surface, (0, 0))
        
        # Dessiner les particules qui "voyagent"
        for particle in self.particles:
            progress = particle['progress']
            life = particle['life']
            
            x = self.neuron_from.x + (self.neuron_to.x - self.neuron_from.x) * progress
            y = self.neuron_from.y + (self.neuron_to.y - self.neuron_from.y) * progress
            
            particle_color = (255, 255, 0, int(255 * life))
            particle_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(particle_surface, particle_color, (5, 5), 4)
            screen.blit(particle_surface, (x - 5, y - 5))


class NeuralNetworkVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('🧠 Visualisation du Réseau de Neurones - Snake IA')
        self.clock = pygame.time.Clock()
        
        # Polices
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 20)
        
        # Créer le réseau
        self.model = DQNetwork(11, 256, 3)
        self.setup_network()
        
        # États de démonstration
        self.demo_states = self.generate_demo_states()
        self.current_demo = 0
        self.auto_demo = True
        self.demo_timer = 0
        
        # Effets visuels
        self.particles_background = []
        for _ in range(50):
            self.particles_background.append({
                'x': random.randint(0, WINDOW_WIDTH),
                'y': random.randint(0, WINDOW_HEIGHT),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-0.5, 0.5),
                'size': random.randint(1, 3)
            })
        
        print("🧠 Visualisation du réseau de neurones lancée!")
        print("=" * 60)
        print("Appuyez sur ESPACE pour changer d'état de démonstration")
        print("Appuyez sur A pour activer/désactiver la démo automatique")
        print("Appuyez sur Q pour quitter")
        print("=" * 60)
    
    def setup_network(self):
        """Créer la structure visuelle du réseau"""
        self.layers = []
        
        # Configuration des couches
        layer_config = [
            {'name': 'INPUT', 'size': 11, 'x': 200},
            {'name': 'HIDDEN 1', 'size': 32, 'x': 550},  # On visualise 32 au lieu de 256
            {'name': 'HIDDEN 2', 'size': 32, 'x': 900},
            {'name': 'OUTPUT', 'size': 3, 'x': 1250}
        ]
        
        # Créer les neurones pour chaque couche
        for layer_info in layer_config:
            neurons = []
            size = layer_info['size']
            x = layer_info['x']
            
            # Espacement vertical
            spacing = min(600 / (size + 1), 50)
            start_y = (WINDOW_HEIGHT - spacing * (size - 1)) / 2
            
            for i in range(size):
                y = start_y + i * spacing
                radius = 12 if layer_info['name'] in ['INPUT', 'OUTPUT'] else 8
                neurons.append(Neuron(x, y, radius))
            
            self.layers.append({
                'neurons': neurons,
                'name': layer_info['name']
            })
        
        # Créer les connexions (sélection pour ne pas surcharger)
        self.connections = []
        for i in range(len(self.layers) - 1):
            layer_from = self.layers[i]['neurons']
            layer_to = self.layers[i + 1]['neurons']
            
            # Créer un sous-ensemble de connexions pour la visualisation
            num_connections = min(len(layer_from) * len(layer_to), 300)
            for _ in range(num_connections):
                neuron_from = random.choice(layer_from)
                neuron_to = random.choice(layer_to)
                weight = random.uniform(-1, 1)
                self.connections.append(Connection(neuron_from, neuron_to, weight))
    
    def generate_demo_states(self):
        """Générer des états de démonstration"""
        return [
            # Danger devant
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            # Danger à droite
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            # Danger à gauche
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            # Nourriture à gauche
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            # Nourriture à droite
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            # Situation complexe
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        ]
    
    def forward_pass(self, state):
        """Faire une passe avant et récupérer les activations"""
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        # Layer 1
        x1 = self.model.linear1(state_tensor)
        a1 = torch.relu(x1)
        
        # Layer 2
        x2 = self.model.linear2(a1)
        a2 = torch.relu(x2)
        
        # Layer 3
        output = self.model.linear3(a2)
        
        return state, a1[:32].detach().numpy(), a2[:32].detach().numpy(), output.detach().numpy()
    
    def update_network(self, state):
        """Mettre à jour le réseau avec un nouvel état"""
        activations = self.forward_pass(state)
        
        # Mettre à jour les neurones
        for i, layer_activations in enumerate(activations):
            for j, neuron in enumerate(self.layers[i]['neurons']):
                if j < len(layer_activations):
                    neuron.target_activation = float(layer_activations[j])
        
        # Activer les connexions
        for conn in self.connections:
            # Vérifier si les neurones sont actifs
            if conn.neuron_from.activation > 0.3 or conn.neuron_from.activation < -0.3:
                conn.active = True
                conn.add_particle()
            else:
                conn.active = False
    
    def draw_background_particles(self):
        """Dessiner des particules de fond pour l'ambiance"""
        for particle in self.particles_background:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Rebondir sur les bords
            if particle['x'] < 0 or particle['x'] > WINDOW_WIDTH:
                particle['vx'] *= -1
            if particle['y'] < 0 or particle['y'] > WINDOW_HEIGHT:
                particle['vy'] *= -1
            
            alpha = random.randint(20, 60)
            color = BLUE if random.random() < 0.5 else PURPLE
            particle_surface = pygame.Surface((particle['size']*2, particle['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surface, (*color, alpha), 
                             (particle['size'], particle['size']), particle['size'])
            self.screen.blit(particle_surface, (particle['x'], particle['y']))
    
    def draw_labels(self):
        """Dessiner les labels des couches"""
        # Titre principal
        title = self.font_large.render("🧠 RÉSEAU DE NEURONES - DEEP Q-LEARNING", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Labels des couches
        y_label = 100
        for i, layer in enumerate(self.layers):
            # Nom de la couche
            label = self.font_medium.render(layer['name'], True, CYAN)
            x_pos = layer['neurons'][0].x
            label_rect = label.get_rect(center=(x_pos, y_label))
            self.screen.blit(label, label_rect)
            
            # Nombre de neurones
            size_text = self.font_small.render(f"({len(layer['neurons'])} neurones)", True, LIGHT_GRAY)
            size_rect = size_text.get_rect(center=(x_pos, y_label + 30))
            self.screen.blit(size_text, size_rect)
        
        # Labels des inputs
        input_labels = [
            "Danger devant",
            "Danger droite",
            "Danger gauche",
            "Direction LEFT",
            "Direction RIGHT",
            "Direction UP",
            "Direction DOWN",
            "Food gauche",
            "Food droite",
            "Food haut",
            "Food bas"
        ]
        
        for i, (neuron, label_text) in enumerate(zip(self.layers[0]['neurons'], input_labels)):
            label = self.font_small.render(label_text, True, LIGHT_GRAY)
            self.screen.blit(label, (20, neuron.y - 8))
        
        # Labels des outputs
        output_labels = ["⬆️ TOUT DROIT", "↪️ DROITE", "↩️ GAUCHE"]
        colors = [GREEN, YELLOW, ORANGE]
        
        for i, (neuron, label_text, color) in enumerate(zip(self.layers[-1]['neurons'], output_labels, colors)):
            label = self.font_medium.render(label_text, True, color)
            self.screen.blit(label, (neuron.x + 40, neuron.y - 12))
            
            # Valeur d'activation
            activation_value = neuron.activation
            value_text = self.font_small.render(f"{activation_value:.2f}", True, WHITE)
            self.screen.blit(value_text, (neuron.x + 40, neuron.y + 15))
    
    def draw_info_panel(self):
        """Dessiner le panneau d'information"""
        panel_y = WINDOW_HEIGHT - 80
        
        # Fond du panneau
        panel_rect = pygame.Rect(0, panel_y, WINDOW_WIDTH, 80)
        panel_surface = pygame.Surface((WINDOW_WIDTH, 80), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (*BLACK, 200), panel_rect)
        self.screen.blit(panel_surface, (0, panel_y))
        
        # Informations
        demo_text = f"Démo {self.current_demo + 1}/{len(self.demo_states)}"
        auto_text = "AUTO: ON" if self.auto_demo else "AUTO: OFF"
        
        info_text = self.font_small.render(
            f"{demo_text} | {auto_text} | ESPACE: Changer | A: Auto ON/OFF | Q: Quitter",
            True, WHITE
        )
        info_rect = info_text.get_rect(center=(WINDOW_WIDTH // 2, panel_y + 25))
        self.screen.blit(info_text, info_rect)
        
        # Explication
        explain = self.font_small.render(
            "Les neurones s'allument selon l'état du jeu • Les particules montrent le flux d'information",
            True, CYAN
        )
        explain_rect = explain.get_rect(center=(WINDOW_WIDTH // 2, panel_y + 55))
        self.screen.blit(explain, explain_rect)
    
    def run(self):
        """Boucle principale"""
        running = True
        
        while running:
            # Événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.current_demo = (self.current_demo + 1) % len(self.demo_states)
                        print(f"→ Démonstration {self.current_demo + 1}")
                    if event.key == pygame.K_a:
                        self.auto_demo = not self.auto_demo
                        print(f"Auto-démo: {'ON' if self.auto_demo else 'OFF'}")
                    if event.key == pygame.K_q:
                        running = False
            
            # Auto-démo
            if self.auto_demo:
                self.demo_timer += 1
                if self.demo_timer > 120:  # Changer toutes les 2 secondes
                    self.demo_timer = 0
                    self.current_demo = (self.current_demo + 1) % len(self.demo_states)
            
            # Mettre à jour le réseau
            current_state = self.demo_states[self.current_demo]
            self.update_network(current_state)
            
            # Mettre à jour les neurones et connexions
            for layer in self.layers:
                for neuron in layer['neurons']:
                    neuron.update()
            
            for conn in self.connections:
                conn.update()
            
            # Dessiner
            self.screen.fill(BLACK)
            self.draw_background_particles()
            
            # Dessiner connexions puis neurones
            for conn in self.connections:
                conn.draw(self.screen)
            
            for layer in self.layers:
                for neuron in layer['neurons']:
                    neuron.draw(self.screen)
            
            self.draw_labels()
            self.draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        print("\n✅ Visualisation terminée!")


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     🧠 VISUALISATION DU RÉSEAU DE NEURONES 🧠            ║
    ╚════════════════════════════════════════════════════════════╝
    
    Cette visualisation montre comment fonctionne le cerveau de l'IA!
    
    🎨 Tu vas voir:
       • Les neurones qui s'allument en temps réel
       • Les connexions entre les neurones
       • Le flux d'information (particules jaunes)
       • Les décisions prises par l'IA
    
    🎮 Contrôles:
       • ESPACE: Changer de scénario
       • A: Auto-démo ON/OFF
       • Q: Quitter
    
    ⏱️  Chargement...
    """)
    
    visualizer = NeuralNetworkVisualizer()
    visualizer.run()
