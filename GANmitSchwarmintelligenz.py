import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import copy
import random
import os
import json

# Initialisiere colorama
init(autoreset=True)

# Prüfen, ob GPU verfügbar ist, und entsprechend das Device setzen
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")

# Erstelle Ausgabeverzeichnis
os.makedirs("./swarm_gan_outputs", exist_ok=True)

# Generator-Netzwerk mit verbesserten Aktivierungsfunktionen
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size  # Speichere Architekturinformationen
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(hidden_size * 2, output_size),
            nn.Tanh()
        )
        
        # Gewichte mit He-Initialisierung
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Discriminator-Netzwerk mit verbesserten Aktivierungsfunktionen
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size  # Speichere Architekturinformationen
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Gewichte mit He-Initialisierung
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Funktion zur Überwachung aktiver Neuronen
def count_active_neurons(model, x):
    """Zählt aktive Neuronen nach Schicht-spezifischen Forward-Passes"""
    model.eval()  # Evaluierungsmodus setzen
    with torch.no_grad():
        # Für Generator
        if isinstance(model, Generator):
            # Für die erste Schicht
            h1 = model.model[0](x)  # Linear
            h1 = model.model[1](h1)  # BatchNorm
            h1 = F.leaky_relu(h1, 0.2)  # LeakyReLU
            active1 = (h1 > 0).float().mean().item() * 100
            
            # Für die zweite Schicht
            h2 = model.model[3](h1)  # Linear
            h2 = model.model[4](h2)  # BatchNorm
            h2 = F.leaky_relu(h2, 0.2)  # LeakyReLU
            active2 = (h2 > 0).float().mean().item() * 100
            
            # Durchschnitt der aktiven Neuronen
            active_avg = (active1 + active2) / 2
        
        # Für Discriminator
        else:
            # Für die erste Schicht
            h1 = model.model[0](x)  # Linear
            h1 = F.leaky_relu(h1, 0.2)  # LeakyReLU
            active1 = (h1 > 0).float().mean().item() * 100
            
            # Für die zweite Schicht
            h2 = model.model[3](h1)  # Linear
            h2 = F.leaky_relu(h2, 0.2)  # LeakyReLU
            active2 = (h2 > 0).float().mean().item() * 100
            
            # Durchschnitt der aktiven Neuronen
            active_avg = (active1 + active2) / 2
    
    model.train()  # Zurück zum Trainingsmodus
    return active_avg

# Funktion zum Generieren und Visualisieren von Punkten
def visualize_points(generator, epoch, save_path=None, title_prefix=""):
    """Generiert und visualisiert Punkte, optional mit Speicherfunktion"""
    generator.eval()  # Evaluierungsmodus
    with torch.no_grad():
        z = torch.randn(500, 2, device=device)  # 500 Datenpunkte generieren
        generated_points = generator(z).cpu().numpy()
    
    # Erstelle einen Einheitskreis
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    
    # Berechne, welche Punkte innerhalb des Einheitskreises liegen
    distances = np.sqrt(generated_points[:, 0]**2 + generated_points[:, 1]**2)
    inside_points = generated_points[distances <= 1]
    outside_points = generated_points[distances > 1]
    
    # Pi-Schätzung
    pi_estimate = 4 * (len(inside_points) / len(generated_points))
    
    # Visualisierung
    plt.figure(figsize=(8, 8))
    plt.plot(circle_x, circle_y, 'r-', alpha=0.5)
    plt.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', alpha=0.5, label='Innerhalb')
    plt.scatter(outside_points[:, 0], outside_points[:, 1], color='red', alpha=0.5, label='Außerhalb')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'{title_prefix} Epoch {epoch}: π-Schätzung = {pi_estimate:.6f}')
    plt.legend()
    
    if save_path:
        try:
            plt.savefig(f"{save_path}/{title_prefix.replace(' ', '_')}_points_epoch_{epoch}.png")
        except Exception as e:
            print(f"Warnung: Konnte Grafik nicht speichern: {e}")
        plt.close()
    else:
        plt.show()
    
    generator.train()  # Zurück zum Trainingsmodus
    return pi_estimate, distances

# Einzelner GAN-Agent im Schwarm
class GANAgent:
    def __init__(self, agent_id, input_size=2, hidden_size=16, output_size=2, learning_rate=0.0005):
        self.agent_id = agent_id
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Erstelle Generator und Discriminator
        self.generator = Generator(input_size, hidden_size, output_size).to(device)
        self.discriminator = Discriminator(output_size, hidden_size).to(device)
        
        # Verlustfunktion
        self.criterion = nn.BCELoss()
        
        # Optimierer
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Tracking-Metriken
        self.metrics = {
            'loss_history': {'d_loss': [], 'g_loss': []},
            'pi_estimates': [],
            'active_neurons': {'gen': [], 'dis': []},
            'fitness_score': 0
        }
        
        # Statusinformationen
        self.best_model_state = None
        self.best_fitness = -float('inf')
        self.iteration = 0
        
    def train_step(self, batch_size=500):
        """Führt einen einzelnen Trainingsschritt durch"""
        # Training des Discriminators
        self.optimizer_D.zero_grad()
        
        # Generiere zufällige Punkte im latenten Raum als Batch
        z = torch.randn(batch_size, 2, device=device)
        with torch.no_grad():
            generated_points = self.generator(z)
        
        # Berechne, ob Punkte im Einheitskreis liegen
        distances = torch.sum(generated_points**2, dim=1)
        real_labels = (distances <= 1).float().view(-1, 1)
        fake_labels = torch.zeros_like(real_labels, device=device)
        
        # Discriminator-Vorhersagen
        d_output_real = self.discriminator(generated_points)
        
        # Discriminator-Verlust
        d_loss = self.criterion(d_output_real, real_labels)
        d_loss.backward()
        self.optimizer_D.step()
        
        # Training des Generators
        self.optimizer_G.zero_grad()
        
        # Generiere neue Punkte
        z = torch.randn(batch_size, 2, device=device)
        generated_points = self.generator(z)
        
        # Berechne, ob Punkte im Einheitskreis liegen
        distances = torch.sum(generated_points**2, dim=1)
        target_labels = (distances <= 1).float().view(-1, 1)
        
        # Generator will den Discriminator täuschen
        g_output = self.discriminator(generated_points)
        g_loss = self.criterion(g_output, target_labels)
        g_loss.backward()
        self.optimizer_G.step()
        
        # Tracking aktualisieren
        self.metrics['loss_history']['d_loss'].append(d_loss.item())
        self.metrics['loss_history']['g_loss'].append(g_loss.item())
        self.iteration += 1
        
        return d_loss.item(), g_loss.item()
    
    def evaluate(self, epoch, save_visualization=False):
        """Evaluiert den aktuellen Zustand des Agenten"""
        # Berechne aktive Neuronen
        z_sample = torch.randn(500, 2, device=device)
        active_neurons_gen = count_active_neurons(self.generator, z_sample)
        gen_points = self.generator(z_sample)
        active_neurons_dis = count_active_neurons(self.discriminator, gen_points)
        
        # Aktualisiere Tracking
        self.metrics['active_neurons']['gen'].append(active_neurons_gen)
        self.metrics['active_neurons']['dis'].append(active_neurons_dis)
        
        # Führe Visualisierung durch
        save_path = "./swarm_gan_outputs" if save_visualization else None
        pi_est, distances = visualize_points(
            self.generator, epoch, 
            save_path=save_path, 
            title_prefix=f"Agent {self.agent_id}"
        )
        self.metrics['pi_estimates'].append(pi_est)
        
        # Berechne Fitness-Score basierend auf mehreren Metriken
        # 1. Gleichmäßige Verteilung der Punkte innerhalb/außerhalb des Kreises
        # 2. Hohe Aktivierung der Neuronen
        # 3. Niedrige Verluste
        
        # Fitness für gleichmäßige Verteilung (genau 50% der Punkte sollten im Kreis sein für optimale Ressourcennutzung)
        # Korrektur hier: Verwenden von NumPy-Funktionen statt PyTorch-Funktionen
        in_circle_ratio = (distances <= 1).astype(float).mean()
        distribution_fitness = 1 - abs(in_circle_ratio - 0.5) * 2  # 1 ist perfekt, 0 ist schlecht
        
        # Fitness für Neuronenaktivierung (höher ist besser)
        activation_fitness = (active_neurons_gen + active_neurons_dis) / 200  # Maximal 1
        
        # Fitness für niedrige Verluste
        d_loss = self.metrics['loss_history']['d_loss'][-1]
        g_loss = self.metrics['loss_history']['g_loss'][-1]
        loss_fitness = max(0, 1 - (d_loss + g_loss) / 2)  # Maximal 1
        
        # Kombinierte Fitness
        fitness = (distribution_fitness * 0.5) + (activation_fitness * 0.3) + (loss_fitness * 0.2)
        self.metrics['fitness_score'] = fitness
        
        # Speichere den besten Zustand
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_model_state = {
                'generator': copy.deepcopy(self.generator.state_dict()),
                'discriminator': copy.deepcopy(self.discriminator.state_dict())
            }
        
        return {
            'fitness': fitness,
            'pi_estimate': pi_est,
            'active_neurons_gen': active_neurons_gen,
            'active_neurons_dis': active_neurons_dis,
            'd_loss': d_loss,
            'g_loss': g_loss
        }
    
    def load_best_state(self):
        """Lädt den besten gespeicherten Zustand"""
        if self.best_model_state is not None:
            self.generator.load_state_dict(self.best_model_state['generator'])
            self.discriminator.load_state_dict(self.best_model_state['discriminator'])
    
    def mutate_hyperparameters(self, mutation_rate=0.1):
        """Mutiert Hyperparameter mit einer bestimmten Wahrscheinlichkeit"""
        if random.random() < mutation_rate:
            # Mutiere Lernrate
            self.learning_rate *= random.uniform(0.5, 1.5)
            
            # Aktualisiere Optimierer mit neuer Lernrate
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = self.learning_rate
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = self.learning_rate
            
            print(f"{Fore.YELLOW}Agent {self.agent_id} mutierte Lernrate auf {self.learning_rate:.6f}{Style.RESET_ALL}")
    
    def crossover_with(self, other_agent, crossover_rate=0.7):
        """Führt Crossover mit einem anderen Agenten durch, berücksichtigt unterschiedliche Architekturgrößen"""
        if random.random() < crossover_rate:
            # Prüfe, ob die Architekturen kompatibel sind
            if self.generator.hidden_size == other_agent.generator.hidden_size:
                # Generator-Crossover (vollständig)
                if random.random() < 0.5:
                    self.generator.load_state_dict(other_agent.generator.state_dict())
                    print(f"{Fore.GREEN}Agent {self.agent_id} übernahm Generator von Agent {other_agent.agent_id}{Style.RESET_ALL}")
            else:
                # Unterschiedliche Architekturgrößen
                print(f"{Fore.YELLOW}Generator-Crossover zwischen Agent {self.agent_id} und {other_agent.agent_id} nicht möglich (unterschiedliche Größen){Style.RESET_ALL}")
                print(f"  Agent {self.agent_id}: {self.generator.hidden_size} Neuronen, Agent {other_agent.agent_id}: {other_agent.generator.hidden_size} Neuronen")
                
                # Alternative: Wissenstransfer durch Training
                self.knowledge_transfer_from(other_agent, 'generator', 100)
            
            # Discriminator-Crossover mit Kompatibilitätsprüfung
            if self.discriminator.hidden_size == other_agent.discriminator.hidden_size:
                # Discriminator-Crossover (vollständig)
                if random.random() < 0.5:
                    self.discriminator.load_state_dict(other_agent.discriminator.state_dict())
                    print(f"{Fore.GREEN}Agent {self.agent_id} übernahm Discriminator von Agent {other_agent.agent_id}{Style.RESET_ALL}")
            else:
                # Unterschiedliche Architekturgrößen
                print(f"{Fore.YELLOW}Discriminator-Crossover zwischen Agent {self.agent_id} und {other_agent.agent_id} nicht möglich (unterschiedliche Größen){Style.RESET_ALL}")
                
                # Alternative: Wissenstransfer durch Training
                self.knowledge_transfer_from(other_agent, 'discriminator', 100)
    
    def knowledge_transfer_from(self, teacher_agent, network_type, steps=100):
        """
        Überträgt Wissen von einem Teacher-Agenten durch Distillation
        Diese Methode funktioniert auch bei unterschiedlichen Architekturgrößen
        """
        print(f"{Fore.BLUE}Agent {self.agent_id} lernt von Agent {teacher_agent.agent_id} durch Wissenstransfer ({network_type}){Style.RESET_ALL}")
        
        # Speichere beide Modelle im Eval-Modus
        teacher_agent.generator.eval()
        teacher_agent.discriminator.eval()
        
        if network_type == 'generator':
            # Einfache Distillation für Generator: Der Schüler lernt, den gleichen Output wie der Lehrer zu erzeugen
            optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Generiere Eingabedaten (latent vectors)
                z = torch.randn(100, 2, device=device)
                
                # Teacher-Ausgabe (mit deaktiviertem Gradienten)
                with torch.no_grad():
                    teacher_output = teacher_agent.generator(z)
                
                # Student-Ausgabe
                student_output = self.generator(z)
                
                # Berechne MSE-Verlust zwischen den Ausgaben
                loss = F.mse_loss(student_output, teacher_output)
                
                # Optimiere den Schüler
                loss.backward()
                optimizer.step()
                
                if step % 20 == 0:
                    print(f"  Wissenstransfer Schritt {step}/{steps}: Loss={loss.item():.6f}")
        
        elif network_type == 'discriminator':
            # Distillation für Discriminator: Der Schüler lernt, die gleichen Wahrscheinlichkeiten wie der Lehrer zu erzeugen
            optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Generiere zufällige Punkte
                z = torch.randn(100, 2, device=device)
                with torch.no_grad():
                    generated_points = teacher_agent.generator(z)
                
                # Teacher-Ausgabe (mit deaktiviertem Gradienten)
                with torch.no_grad():
                    teacher_output = teacher_agent.discriminator(generated_points)
                
                # Student-Ausgabe
                student_output = self.discriminator(generated_points)
                
                # Berechne MSE-Verlust zwischen den Ausgaben
                loss = F.mse_loss(student_output, teacher_output)
                
                # Optimiere den Schüler
                loss.backward()
                optimizer.step()
                
                if step % 20 == 0:
                    print(f"  Wissenstransfer Schritt {step}/{steps}: Loss={loss.item():.6f}")
        
        # Zurück zum Trainingsmodus
        teacher_agent.generator.train()
        teacher_agent.discriminator.train()

# Schwarmintelligenz für GANs
class SwarmGAN:
    def __init__(self, num_agents=5, hidden_sizes=None, learning_rates=None):
        """Initialisiert einen Schwarm von GAN-Agenten"""
        self.num_agents = num_agents
        
        # Standardwerte für Hyperparameter
        if hidden_sizes is None:
            hidden_sizes = [16, 24, 32, 48, 64][:num_agents]
            # Wenn mehr Agenten als vorgegebene Werte, fülle mit zufälligen Werten auf
            while len(hidden_sizes) < num_agents:
                hidden_sizes.append(random.choice([16, 24, 32, 48, 64]))
        
        if learning_rates is None:
            learning_rates = [0.0001, 0.0003, 0.0005, 0.0008, 0.001][:num_agents]
            while len(learning_rates) < num_agents:
                learning_rates.append(random.choice([0.0001, 0.0003, 0.0005, 0.0008, 0.001]))
        
        # Erstelle Agenten
        self.agents = []
        for i in range(num_agents):
            agent = GANAgent(
                agent_id=i,
                hidden_size=hidden_sizes[i],
                learning_rate=learning_rates[i]
            )
            self.agents.append(agent)
        
        # Tracking-Metriken für den gesamten Schwarm
        self.metrics = {
            'best_agent_per_epoch': [],
            'avg_fitness_per_epoch': [],
            'best_fitness_overall': -float('inf'),
            'best_agent_overall': None,
            'generation': 0
        }
    
    def train_generation(self, steps_per_agent=1000, batch_size=500):
        """Trainiert alle Agenten für eine Generation"""
        self.metrics['generation'] += 1
        generation = self.metrics['generation']
        
        print(f"\n{Fore.CYAN}===== Generation {generation} ====={Style.RESET_ALL}")
        print(f"Training {self.num_agents} Agenten für jeweils {steps_per_agent} Schritte...")
        
        # Trainiere jeden Agenten
        for agent_idx, agent in enumerate(self.agents):
            print(f"\n{Fore.BLUE}Training Agent {agent_idx} (Neuronen: {agent.hidden_size}, LR: {agent.learning_rate:.6f}){Style.RESET_ALL}")
            
            # Trainiere für die angegebene Anzahl von Schritten
            for step in range(steps_per_agent):
                d_loss, g_loss = agent.train_step(batch_size)
                
                # Zeige Fortschritt an
                if step % (steps_per_agent // 10) == 0:
                    print(f"Schritt {step}/{steps_per_agent}: D_loss: {d_loss:.6f}, G_loss: {g_loss:.6f}")
            
            # Evaluiere nach dem Training
            eval_results = agent.evaluate(
                epoch=generation,
                save_visualization=(agent_idx < 3)  # Speichere nur für die ersten 3 Agenten
            )
            
            print(f"Agent {agent_idx} Fitness: {eval_results['fitness']:.4f}")
            print(f"  Aktive Neuronen: Gen {eval_results['active_neurons_gen']:.2f}%, Dis {eval_results['active_neurons_dis']:.2f}%")
            print(f"  Pi-Schätzung: {eval_results['pi_estimate']:.6f}")
        
        # Finde den besten Agenten dieser Generation
        best_agent_idx = max(range(len(self.agents)), key=lambda i: self.agents[i].metrics['fitness_score'])
        best_agent = self.agents[best_agent_idx]
        best_fitness = best_agent.metrics['fitness_score']
        
        print(f"\n{Fore.GREEN}Bester Agent in Generation {generation}: Agent {best_agent_idx} mit Fitness {best_fitness:.6f}{Style.RESET_ALL}")
        
        # Aktualisiere Schwarm-Metriken
        self.metrics['best_agent_per_epoch'].append(best_agent_idx)
        self.metrics['avg_fitness_per_epoch'].append(sum(agent.metrics['fitness_score'] for agent in self.agents) / len(self.agents))
        
        # Aktualisiere den besten Agenten insgesamt
        if best_fitness > self.metrics['best_fitness_overall']:
            self.metrics['best_fitness_overall'] = best_fitness
            self.metrics['best_agent_overall'] = best_agent_idx
            
            # Speichere die Modelle des besten Agenten
            try:
                torch.save(best_agent.generator.state_dict(), "./swarm_gan_outputs/best_generator.pth")
                torch.save(best_agent.discriminator.state_dict(), "./swarm_gan_outputs/best_discriminator.pth")
            except Exception as e:
                print(f"Warnung: Konnte beste Modelle nicht speichern: {e}")
            
            print(f"{Fore.GREEN}Neuer bester Agent insgesamt: Agent {best_agent_idx} in Generation {generation}{Style.RESET_ALL}")
        
        return best_agent_idx, best_fitness
    
    def apply_swarm_intelligence(self):
        """Wendet Schwarmintelligenz-Prinzipien an, um Agenten zu verbessern"""
        print(f"\n{Fore.YELLOW}Anwenden von Schwarmintelligenz...{Style.RESET_ALL}")
        
        # Sortiere Agenten nach Fitness
        sorted_agents = sorted(self.agents, key=lambda a: a.metrics['fitness_score'], reverse=True)
        
        # Behalte die besten 2 Agenten unverändert (Elitismus)
        elite_agents = sorted_agents[:2]
        other_agents = sorted_agents[2:]
        
        print(f"Elite-Agenten: {[a.agent_id for a in elite_agents]}")
        
        # Für alle nicht-Elite-Agenten
        for agent in other_agents:
            # 1. Crossover mit einem zufälligen Elite-Agenten
            elite_partner = random.choice(elite_agents)
            agent.crossover_with(elite_partner)
            
            # 2. Mutiere Hyperparameter
            agent.mutate_hyperparameters(mutation_rate=0.3)
            
            # 3. Lade besten eigenen Zustand
            agent.load_best_state()
        
        print(f"{Fore.YELLOW}Schwarmintelligenz-Anpassungen abgeschlossen.{Style.RESET_ALL}")
    
    def visualize_swarm_performance(self):
        """Visualisiert die Performance des Schwarms über Generationen"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Durchschnittliche Fitness pro Generation
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['avg_fitness_per_epoch'], 'b-', label='Durchschnittliche Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Durchschnittliche Fitness pro Generation')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Beste Agenten pro Generation
        plt.subplot(2, 2, 2)
        x = range(len(self.metrics['best_agent_per_epoch']))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
        for i in range(self.num_agents):
            mask = [agent_idx == i for agent_idx in self.metrics['best_agent_per_epoch']]
            if any(mask):
                plt.scatter([x[j] for j in range(len(x)) if mask[j]], 
                           [1 for j in range(len(x)) if mask[j]], 
                           c=[colors[i]], label=f'Agent {i}')
        
        plt.yticks([])
        plt.xlabel('Generation')
        plt.title('Bester Agent pro Generation')
        plt.legend()
        
        # Plot 3: Pi-Schätzungen für jeden Agenten
        plt.subplot(2, 2, 3)
        for i, agent in enumerate(self.agents):
            if agent.metrics['pi_estimates']:
                plt.plot(agent.metrics['pi_estimates'], label=f'Agent {i}')
        plt.axhline(y=np.pi, color='r', linestyle='--', label='π')
        plt.xlabel('Evaluation')
        plt.ylabel('π-Schätzung')
        plt.title('π-Schätzungen der Agenten')
        plt.grid(True)
        plt.legend()
        
        # Plot 4: Aktive Neuronen für jeden Agenten
        plt.subplot(2, 2, 4)
        for i, agent in enumerate(self.agents):
            if agent.metrics['active_neurons']['gen']:
                plt.plot(agent.metrics['active_neurons']['gen'], 
                         label=f'Agent {i} (Gen)')
        plt.xlabel('Evaluation')
        plt.ylabel('Aktive Neuronen (%)')
        plt.title('Aktive Neuronen im Generator')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        try:
            plt.savefig('./swarm_gan_outputs/swarm_performance.png')
        except Exception as e:
            print(f"Warnung: Konnte Grafik nicht speichern: {e}")
        plt.close()

# Hauptfunktion zum Training des Schwarms
def train_swarm_gan(num_agents=5, generations=10, steps_per_generation=500):
    """Trainiert einen Schwarm von GANs mit Schwarmintelligenz"""
    print(f"{Fore.CYAN}Starte Training des GAN-Schwarms mit {num_agents} Agenten für {generations} Generationen{Style.RESET_ALL}")
    
    # Erstelle Schwarm mit verschiedenen Hyperparametern
    swarm = SwarmGAN(num_agents=num_agents)
    
    # Training über mehrere Generationen
    for gen in range(generations):
        # Trainiere eine Generation
        best_agent_idx, best_fitness = swarm.train_generation(steps_per_agent=steps_per_generation)
        
        # Visualisiere Schwarm-Performance nach jeder Generation
        swarm.visualize_swarm_performance()
        
        # Wende Schwarmintelligenz an (außer nach der letzten Generation)
        if gen < generations - 1:
            swarm.apply_swarm_intelligence()
    
    # Finde den besten Agenten insgesamt
    best_agent_overall_idx = swarm.metrics['best_agent_overall']
    best_agent_overall = swarm.agents[best_agent_overall_idx]
    
    print(f"\n{Fore.GREEN}Training abgeschlossen!{Style.RESET_ALL}")
    print(f"Bester Agent: Agent {best_agent_overall_idx} mit Fitness {best_agent_overall.metrics['fitness_score']:.6f}")
    
    # Finale Visualisierung des besten Agenten
    best_agent_overall.evaluate(epoch="final", save_visualization=True)
    
    return swarm

# Hauptausführung
if __name__ == "__main__":
    # Konfiguriere Parameter für schnelleres Training
    NUM_AGENTS = 5           # Anzahl der GAN-Agenten im Schwarm
    GENERATIONS = 5          # Reduzierte Anzahl der Generationen
    STEPS_PER_GEN = 200      # Reduzierte Anzahl der Trainingsschritte pro Generation
    
    # Trainiere den Schwarm
    swarm = train_swarm_gan(
        num_agents=NUM_AGENTS,
        generations=GENERATIONS,
        steps_per_generation=STEPS_PER_GEN
    )
    
    print(f"{Fore.GREEN}Schwarmintelligenz-GAN-Training abgeschlossen.{Style.RESET_ALL}")
    print(f"Ergebnisse wurden in ./swarm_gan_outputs/ gespeichert.")
