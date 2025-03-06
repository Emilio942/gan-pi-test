import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Versuche tqdm zu importieren, wenn nicht vorhanden, erstelle einfache Alternative
try:
    from tqdm import tqdm
except ImportError:
    # Einfache Alternative zu tqdm, falls nicht installiert
    class SimpleTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable)
            
        def __iter__(self):
            print(f"{self.desc}: 0/{self.total}")
            for i, item in enumerate(self.iterable):
                if i % (self.total // 10) == 0 and i > 0:
                    print(f"{self.desc}: {i}/{self.total}")
                yield item
                
        def set_description(self, desc):
            self.desc = desc
            
    tqdm = SimpleTqdm

# Initialisiere colorama
init(autoreset=True)

# Prüfen, ob GPU verfügbar ist, und entsprechend das Device setzen
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Verwende Device: {device}")

# Generator-Netzwerk mit Batch-Normalisierung und verbesserten Aktivierungsfunktionen
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
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
        
        # Gewichte mit He-Initialisierung (für ReLU/LeakyReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Discriminator-Netzwerk mit Dropout und verbesserten Aktivierungsfunktionen
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(Discriminator, self).__init__()
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

# Verbesserte Funktion zur Überwachung aktiver Neuronen
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

# Funktion zum Speichern und Visualisieren von generierten Punkten
def visualize_points(generator, epoch, save_path=None):
    """Generiert und visualisiert Punkte, optional mit Speicherfunktion"""
    generator.eval()  # Evaluierungsmodus
    with torch.no_grad():
        z = torch.randn(1000, 2, device=device)
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
    plt.title(f'Epoch {epoch}: π-Schätzung = {pi_estimate:.6f}')
    plt.legend()
    
    if save_path:
        try:
            plt.savefig(f"{save_path}/generated_points_epoch_{epoch}.png")
        except Exception as e:
            print(f"Warnung: Konnte Grafik nicht speichern: {e}")
        plt.close()
    else:
        plt.show()
    
    generator.train()  # Zurück zum Trainingsmodus
    return pi_estimate

# Funktion zur Durchführung des GAN-Trainings und zur Rückgabe der Ergebnisse
def train_and_evaluate_gan(generator, discriminator, num_neurons_gen, num_neurons_dis, learning_rate, epochs, batch_size=64):
    # Verschiebe Modelle auf das richtige Device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Einfachere Verlustfunktion für mehr Stabilität
    criterion = nn.BCELoss()
    
    # Optimierer mit Gewichtsabfall für Regularisierung
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=1e-5)
    
    # Speicher für den letzten "guten" Zustand (nur Modelparameter, nicht Optimierer)
    best_generator_state = None
    best_discriminator_state = None
    best_epoch = 0
    best_loss = float('inf')
    
    # Tracking-Metriken
    loss_history = {'d_loss': [], 'g_loss': []}
    activation_history = {'generator': [], 'discriminator': []}
    pi_estimates = []
    
    # Progress bar für bessere Übersicht
    pbar = tqdm(range(epochs), desc="Training GAN")
    
    for epoch in pbar:
        # Training des Discriminators
        optimizer_D.zero_grad()
        
        # Generiere zufällige Punkte im latenten Raum als Batch
        z = torch.randn(batch_size, 2, device=device)
        generated_points = generator(z)
        
        # Berechne, ob Punkte im Einheitskreis liegen
        distances = torch.sum(generated_points**2, dim=1)
        real_labels = (distances <= 1).float().view(-1, 1)
        fake_labels = torch.zeros_like(real_labels, device=device)
        
        # Discriminator-Ausgaben
        d_output_real = discriminator(generated_points.detach())
        d_output_fake = torch.zeros_like(d_output_real, device=device)
        
        # Discriminator-Verluste
        d_loss_real = criterion(d_output_real, real_labels)
        d_loss_fake = criterion(d_output_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        
        # Gradienten-Clipping 
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()
        
        # Training des Generators
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, 2, device=device)
        generated_points = generator(z)
        distances = torch.sum(generated_points**2, dim=1)
        target_labels = (distances <= 1).float().view(-1, 1)
        
        # Generator will den Discriminator überlisten
        g_output = discriminator(generated_points)
        g_loss = criterion(g_output, target_labels)
        g_loss.backward()
        
        # Gradienten-Clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()
        
        # Tracking-Metriken aktualisieren
        loss_history['d_loss'].append(d_loss.item())
        loss_history['g_loss'].append(g_loss.item())
        
        # Überprüfen auf NaN oder Inf-Werte
        if torch.isnan(d_loss) or torch.isnan(g_loss) or torch.isinf(d_loss) or torch.isinf(g_loss):
            print(f"{Fore.RED}NaN oder Inf-Werte in Verlusten erkannt. Zurücksetzen auf den letzten stabilen Zustand.{Style.RESET_ALL}")
            if best_generator_state is not None:
                generator.load_state_dict(best_generator_state)
                discriminator.load_state_dict(best_discriminator_state)
            continue
        
        # Speichere das beste Modell basierend auf dem kombinierten Verlust
        combined_loss = d_loss.item() + g_loss.item()
        if combined_loss < best_loss:
            best_loss = combined_loss
            best_epoch = epoch
            best_generator_state = generator.state_dict().copy()  # Tiefe Kopie, um Speicher zu schonen
            best_discriminator_state = discriminator.state_dict().copy()
        
        # Berechne Aktivierungsstatistik periodisch, um Ressourcen zu sparen
        if epoch % 1000 == 0:
            # Überwache aktive Neuronen
            z_sample = torch.randn(batch_size, 2, device=device)
            active_neurons_gen = count_active_neurons(generator, z_sample)
            gen_points = generator(z_sample)
            active_neurons_dis = count_active_neurons(discriminator, gen_points)
            
            activation_history['generator'].append(active_neurons_gen)
            activation_history['discriminator'].append(active_neurons_dis)
            
            # Pi-Schätzung und Visualisierung
            if epoch % 5000 == 0:
                pi_est = visualize_points(generator, epoch, save_path="./gan_outputs")
                pi_estimates.append(pi_est)
                
                # Aktualisiere die Fortschrittsanzeige
                pbar.set_description(f"D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, π: {pi_est:.4f}")
                
                # Debugging-Ausgabe
                print(f"\nEpoch {epoch}: D_loss: {Fore.RED}{d_loss.item():.6f}{Style.RESET_ALL}, G_loss: {Fore.GREEN}{g_loss.item():.6f}{Style.RESET_ALL}")
                print(f"Pi-Schätzung: {Fore.YELLOW}{pi_est:.6f}{Style.RESET_ALL}")
                print(f"Aktive Neuronen Generator: {Fore.BLUE}{active_neurons_gen:.2f}%{Style.RESET_ALL}, Discriminator: {Fore.MAGENTA}{active_neurons_dis:.2f}%{Style.RESET_ALL}")
                print(f"Beste Epoche bisher: {best_epoch} mit Verlust: {best_loss:.6f}")
    
    # Lade das beste Modell zum Schluss
    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
        discriminator.load_state_dict(best_discriminator_state)
    
    # Finale Auswertung
    z = torch.randn(1000, 2, device=device)
    generated_points = generator(z)
    distances = torch.sum(generated_points.cpu()**2, dim=1)
    real_labels = (distances <= 1).float()
    
    # Metriken für die Analyse
    pi_estimate = 4 * real_labels.mean().item()
    
    # Visualisiere das endgültige Ergebnis
    final_pi = visualize_points(generator, epochs, save_path="./gan_outputs")
    
    # Speichere Modelle, falls möglich
    try:
        torch.save(generator.state_dict(), "best_generator.pth")
        torch.save(discriminator.state_dict(), "best_discriminator.pth")
    except Exception as e:
        print(f"Warnung: Konnte Modelle nicht speichern: {e}")
    
    return {
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
        'pi_estimate': final_pi,
        'active_neurons_gen': activation_history['generator'][-1] if activation_history['generator'] else 0,
        'active_neurons_dis': activation_history['discriminator'][-1] if activation_history['discriminator'] else 0,
        'best_epoch': best_epoch,
        'loss_history': loss_history,
        'activation_history': activation_history,
        'pi_estimates': pi_estimates,
        'learning_rate': learning_rate
    }

# Hauptfunktion zur Durchführung der Runden
def run_experiment(total_rounds, initial_neurons, neuron_step, initial_learning_rate, epochs_per_round, batch_size=64):
    import os
    import json
    
    # Stelle sicher, dass Ausgabeverzeichnis existiert
    try:
        os.makedirs("./gan_outputs", exist_ok=True)
    except Exception as e:
        print(f"Warnung: Konnte Ausgabeverzeichnis nicht erstellen: {e}")
        print("Visualisierungen und Ergebnisdateien werden nicht gespeichert.")
    
    results = []
    current_learning_rate = initial_learning_rate
    
    # Definiere Funktion für einen einzelnen Durchlauf
    def run_single_round(round_num):
        num_neurons_gen = initial_neurons + (round_num - 1) * neuron_step
        num_neurons_dis = num_neurons_gen
        
        print(f"\n=== {Fore.CYAN}Runde {round_num}/{total_rounds}{Style.RESET_ALL} mit {Fore.BLUE}{num_neurons_gen} Neuronen im Generator{Style.RESET_ALL} und {Fore.MAGENTA}{num_neurons_dis} Neuronen im Discriminator{Style.RESET_ALL} ===")
        
        lr = current_learning_rate * (1.1 ** (round_num - 1))  # Lernrate für diese Runde
        
        generator = Generator(input_size=2, hidden_size=num_neurons_gen, output_size=2)
        discriminator = Discriminator(input_size=2, hidden_size=num_neurons_dis)
        
        result = train_and_evaluate_gan(
            generator, 
            discriminator, 
            num_neurons_gen, 
            num_neurons_dis, 
            lr, 
            epochs_per_round,
            batch_size
        )
        
        result['round'] = round_num
        result['neurons'] = num_neurons_gen
        
        return result
    
    # Einfache sequentielle Ausführung ohne Threading
    for round_num in range(1, total_rounds + 1):
        try:
            result = run_single_round(round_num)
            results.append(result)
            
            # Speichere Zwischenergebnisse, wenn möglich
            try:
                with open(f"./gan_outputs/round_{round_num}_results.json", 'w') as f:
                    json.dump({k: v for k, v in result.items() if not isinstance(v, dict) and not isinstance(v, list)}, f, indent=4)
            except Exception as e:
                print(f"Konnte Ergebnis nicht speichern: {e}")
        except Exception as exc:
            print(f'{Fore.RED}Runde {round_num} erzeugte eine Ausnahme: {exc}{Style.RESET_ALL}')
    
    # Sortiere Ergebnisse nach Rundennummer
    results.sort(key=lambda x: x['round'])
    
    # Erstelle Zusammenfassung und Visualisierung nur wenn Ergebnisse vorliegen
    if results:
        create_summary_visualization(results)
    else:
        print("Keine Ergebnisse zur Zusammenfassung vorhanden.")
    
    return results

# Funktion zur Erstellung einer Zusammenfassung und Visualisierung
def create_summary_visualization(results):
    import matplotlib.pyplot as plt
    
    # Prüfen, ob Ergebnisse vorhanden sind
    if not results:
        print("Keine Ergebnisse zur Visualisierung vorhanden.")
        return
    
    # Extrahiere relevante Daten
    rounds = [result['round'] for result in results]
    neurons = [result['neurons'] for result in results]
    pi_estimates = [result['pi_estimate'] for result in results]
    active_neurons_gen = [result['active_neurons_gen'] for result in results]
    active_neurons_dis = [result['active_neurons_dis'] for result in results]
    
    # Erstelle Grafiken
    plt.figure(figsize=(15, 10))
    
    # Pi-Schätzung
    plt.subplot(2, 2, 1)
    plt.plot(neurons, pi_estimates, 'bo-')
    plt.axhline(y=np.pi, color='r', linestyle='--', label=f'Exakter Wert: {np.pi:.6f}')
    plt.xlabel('Anzahl der Neuronen')
    plt.ylabel('π-Schätzung')
    plt.title('π-Schätzung vs. Neuronenzahl')
    plt.grid(True)
    plt.legend()
    
    # Aktive Neuronen
    plt.subplot(2, 2, 2)
    plt.plot(neurons, active_neurons_gen, 'go-', label='Generator')
    plt.plot(neurons, active_neurons_dis, 'mo-', label='Discriminator')
    plt.xlabel('Anzahl der Neuronen')
    plt.ylabel('Aktive Neuronen (%)')
    plt.title('Aktive Neuronen vs. Neuronenzahl')
    plt.grid(True)
    plt.legend()
    
    # Approximationsfehler
    errors = [abs(est - np.pi) for est in pi_estimates]
    plt.subplot(2, 2, 3)
    plt.plot(neurons, errors, 'ro-')
    plt.xlabel('Anzahl der Neuronen')
    plt.ylabel('Absoluter Fehler')
    plt.title('Approximationsfehler vs. Neuronenzahl')
    plt.grid(True)
    
    # Fehler auf logarithmischer Skala
    plt.subplot(2, 2, 4)
    plt.semilogy(neurons, errors, 'ro-')
    plt.xlabel('Anzahl der Neuronen')
    plt.ylabel('Absoluter Fehler (log)')
    plt.title('Log-Fehler vs. Neuronenzahl')
    plt.grid(True)
    
    plt.tight_layout()
    try:
        plt.savefig('./gan_outputs/summary_results.png')
    except Exception as e:
        print(f"Warnung: Konnte Grafik nicht speichern: {e}")
    plt.close()
    
    # Speichere Zusammenfassung als Text
    try:
        with open('./gan_outputs/summary.txt', 'w') as f:
            f.write("GAN-Training Zusammenfassung\n")
            f.write("========================\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Runde {result['round']} (Neuronen: {result['neurons']}):\n")
                f.write(f"  π-Schätzung: {result['pi_estimate']:.6f} (Fehler: {abs(result['pi_estimate'] - np.pi):.6f})\n")
                f.write(f"  Aktive Neuronen: Generator {result['active_neurons_gen']:.2f}%, Discriminator {result['active_neurons_dis']:.2f}%\n")
                f.write(f"  Beste Epoche: {result['best_epoch']}\n")
                f.write("\n")
            
            # Finde die beste Runde basierend auf dem geringsten π-Approximationsfehler
            best_round_idx = np.argmin([abs(est - np.pi) for est in pi_estimates])
            f.write(f"\nBeste Approximation: Runde {results[best_round_idx]['round']} mit π = {pi_estimates[best_round_idx]:.8f}\n")
            f.write(f"Absoluter Fehler: {abs(pi_estimates[best_round_idx] - np.pi):.8f}\n")
    except Exception as e:
        print(f"Warnung: Konnte Zusammenfassung nicht speichern: {e}")

# Parameter des Experiments - Angepasst für bessere Ressourcennutzung
if __name__ == "__main__":
    # Einstellbare Parameter
    total_rounds = 3             # Weniger Runden für schnellere Ausführung
    initial_neurons = 16         # Mehr Anfangsneuronen
    neuron_step = 16             # Größere Schritte
    initial_learning_rate = 0.0005  # Höhere initiale Lernrate
    epochs_per_round = 10000     # Weniger Epochen pro Runde
    
    # Batch-Größe an verfügbare Hardware anpassen
    batch_size = 64
    if torch.cuda.is_available():
        # Wenn GPU verfügbar, größere Batches verwenden
        batch_size = 128
    
    # Experiment durchführen
    results = run_experiment(
        total_rounds, 
        initial_neurons, 
        neuron_step, 
        initial_learning_rate, 
        epochs_per_round,
        batch_size
    )
    
    print(f"\n{Fore.GREEN}Experiment abgeschlossen. Ergebnisse wurden in ./gan_outputs/ gespeichert.{Style.RESET_ALL}")
