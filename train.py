import sys
import os
import torch
from datetime import datetime

# --- GESTION DES CHEMINS (Indispensable pour Jean Zay) ---
# On récupère le dossier courant (racine du projet)
project_root = os.getcwd()

# On ajoute 'src' au chemin pour pouvoir faire "from config import Config"
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# --- IMPORTS ---
try:
    from config import Config
    from src.models.adr import PI_DeepONet_ADR
    from src.training.smart_trainer import train_smart_time_marching
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

def main():
    # 1. SETUP DOSSIER DE SAUVEGARDE
    # On crée un dossier unique horodaté pour ne pas écraser les tests précédents
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # On sauvegarde dans results/run_DATE
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(project_root, "results", run_name)
    
    # On force la config à utiliser ce dossier
    Config.save_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"🚀 Lancement Entraînement Jean Zay")
    print(f"📁 Dossier de sortie : {run_dir}")
    print(f"⚙️  Paramètres :")
    print(f"   - Tmax: {Config.T_max}")
    print(f"   - dt: {Config.dt}")
    print(f"   - Batch Size: {Config.batch_size}")
    print(f"   - Iters/Step: {Config.n_iters_per_step}")

    # 2. DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device détecté : {device}")

    # 3. INITIALISATION MODÈLE
    model = PI_DeepONet_ADR().to(device)

    # 4. ENTRAÎNEMENT (Smart Time Marching)
    # Note: On utilise ici les valeurs par défaut de Config (les "vraies" valeurs)
    # Pas d'override comme dans le test !
    try:
        model = train_smart_time_marching(
            model,
            bounds=Config.ranges,
            n_warmup=Config.n_warmup,
            n_iters_per_step=Config.n_iters_per_step
        )
    except Exception as e:
        print(f"❌ Erreur critique pendant l'entraînement : {e}")
        # On sauvegarde quand même l'état actuel en cas de crash
        emergency_path = os.path.join(run_dir, "model_CRASHED.pth")
        torch.save(model.state_dict(), emergency_path)
        print(f"💾 Sauvegarde d'urgence effectuée : {emergency_path}")
        raise e

    # 5. SAUVEGARDE FINALE
    final_model_path = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Modèle final sauvegardé : {final_model_path}")

if __name__ == "__main__":
    main()