import sys
import os
import torch
import yaml
from datetime import datetime

# --- GESTION DES CHEMINS ---
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# --- IMPORTS DÉDIÉS ---
try:
    # On importe le module entier pour pouvoir modifier sa config interne
    import src.training.smart_trainer as trainer_module 
    from src.models.adr import PI_DeepONet_ADR
    print("✅ Imports réussis.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)

def main():
    # 1. SETUP DOSSIER DE SAUVEGARDE (Dès le début)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    # On force le chemin absolu pour éviter les erreurs sur les nœuds de calcul
    run_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. INJECTION DE LA CONFIGURATION
    # C'est ici que la magie opère : on modifie la config DU TRAINER directement
    print(f"🔧 Mise à jour du dossier de sortie dans le Trainer...")
    trainer_module.cfg['audit']['save_dir'] = run_dir
    
    # On récupère aussi une référence locale pour l'affichage
    cfg = trainer_module.cfg 

    print(f"🚀 Lancement Entraînement ADR (Jean Zay)")
    print(f"📁 Dossier de sortie : {run_dir}")
    print(f"⚙️  Paramètres clés :")
    print(f"   - Tmax: {cfg['geometry']['T_max']}")
    print(f"   - Zones: {cfg['time_stepping']['zones']}")
    print(f"   - Points (n_sample): {cfg['training']['n_sample']}")
    print(f"   - Macro Loops: {cfg['training']['nb_loop']}")

    # 3. DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    
    # 4. INITIALISATION MODÈLE
    model = PI_DeepONet_ADR(cfg).to(device)

    # 5. ENTRAÎNEMENT (Smart Time Marching)
    try:
        # On appelle la fonction via le module
        model = trainer_module.train_smart_time_marching(
            model,
            bounds=cfg['physics_ranges']
        )
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        emergency_path = os.path.join(run_dir, "model_CRASHED.pth")
        torch.save(model.state_dict(), emergency_path)
        print(f"🆘 Sauvegarde d'urgence effectuée : {emergency_path}")
        raise e

    # 6. SAUVEGARDE FINALE
    final_model_path = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Modèle final sauvegardé : {final_model_path}")

if __name__ == "__main__":
    main()