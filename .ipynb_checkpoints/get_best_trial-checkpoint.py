import optuna
import yaml
import os
import sys

# --- CONFIGURATION ---
DB_URL = "sqlite:///optuna_study.db"
STUDY_NAME = "ADR_Final_Optimization" # Le nom exact mis dans tune_optuna.py

def main():
    # 1. Chargement de l'étude
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
        print(f"📚 Étude chargée : {len(study.trials)} essais complétés.")
    except Exception as e:
        print(f"❌ Erreur : Impossible de charger l'étude. Vérifie le nom ou le fichier .db.\n{e}")
        return

    # 2. Le Champion
    best_trial = study.best_trial
    print("\n🏆 LE GRAND GAGNANT EST :")
    print(f"   🆔 Trial ID : {best_trial.number}")
    print(f"   📉 Erreur L2 : {best_trial.value:.4%} (C'est ton score à battre !)")
    print("   ⚙️  Hyperparamètres :")
    for key, value in best_trial.params.items():
        print(f"      - {key}: {value}")

    # 3. Génération de la Config "Production"
    # On reprend la structure de ton YAML de base, mais on injecte les gagnants
    # et on REMET les paramètres de temps longs (3 boucles, 6000 iters)
    
    # Définition des modes Fourier (copié du script tuning)
    fourier_presets = {
        "standard": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0],
        "high_freq": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "dense": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    }

    # On charge ton config actuel pour avoir le squelette
    # Assure-toi que src/config_ADR.yaml existe
    base_config_path = "src/config_ADR.yaml"
    with open(base_config_path, 'r') as f:
        final_cfg = yaml.safe_load(f)

    # --- INJECTION DES GAGNANTS ---
    params = best_trial.params
    
    # Architecture
    width = params['width']
    depth = params['depth']
    final_cfg['model']['branch_width'] = width
    final_cfg['model']['trunk_width'] = width
    final_cfg['model']['latent_dim'] = width
    final_cfg['model']['branch_depth'] = depth
    final_cfg['model']['trunk_depth'] = depth - 1
    
    # Fourier
    mode = params['fourier_mode']
    final_cfg['model']['sFourier'] = fourier_presets[mode]
    final_cfg['model']['nFourier'] = len(final_cfg['model']['sFourier']) * 2

    # Training Params
    final_cfg['training']['learning_rate'] = params['lr']
    
    # Loss Weights
    final_cfg['loss_weights']['weight_ic_init'] = params['w_ic_init']
    final_cfg['loss_weights']['weight_bc'] = params['w_bc']
    final_cfg['loss_weights']['first_w_res'] = params['w_res_target']

    # --- RESTAURATION DES PARAMÈTRES "LOURDS" (PRODUCTION) ---
    print("\n💪 Passage en mode PRODUCTION (Paramètres lourds activés)")
    final_cfg['training']['n_warmup'] = 7000          # On remet le warmup long
    final_cfg['training']['n_iters_per_step'] = 6000  # On remet 6000 itérations
    final_cfg['training']['n_iters_correction'] = 8000
    final_cfg['training']['nb_loop'] = 3              # On remet 3 boucles macro
    final_cfg['training']['threshold'] = 0.01         # On vise 1% maintenant !

    # Sauvegarde
    output_filename = "final_optimized_config.yaml"
    with open(output_filename, 'w') as f:
        yaml.dump(final_cfg, f)
    
    print(f"💾 Configuration sauvegardée dans : {output_filename}")
    print("🚀 Tu n'as plus qu'à lancer l'entraînement final avec ce fichier !")

if __name__ == "__main__":
    main()