# Création des dossiers
New-Item -Path 'mon_projet', 'mon_projet/data', 'mon_projet/data/raw', 'mon_projet/data/processed', 'mon_projet/models', 'mon_projet/wrappers', 'mon_projet/optimizers', 'mon_projet/tests', 'mon_projet/notebooks' -ItemType Directory

# Création des fichiers __init__.py dans les sous-dossiers pour les rendre des modules Python
"mon_projet/models", "mon_projet/wrappers", "mon_projet/optimizers", "mon_projet/tests" | ForEach-Object {
    New-Item -Path "$_/__init__.py" -ItemType File
}

# Création des autres fichiers de base
New-Item -Path 'mon_projet/requirements.txt', 'mon_projet/main.py', 'mon_projet/utils.py' -ItemType File

# Exemple pour créer un modèle Python spécifique
New-Item -Path 'mon_projet/models/simple1dcnn.py' -ItemType File
New-Item -Path 'mon_projet/wrappers/pytorch_wrapper.py' -ItemType File

# Exemple pour créer des scripts de test
New-Item -Path 'mon_projet/tests/test_model.py', 'mon_projet/tests/test_optimizer.py' -ItemType File
