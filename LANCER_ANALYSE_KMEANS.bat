@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environnement .venv introuvable.
    echo Creez-le avec : py -m venv .venv
    echo Puis installez : .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Analyse K=4 : structure naturelle...
.venv\Scripts\python entrainer_kmeans.py --k-final 4 --sortie resultats\kmeans_k4
if errorlevel 1 goto :erreur

echo Analyse K=9 : comparaison avec les neuf cultures...
.venv\Scripts\python entrainer_kmeans.py --k-final 9 --sortie resultats\kmeans_k9
if errorlevel 1 goto :erreur

echo Les deux analyses K-means sont terminees.
pause
exit /b 0

:erreur
echo Une erreur a interrompu l'analyse.
pause
exit /b 1
