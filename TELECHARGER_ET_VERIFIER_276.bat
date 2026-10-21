@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERREUR : environnement .venv introuvable.
    echo Copier d'abord ce correctif dans AgriSLIC_KMeans_Examen.
    pause
    exit /b 1
)
echo [1/2] Telechargement de la tuile 276 depuis la source publique...
"%PY%" telecharger_dataset_lite.py --tuiles 276 --racine dataset_lite
if errorlevel 1 goto erreur
echo.
echo [2/2] Verification de correspondance avec les 64 caracteristiques officielles...
"%PY%" verifier_tuile_test.py --tuile 276 --dataset dataset_lite
if errorlevel 1 goto erreur
echo.
echo SUCCES : fermer puis relancer Streamlit.
echo .venv\Scripts\python.exe -m streamlit run app.py
pause
exit /b 0
:erreur
echo.
echo Le telechargement ou la verification a echoue.
echo Ne pas utiliser la carte de test avant correction.
pause
exit /b 1
