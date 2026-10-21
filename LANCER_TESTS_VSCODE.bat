@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environnement .venv introuvable. Lire GUIDE_TESTS_VSCODE.md
  pause
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
%PY% --version
%PY% -m pip check || goto error
%PY% -m py_compile telecharger_dataset_lite.py extraire_caracteristiques_slic.py entrainer_kmeans.py app.py verifier_projet.py || goto error
%PY% verifier_projet.py || goto error
if not exist "verification" mkdir verification
%PY% extraire_caracteristiques_slic.py --dataset test_dataset --sortie verification\extraction_test.csv --limite-tuiles 1 || goto error
echo.
echo TESTS ESSENTIELS TERMINES AVEC SUCCES.
pause
exit /b 0
:error
echo.
echo ERREUR DE TEST: envoyer une capture de ce terminal.
pause
exit /b 1
