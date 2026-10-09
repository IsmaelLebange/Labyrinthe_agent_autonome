#!/bin/sh
set -eu

if [ -z "${VNC_PASSWORD:-}" ]; then
    echo "VNC_PASSWORD doit être défini dans les variables Render."
    exit 1
fi

if [ "${#VNC_PASSWORD}" -ne 8 ]; then
    echo "VNC_PASSWORD doit contenir exactement 8 caractères pour l'authentification VNC."
    exit 1
fi

x11vnc -storepasswd "$VNC_PASSWORD" /tmp/vnc.pass >/dev/null

echo "Démarrage de l'écran virtuel Xvfb..."
Xvfb :99 -screen 0 1360x900x24 -nolisten tcp -ac &
sleep 1

echo "Démarrage du gestionnaire de fenêtres..."
openbox --sm-disable --display :99 &
sleep 1

echo "Lancement de l'interface Tkinter connectée à l'API..."
python -u -m "${GUI_MODULE:-interface.app_api}" &
echo "Démarrage du serveur VNC..."
x11vnc -display :99 -rfbport 5900 -localhost -rfbauth /tmp/vnc.pass \
    -forever -shared -noxdamage -noscr -repeat &

echo "Démarrage de noVNC sur le port ${PORT:-10000}..."
exec websockify --web=/usr/share/novnc "${PORT:-10000}" 127.0.0.1:5900
