# ...existing code...
import os
import json
import sqlite3
import threading
import logging
import queue
import time
from datetime import datetime
import importlib
import importlib.machinery
import importlib.util

# optional dependencies
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    from flask import Flask, request, jsonify
except Exception:
    Flask = None

# tentatives d'import des modules du projet pour intégration
def safe_import(name):
    """
    Essaie:
    - importlib.import_module(name)
    - si échoue, recherche un fichier <name>.py dans le dossier pages puis dans le root du projet
      et le charge par chemin (utile pour fichiers dont le nom commence par un chiffre).
    Retourne le module ou None.
    """
    try:
        return importlib.import_module(name)
    except Exception:
        # Try loading by path relative to this file (pages/) and project root
        base_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(base_dir)
        candidates = [
            os.path.join(base_dir, f"{name}.py"),
            os.path.join(project_root, f"{name}.py"),
            os.path.join(project_root, "pages", f"{name}.py"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    loader = importlib.machinery.SourceFileLoader(name, candidate)
                    spec = importlib.util.spec_from_loader(name, loader)
                    module = importlib.util.module_from_spec(spec)
                    loader.exec_module(module)
                    return module
                except Exception:
                    logging.getLogger("IoT_Monitoring").exception("safe_import: erreur chargement par path %s", candidate)
        return None

Dashboard = safe_import("1_Dashboard")
Agro = safe_import("3_Agro_Monitoring")
Disease = safe_import("6_Disease_Detection")
Fertilizer = safe_import("7_Smart_Fertilizer")
Satellite = safe_import("9_Satellite_Imagery")
Climate = safe_import("8_Climate_Forecasting")
DataUpload = safe_import("5_Data_Upload")
Corrections = safe_import("13_Corrections_Admin")
Social = safe_import("10_Social_Network")
Yield = safe_import("2_Yield_Prediction")
Home = safe_import("Home")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IoT_Monitoring")

DB_PATH = os.path.join(os.path.dirname(__file__), "iot_monitoring.db")

# ...existing code...
class IoTMonitoring:
    def __init__(self,
                 mqtt_broker="localhost",
                 mqtt_port=1883,
                 mqtt_topics=("sensors/#",),
                 http_port=5001,
                 csv_folder=os.path.join(os.path.dirname(__file__), "ingest_csv")):
        # ...existing code...
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topics = mqtt_topics
        self.http_port = http_port
        self.csv_folder = csv_folder
        self.queue = queue.Queue()
        self._stop = threading.Event()
        self._init_db()

        # init clients
        if mqtt:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
        else:
            self.mqtt_client = None
            log.warning("paho-mqtt non disponible, MQTT désactivé")

        if Flask:
            self.app = Flask("iot_ingest")
            self._setup_http_routes()
        else:
            self.app = None
            log.warning("Flask non disponible, HTTP endpoint désactivé")

    # Database
    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source TEXT,
                field_id TEXT,
                payload TEXT
            )
        """)
        conn.commit()
        conn.close()
        log.info("SQLite initialisé: %s", DB_PATH)

    def _store(self, source, field_id, payload):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO sensor_data (timestamp, source, field_id, payload) VALUES (?, ?, ?, ?)",
                  (datetime.utcnow().isoformat(), source, field_id, json.dumps(payload)))
        conn.commit()
        conn.close()
        log.debug("Donnée stockée: %s %s", source, field_id)

    # MQTT
    def start_mqtt(self):
        if not self.mqtt_client:
            return
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            threading.Thread(target=self.mqtt_client.loop_forever, daemon=True).start()
            log.info("MQTT démarré sur %s:%s", self.mqtt_broker, self.mqtt_port)
        except Exception as e:
            log.exception("Erreur démarrage MQTT: %s", e)

    def _on_connect(self, client, userdata, flags, rc):
        log.info("MQTT connecté rc=%s", rc)
        for t in self.mqtt_topics:
            client.subscribe(t)
            log.info("Subscribed to %s", t)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            payload = {"raw": msg.payload.decode(errors="ignore")}
        record = {"source": "mqtt", "topic": msg.topic, "payload": payload}
        self.queue.put(record)

    # HTTP ingestion
    def _setup_http_routes(self):
        app = self.app

        @app.route("/ingest", methods=["POST"])
        def ingest():
            data = request.get_json(force=True, silent=True) or {}
            field_id = data.get("field_id", "unknown")
            record = {"source": "http", "topic": f"http/{field_id}", "payload": data}
            self.queue.put(record)
            return jsonify({"ok": True}), 201

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"}), 200

    def start_http(self):
        if not self.app:
            return
        threading.Thread(target=lambda: self.app.run(port=self.http_port, host="0.0.0.0"), daemon=True).start()
        log.info("HTTP ingestion démarrée sur port %s", self.http_port)

    # CSV ingestion (fichier déposé)
    def start_csv_watcher(self, poll_interval=10):
        os.makedirs(self.csv_folder, exist_ok=True)

        def watcher():
            seen = set()
            while not self._stop.is_set():
                for fname in os.listdir(self.csv_folder):
                    path = os.path.join(self.csv_folder, fname)
                    if path in seen:
                        continue
                    if not fname.lower().endswith(".csv"):
                        continue
                    try:
                        with open(path, "r", encoding="utf8") as f:
                            header = f.readline().strip().split(",")
                            for line in f:
                                vals = line.strip().split(",")
                                data = dict(zip(header, vals))
                                record = {"source": "csv", "topic": fname, "payload": data}
                                self.queue.put(record)
                        seen.add(path)
                        log.info("Ingest CSV: %s", path)
                    except Exception:
                        log.exception("Erreur lecture CSV %s", path)
                time.sleep(poll_interval)

        threading.Thread(target=watcher, daemon=True).start()
        log.info("CSV watcher démarré dans %s", self.csv_folder)

    # Processing loop
    def start_processor(self):
        def processor():
            while not self._stop.is_set():
                try:
                    item = self.queue.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    topic = item.get("topic", "")
                    payload = item.get("payload", {})
                    source = item.get("source", "unknown")
                    field_id = payload.get("field_id") or payload.get("id") or "unknown"
                    self._store(source, field_id, payload)
                    enriched = self._enrich_data(field_id, payload)
                    self._check_alerts(field_id, enriched)
                    self._notify_others(field_id, enriched)
                    # forward to remote upload if available
                    if DataUpload and hasattr(DataUpload, "upload_record"):
                        try:
                            DataUpload.upload_record(enriched)
                        except Exception:
                            log.exception("Erreur DataUpload.upload_record")
                except Exception:
                    log.exception("Erreur traitement item")
        threading.Thread(target=processor, daemon=True).start()
        log.info("Processor démarré")

    # Enrichment: appel aux modules Satellite/Agro/Climate si présents
    def _enrich_data(self, field_id, payload):
        enriched = {
            "ingest_at": datetime.utcnow().isoformat(),
            "field_id": field_id,
            "raw": payload,
        }
        # satellite NDVI
        try:
            if Satellite and hasattr(Satellite, "get_latest_ndvi"):
                ndvi = Satellite.get_latest_ndvi(field_id)
                enriched["ndvi"] = ndvi
        except Exception:
            log.exception("Erreur récupération NDVI")

        # soil sensors / historical soil map
        try:
            if Agro and hasattr(Agro, "get_soil_profile"):
                soil = Agro.get_soil_profile(field_id)
                enriched["soil_profile"] = soil
        except Exception:
            log.exception("Erreur récupération profil sol")

        # climate forecast
        try:
            if Climate and hasattr(Climate, "get_forecast"):
                fc = Climate.get_forecast(field_id)
                enriched["forecast"] = fc
        except Exception:
            log.exception("Erreur récupération forecast")

        return enriched

    # Alerts: seuils simples, escalate via Dashboard / Corrections_Admin / Social_Network
    def _check_alerts(self, field_id, enriched):
        raw = enriched.get("raw", {})
        alerts = []
        try:
            temp = float(raw.get("temperature", raw.get("temp", float("nan"))))
        except Exception:
            temp = None
        try:
            hum = float(raw.get("humidity", raw.get("hum", raw.get("soil_moisture", float("nan")))))
        except Exception:
            hum = None
        try:
            ph = float(raw.get("ph", float("nan")))
        except Exception:
            ph = None

        if temp is not None and (temp < -2 or temp > 45):
            alerts.append({"type": "temperature", "value": temp})
        if hum is not None and (hum < 15 or hum > 98):
            alerts.append({"type": "humidity", "value": hum})
        if ph is not None and (ph < 4.5 or ph > 8.5):
            alerts.append({"type": "ph", "value": ph})

        if alerts:
            message = {"field_id": field_id, "alerts": alerts, "enriched": enriched}
            log.warning("ALERTS: %s", message)
            # Dashboard notification
            try:
                if Dashboard and hasattr(Dashboard, "push_alert"):
                    Dashboard.push_alert(message)
            except Exception:
                log.exception("Erreur push_alert Dashboard")
            # Corrections admin
            try:
                if Corrections and hasattr(Corrections, "notify_admin"):
                    Corrections.notify_admin(message)
            except Exception:
                log.exception("Erreur notify_admin")
            # Social / messaging
            try:
                if Social and hasattr(Social, "post_alert"):
                    Social.post_alert(message)
            except Exception:
                log.exception("Erreur post_alert Social_Network")

    # Notify other modules for enrichment or ML retrain
    def _notify_others(self, field_id, enriched):
        try:
            if Yield and hasattr(Yield, "on_new_observation"):
                Yield.on_new_observation(enriched)
        except Exception:
            log.exception("Erreur Yield.on_new_observation")
        try:
            if Disease and hasattr(Disease, "ingest_observation"):
                Disease.ingest_observation(enriched)
        except Exception:
            log.exception("Erreur Disease.ingest_observation")
        try:
            if Fertilizer and hasattr(Fertilizer, "ingest_npk"):
                Fertilizer.ingest_npk(enriched)
        except Exception:
            log.exception("Erreur Fertilizer.ingest_npk")

    # lifecycle
    def start(self):
        log.info("Démarrage IoT Monitoring")
        self.start_mqtt()
        self.start_http()
        self.start_csv_watcher()
        self.start_processor()

    def stop(self):
        log.info("Arrêt IoT Monitoring")
        self._stop.set()
        try:
            if self.mqtt_client:
                self.mqtt_client.disconnect()
        except Exception:
            pass


# CLI / simple runner
if __name__ == "__main__":
    monitor = IoTMonitoring()
    monitor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        log.info("Process terminé")
# ...existing code...
