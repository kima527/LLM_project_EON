import os
import re
import json
from dotenv import load_dotenv
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(os.path.dirname(script_dir))

# Konfiguration
@dataclass
class Config:
    """Konfigurationsklasse für den Anonymisierer"""
    input_folder: str
    output_folder: str
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10  # Anzahl der Dateien pro Batch


# Logging Setup
def setup_logging(log_file: str = "anonymizer.log") -> logging.Logger:
    """Konfiguriert das Logging"""
    logger = logging.getLogger("EmailAnonymizer")
    logger.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class EmailAnonymizer:
    """Klasse zur Anonymisierung von E-Mail-Daten"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = setup_logging()

        # Erstelle Output-Verzeichnis
        self.config.output_folder.mkdir(parents=True, exist_ok=True)

        # Definiere alle Labels
        self.all_labels = [
            "GIVENNAME", "SURNAME", "DATEOFBIRTH", "PASSWORD", "USERNAME", "LINK",
            "ACCOUNTNUM", "IDCARDNUM", "DRIVERLICENSENUM", "SOCIALNUM", "TAXNUM",
            "CITY", "STREET", "ZIPCODE", "BUILDINGNUM",
            "CREDITCARDNUMBER", "BIC", "AMOUNT", "IBAN",
            "TELEPHONENUM", "EMAIL",
            "BUNDLE_CODE", "CONTRACT_NUMBER", "METER_NUMBER", "METER_AMOUNT",
            "CUSTOMER_NUMBER", "COMPANY_REGISTER", "ACCOUNT_CONTRACT_NUMBER", "INVOICE_NUMBER",
            "DAY", "MONTH", "YEAR", "DATE",
            "ORGANISATION", "GENERIC_NUMBER"
        ]

        # Statistiken
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now()
        }

    def create_system_prompt(self) -> str:
        """Erstellt den System-Prompt für die Anonymisierung"""
        return f"""
Du bist ein Anonymisierungs-Modul für sensible Texte wie E-Mails. 
Deine Aufgabe ist es, personenbezogene Daten zu erkennen und durch passende Labels zu ersetzen.

Nutze diese Labels exakt und vollständig:
{chr(10).join(self.all_labels)}

WICHTIGE REGELN:
1. Ersetze nur tatsächlich sensible Daten durch Labels
2. Behalte den Satzbau und die Struktur bei
3. Verwende Labels im Format [LABEL_NAME]
4. Gib nur den anonymisierten Text zurück, ohne weitere Erklärungen
5. Bei Unsicherheit: Lieber zu vorsichtig als zu wenig anonymisieren

Beispiel:
Input: "Hallo Max Mustermann, Ihre IBAN DE123456789 wurde gespeichert."
Output: "Hallo [GIVENNAME] [SURNAME], Ihre IBAN [IBAN] wurde gespeichert."
"""

    def anonymize_text(self, text: str) -> Optional[str]:
        """
        Anonymisiert einen Text mit GPT-4

        Args:
            text: Der zu anonymisierende Text

        Returns:
            Anonymisierter Text oder None bei Fehler
        """
        if not text.strip():
            self.logger.warning("Leerer Text übermittelt")
            return text

        system_prompt = self.create_system_prompt()

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=self.config.temperature
                )

                anonymized_text = response.choices[0].message.content.strip()

                # Validierung: Prüfe ob die Antwort plausibel ist
                if self.validate_anonymization(text, anonymized_text):
                    return anonymized_text
                else:
                    self.logger.warning(f"Validierung fehlgeschlagen bei Versuch {attempt + 1}")

            except Exception as e:
                self.logger.error(f"API-Fehler bei Versuch {attempt + 1}: {str(e)}")

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

        return None

    def validate_anonymization(self, original: str, anonymized: str) -> bool:
        """
        Validiert die Anonymisierung

        Args:
            original: Originaltext
            anonymized: Anonymisierter Text

        Returns:
            True wenn Validierung erfolgreich
        """
        # Grundlegende Validierung
        if not anonymized or len(anonymized) < 10:
            return False

        # Prüfe ob Labels verwendet wurden
        label_pattern = r'\[([A-Z_]+)\]'
        labels_found = re.findall(label_pattern, anonymized)

        # Wenn keine Labels gefunden wurden, könnte es ein Problem geben
        # (außer der Text enthält wirklich keine sensiblen Daten)
        if not labels_found and len(original) > 100:
            self.logger.warning("Keine Labels im anonymisierten Text gefunden")

        return True

    def process_file(self, file_path: Path) -> bool:
        """
        Verarbeitet eine einzelne Datei

        Args:
            file_path: Pfad zur Datei

        Returns:
            True wenn erfolgreich verarbeitet
        """
        try:
            # Originaltext laden
            with open(file_path, "r", encoding="utf-8") as f:
                original_text = f.read()

            self.logger.info(f"Verarbeite: {file_path.name} ({len(original_text)} Zeichen)")

            # Anonymisieren
            anonymized_text = self.anonymize_text(original_text)

            if anonymized_text is None:
                self.logger.error(f"Anonymisierung fehlgeschlagen für: {file_path.name}")
                return False

            # Speichern
            output_path = self.config.output_folder / file_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(anonymized_text)

            self.logger.info(f"Erfolgreich gespeichert: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten von {file_path.name}: {str(e)}")
            return False

    def process_all_files(self) -> Dict[str, int]:
        """
        Verarbeitet alle Dateien im Input-Verzeichnis

        Returns:
            Statistiken der Verarbeitung
        """
        # Finde alle .txt Dateien
        files = list(self.config.input_folder.glob("*.txt"))

        if not files:
            self.logger.warning(f"Keine .txt Dateien in {self.config.input_folder} gefunden")
            return self.stats

        self.logger.info(f"Starte Verarbeitung von {len(files)} Dateien")

        # Verarbeite Dateien
        for file_path in files:
            self.stats["processed"] += 1

            if self.process_file(file_path):
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1

        # Berechne Gesamtzeit
        end_time = datetime.now()
        duration = end_time - self.stats["start_time"]

        self.logger.info(f"""
Verarbeitung abgeschlossen:
- Verarbeitete Dateien: {self.stats['processed']}
- Erfolgreich: {self.stats['successful']}
- Fehlgeschlagen: {self.stats['failed']}
- Gesamtzeit: {duration}
        """)

        return self.stats

    def save_statistics(self, output_file: str = "anonymization_stats.json"):
        """Speichert die Statistiken in einer JSON-Datei"""
        stats_with_time = {
            **self.stats,
            "start_time": self.stats["start_time"].isoformat(),
            "end_time": datetime.now().isoformat()
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats_with_time, f, indent=2, ensure_ascii=False)


def main():
    """Hauptfunktion"""
    # Konfiguration - WICHTIG: API-Key aus Umgebungsvariable laden!
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ FEHLER: OPENAI_API_KEY Umgebungsvariable nicht gesetzt!")
        print("Setzen Sie die Variable mit: export OPENAI_API_KEY='your-api-key'")
        return

    config = Config(
        input_folder=os.path.join(project_root, "TestingData", "AllOriginalEmails"),
        output_folder=os.path.join(project_root, "SecondModel_Open_AI", "Anonymized_Output"),
        api_key=api_key,
        model="gpt-4o",
        temperature=0,
        max_retries=3,
        retry_delay=1.0
    )

    # Anonymisierer erstellen und ausführen
    anonymizer = EmailAnonymizer(config)
    stats = anonymizer.process_all_files()

    # Statistiken speichern
    anonymizer.save_statistics()

    print(f"\n🎉 Verarbeitung abgeschlossen!")
    print(f"✅ Erfolgreich: {stats['successful']}")
    print(f"❌ Fehlgeschlagen: {stats['failed']}")


if __name__ == "__main__":
    main()