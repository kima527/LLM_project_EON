import os
import re
from collections import defaultdict
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from pathlib import Path
from openai import OpenAI
#funktioneirt gut
# OpenAI API Setup
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(os.path.dirname(script_dir))


load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)




class PIIranhaEvaluator:
    def __init__(self, piiranha_dir: str, groundtruth_dir: str):
        self.piiranha_dir = Path(piiranha_dir)
        self.groundtruth_dir = Path(groundtruth_dir)
        self.system_prompt = """Du bist ein präziser Evaluator für Anonymisierungs-Qualität. Du vergleichst zwei Versionen eines Textes.

🎯 **AUFGABE:** 
- **GroundTruth:** Zeigt WO anonymisiert werden SOLLTE (mit [LABELS])
- **Piiranha:** Zeigt was tatsächlich anonymisiert WURDE
- **Bewerte:** Hat Piiranha die richtige Anonymisierung durchgeführt?

🔍 **KRITISCHES VERSTÄNDNIS:**
❌ **NICHT ANONYMISIERT = SICHERHEITSPROBLEM!**
- Wenn GroundTruth `[IBAN]` hat, aber Piiranha zeigt `DE40926301819767992111` → **GEFAHR! Echte IBAN sichtbar!**
- Wenn GroundTruth `[GIVENNAME]` hat, aber Piiranha zeigt `Leo` → **GEFAHR! Echter Name sichtbar!**
- Wenn GroundTruth `[AMOUNT]` hat, aber Piiranha zeigt `148 €` → **GEFAHR! Echter Betrag sichtbar!**
⚠️ **FALSCHES LABEL (SEMANTISCHER FEHLER) = FALSE POSITIVE!**
- Wenn GroundTruth `[CUSTOMER_NUMBER]` hat, aber Piiranha `[ACCOUNTNR]` verwendet → **FEHLER!**
- Das bedeutet: Piiranha hat einen **nicht passenden Labeltyp** verwendet.
- Solche Fehler zählen als **False Positives** für `[ACCOUNTNR]` UND als **False Negatives** für `[CUSTOMER_NUMBER]`

✅ **KORREKT ANONYMISIERT:**
- Wenn GroundTruth `[IBAN]` hat und Piiranha auch `[IBAN]` → **SICHER!**
- Wenn GroundTruth `[GIVENNAME]` hat und Piiranha auch `[GIVENNAME]` → **SICHER!**

📊 **KLASSIFIKATION:**
🟢 **True Positive (TP):** GT hat `[LABEL]` UND Piiranha hat auch `[LABEL]` 
🔴 **False Negative (FN):** GT hat `[LABEL]` ABER Piiranha hat ECHTE DATEN (KRITISCH!)
🟡 **False Positive (FP):** GT hat KEINE Labels ABER Piiranha hat `[LABEL]` (überanonymisiert)

📚 **DETAILLIERTE BEISPIELE:**

**BEISPIEL 1 - Alles korrekt anonymisiert:**
GroundTruth: "Herr [GIVENNAME] [SURNAME] hat [AMOUNT] überwiesen."
Piiranha: "Herr [GIVENNAME] [SURNAME] hat [AMOUNT] überwiesen."
➤ Analyse: 
- [GIVENNAME]: TP=1 (korrekt anonymisiert)
- [SURNAME]: TP=1 (korrekt anonymisiert)  
- [AMOUNT]: TP=1 (korrekt anonymisiert)
➤ Ergebnis:
[GIVENNAME]    100.0%   100.0%   100.0%   1   0   0
[SURNAME]      100.0%   100.0%   100.0%   1   0   0
[AMOUNT]       100.0%   100.0%   100.0%   1   0   0

**BEISPIEL 2 - Nichts anonymisiert (kritisch!):**
GroundTruth: "Herr [GIVENNAME] [SURNAME] hat [AMOUNT] überwiesen."
Piiranha: "Herr Max Schmidt hat 500€ überwiesen."
➤ Analyse:
- [GIVENNAME]: FN=1 (echter Name "Max" sichtbar!)
- [SURNAME]: FN=1 (echter Name "Schmidt" sichtbar!)
- [AMOUNT]: FN=1 (echter Betrag "500€" sichtbar!)
➤ Ergebnis:
[GIVENNAME]    0.0%     0.0%     0.0%     0   0   1
[SURNAME]      0.0%     0.0%     0.0%     0   0   1
[AMOUNT]       0.0%     0.0%     0.0%     0   0   1

**BEISPIEL 3 - Gemischte Resultate:**
GroundTruth: "den Abschlag von[AMOUNT] für [MONTH] auf das Konto [IBAN]. Sie müssten nur den Betrag umbuchen auf [IBAN]. Dipl.-Ing. [GIVENNAME] [SURNAME]"
Piiranha: "den Abschlag von 148€ für August auf das Konto DE40926301819767992111. Sie müssten nur den Betrag umbuchen auf DE57671088940322928241. Dipl.-Ing. Leo [SURNAME]"
➤ Analyse:
- [AMOUNT]: FN=1 (echter Betrag "148€" sichtbar!)
- [MONTH]: FN=1 (echter Monat "August" sichtbar!)
- [IBAN] (erste): FN=1 (echte IBAN "DE40..." sichtbar!)
- [IBAN] (zweite): FN=1 (echte IBAN "DE57..." sichtbar!)
- [GIVENNAME]: FN=1 (echter Name "Leo" sichtbar!)
- [SURNAME]: TP=1 (korrekt anonymisiert!)
➤ Ergebnis:
[AMOUNT]       0.0%     0.0%     0.0%     0   0   1
[MONTH]        0.0%     0.0%     0.0%     0   0   1
[IBAN]         0.0%     0.0%     0.0%     0   0   2
[GIVENNAME]    0.0%     0.0%     0.0%     0   0   1
[SURNAME]      100.0%   100.0%   100.0%   1   0   0

**BEISPIEL 4 - Teilweise korrekt:**
GroundTruth: "Von [GIVENNAME] an [IBAN] mit [AMOUNT]."
Piiranha: "Von [GIVENNAME] an DE123456789 mit [AMOUNT]."
➤ Analyse:
- [GIVENNAME]: TP=1 (korrekt anonymisiert)
- [IBAN]: FN=1 (echte IBAN "DE123456789" sichtbar!)
- [AMOUNT]: TP=1 (korrekt anonymisiert)
➤ Ergebnis:
[GIVENNAME]    100.0%   100.0%   100.0%   1   0   0
[IBAN]         0.0%     0.0%     0.0%     0   0   1
[AMOUNT]       100.0%   100.0%   100.0%   1   0   0

**BEISPIEL 5 - Komplexer realer Fall:**
GroundTruth: "Übergabedatum: [DATE] Mieter: [GIVENNAME] [SURNAME] ([TELEPHONENUM]) [GIVENNAME] [SURNAME] ([TELEPHONENUM]) Vermieter: Dr. [GIVENNAME] [SURNAME] Zählernummer: [METER_NUMBER] Zählerstand: [METER_AMOUNT] Vielen Dank [GIVENNAME] [SURNAME]"
Piiranha: "Übergabedatum: 01.08.2023 Mieter: Isabelle Eckbauer [TELEPHONENUM]) [GIVENNAME] [SURNAME] [TELEPHONENUM]) Vermieter: Dr. FranzXaver Huhn Zählernummer: 1ISK0070547123 Zählerstand: 0022386623 Vielen Dank [GIVENNAME] Eckbauer"
➤ Analyse:
- [DATE]: FN=1 (echtes Datum "01.08.2023" sichtbar!)
- [GIVENNAME]: TP=2, FN=2 (2 korrekt anonymisiert, aber "Isabelle" und "FranzXaver" sichtbar!)
- [SURNAME]: TP=0, FN=3 (alle echten Namen "Eckbauer", "Huhn", "Eckbauer" sichtbar!)
- [TELEPHONENUM]: TP=2, FN=0 (beide korrekt anonymisiert!)
- [METER_NUMBER]: FN=1 (echte Nummer "1ISK0070547123" sichtbar!)
- [METER_AMOUNT]: FN=1 (echter Stand "0022386623" sichtbar!)
➤ Ergebnis:
[DATE]         0.0%     0.0%     0.0%     0   0   1
[GIVENNAME]    100.0%   50.0%    66.7%    2   0   2
[SURNAME]      0.0%     0.0%     0.0%     0   0   3
[TELEPHONENUM] 100.0%   100.0%   100.0%   2   0   0
[METER_NUMBER] 0.0%     0.0%     0.0%     0   0   1
[METER_AMOUNT] 0.0%     0.0%     0.0%     0   0   1

🔍 **ANALYSE-METHODE:**
1. Finde alle [LABELS] im GroundTruth
2. Für jeden GT-Label: Schaue an entsprechender Stelle im Piiranha
   - Steht dort [SAME_LABEL]? → TP
   - Stehen dort echte Daten? → FN (KRITISCH!)
3. Zusätzliche [LABELS] im Piiranha ohne GT-Entsprechung → FP

📤 **AUSGABEFORMAT (EXAKT):**
================================================================================
📁 BEISPIEL: [Dateiname]
🎯 GESAMTABDECKUNG: [xx.x]%   Erkannte Spans: [TP]/[Gesamtzahl GT-Labels]
📈 METRIKEN PRO LABEL-TYP:
--------------------------------------------------------------------------------
Label Type     Precision    Recall    F1-Score    TP    FP    FN
[LABEL]        [xx.x]%      [xx.x]%   [xx.x]%     [n]   [n]   [n]
================================================================================"""

        self.all_file_results = []
        self.aggregated_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    def evaluate_file_pair(self, piiranha_file: str, groundtruth_file: str) -> Dict:
        """Evaluiert ein Dateipaar mit OpenAI API"""
        try:
            # Dateien einlesen
            with open(self.piiranha_dir / piiranha_file, 'r', encoding='utf-8') as f:
                piiranha_text = f.read().strip()

            with open(self.groundtruth_dir / groundtruth_file, 'r', encoding='utf-8') as f:
                groundtruth_text = f.read().strip()

            # User-Prompt für diese spezifische Dateipaar
            user_prompt = f"""Evaluiere dieses Dateipaar:

**GroundTruth Text:**
{groundtruth_text}

**Piiranha Text:**
{piiranha_text}

Analysiere sehr genau und gib die Metriken im angegebenen Format aus."""

            # API Call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0
            )

            ai_response = response.choices[0].message.content

            # Parse die Antwort
            parsed_result = self.parse_ai_response(ai_response, piiranha_file, groundtruth_file)
            return parsed_result

        except Exception as e:
            print(f"❌ Fehler beim Verarbeiten von {piiranha_file} und {groundtruth_file}: {e}")
            return None

    def parse_ai_response(self, ai_response: str, piiranha_file: str, groundtruth_file: str) -> Dict:
        """Parst die OpenAI Antwort und extrahiert Metriken"""
        result = {
            'piiranha_file': piiranha_file,
            'groundtruth_file': groundtruth_file,
            'ai_response': ai_response,
            'coverage': 0.0,
            'total_spans': '0/0',
            'metrics': {}
        }

        try:
            # Extrahiere Coverage
            coverage_match = re.search(r'GESAMTABDECKUNG:\s*(\d+\.?\d*)%', ai_response)
            if coverage_match:
                result['coverage'] = float(coverage_match.group(1))

            # Extrahiere Spans
            spans_match = re.search(r'Erkannte Spans:\s*(\d+)/(\d+)', ai_response)
            if spans_match:
                result['total_spans'] = f"{spans_match.group(1)}/{spans_match.group(2)}"

            # Extrahiere Metriken pro Label
            # Suche nach Tabellen-Zeilen mit Metriken
            metric_pattern = r'\[([A-Z_]+)\]\s+(\d+\.?\d*)%\s+(\d+\.?\d*)%\s+(\d+\.?\d*)%\s+(\d+)\s+(\d+)\s+(\d+)'

            for match in re.finditer(metric_pattern, ai_response):
                label_type = match.group(1)
                precision = float(match.group(2))
                recall = float(match.group(3))
                f1 = float(match.group(4))
                tp = int(match.group(5))
                fp = int(match.group(6))
                fn = int(match.group(7))

                result['metrics'][label_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }

                # Für aggregierte Statistiken
                self.aggregated_metrics[label_type]['tp'] += tp
                self.aggregated_metrics[label_type]['fp'] += fp
                self.aggregated_metrics[label_type]['fn'] += fn

        except Exception as e:
            print(f"⚠️ Fehler beim Parsen der AI-Antwort: {e}")

        return result

    def run_evaluation(self):
        """Führt die komplette Evaluierung durch"""
        print("🚀 Starte PIIranha Anonymisierung Evaluierung mit OpenAI...")
        print("=" * 80)

        # Finde alle Dateipaare
        piiranha_files = sorted(
            [f for f in os.listdir(self.piiranha_dir) if f.startswith('piiranha_') and f.endswith('.txt')])
        groundtruth_files = sorted(
            [f for f in os.listdir(self.groundtruth_dir) if f.startswith('groundtruth_') and f.endswith('.txt')])

        print(f"📂 Gefunden: {len(piiranha_files)} Piiranha-Dateien, {len(groundtruth_files)} GroundTruth-Dateien")

        if len(piiranha_files) != len(groundtruth_files):
            print("⚠️ Warnung: Unterschiedliche Anzahl von Dateien in den Ordnern!")

        # Evaluiere jedes Dateipaar
        for i in range(min(len(piiranha_files), len(groundtruth_files))):
            pf = piiranha_files[i]
            gf = groundtruth_files[i]

            print(f"\n🔍 Verarbeite Paar {i + 1}/{min(len(piiranha_files), len(groundtruth_files))}: {gf} vs. {pf}")

            result = self.evaluate_file_pair(pf, gf)
            if result:
                self.all_file_results.append(result)

                # Zeige das Ergebnis an
                print(result['ai_response'])
            else:
                print(f"❌ Fehler bei der Verarbeitung von {pf} und {gf}")

        # Finale Zusammenfassung
        self.print_summary()

    def print_summary(self):
        """Druckt die finale Zusammenfassung aller Dateien"""
        print("\n" + "=" * 80)
        print("🏁 GESAMTERGEBNISSE ÜBER ALLE BEISPIELE")
        print("=" * 80)

        if not self.all_file_results:
            print("❌ Keine Ergebnisse zum Zusammenfassen!")
            return

        # Berechne durchschnittliche Abdeckung
        total_coverage = sum(result['coverage'] for result in self.all_file_results)
        avg_coverage = total_coverage / len(self.all_file_results)

        print(f"🎯 DURCHSCHNITTLICHE ABDECKUNG: {avg_coverage:.1f}%")
        print(f"📊 Verarbeitete Dateien: {len(self.all_file_results)}")

        print("\n📈 AGGREGIERTE METRIKEN PRO LABEL-TYP:")
        print("-" * 80)
        print(f"{'Label Type':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 80)

        # Berechne aggregierte Metriken
        for label_type in sorted(self.aggregated_metrics.keys()):
            metrics = self.aggregated_metrics[label_type]
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']

            precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Sichere Formatierung für lange Label-Namen
            label_display = f"[{label_type}]"
            padding = max(0, 15 - len(label_display))

            print(f"{label_display}{' ' * padding} "
                  f"{precision:<10.1f} "
                  f"{recall:<8.1f} "
                  f"{f1:<10.1f} "
                  f"{tp:<4} "
                  f"{fp:<4} "
                  f"{fn:<4}")

        # Gesamtmetriken
        total_tp = sum(m['tp'] for m in self.aggregated_metrics.values())
        total_fp = sum(m['fp'] for m in self.aggregated_metrics.values())
        total_fn = sum(m['fn'] for m in self.aggregated_metrics.values())

        if total_tp + total_fp + total_fn > 0:
            overall_precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
            overall_recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
            overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (
                                                                                                                        overall_precision + overall_recall) > 0 else 0.0

            print("-" * 80)
            print(f"{'GESAMT':<15} "
                  f"{overall_precision:<10.1f} "
                  f"{overall_recall:<8.1f} "
                  f"{overall_f1:<10.1f} "
                  f"{total_tp:<4} "
                  f"{total_fp:<4} "
                  f"{total_fn:<4}")


def main():
    # Pfade zu den Ordnern
    piiranha_dir = os.path.join(project_root, "TestingData", "PIIRANHA_BaseModel_Anonymized_EMails")
    groundtruth_dir = os.path.join(project_root, "TestingData", "GroundTruthDataset")

    # Erstelle Evaluator und führe Evaluierung durch
    evaluator = PIIranhaEvaluator(piiranha_dir, groundtruth_dir)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()