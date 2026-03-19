"""
Prediction Logger - Sauvegarde des prédictions et feedback utilisateur

Permet de logger les diagnostics pour améliorer le modèle et analyser les performances.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import uuid

class PredictionLogger:
    def __init__(self, log_dir="prediction_logs"):
        """
        Initialize the prediction logger.

        Args:
            log_dir: Directory to store prediction logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Fichiers de log
        self.predictions_file = self.log_dir / "predictions.jsonl"
        self.feedback_file = self.log_dir / "feedback.jsonl"
        self.stats_file = self.log_dir / "statistics.json"

    def log_prediction(self, image_path, predictions, diagnosis_result, user_id=None):
        """
        Log a prediction event.

        Args:
            image_path: Path to the analyzed image
            predictions: List of predictions from the model
            diagnosis_result: Complete diagnosis result
            user_id: Optional user identifier
        """
        prediction_id = str(uuid.uuid4())

        log_entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "image_path": str(image_path),
            "predictions": predictions,
            "top_disease": diagnosis_result.get("top_disease"),
            "confidence_threshold": diagnosis_result.get("confidence_threshold", 0.3),
            "explanations_count": len(diagnosis_result.get("explanations", {})),
            "sources_count": len(diagnosis_result.get("sources", {})),
            "unknown_disease": diagnosis_result.get("unknown_disease", False),
            "faiss_validation": diagnosis_result.get("faiss_validation", {}),
            "error_detected": self._detect_prediction_errors(diagnosis_result)
        }

        # Sauvegarder dans le fichier JSONL
        with open(self.predictions_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')

        # Intelligent saving: Auto-create training data for errors (improvement #6)
        if log_entry.get("error_detected"):
            self._auto_create_training_data(log_entry)

        return prediction_id

    def log_feedback(self, prediction_id, user_feedback, correct_disease=None, additional_notes=None):
        """
        Log user feedback on a prediction.

        Args:
            prediction_id: ID of the prediction to provide feedback on
            user_feedback: User's feedback (correct/incorrect/unsure)
            correct_disease: If incorrect, what was the actual disease
            additional_notes: Any additional user notes
        """
        feedback_entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "user_feedback": user_feedback,  # "correct", "incorrect", "unsure"
            "correct_disease": correct_disease,
            "additional_notes": additional_notes
        }

        # Sauvegarder dans le fichier JSONL
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            json.dump(feedback_entry, f, ensure_ascii=False)
            f.write('\n')

    def get_statistics(self):
        """
        Calculate and return prediction statistics.

        Returns:
            dict: Statistics about predictions and feedback
        """
        stats = {
            "total_predictions": 0,
            "total_feedback": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "unsure_predictions": 0,
            "avg_confidence": 0.0,
            "common_misdiagnoses": {},
            "last_updated": datetime.now().isoformat()
        }

        # Lire les prédictions
        predictions = []
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))

        stats["total_predictions"] = len(predictions)

        # Calculer la confiance moyenne
        if predictions:
            confidences = []
            for pred in predictions:
                if pred["predictions"]:
                    confidences.append(pred["predictions"][0]["confidence"])
            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)

        # Lire le feedback
        feedback_entries = []
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        feedback_entries.append(json.loads(line))

        stats["total_feedback"] = len(feedback_entries)

        # Analyser le feedback
        misdiagnoses = {}
        for feedback in feedback_entries:
            if feedback["user_feedback"] == "correct":
                stats["correct_predictions"] += 1
            elif feedback["user_feedback"] == "incorrect":
                stats["incorrect_predictions"] += 1
                # Noter les erreurs de diagnostic
                if feedback.get("correct_disease"):
                    key = f"{feedback.get('predicted_disease', 'unknown')} → {feedback['correct_disease']}"
                    misdiagnoses[key] = misdiagnoses.get(key, 0) + 1
            elif feedback["user_feedback"] == "unsure":
                stats["unsure_predictions"] += 1

        stats["common_misdiagnoses"] = dict(sorted(misdiagnoses.items(), key=lambda x: x[1], reverse=True)[:10])

        # Sauvegarder les statistiques
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def export_training_data(self, output_file="training_data.jsonl"):
        """
        Export logged predictions as potential training data.

        Args:
            output_file: Output file for training data
        """
        training_data = []

        if self.predictions_file.exists() and self.feedback_file.exists():
            # Charger les prédictions
            predictions = {}
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        predictions[pred["prediction_id"]] = pred

            # Charger le feedback et créer les données d'entraînement
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        feedback = json.loads(line)
                        pred_id = feedback["prediction_id"]

                        if pred_id in predictions:
                            pred = predictions[pred_id]

                            # Créer une entrée d'entraînement
                            training_entry = {
                                "image_path": pred["image_path"],
                                "predicted_disease": pred.get("top_disease"),
                                "predicted_confidence": pred["predictions"][0]["confidence"] if pred["predictions"] else 0,
                                "user_feedback": feedback["user_feedback"],
                                "correct_disease": feedback.get("correct_disease"),
                                "notes": feedback.get("additional_notes"),
                                "timestamp": feedback["timestamp"]
                            }

                            training_data.append(training_entry)

        # Sauvegarder les données d'entraînement
        output_path = self.log_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in training_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ Données d'entraînement exportées: {len(training_data)} entrées dans {output_path}")
        return training_data

    def _detect_prediction_errors(self, diagnosis_result):
        """
        Detect potential prediction errors for intelligent saving.

        Args:
            diagnosis_result: Complete diagnosis result

        Returns:
            dict: Error detection results
        """
        errors = {
            "has_errors": False,
            "error_types": [],
            "severity": "low"
        }

        # Check for unknown disease detection
        if diagnosis_result.get("unknown_disease"):
            errors["has_errors"] = True
            errors["error_types"].append("unknown_disease_detected")
            errors["severity"] = "high"

        # Check FAISS validation warnings
        faiss_validation = diagnosis_result.get("faiss_validation", {})
        if faiss_validation.get("warnings"):
            errors["has_errors"] = True
            errors["error_types"].append("faiss_validation_warning")
            errors["severity"] = "medium"

        # Check low confidence
        predictions = diagnosis_result.get("predictions", [])
        if predictions and predictions[0]["confidence"] < 0.3:
            errors["has_errors"] = True
            errors["error_types"].append("low_confidence")
            errors["severity"] = "medium"

        # Check if similar images show different diseases
        similar_images = diagnosis_result.get("similar_images", [])
        if similar_images and predictions:
            top_pred = predictions[0]["disease"]
            similar_diseases = [img["disease"] for img in similar_images[:3]]
            if top_pred not in similar_diseases:
                errors["has_errors"] = True
                errors["error_types"].append("faiss_disease_mismatch")
                errors["severity"] = "high"

        return errors

    def _auto_create_training_data(self, log_entry):
        """
        Automatically create training data from detected errors.

        Args:
            log_entry: Prediction log entry with error detection
        """
        try:
            training_file = self.log_dir / "auto_training_data.jsonl"

            # Create training entry from error
            training_entry = {
                "image_path": log_entry["image_path"],
                "predicted_disease": log_entry.get("top_disease"),
                "confidence": log_entry["predictions"][0]["confidence"] if log_entry["predictions"] else 0,
                "error_types": log_entry["error_detected"]["error_types"],
                "severity": log_entry["error_detected"]["severity"],
                "unknown_disease": log_entry.get("unknown_disease", False),
                "faiss_validation": log_entry.get("faiss_validation", {}),
                "similar_images": [],  # Would be populated if available
                "timestamp": log_entry["timestamp"],
                "auto_generated": True,
                "needs_user_verification": True
            }

            # Append to training data file
            with open(training_file, 'a', encoding='utf-8') as f:
                json.dump(training_entry, f, ensure_ascii=False)
                f.write('\n')

            print(f"📝 Auto-created training data for error detection: {log_entry['prediction_id']}")

        except Exception as e:
            print(f"Error auto-creating training data: {e}")

    def get_error_statistics(self):
        """
        Get statistics about detected errors for model improvement.

        Returns:
            dict: Error statistics
        """
        error_stats = {
            "total_predictions": 0,
            "predictions_with_errors": 0,
            "error_types_count": {},
            "severity_distribution": {"low": 0, "medium": 0, "high": 0},
            "auto_training_data_created": 0
        }

        if self.predictions_file.exists():
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        error_stats["total_predictions"] += 1

                        error_detected = pred.get("error_detected", {})
                        if error_detected.get("has_errors"):
                            error_stats["predictions_with_errors"] += 1

                            # Count error types
                            for error_type in error_detected.get("error_types", []):
                                error_stats["error_types_count"][error_type] = \
                                    error_stats["error_types_count"].get(error_type, 0) + 1

                            # Count severity
                            severity = error_detected.get("severity", "low")
                            error_stats["severity_distribution"][severity] += 1

        # Count auto-generated training data
        training_file = self.log_dir / "auto_training_data.jsonl"
        if training_file.exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                error_stats["auto_training_data_created"] = sum(1 for line in f if line.strip())

        return error_stats