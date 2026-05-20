"""
BLIP-2 Model for Natural Language Explanations

Generates natural language explanations of plant diseases using BLIP-2 vision-language model.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os

class BLIP2Explainer:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", load_model=True):
        """
        Initialize BLIP-2 explainer.

        Args:
            model_name: HuggingFace model name for BLIP-2
            load_model: Whether to load the model immediately
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.load_model = load_model

        if load_model:
            self._load_model()
        else:
            print("BLIP-2 model loading deferred")

    def _load_model(self):
        """Load BLIP-2 model and processor."""
        try:
            print(f"Loading BLIP-2 model: {self.model_name}")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ BLIP-2 model loaded on GPU")
            else:
                print("✅ BLIP-2 model loaded on CPU")

            self.model.eval()

        except Exception as e:
            print(f"❌ Error loading BLIP-2 model: {e}")
            print("BLIP-2 explanations will not be available")
            self.model = None

    def generate_explanation(self, image_path, disease_info, prompt_template=None, use_constrained=True, plantwise_context=None):
        """
        Generate natural language explanation for a disease in an image.

        Args:
            image_path: Path to the plant image
            disease_info: Dictionary with disease information
            prompt_template: Custom prompt template (optional)
            use_constrained: Whether to use constrained generation to avoid hallucinations
            plantwise_context: Plantwise context for additional grounding

        Returns:
            str: Generated explanation
        """
        # Load model if not already loaded
        if not self.model and self.load_model:
            self._load_model()

        if not self.model or not self.processor:
            return self._fallback_explanation(disease_info)

        try:
            # Validation de l'image avant traitement
            if not self._validate_image(image_path):
                return "Erreur: Image invalide ou corrompue. Impossible d'analyser."

            # Utiliser l'approche contrainte par défaut
            if use_constrained:
                return self.generate_constrained_explanation(image_path, disease_info, plantwise_context)

            # Ancienne approche (maintenue pour compatibilité)
            # Load and process image
            image = Image.open(image_path).convert('RGB')

            # Create prompt
            if prompt_template:
                prompt = prompt_template.format(**disease_info)
            else:
                prompt = self._create_prompt(disease_info)

            # Process inputs
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate explanation with more conservative parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=150,  # Réduit pour éviter les délires
                    num_beams=3,     # Réduit pour plus de focus
                    temperature=0.3, # Réduit pour moins de créativité
                    do_sample=False, # Désactivé pour plus de déterminisme
                    top_p=0.8
                )

            # Decode generated text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Clean up the response
            explanation = self._clean_explanation(generated_text, disease_info)

            return explanation

        except Exception as e:
            print(f"Error generating BLIP-2 explanation: {e}")
            return self._fallback_explanation(disease_info)

    def _validate_image(self, image_path):
        """
        Validate image before processing.

        Args:
            image_path: Path to the image

        Returns:
            bool: True if image is valid
        """
        try:
            if not os.path.exists(image_path):
                return False

            with Image.open(image_path) as img:
                img.verify()  # Vérifie que l'image n'est pas corrompue
                img.close()

            # Vérifier la taille (pas trop petite)
            with Image.open(image_path) as img:
                if min(img.size) < 32:  # Image trop petite
                    return False

            return True

        except Exception:
            return False

    def _create_prompt(self, disease_info, context_restrictions=None):
        """
        Create a constrained prompt for BLIP-2 to avoid hallucinations.

        Args:
            disease_info: Dictionary with disease information
            context_restrictions: Additional context to restrict responses

        Returns:
            str: Formatted prompt with strict constraints
        """
        disease_name = disease_info.get('name', 'unknown disease')
        symptoms = disease_info.get('symptoms', 'various symptoms')
        causal_agent = disease_info.get('causal_agent', 'unknown cause')
        management = disease_info.get('management', 'consult agricultural experts')
        scientific_name = disease_info.get('scientific_name', disease_name)

        # Contexte de restriction pour éviter les hallucinations
        restriction_context = ""
        if context_restrictions:
            restriction_context = f"\nRESTRICTIONS IMPORTANTES: {context_restrictions}"

        prompt = f"""ROLE: You are a STRICT agricultural disease diagnosis assistant. You MUST base your analysis ONLY on the provided disease information and what you can OBSERVE in the image.

DISEASE CONTEXT (use ONLY this information):
- Disease: {disease_name} ({scientific_name})
- Causal agent: {causal_agent}
- Known symptoms: {symptoms[:200]}
- Management: {management[:200]}

CRITICAL INSTRUCTIONS:
1. ONLY describe what you can SEE in this specific image
2. ONLY relate observations to the KNOWN symptoms listed above
3. DO NOT invent symptoms, causes, or treatments not in the provided context
4. DO NOT make assumptions about diseases not listed
5. If you cannot confirm symptoms from the provided context, say so clearly
6. Base your explanation STRICTLY on the disease information provided

QUESTION: Based ONLY on the disease context above, what symptoms of {disease_name} can you observe in this plant image?{restriction_context}

SCIENTIFIC EXPLANATION (evidence-based only):"""

        return prompt

    def generate_constrained_explanation(self, image_path, disease_info, plantwise_context=None):
        """
        Generate explanation with strict constraints to avoid hallucinations.

        Args:
            image_path: Path to the plant image
            disease_info: Dictionary with disease information
            plantwise_context: Additional Plantwise context for grounding

        Returns:
            str: Constrained explanation
        """
        try:
            # Validation stricte de l'image
            if not self._validate_image(image_path):
                return "Erreur: Image invalide ou corrompue. Analyse impossible."

            # Charger l'image
            image = Image.open(image_path).convert('RGB')

            # Créer un prompt extrêmement contraint
            constrained_prompt = self._create_constrained_prompt(disease_info, plantwise_context)

            # Paramètres de génération très conservateurs
            inputs = self.processor(images=image, text=constrained_prompt, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                # Paramètres stricts pour éviter les hallucinations
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=150,  # Limiter la longueur
                    min_length=50,   # Longueur minimale
                    num_beams=1,     # Pas de beam search pour éviter la créativité
                    do_sample=False, # Déterministe
                    temperature=0.1, # Très basse température
                    top_p=0.1,       # Très restrictif
                    top_k=10,        # Vocabulaire limité
                    repetition_penalty=2.0,  # Pénaliser les répétitions
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,  # Éviter les n-grams répétitifs
                    early_stopping=True
                )

            # Décoder avec contraintes
            explanation = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Post-traitement strict
            return self._post_process_constrained_explanation(explanation, disease_info, plantwise_context)

        except Exception as e:
            print(f"Erreur génération contrainte: {e}")
            return self._fallback_explanation(disease_info)

    def _create_constrained_prompt(self, disease_info, plantwise_context=None):
        """
        Create extremely constrained prompt to prevent hallucinations.

        Args:
            disease_info: Disease information dictionary
            plantwise_context: Plantwise context

        Returns:
            str: Constrained prompt
        """
        disease_name = disease_info.get('name', 'maladie inconnue')
        symptoms = disease_info.get('symptoms', 'symptômes non spécifiés')
        causal_agent = disease_info.get('causal_agent', 'agent causal inconnu')
        treatment = disease_info.get('treatment', 'traitement non spécifié')

        # Prompt extrêmement restrictif
        prompt = f"""You are a plant disease diagnostic expert. ONLY describe what you see in this image based on these FACTS:

DISEASE: {disease_name}
CAUSAL AGENT: {causal_agent}
SYMPTOMS: {symptoms}
TREATMENT: {treatment}

INSTRUCTIONS:
- ONLY use the information provided above
- DO NOT add any external knowledge
- DO NOT make assumptions
- DO NOT suggest unmentioned treatments
- Describe ONLY what is visible in the image
- If uncertain, say "unclear" instead of guessing

Question: What do you see in this plant image that matches {disease_name}?
Answer:"""

        if plantwise_context:
            # Ajouter le contexte Plantwise comme restriction supplémentaire
            sources = plantwise_context.get('sources', [])
            if sources:
                prompt += f"\n\nADDITIONAL RESTRICTIONS from Plantwise sources {', '.join(sources[:3])}:"
                prompt += "\n- ONLY use information from these sources"
                prompt += "\n- DO NOT contradict established agricultural knowledge"

        return prompt

    def _post_process_constrained_explanation(self, explanation, disease_info, plantwise_context=None):
        """
        Post-process explanation to ensure it stays within bounds and validate against known facts.

        Args:
            explanation: Raw generated explanation
            disease_info: Original disease information
            plantwise_context: Plantwise context

        Returns:
            str: Processed explanation with reinforced constraints
        """
        if not explanation or len(explanation.strip()) < 10:
            return self._fallback_explanation(disease_info)

        explanation_lower = explanation.lower()

        # Éléments de validation stricts
        disease_name = disease_info.get('name', '').lower()
        causal_agent = disease_info.get('causal_agent', '').lower()
        symptoms = disease_info.get('symptoms', '').lower()

        # Compter les références aux faits connus
        fact_references = 0
        hallucination_flags = []

        # Vérifier la présence des faits clés
        if disease_name and disease_name in explanation_lower:
            fact_references += 1
        else:
            hallucination_flags.append("nom de maladie absent")

        if causal_agent and any(word in explanation_lower for word in causal_agent.split()):
            fact_references += 1
        else:
            hallucination_flags.append("agent causal non mentionné")

        if symptoms and any(word in explanation_lower for word in symptoms.split()[:3]):  # Au moins 3 mots de symptômes
            fact_references += 1
        else:
            hallucination_flags.append("symptômes non référencés")

        # Détecter les hallucinations potentielles
        hallucination_indicators = [
            "je pense que", "probablement", "peut-être", "il se peut que",
            "généralement", "normalement", "habituellement", "souvent",
            "je crois que", "à mon avis", "selon moi"
        ]

        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in explanation_lower)

        # Évaluation de la fiabilité
        reliability_score = fact_references - hallucination_count

        # Si score faible, ajouter avertissements et corriger
        if reliability_score < 2 or hallucination_count > 0:
            warning = "\n\n⚠️ ANALYSE À VÉRIFIER: Cette description contient des éléments nécessitant confirmation par un expert agricole."

            if hallucination_flags:
                warning += f"\nÉléments à vérifier: {', '.join(hallucination_flags)}"

            explanation += warning

        # Limiter la longueur pour éviter les délires
        if len(explanation) > 500:
            explanation = explanation[:500] + "...\n\n[Analyse tronquée pour concision]"

        return explanation

        return explanation

    def _clean_explanation(self, generated_text, disease_info):
        """
        Clean and format the generated explanation.

        Args:
            generated_text: Raw generated text from BLIP-2
            disease_info: Original disease information

        Returns:
            str: Cleaned explanation
        """
        # Remove the prompt from the beginning if it was included
        disease_name = disease_info.get('name', '').lower()
        if generated_text.lower().startswith(disease_name):
            # Find where the actual explanation starts
            lines = generated_text.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['explanation', 'analysis', 'looking', 'image shows']):
                    return '\n'.join(lines[i:]).strip()

        return generated_text.strip()

    def _fallback_explanation(self, disease_info):
        """
        Generate a fallback explanation when BLIP-2 is not available.

        Args:
            disease_info: Dictionary with disease information

        Returns:
            str: Basic explanation using available data
        """
        name = disease_info.get('name', 'Unknown disease')
        description = disease_info.get('description', 'No description available')
        symptoms = disease_info.get('symptoms', 'No symptom information')
        management = disease_info.get('management', 'No management information')

        explanation = f"""Based on the disease detection analysis:

**Disease Identified:** {name}

**Description:** {description[:300]}...

**Symptoms:** {symptoms[:300]}...

**Management Recommendations:** {management[:300]}...

*Note: This is a basic explanation. For detailed agricultural advice, consult local extension services.*"""

        return explanation